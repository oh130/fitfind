"""
Search Engine API — port 8002

엔드포인트:
  POST /search             텍스트/이미지/hybrid 검색
  POST /cross-similarity   상품 간 임베딩 유사도 계산
  GET  /health
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

from search_engine import MultimodalSearchEngine

LOGGER = logging.getLogger(__name__)

EMBEDDING_DIM = 512
TEST_SAMPLE_SIZE = 500
DATA_PATH = Path("/app/data/processed/articles_feature.csv")

# test/production 인덱스 캐시 경로 분리
TEST_INDEX_PATH = Path("/app/data/faiss_index/search_test.index")
TEST_META_PATH  = Path("/app/data/faiss_index/search_test_metadata.json")
PROD_INDEX_PATH = Path("/app/data/faiss_index/search.index")
PROD_META_PATH  = Path("/app/data/faiss_index/search_metadata.json")

# 전역 객체 (search_engine.embedder에서 노출 — 별도 로딩 없음)
clip_model: CLIPModel
clip_processor: CLIPProcessor
search_engine: MultimodalSearchEngine
product_metadata: dict[str, dict[str, Any]] = {}


def image_url_for_article(article_id: str) -> str:
    return f"/api/images/{article_id}"


def _embed_df(engine: MultimodalSearchEngine, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """engine.embedder로 DataFrame을 임베딩하고 FAISS 인덱스를 빌드한다."""
    embeddings: list[np.ndarray] = []
    metadatas: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        article_id = str(row.get("article_id", "")).strip()
        if not article_id:
            continue

        text = " ".join(filter(None, [
            row.get("prod_name", ""),
            row.get("product_type_name", ""),
            row.get("colour_group_name", ""),
            row.get("department_name", ""),
        ])) or article_id

        vec = engine.embedder.embed_text(text)  # CLIP 추론 (전역 모델 재사용)
        if vec.shape[0] != engine.dimension:
            LOGGER.warning("임베딩 shape 불일치 (article_id=%s) — 건너뜀", article_id)
            continue

        embeddings.append(vec)
        metadatas.append({
            "article_id": article_id,
            "product_id": article_id,
            "name": row.get("prod_name", ""),
            "price": 0.0,
            "category": row.get("category", ""),
            "main_category": row.get("main_category", ""),
            "color": row.get("colour_group_name", ""),
        })

    if not embeddings:
        return {}

    engine.build_index(
        embeddings=np.stack(embeddings).astype(np.float32),
        item_ids=list(range(len(embeddings))),
        metadatas=metadatas,
    )
    LOGGER.info("FAISS 인덱스 빌드 완료: %d건", len(embeddings))
    return {str(m["product_id"]): m for m in metadatas}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_model, clip_processor, search_engine, product_metadata

    mode = os.getenv("SEARCH_ENGINE_MODE", "test")
    index_path = TEST_INDEX_PATH if mode == "test" else PROD_INDEX_PATH
    meta_path  = TEST_META_PATH  if mode == "test" else PROD_META_PATH
    sample_size = TEST_SAMPLE_SIZE if mode == "test" else None

    if index_path.exists() and meta_path.exists():
        # 캐시 히트: 저장된 인덱스 로드 (수초, CLIP 1회 로딩)
        LOGGER.info("%s 모드 — 저장된 FAISS 인덱스 로드: %s", mode, index_path)
        search_engine = MultimodalSearchEngine.load_from_artifacts(str(index_path), str(meta_path))
        product_metadata = {
            item.product_id: {"product_id": item.product_id, "name": item.name, "price": item.price}
            for item in search_engine.items
        }
        LOGGER.info("FAISS 인덱스 로드 완료: %d건", len(search_engine.items))

    else:
        # 캐시 미스: 임베딩 후 저장 (CLIP 1회 로딩)
        search_engine = MultimodalSearchEngine("test")  # CLIP 로드 (더미 인덱스로 초기화)

        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
            LOGGER.info("%s 모드 — %d건 임베딩 시작 (최초 1회, 이후 캐시 사용)", mode, len(df))
            product_metadata = _embed_df(search_engine, df)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            search_engine.save_index(str(index_path), str(meta_path))
            LOGGER.info("FAISS 인덱스 저장 완료: %s", index_path)
        else:
            LOGGER.warning("articles_feature.csv 없음 — 더미 12개로 실행")
            product_metadata = {}

    # CLIP을 별도로 로딩하지 않고 search_engine.embedder에서 노출 (중복 로딩 제거)
    clip_model = search_engine.embedder.model
    clip_processor = search_engine.embedder.processor

    yield


app = FastAPI(title="Search Engine", lifespan=lifespan)


# ── 요청/응답 스키마 ──────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = ""
    image_base64: str | None = None
    top_k: int = 10


class CrossSimilarityRequest(BaseModel):
    article_ids: list[str]


# ── 엔드포인트 ────────────────────────────────────────────────

@app.post("/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    start = time.perf_counter()
    has_text = bool(req.query.strip())
    has_image = bool(req.image_base64)

    if not has_text and not has_image:
        raise HTTPException(status_code=400, detail="query 또는 image_base64 중 하나는 필요합니다.")

    text_emb = None
    image_emb = None

    if has_text:
        text_emb = search_engine.embedder.embed_text(req.query)

    if has_image:
        try:
            image_bytes = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_emb = search_engine.embedder.embed_image(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {e}")

    if has_text and has_image:
        search_type = "hybrid"
    elif has_image:
        search_type = "image"
    else:
        search_type = "text"

    #if search_engine is None:
    #raise HTTPException(status_code=503, detail="Search engine not initialized")

    '''if not search_engine._is_built:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "search_type": search_type,
            "results": [],
            "latency_ms": round(latency_ms, 2),
            "total_count": 0,
        }'''

    raw_results = search_engine.search(
        query_type=search_type,
        embedding=text_emb if search_type == "text" else (image_emb if search_type == "image" else None),
        text_embedding=text_emb if search_type == "hybrid" else None,
        image_embedding=image_emb if search_type == "hybrid" else None,
        top_k=req.top_k,
    )

    results = []
    for r in raw_results:
        meta = r.metadata or {}
        product_id = str(meta.get("product_id", r.item_id))
        results.append({
            "product_id": product_id,
            "name": meta.get("name", ""),
            "score": round(r.score, 4),
            "price": float(meta.get("price", 0.0)),
            "image_url": image_url_for_article(product_id),
        })

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "search_type": search_type,
        "results": results,
        "latency_ms": round(latency_ms, 2),
        "total_count": len(results),
    }


def _normalize_article_id(article_id: str) -> str:
    normalized = str(article_id).strip()
    if normalized.isdigit():
        return normalized.zfill(10)
    return normalized


def _item_index_by_article_id() -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, item in enumerate(search_engine.items):
        candidate_ids = {
            str(item.product_id),
            str(item.metadata.get("product_id", "")),
            str(item.metadata.get("article_id", "")),
        }
        for candidate_id in candidate_ids:
            normalized = _normalize_article_id(candidate_id)
            if normalized:
                lookup[normalized] = index
    return lookup


def _embedding_for_item(index: int) -> np.ndarray | None:
    if search_engine._embeddings is not None and 0 <= index < len(search_engine._embeddings):
        return np.asarray(search_engine._embeddings[index], dtype=np.float32)

    if search_engine.index is not None:
        try:
            return np.asarray(search_engine.index.reconstruct(index), dtype=np.float32)
        except Exception:
            LOGGER.debug("Failed to reconstruct embedding for index=%s; falling back to re-embedding.", index)

    if 0 <= index < len(search_engine.items):
        item = search_engine.items[index]
        text = item.description or item.name or item.product_id
        return search_engine.embedder.embed_item(text=text, image=None)
    return None


def _normalized_embedding_for_item(index: int) -> np.ndarray | None:
    embedding = _embedding_for_item(index)
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector / norm


@app.post("/cross-similarity")
async def cross_similarity(req: CrossSimilarityRequest) -> dict[str, Any]:
    """Return pairwise item similarity for outfit-set construction."""

    start = time.perf_counter()
    requested_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw_id in req.article_ids[:100]:
        normalized_id = _normalize_article_id(raw_id)
        if normalized_id and normalized_id not in seen_ids:
            requested_ids.append(normalized_id)
            seen_ids.add(normalized_id)

    index_lookup = _item_index_by_article_id()
    vectors: dict[str, np.ndarray] = {}
    missing_article_ids: list[str] = []
    for article_id in requested_ids:
        item_index = index_lookup.get(article_id)
        if item_index is None:
            missing_article_ids.append(article_id)
            continue
        vector = _normalized_embedding_for_item(item_index)
        if vector is None:
            missing_article_ids.append(article_id)
            continue
        vectors[article_id] = vector

    similarity: dict[str, dict[str, float]] = {}
    for left_id, left_vector in vectors.items():
        row: dict[str, float] = {}
        for right_id, right_vector in vectors.items():
            if left_id == right_id:
                continue
            row[right_id] = round(float(np.dot(left_vector, right_vector)), 6)
        similarity[left_id] = row

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "similarity": similarity,
        "article_ids": list(vectors.keys()),
        "missing_article_ids": missing_article_ids,
        "latency_ms": round(latency_ms, 2),
        "total_count": len(vectors),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_size": len(search_engine) if search_engine._is_built else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=False)

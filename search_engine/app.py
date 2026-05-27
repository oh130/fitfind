"""
Search Engine API on port 8002.

This FastAPI layer intentionally stays thin:
- build/load a reusable MultimodalSearchEngine
- decode request payloads
- delegate text/image/hybrid retrieval to search_engine.py
- expose health and optional item image endpoints
"""

from __future__ import annotations

import base64
import csv
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from search_engine import MultimodalSearchEngine

LOGGER = logging.getLogger(__name__)

FILE_DIR = Path(__file__).resolve().parent
if (FILE_DIR / "data").exists():
    DEFAULT_PROJECT_DATA_DIR = FILE_DIR / "data"
else:
    DEFAULT_PROJECT_DATA_DIR = FILE_DIR.parent / "data"
DEFAULT_CACHE_DIR = DEFAULT_PROJECT_DATA_DIR / "faiss_index"
TEST_INDEX_PATH = DEFAULT_CACHE_DIR / "search_test_v2.index"
TEST_META_PATH = DEFAULT_CACHE_DIR / "search_test_v2_metadata.json"
DEV_INDEX_PATH = DEFAULT_CACHE_DIR / "search_dev_v2.index"
DEV_META_PATH = DEFAULT_CACHE_DIR / "search_dev_v2_metadata.json"
PROD_INDEX_PATH = DEFAULT_CACHE_DIR / "search_v2.index"
PROD_META_PATH = DEFAULT_CACHE_DIR / "search_v2_metadata.json"
PRICE_KRW_FACTOR = 1_000_000

search_engine: MultimodalSearchEngine
article_meta: dict[str, dict[str, Any]] = {}


def _configured_data_root() -> str | None:
    return os.getenv("SEARCH_ENGINE_DATA_ROOT") or os.getenv("DATA_ROOT") or str(DEFAULT_PROJECT_DATA_DIR)


PROCESSED_DATA_DIR = Path(_configured_data_root() or DEFAULT_PROJECT_DATA_DIR)
if PROCESSED_DATA_DIR.name != "processed":
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / "processed"
ARTICLES_PATH = PROCESSED_DATA_DIR / "articles_feature.csv"
ITEM_FEATURES_CANDIDATES = [
    PROCESSED_DATA_DIR / "item_features_test.csv",
    PROCESSED_DATA_DIR / "item_features_dev.csv",
    PROCESSED_DATA_DIR / "item_features.csv",
]


def _artifact_paths(mode: str) -> tuple[Path, Path]:
    if mode == "test":
        return TEST_INDEX_PATH, TEST_META_PATH
    if mode == "dev":
        return DEV_INDEX_PATH, DEV_META_PATH
    return PROD_INDEX_PATH, PROD_META_PATH


def _normalize_article_id(value: object) -> str:
    raw = str(value or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return raw
    return digits[-10:].zfill(10)


def _load_article_meta() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    if not ARTICLES_PATH.exists():
        return meta

    with ARTICLES_PATH.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            article_id = str(row.get("article_id", "")).strip()
            if not article_id:
                continue
            meta[article_id] = {
                "name": row.get("prod_name", "") or row.get("name", ""),
                "category": row.get("category", ""),
                "color": row.get("color", ""),
                "product_type": row.get("product_type", ""),
                "price": 0,
            }

    for path in ITEM_FEATURES_CANDIDATES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                article_id = str(row.get("article_id", "")).strip()
                if article_id not in meta:
                    continue
                try:
                    raw_price = float(row.get("avg_price", 0) or 0)
                    meta[article_id]["price"] = int(raw_price * PRICE_KRW_FACTOR)
                except (TypeError, ValueError):
                    continue
        break
    return meta


def _item_image_bytes(item: Any) -> bytes | None:
    image_base64 = str(getattr(item, "metadata", {}).get("_image_base64", "")).strip()
    if image_base64:
        try:
            return base64.b64decode(image_base64)
        except Exception:
            pass

    image_path = _resolve_item_image_path(item)
    if image_path is not None:
        try:
            return image_path.read_bytes()
        except OSError:
            return None

    if item.image is not None:
        buffer = io.BytesIO()
        item.image.save(buffer, format="PNG")
        return buffer.getvalue()
    return None


def _artifacts_match(meta_path: Path, mode: str) -> bool:
    return meta_path.exists() and MultimodalSearchEngine._cached_artifacts_are_current(meta_path, mode)


def _build_or_load_engine(mode: str) -> MultimodalSearchEngine:
    data_root = _configured_data_root()
    index_path, meta_path = _artifact_paths(mode)
    if index_path.exists() and meta_path.exists() and not _artifacts_match(meta_path, mode):
        LOGGER.warning("Cached %s artifacts at %s do not match expected format. Rebuilding.", mode, index_path)
    return MultimodalSearchEngine.load_cached_or_build(
        mode=mode,
        data_root=data_root,
        cache_dir=index_path.parent,
    )


def image_url_for_article(article_id: str) -> str:
    return f"/api/images/{article_id}"


def _decode_request_image_bytes(image_base64: str | None) -> bytes | None:
    if not image_base64:
        return None
    try:
        return base64.b64decode(image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image_base64: {exc}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine, article_meta

    mode = os.getenv("SEARCH_ENGINE_MODE", "test").strip().lower() or "test"
    search_engine = _build_or_load_engine(mode)
    article_meta = _load_article_meta()
    _warm_up_search_engine()
    yield


app = FastAPI(title="Search Engine", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = ""
    image_base64: str | None = None
    top_k: int = 10
    use_cache: bool = True


def _base64_payload_from_item(item: Any) -> tuple[str, str]:
    image_base64 = str(getattr(item, "metadata", {}).get("_image_base64", "")).strip()
    if image_base64:
        return image_base64, "image/jpeg"

    image_path = _resolve_item_image_path(item)
    if image_path is not None:
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        return base64.b64encode(image_path.read_bytes()).decode("utf-8"), mime_type

    if item.image is not None:
        buffer = io.BytesIO()
        item.image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/png"

    raise HTTPException(status_code=404, detail="Image not available for this item")


def _resolve_item_image_path(item: Any) -> Path | None:
    if item.image_path:
        candidate = Path(item.image_path)
        if candidate.exists():
            return candidate

    normalized_id = _normalize_article_id(getattr(item, "product_id", ""))
    if not normalized_id:
        return None

    candidate_roots = []
    try:
        candidate_roots.extend(search_engine._candidate_image_roots())  # type: ignore[attr-defined]
    except Exception:
        pass

    for root in candidate_roots:
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = Path(root) / normalized_id[:3] / f"{normalized_id}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _warm_up_search_engine() -> None:
    try:
        search_engine.search(query="black dress", top_k=1)
        sample_item = next((item for item in search_engine.items if item.image is not None or item.image_path), None)
        if sample_item is None:
            return
        image_bytes = _item_image_bytes(sample_item)
        if not image_bytes:
            return
        # Prime image-only and hybrid paths so first uncached user queries do not pay
        # the one-time model/kernel warm-up cost.
        search_engine.search(image_bytes=image_bytes, top_k=1)
        warm_query = sample_item.name or sample_item.description or "fashion item"
        search_engine.search(query=warm_query, image_bytes=image_bytes, top_k=1)
    except Exception as exc:
        LOGGER.warning("Search engine warm-up skipped due to error: %s", exc)


@app.post("/search")
@app.post("/api/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    if not req.query.strip() and not req.image_base64:
        raise HTTPException(status_code=400, detail="query or image_base64 is required")

    started = time.perf_counter()
    image_bytes = _decode_request_image_bytes(req.image_base64)
    response = search_engine.search(
        query=req.query,
        image_bytes=image_bytes,
        top_k=max(1, int(req.top_k)),
        use_cache=bool(req.use_cache),
    )
    response["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)

    for result in response.get("results", []):
        product_id = str(result.get("product_id", "")).strip()
        if product_id:
            meta = article_meta.get(product_id, {})
            result["image_url"] = image_url_for_article(product_id)
            result["name"] = meta.get("name") or result.get("name") or product_id
            result["category"] = meta.get("category", "")
            result["color"] = meta.get("color", "")
            result["product_type"] = meta.get("product_type", "")
            result["price"] = meta.get("price", 0)
        else:
            result.setdefault("category", "")
            result.setdefault("color", "")
            result.setdefault("product_type", "")
    return response


@app.get("/api/images/{article_id}")
async def get_item_image(
    article_id: str,
    request: Request,
    format: str = Query("binary", pattern="^(binary|base64)$"),
):
    item = search_engine.find_item(article_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    wants_base64 = format == "base64" or "application/json" in request.headers.get("accept", "").lower()
    if wants_base64:
        image_base64, mime_type = _base64_payload_from_item(item)
        normalized_id = _normalize_article_id(article_id)
        return JSONResponse(
            {
                "article_id": str(article_id),
                "normalized_article_id": normalized_id or str(article_id),
                "image_base64": image_base64,
                "content_type": mime_type,
            }
        )

    resolved_path = _resolve_item_image_path(item)
    if resolved_path is not None:
        return FileResponse(resolved_path)

    if item.image is not None:
        buffer = io.BytesIO()
        item.image.save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")

    raise HTTPException(status_code=404, detail="Image not available for this item")


@app.get("/health")
async def health() -> dict[str, Any]:
    image_backed_items = 0
    for item in search_engine.items:
        if item.image is not None or item.image_path:
            image_backed_items += 1

    return {
        "status": "ok",
        "mode": search_engine.mode,
        "index_size": len(search_engine) if search_engine._is_built else 0,
        "image_backed_items": image_backed_items,
        "data_root": str(search_engine.data_root),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=False)

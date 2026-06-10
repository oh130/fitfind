from __future__ import annotations

import base64
import contextlib
import io
import json
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

try:
    from query_expansion import expand_fashion_query
except ImportError:  # pragma: no cover - supports package-style imports in tests
    from search_engine.query_expansion import expand_fashion_query  # type: ignore[no-redef]

LOGGER = logging.getLogger(__name__)

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
DEFAULT_TOP_K = 10
DEFAULT_PORT = 8002
DEFAULT_RANDOM_SEED = 42
DEFAULT_DEV_SAMPLE_SIZE = 600
SEARCH_PAIR_MANIFEST_DEV = "search_pairs_dev.csv"
SEARCH_PAIR_MANIFEST_PROD = "search_pairs_production.csv"
SEARCH_INDEX_FILENAMES = {
    "test": ("search_test_v2.index", "search_test_v2_metadata.json"),
    "dev": ("search_dev_v2.index", "search_dev_v2_metadata.json"),
    "production": ("search_v2.index", "search_v2_metadata.json"),
}
SEARCH_INDEX_FORMAT = "clip_multimodal_v4"
SUPPORTED_SEARCH_INDEX_FORMATS = {SEARCH_INDEX_FORMAT}
IMAGE_COLOR_RERANK_MIN_POOL = int(os.getenv("IMAGE_COLOR_RERANK_MIN_POOL", "100"))
IMAGE_COLOR_RERANK_MULTIPLIER = int(os.getenv("IMAGE_COLOR_RERANK_MULTIPLIER", "4"))
IMAGE_COLOR_EXACT_BOOST = float(os.getenv("IMAGE_COLOR_EXACT_BOOST", "1.08"))
IMAGE_COLOR_NEUTRAL_BOOST = float(os.getenv("IMAGE_COLOR_NEUTRAL_BOOST", "1.03"))
IMAGE_COLOR_MISMATCH_PENALTY = float(os.getenv("IMAGE_COLOR_MISMATCH_PENALTY", "0.95"))
IMAGE_COLOR_RERANK_MIN_CONFIDENCE = float(os.getenv("IMAGE_COLOR_RERANK_MIN_CONFIDENCE", "0.35"))
CSV_STRING_COLUMNS = {
    "article_id": str,
    "product_code": str,
    "prod_name": str,
    "product_type_name": str,
    "product_group_name": str,
    "graphical_appearance_name": str,
    "colour_group_name": str,
    "perceived_colour_master_name": str,
    "perceived_colour_value_name": str,
    "department_name": str,
    "index_name": str,
    "index_group_name": str,
    "section_name": str,
    "garment_group_name": str,
    "detail_desc": str,
    "category": str,
    "main_category": str,
    "color": str,
    "category_l1": str,
    "category_l2": str,
    "category_l3": str,
}


class _NumpyInnerProductIndex:
    # FAISS를 쓸 수 없는 환경에서만 사용하는 최소 기능 대체 인덱스다.
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of shape (n, {self.dimension})")
        self.vectors = np.vstack([self.vectors, vectors]) if len(self.vectors) else vectors.copy()

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if self.vectors.size == 0:
            empty_scores = np.empty((query_vec.shape[0], 0), dtype=np.float32)
            empty_indices = np.empty((query_vec.shape[0], 0), dtype=np.int64)
            return empty_scores, empty_indices

        scores = query_vec @ self.vectors.T
        k = min(int(top_k), self.vectors.shape[0])
        indices = np.argsort(-scores, axis=1)[:, :k]
        rows = np.arange(scores.shape[0])[:, None]
        top_scores = scores[rows, indices]
        return top_scores.astype(np.float32), indices.astype(np.int64)


@dataclass
class SearchItem:
    product_id: str
    name: str
    price: float
    description: str
    image: Optional[Image.Image]
    image_path: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    item_id: str
    score: float
    metadata: Dict[str, Any]


COLOR_FAMILY_ALIASES: dict[str, str] = {
    "black": "black",
    "grey": "gray",
    "gray": "gray",
    "silver": "gray",
    "white": "white",
    "off white": "white",
    "cream": "white",
    "beige": "beige",
    "brown": "brown",
    "blue": "blue",
    "navy": "blue",
    "turquoise": "blue",
    "red": "red",
    "burgundy": "red",
    "pink": "pink",
    "purple": "purple",
    "lilac": "purple",
    "green": "green",
    "khaki": "green",
    "yellow": "yellow",
    "orange": "orange",
}
NEUTRAL_COLOR_FAMILIES = {"black", "gray", "white"}


def normalize_color_family(color_text: Any) -> Optional[str]:
    text = str(color_text or "").strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z\s/,-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    for alias, family in COLOR_FAMILY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return family
    return None


def color_family_from_rgb(red: int, green: int, blue: int) -> str:
    r, g, b = red / 255.0, green / 255.0, blue / 255.0
    max_value = max(r, g, b)
    min_value = min(r, g, b)
    chroma = max_value - min_value

    if max_value < 0.22:
        return "black"
    if chroma < 0.10:
        if max_value > 0.86:
            return "white"
        return "gray"

    if max_value == r:
        hue = ((g - b) / chroma) % 6
    elif max_value == g:
        hue = ((b - r) / chroma) + 2
    else:
        hue = ((r - g) / chroma) + 4
    hue_degrees = hue * 60

    if hue_degrees < 20 or hue_degrees >= 345:
        return "red"
    if hue_degrees < 45:
        return "orange"
    if hue_degrees < 70:
        return "yellow"
    if hue_degrees < 165:
        return "green"
    if hue_degrees < 255:
        return "blue"
    if hue_degrees < 295:
        return "purple"
    return "pink"


def dominant_color_signal_from_image(image: Image.Image) -> tuple[Optional[str], float]:
    try:
        rgb_image = image.convert("RGB")
    except Exception:
        return None, 0.0

    width, height = rgb_image.size
    if width <= 0 or height <= 0:
        return None, 0.0

    left = int(width * 0.15)
    top = int(height * 0.15)
    right = max(left + 1, int(width * 0.85))
    bottom = max(top + 1, int(height * 0.85))
    rgb_image = rgb_image.crop((left, top, right, bottom)).resize((64, 64))

    counts: dict[str, float] = {}
    for red, green, blue in rgb_image.getdata():
        family = color_family_from_rgb(int(red), int(green), int(blue))
        counts[family] = counts.get(family, 0.0) + 1.0

    if not counts:
        return None, 0.0

    total = sum(counts.values())
    white_share = counts.get("white", 0.0) / total if total else 0.0
    if white_share > 0.45:
        non_white = {family: count for family, count in counts.items() if family != "white"}
        if non_white and (sum(non_white.values()) / total) >= 0.08:
            family = max(non_white, key=non_white.get)
            return family, non_white[family] / total
    family = max(counts, key=counts.get)
    return family, counts[family] / total


def dominant_color_family_from_image(image: Image.Image) -> Optional[str]:
    family, _confidence = dominant_color_signal_from_image(image)
    return family


def encode_image_file(path: Path | str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def normalize_article_id(value: object) -> str:
    raw = str(value or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return raw
    return digits[-10:].zfill(10)


class OpenAIClipEmbedder:
    # 텍스트/이미지를 OpenAI CLIP 공통 임베딩 공간으로 변환한다.
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        device: Optional[str] = None,
        fail_on_load_error: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fail_on_load_error = fail_on_load_error
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.dim: Optional[int] = None
        self._load_error: Optional[Exception] = None
        self._text_cache: Dict[str, np.ndarray] = {}
        self._image_cache: Dict[str, np.ndarray] = {}
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None and self.dim is not None:
            return
        try:
            LOGGER.info("Loading CLIP model: %s", self.model_name)
            # 오프라인 캐시를 먼저 확인하고, 없을 때만 원격 다운로드를 시도한다.
            load_attempts = (
                {"local_files_only": True},
                {"local_files_only": False},
            )
            last_error: Optional[Exception] = None
            for kwargs in load_attempts:
                try:
                    with self._suppress_model_load_output():
                        self.processor = CLIPProcessor.from_pretrained(self.model_name, **kwargs)
                        self.model = CLIPModel.from_pretrained(self.model_name, **kwargs)
                    break
                except Exception as exc:
                    last_error = exc
                    self.processor = None
                    self.model = None
                    if kwargs.get("local_files_only"):
                        LOGGER.info("CLIP model not found in local cache, retrying with remote download enabled")
                    else:
                        raise
            if self.model is None or self.processor is None:
                raise last_error or RuntimeError(f"Unable to load CLIP model '{self.model_name}'")
            self.model.to(self.device)
            self.model.eval()
            self.dim = int(getattr(self.model.config, "projection_dim", 512))
            LOGGER.info("CLIP model ready on %s with dim=%d", self.device, self.dim)
        except Exception as exc:  # pragma: no cover
            self._load_error = exc
            message = (
                f"Failed to load OpenAI CLIP model '{self.model_name}'. "
                "This project requires a trained CLIP model and does not support an untrained fallback. "
                f"Original error: {exc}"
            )
            LOGGER.exception(message)
            if self.fail_on_load_error:
                raise RuntimeError(message) from exc

    @staticmethod
    @contextlib.contextmanager
    def _suppress_model_load_output():
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            from transformers.utils import logging as transformers_logging

            previous_level = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
        except Exception:
            transformers_logging = None
            previous_level = None

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                yield
        finally:
            if transformers_logging is not None and previous_level is not None:
                try:
                    transformers_logging.set_verbosity(previous_level)
                except Exception:
                    pass

    def _require_components(self) -> Tuple[CLIPModel, CLIPProcessor, int]:
        self._ensure_loaded()
        if self.model is None or self.processor is None or self.dim is None:
            message = (
                f"OpenAI CLIP model '{self.model_name}' is unavailable. "
                "A trained CLIP checkpoint is required for this search engine."
            )
            if self._load_error is not None:
                message = f"{message} Last error: {self._load_error}"
            raise RuntimeError(message)
        return self.model, self.processor, self.dim

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        model, processor, dim = self._require_components()
        value = (text or "").strip()
        if not value:
            return np.zeros(dim, dtype=np.float32)
        if use_cache:
            cached = self._text_cache.get(value)
            if cached is not None:
                return cached.copy()
        # text tower의 pooled output을 projection layer로 512차원 CLIP 공간에 투영한다.
        inputs = processor(text=[value], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        with torch.no_grad():
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            pooled = text_outputs[1]
            features = model.text_projection(pooled)
            features = torch.nn.functional.normalize(features, dim=-1)
        embedding = self._normalize(features[0].detach().cpu().numpy())
        if use_cache:
            self._text_cache[value] = embedding.copy()
        return embedding

    def embed_image(self, image: Image.Image) -> np.ndarray:
        model, processor, _ = self._require_components()
        rgb_image = image.convert("RGB")
        # vision tower의 pooled output을 projection layer로 512차원 CLIP 공간에 투영한다.
        inputs = processor(images=rgb_image, return_tensors="pt")
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_outputs[1]
            features = model.visual_projection(pooled)
            features = torch.nn.functional.normalize(features, dim=-1)
        return self._normalize(features[0].detach().cpu().numpy())

    @staticmethod
    def _hash_image_bytes(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def register_image_bytes_embedding(self, image_bytes: bytes, embedding: np.ndarray) -> None:
        self._image_cache[self._hash_image_bytes(image_bytes)] = self._normalize(embedding).copy()

    def register_image_file_embedding(self, image_path: str | Path, embedding: np.ndarray) -> None:
        path = Path(image_path)
        if not path.exists():
            return
        try:
            self.register_image_bytes_embedding(path.read_bytes(), embedding)
        except Exception:
            return

    def embed_image_bytes(self, image_bytes: bytes, use_cache: bool = True) -> np.ndarray:
        _, _, dim = self._require_components()
        if not image_bytes:
            return np.zeros(dim, dtype=np.float32)
        cache_key = self._hash_image_bytes(image_bytes)
        if use_cache:
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        embedding = self.embed_image(image)
        if use_cache:
            self._image_cache[cache_key] = embedding.copy()
        return embedding

    def combine_embeddings(self, vectors: Sequence[np.ndarray]) -> np.ndarray:
        # hybrid 검색은 텍스트/이미지 벡터를 평균낸 뒤 다시 정규화한다.
        _, _, dim = self._require_components()
        usable = [self._normalize(vec) for vec in vectors if vec is not None and np.any(vec)]
        if not usable:
            return np.zeros(dim, dtype=np.float32)
        combined = np.mean(np.stack(usable).astype(np.float32), axis=0)
        return self._normalize(combined)

    def embed_item(self, text: str, image: Optional[Image.Image] = None) -> np.ndarray:
        vectors: List[np.ndarray] = []
        if text and text.strip():
            vectors.append(self.embed_text(text))
        if image is not None:
            vectors.append(self.embed_image(image))
        return self.combine_embeddings(vectors)

    def embed_query(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, str]:
        has_text = bool(text and text.strip())
        has_image = image is not None or bool(image_bytes)
        image_embedding = None
        if image_bytes:
            image_embedding = self.embed_image_bytes(image_bytes, use_cache=use_cache)
        elif image is not None:
            image_embedding = self.embed_image(image)
        if has_text and has_image:
            return self.combine_embeddings([self.embed_text(text or "", use_cache=use_cache), image_embedding]), "hybrid"
        if has_image:
            return image_embedding, "image"
        return self.embed_text(text or "", use_cache=use_cache), "text"

    def project_external_embedding(self, embedding: np.ndarray, modality: str = "text") -> np.ndarray:
        # app.py 등 외부 코드가 만든 임베딩도 CLIP 검색 공간 차원에 맞춰 흡수한다.
        model, _, dim = self._require_components()
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.size == 0:
            return np.zeros(dim, dtype=np.float32)

        if vector.ndim == 1 and vector.shape[0] == dim:
            return self._normalize(vector)

        if vector.ndim >= 2 and vector.shape[-1] == dim:
            flattened = vector.reshape(-1, dim)
            return self._normalize(flattened.mean(axis=0))

        if modality == "image":
            hidden_dim = int(getattr(model.config.vision_config, "hidden_size", dim))
            projection = model.visual_projection
        else:
            hidden_dim = int(getattr(model.config.text_config, "hidden_size", dim))
            projection = model.text_projection

        if vector.ndim == 1 and vector.shape[0] == hidden_dim:
            tensor = torch.from_numpy(vector).to(self.device).unsqueeze(0)
            with torch.no_grad():
                projected = projection(tensor)
                projected = torch.nn.functional.normalize(projected, dim=-1)
            return self._normalize(projected[0].detach().cpu().numpy())

        if vector.ndim >= 2 and vector.shape[-1] == hidden_dim:
            flattened = vector.reshape(-1, hidden_dim)
            pooled = flattened.mean(axis=0, dtype=np.float32)
            tensor = torch.from_numpy(pooled).to(self.device).unsqueeze(0)
            with torch.no_grad():
                projected = projection(tensor)
                projected = torch.nn.functional.normalize(projected, dim=-1)
            return self._normalize(projected[0].detach().cpu().numpy())

        flattened = vector.reshape(-1).astype(np.float32)
        if flattened.shape[0] > dim:
            flattened = flattened[:dim]
        elif flattened.shape[0] < dim:
            flattened = np.pad(flattened, (0, dim - flattened.shape[0]))
        return self._normalize(flattened)


class MultimodalSearchEngine:
    """OpenAI CLIP + FAISS(HNSW) based multimodal search engine."""

    def __init__(
        self,
        mode: str = "test",
        data_root: Optional[str] = None,
        top_k_default: int = DEFAULT_TOP_K,
        clip_model_name: str = CLIP_MODEL_NAME,
    ) -> None:
        self.mode = (mode or "test").lower().strip()
        self.data_root = self._resolve_runtime_data_root(data_root)
        self.top_k_default = int(top_k_default)
        self.random_seed = int(os.getenv("SEARCH_ENGINE_RANDOM_SEED", str(DEFAULT_RANDOM_SEED)))
        self.dev_sample_size = int(os.getenv("SEARCH_ENGINE_DEV_SAMPLE_SIZE", str(DEFAULT_DEV_SAMPLE_SIZE)))
        self.embedder = OpenAIClipEmbedder(model_name=clip_model_name)
        self.items: List[SearchItem] = []
        self.item_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._text_embeddings: Optional[np.ndarray] = None
        self._image_embeddings: Optional[np.ndarray] = None
        self._image_item_indices: np.ndarray = np.empty((0,), dtype=np.int64)
        self._image_hash_to_item_index: Dict[str, int] = {}
        self._aux_vectors_path: Optional[Path] = None
        self._source_articles_path: Optional[Path] = None
        self.index: Any = None
        self.text_index: Any = None
        self.image_index: Any = None
        self.dimension = int(self.embedder.dim or 512)
        self._is_built = False

        if self.mode == "production":
            self.items = self._load_production_items()
        elif self.mode == "dev":
            self.items = self._load_dev_items()
        else:
            self.items = self._build_dummy_items()

        self._build_index()

    @staticmethod
    def _resolve_data_root(data_root: Optional[str]) -> Path:
        if data_root:
            path = Path(data_root)
            if (path / "processed" / "item_master_dev.csv").exists() or (path / "processed" / "articles_feature.csv").exists():
                return path / "processed"
            return path

        file_dir = Path(__file__).resolve().parent
        project_root = file_dir.parent
        # docker-compose와 로컬 실행을 모두 지원하기 위해 후보 경로를 순서대로 확인한다.
        candidates = [
            os.getenv("SEARCH_ENGINE_DATA_ROOT"),
            os.getenv("DATA_ROOT"),
            file_dir / "data" / "processed",
            project_root / "data" / "processed",
            file_dir / "data",
            project_root / "data",
            Path("/app/data"),
            Path("/app/data/processed"),
        ]

        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if (path / "item_master_dev.csv").exists() or (path / "articles_feature.csv").exists() or (path / "item_master.csv").exists():
                return path

        for candidate in candidates:
            if candidate:
                return Path(candidate)
        return project_root / "data" / "processed"

    @staticmethod
    def _resolve_runtime_data_root(data_root: Optional[str]) -> Path:
        if data_root:
            path = Path(data_root)
            if (path / "articles_feature.csv").exists():
                return path
            if (path / "item_master_dev.csv").exists():
                return path
            if (path / "item_master.csv").exists():
                return path
            if (path / "processed" / "articles_feature.csv").exists():
                return path / "processed"
            if (path / "processed" / "item_master_dev.csv").exists():
                return path / "processed"
            return path

        env_root = os.getenv("SEARCH_ENGINE_DATA_ROOT") or os.getenv("DATA_ROOT")
        file_dir = Path(__file__).resolve().parent
        project_root = file_dir.parent
        candidates = [
            Path(env_root) if env_root else None,
            file_dir / "data" / "processed",
            project_root / "data" / "processed",
            file_dir / "data",
            project_root / "data",
            Path("/app/data"),
            Path("/app/data/processed"),
        ]

        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if (path / "articles_feature.csv").exists():
                return path
            if (path / "item_master_dev.csv").exists():
                return path
            if (path / "item_master.csv").exists():
                return path

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return Path(candidate)
        return project_root / "data" / "processed"

    def _build_dummy_items(self) -> List[SearchItem]:
        # test 모드에서는 더미 이미지와 설명을 직접 만들어 즉시 검색 가능 상태로 만든다.
        palette = [
            (231, 76, 60),
            (52, 152, 219),
            (46, 204, 113),
            (155, 89, 182),
            (241, 196, 15),
            (230, 126, 34),
            (236, 240, 241),
            (52, 73, 94),
        ]
        samples = [
            ("100001", "Women Casual White Shirt", 29.9, "women apparel shirt white cotton casual"),
            ("100002", "Men Denim Jacket", 79.0, "men outerwear denim jacket blue casual"),
            ("100003", "Slim Fit Black Jeans", 49.5, "men bottoms black jeans slim fit"),
            ("100004", "Floral Summer Dress", 59.9, "women dress floral summer lightweight"),
            ("100005", "Kids Sports Sneakers", 39.9, "kids shoes sporty comfortable white"),
            ("100006", "Warm Wool Coat", 129.0, "women outerwear coat wool winter beige"),
            ("100007", "Canvas Shoulder Bag", 34.9, "accessories bag canvas casual neutral"),
            ("100008", "Striped Knit Sweater", 44.9, "women knit sweater striped warm casual"),
            ("100009", "Formal Navy Trousers", 54.9, "men trousers formal navy office"),
            ("100010", "Printed Long Sleeve Tee", 24.9, "men t-shirt printed long sleeve casual"),
            ("100011", "Pleated Midi Skirt", 39.9, "women skirt pleated midi elegant"),
            ("100012", "Running Shorts", 19.9, "men shorts sport running breathable"),
        ]

        items: List[SearchItem] = []
        for idx, (product_id, name, price, desc) in enumerate(samples):
            color = palette[idx % len(palette)]
            img = Image.new("RGB", (128, 128), color)
            draw = ImageDraw.Draw(img)
            draw.rectangle((16, 16, 112, 112), outline=(255, 255, 255), width=4)
            draw.text((14, 52), name[:12], fill=(255, 255, 255))
            items.append(
                SearchItem(
                    product_id=product_id,
                    name=name,
                    price=float(price),
                    description=desc,
                    image=img,
                    image_path=None,
                    metadata={
                        "mode": "test",
                        "product_id": product_id,
                        "category": name.split()[0].lower(),
                        "name": name,
                        "description": desc,
                        "price": float(price),
                    },
                )
            )
        LOGGER.info("Prepared %d dummy items for test mode", len(items))
        return items

    def _article_table_candidates(self, mode_label: str) -> List[Path]:
        local_root = Path(__file__).resolve().parent
        if mode_label == "dev":
            preferred = [
                self.data_root / "articles_feature.csv",
                self.data_root / "item_master_dev.csv",
                self.data_root / SEARCH_PAIR_MANIFEST_DEV,
                local_root / SEARCH_PAIR_MANIFEST_DEV,
            ]
        elif mode_label == "production":
            preferred = [
                local_root / SEARCH_PAIR_MANIFEST_PROD,
                self.data_root / SEARCH_PAIR_MANIFEST_PROD,
                self.data_root / "item_master.csv",
                self.data_root / "articles_feature.csv",
                self.data_root / "item_master_dev.csv",
            ]
        else:
            preferred = [self.data_root / "item_master_dev.csv", self.data_root / "articles_feature.csv"]
        return preferred

    def _feature_article_candidates(self, mode_label: str) -> List[Path]:
        base_candidates = [
            self.data_root / "articles_feature.csv",
            self.data_root / "item_features_dev.csv",
            self.data_root / "item_features.csv",
            self.data_root / "item_master_dev.csv",
            self.data_root / "item_master.csv",
            self.data_root.parent / "processed" / "articles_feature.csv",
            self.data_root.parent / "processed" / "item_features_dev.csv",
            self.data_root.parent / "processed" / "item_features.csv",
            self.data_root.parent / "processed" / "item_master_dev.csv",
            self.data_root.parent / "processed" / "item_master.csv",
        ]
        if mode_label == "production":
            return base_candidates
        if mode_label == "dev":
            return base_candidates
        return []

    def _merge_article_features(self, raw_articles: pd.DataFrame, mode_label: str) -> pd.DataFrame:
        merged = raw_articles.copy()
        merged["article_id"] = merged["article_id"].apply(self._normalize_article_id)

        for path in self._feature_article_candidates(mode_label):
            if not path.exists():
                continue
            try:
                feature_df = pd.read_csv(path, dtype=CSV_STRING_COLUMNS).fillna("")
            except Exception as exc:
                LOGGER.warning("Failed to load feature article table from %s: %s", path, exc)
                continue

            if "article_id" not in feature_df.columns:
                continue

            feature_df["article_id"] = feature_df["article_id"].apply(self._normalize_article_id)
            feature_df = feature_df.drop_duplicates(subset=["article_id"])
            merged = merged.merge(feature_df, on="article_id", how="left", suffixes=("", "_feature"))

            for column in list(merged.columns):
                if not column.endswith("_feature"):
                    continue
                base_column = column[:-8]
                if base_column not in merged.columns:
                    merged[base_column] = merged[column]
                else:
                    merged[base_column] = merged[base_column].where(
                        merged[base_column].astype(str).str.strip().ne(""),
                        merged[column],
                    )
                merged = merged.drop(columns=[column])
            return merged.fillna("")

        return merged.fillna("")

    def _load_article_table(self, mode_label: str) -> Tuple[pd.DataFrame, Path]:
        for path in self._article_table_candidates(mode_label):
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path, dtype=CSV_STRING_COLUMNS).fillna("")
                if not frame.empty:
                    return frame, path
            except Exception as exc:
                LOGGER.warning("Failed to load article table from %s: %s", path, exc)
        raise FileNotFoundError(
            f"No processed article table found for mode={mode_label} under {self.data_root}. "
            "Expected item_master_dev.csv, item_master.csv, or articles_feature.csv."
        )

    @staticmethod
    def _normalize_article_id(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        digits = "".join(ch for ch in raw if ch.isdigit())
        if not digits:
            return raw
        return digits[-10:].zfill(10)

    @staticmethod
    def _extract_price_from_row(row: pd.Series) -> float:
        for key in ("price", "price_mean", "avg_price"):
            value = row.get(key, "")
            try:
                if value != "":
                    return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _manifest_output_path(self, mode_label: str) -> Path:
        processed_root = Path(__file__).resolve().parent
        filename = SEARCH_PAIR_MANIFEST_DEV if mode_label == "dev" else SEARCH_PAIR_MANIFEST_PROD
        return processed_root / filename

    def _write_pair_manifest(self, items: Sequence[SearchItem], mode_label: str, source_articles_path: Path) -> None:
        manifest_path = self._manifest_output_path(mode_label)
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            rows: List[Dict[str, Any]] = []
            for item in items:
                metadata = dict(item.metadata or {})
                rows.append(
                    {
                        "article_id": item.product_id,
                        "product_id": item.product_id,
                        "prod_name": metadata.get("product_name", item.name),
                        "product_type_name": metadata.get("product_type_name", ""),
                        "product_group_name": metadata.get("product_group_name", ""),
                        "graphical_appearance_name": metadata.get("graphical_appearance_name", ""),
                        "colour_group_name": metadata.get("colour_group_name", ""),
                        "perceived_colour_master_name": metadata.get("perceived_colour_master_name", ""),
                        "perceived_colour_value_name": metadata.get("perceived_colour_value_name", ""),
                        "department_name": metadata.get("department_name", ""),
                        "index_name": metadata.get("index_name", ""),
                        "index_group_name": metadata.get("index_group_name", ""),
                        "section_name": metadata.get("section_name", ""),
                        "garment_group_name": metadata.get("garment_group_name", ""),
                        "detail_desc": metadata.get("detail_desc", ""),
                        "category": metadata.get("category", ""),
                        "main_category": metadata.get("main_category", ""),
                        "category_l1": metadata.get("category_l1", ""),
                        "category_l2": metadata.get("category_l2", ""),
                        "category_l3": metadata.get("category_l3", ""),
                        "price_mean": item.price,
                        "image_path": item.image_path or "",
                        "image_available": bool(item.image_path or item.image is not None),
                        "source_articles_path": str(source_articles_path),
                    }
                )
            pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
            LOGGER.info("Wrote %s search pair manifest to %s", mode_label, manifest_path)
        except Exception as exc:
            LOGGER.warning("Failed to write %s pair manifest: %s", mode_label, exc)

    def _expand_query_terms(self, text: str) -> str:
        return expand_fashion_query(text)

    def _build_item_from_article_row(self, row: pd.Series, price_map: Dict[str, float], mode_label: str) -> Optional[SearchItem]:
        article_id = self._article_id(row)
        if not article_id:
            return None

        name = self._build_article_name(row)
        description = self._build_semantic_article_description(row)
        precomputed_image_path = str(row.get("image_path", "")).strip()
        image_path = None
        if precomputed_image_path:
            candidate = Path(precomputed_image_path)
            try:
                if candidate.exists():
                    image_path = candidate
            except OSError:
                image_path = None
        if image_path is None:
            image_path = self._locate_article_image_path(row)
        image = None
        price = float(price_map.get(article_id, self._extract_price_from_row(row)))
        metadata = {
            "mode": mode_label,
            "article_id": article_id,
            "product_id": article_id,
            "name": name,
            "product_code": str(row.get("product_code", "")),
            "product_name": row.get("prod_name", row.get("product_name", name)),
            "product_type_name": row.get("product_type_name", ""),
            "product_group_name": row.get("product_group_name", ""),
            "graphical_appearance_name": row.get("graphical_appearance_name", ""),
            "colour_group_name": row.get("colour_group_name", ""),
            "perceived_colour_master_name": row.get("perceived_colour_master_name", ""),
            "perceived_colour_value_name": row.get("perceived_colour_value_name", ""),
            "index_name": row.get("index_name", ""),
            "index_group_name": row.get("index_group_name", ""),
            "department_name": row.get("department_name", ""),
            "section_name": row.get("section_name", ""),
            "garment_group_name": row.get("garment_group_name", ""),
            "detail_desc": row.get("detail_desc", ""),
            "category": row.get("category", ""),
            "main_category": row.get("main_category", ""),
            "category_l1": row.get("category_l1", ""),
            "category_l2": row.get("category_l2", ""),
            "category_l3": row.get("category_l3", ""),
            "image_name": row.get("image_name", ""),
            "image_path": str(image_path) if image_path is not None else "",
            "price": price,
        }
        return SearchItem(
            product_id=article_id,
            name=name,
            price=price,
            description=description,
            image=image,
            image_path=str(image_path) if image_path is not None else None,
            metadata=metadata,
        )

    def _load_dev_items(self) -> List[SearchItem]:
        articles, articles_path = self._load_article_table("dev")
        self._source_articles_path = articles_path

        price_map = self._load_article_price_map()
        shuffled = articles.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)

        items: List[SearchItem] = []
        for _, row in shuffled.iterrows():
            item = self._build_item_from_article_row(row, price_map=price_map, mode_label="dev")
            if item is None:
                continue
            if item.image_path is None:
                continue
            items.append(item)
            if len(items) >= self.dev_sample_size:
                break

        if not items:
            raise RuntimeError(
                "Dev mode could not find any image-backed items. "
                "Set SEARCH_ENGINE_IMAGE_ROOT to a valid H&M image directory such as D:/imagedata."
            )

        LOGGER.info(
            "Prepared %d dev items with real images from %s (seed=%d)",
            len(items),
            articles_path,
            self.random_seed,
        )
        self._write_pair_manifest(items, mode_label="dev", source_articles_path=articles_path)
        return items

    def _load_production_items(self) -> List[SearchItem]:
        # production 모드에서는 H&M articles.csv를 읽어 상품 메타데이터를 구성한다.
        articles, articles_path = self._load_article_table("production")
        self._source_articles_path = articles_path
        price_map = self._load_article_price_map()

        items: List[SearchItem] = []
        for _, row in articles.iterrows():
            item = self._build_item_from_article_row(row, price_map=price_map, mode_label="production")
            if item is not None:
                items.append(item)
        LOGGER.info("Prepared %d production items from %s", len(items), articles_path)
        self._write_pair_manifest(items, mode_label="production", source_articles_path=articles_path)
        return items

    def _load_article_price_map(self) -> Dict[str, float]:
        candidates = [
            self.data_root / "item_features_dev.csv",
            self.data_root / "item_features.csv",
            self.data_root / "item_master_dev.csv",
            self.data_root / "item_master.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, dtype={"article_id": str}).dropna()
                if "article_id" not in df.columns:
                    continue
                price_col = next((name for name in ("price", "avg_price", "price_mean") if name in df.columns), None)
                if price_col is None:
                    continue
                df["article_id"] = df["article_id"].apply(self._normalize_article_id)
                grouped = pd.to_numeric(df[price_col], errors="coerce").groupby(df["article_id"]).mean()
                return {str(key): float(value) for key, value in grouped.items()}
            except Exception as exc:
                LOGGER.warning("Failed to load prices from %s: %s", path, exc)
        return {}

    @staticmethod
    def _article_id(row: pd.Series) -> str:
        for key in ("article_id", "item_id", "product_id"):
            value = MultimodalSearchEngine._normalize_article_id(row.get(key, ""))
            if value:
                return value
        value = str(row.get("product_code", "")).strip()
        if value:
            return value
        return ""

    @staticmethod
    def _build_article_name(row: pd.Series) -> str:
        candidates = [
            str(row.get("prod_name", "")).strip(),
            str(row.get("product_name", "")).strip(),
            str(row.get("product_type_name", "")).strip(),
            str(row.get("category_l3", "")).strip(),
            str(row.get("detail_desc", "")).strip(),
        ]
        for value in candidates:
            if value:
                return value[:120]
        return "item"

    @staticmethod
    def _build_article_description(row: pd.Series) -> str:
        # CLIP 텍스트 검색 품질을 위해 색상/카테고리/설명 필드를 하나의 문장으로 합친다.
        fields = [
            str(row.get("prod_name", "")).strip(),
            str(row.get("product_type_name", "")).strip(),
            str(row.get("product_group_name", "")).strip(),
            str(row.get("graphical_appearance_name", "")).strip(),
            str(row.get("colour_group_name", "")).strip(),
            str(row.get("perceived_colour_master_name", "")).strip(),
            str(row.get("perceived_colour_value_name", "")).strip(),
            str(row.get("index_name", "")).strip(),
            str(row.get("index_group_name", "")).strip(),
            str(row.get("department_name", "")).strip(),
            str(row.get("section_name", "")).strip(),
            str(row.get("garment_group_name", "")).strip(),
            str(row.get("category", "")).strip(),
            str(row.get("main_category", "")).strip(),
            str(row.get("color", "")).strip(),
            str(row.get("detail_desc", "")).strip(),
        ]
        return " | ".join(field for field in fields if field)

    @staticmethod
    def _build_semantic_article_description(row: pd.Series) -> str:
        name = str(row.get("prod_name", "")).strip()
        product_type = str(row.get("product_type_name", "")).strip()
        product_group = str(row.get("product_group_name", "")).strip()
        color = str(row.get("colour_group_name", "")).strip() or str(row.get("perceived_colour_master_name", "")).strip()
        department = str(row.get("department_name", "")).strip()
        section = str(row.get("section_name", "")).strip()
        garment = str(row.get("garment_group_name", "")).strip()
        appearance = str(row.get("graphical_appearance_name", "")).strip()
        detail_desc = str(row.get("detail_desc", "")).strip()
        category_l1 = str(row.get("category_l1", "")).strip() or str(row.get("index_group_name", "")).strip()
        category_l2 = str(row.get("category_l2", "")).strip() or section
        category_l3 = str(row.get("category_l3", "")).strip() or product_type

        sentences = [
            f"{name}." if name else "",
            f"Product type: {product_type}." if product_type else "",
            f"Product group: {product_group}." if product_group else "",
            f"Color: {color}." if color else "",
            f"Department: {department}." if department else "",
            f"Section: {section}." if section else "",
            f"Garment group: {garment}." if garment else "",
            f"Appearance: {appearance}." if appearance else "",
            f"Description: {detail_desc}." if detail_desc else "",
        ]
        keywords = " ".join(
            value
            for value in [
                name,
                product_type,
                product_group,
                color,
                department,
                section,
                garment,
                category_l1,
                category_l2,
                category_l3,
                appearance,
            ]
            if value
        )
        if keywords:
            sentences.append(f"Keywords: {keywords}.")
        return " ".join(part for part in sentences if part)

    def _locate_article_image(self, row: pd.Series) -> Optional[Image.Image]:
        # 이미지가 있으면 텍스트와 함께 상품 임베딩에 반영하고, 없으면 텍스트만 사용한다.
        image_name = str(row.get("image_name", "")).strip()
        article_id = self._article_id(row)
        candidates: List[Path] = []

        if image_name:
            candidates.extend(
                [
                    self.data_root / "images" / image_name,
                    self.data_root / image_name,
                ]
            )

        if article_id:
            candidates.extend(
                [
                    self.data_root / "images" / f"{article_id}.jpg",
                    self.data_root / "images" / f"{article_id}.png",
                    self.data_root / "images" / article_id,
                    self.data_root / f"{article_id}.jpg",
                ]
            )
            try:
                padded = f"{int(float(article_id)):010d}.jpg"
                candidates.append(self.data_root / "images" / padded)
            except Exception:
                pass

        for path in candidates:
            if path.exists():
                image = encode_image_file(path)
                if image is not None:
                    return image
        return None

    def _candidate_image_roots(self) -> List[Path]:
        env_roots: List[Path] = []
        env_value = (
            os.getenv("SEARCH_ENGINE_IMAGE_ROOTS")
            or os.getenv("SEARCH_ENGINE_IMAGE_ROOT")
            or os.getenv("IMAGE_ROOT")
            or ""
        ).strip()
        if env_value:
            for raw_path in env_value.replace("\n", ";").split(";"):
                cleaned = raw_path.strip()
                if cleaned:
                    env_roots.append(Path(cleaned))

        roots = env_roots + [
            Path("D:/imagedata"),
            self.data_root / "images",
            self.data_root,
            self.data_root.parent / "images",
            self.data_root.parent / "raw" / "images",
        ]
        available: List[Path] = []
        for root in roots:
            try:
                if root.exists():
                    available.append(root)
            except OSError:
                continue
        return available

    def _locate_article_image_path(self, row: pd.Series) -> Optional[Path]:
        image_name = str(row.get("image_name", "")).strip()
        article_id = self._article_id(row)
        candidates: List[Path] = []
        image_roots = self._candidate_image_roots()

        if image_name:
            image_path = Path(image_name)
            for root in image_roots:
                candidates.append(root / image_path)
            candidates.append(self.data_root / image_path)

        if article_id:
            normalized_article_id = article_id
            try:
                normalized_article_id = f"{int(float(article_id)):010d}"
            except Exception:
                pass

            folder_prefixes = {
                "",
                normalized_article_id[:2],
                normalized_article_id[:3],
            }
            filename_variants = [normalized_article_id, article_id]
            for root in image_roots:
                for filename in filename_variants:
                    for folder in folder_prefixes:
                        base = root / folder if folder else root
                        candidates.append(base / f"{filename}.jpg")
                        candidates.append(base / f"{filename}.jpeg")
                        candidates.append(base / f"{filename}.png")
                        candidates.append(base / filename)

        for path in candidates:
            try:
                if path.exists():
                    return path
            except OSError:
                continue
        return None

    def _read_item_image_bytes(self, item: SearchItem) -> Optional[bytes]:
        image_base64 = str(item.metadata.get("_image_base64", "")).strip()
        if image_base64:
            try:
                return base64.b64decode(image_base64)
            except Exception:
                return None
        if item.image_path:
            try:
                return Path(item.image_path).read_bytes()
            except OSError:
                return None
        if item.image is not None:
            buffer = io.BytesIO()
            item.image.save(buffer, format="PNG")
            return buffer.getvalue()
        return None

    @staticmethod
    def _load_item_image_for_embedding(item: SearchItem) -> Optional[Image.Image]:
        if item.image is not None:
            return item.image
        if not item.image_path:
            return None
        return encode_image_file(item.image_path)

    def _metadata_hint_text(self, item: SearchItem) -> str:
        metadata = dict(item.metadata or {})
        tokens: List[str] = []
        for value in (
            metadata.get("colour_group_name", ""),
            metadata.get("perceived_colour_master_name", ""),
            metadata.get("product_type_name", ""),
            metadata.get("garment_group_name", ""),
            metadata.get("section_name", ""),
            metadata.get("department_name", ""),
            metadata.get("product_group_name", ""),
        ):
            token = str(value or "").strip().lower()
            if token and token not in tokens:
                tokens.append(token)
        if not tokens:
            fallback = (item.name or item.description or "").strip().lower()
            return self._expand_query_terms(fallback)
        return self._expand_query_terms(" ".join(tokens))

    def _weighted_combine_vectors(
        self,
        vectors: Sequence[Optional[np.ndarray]],
        weights: Sequence[float],
    ) -> np.ndarray:
        usable: List[np.ndarray] = []
        usable_weights: List[float] = []
        for vector, weight in zip(vectors, weights):
            if vector is None or weight <= 0:
                continue
            array = np.asarray(vector, dtype=np.float32)
            if array.size == 0 or not np.any(array):
                continue
            usable.append(array)
            usable_weights.append(float(weight))
        if not usable:
            return np.zeros(self.dimension, dtype=np.float32)
        combined = np.average(
            np.stack(usable).astype(np.float32),
            axis=0,
            weights=np.asarray(usable_weights, dtype=np.float32),
        )
        return self.embedder._normalize(combined)

    def _build_inner_product_index(self, vectors: np.ndarray) -> Any:
        if faiss is not None:
            index = faiss.IndexHNSWFlat(int(vectors.shape[1]), 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
            index.add(vectors)
            return index
        index = _NumpyInnerProductIndex(int(vectors.shape[1]))
        index.add(vectors)
        return index

    def _score_matrix(self, matrix: Optional[np.ndarray], query_vec: np.ndarray) -> np.ndarray:
        if matrix is None or matrix.size == 0:
            return np.zeros(len(self.items), dtype=np.float32)
        vector = self.embedder._normalize(np.asarray(query_vec, dtype=np.float32))
        return (matrix @ vector).astype(np.float32)

    def _rank_results_from_scores(self, scores: np.ndarray, top_k: int) -> List[SearchResult]:
        if scores.size == 0:
            return []
        order = np.argsort(-scores)[: max(1, int(top_k))]
        hits: List[SearchResult] = []
        for idx in order:
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[int(idx)]
            hits.append(
                SearchResult(
                    item_id=str(item.product_id),
                    score=float(scores[int(idx)]),
                    metadata=dict(item.metadata or {}),
                )
            )
        return hits

    @staticmethod
    def _metadata_color_family(metadata: Dict[str, Any]) -> Optional[str]:
        for key in (
            "color",
            "perceived_colour_master_name",
            "colour_group_name",
            "perceived_colour_value_name",
        ):
            family = normalize_color_family(metadata.get(key))
            if family:
                return family
        return None

    @staticmethod
    def _image_color_signal(image: Optional[Image.Image], image_bytes: Optional[bytes]) -> tuple[Optional[str], float]:
        if image is not None:
            family, confidence = dominant_color_signal_from_image(image)
            if family:
                return family, confidence
        if not image_bytes:
            return None, 0.0
        try:
            with Image.open(io.BytesIO(image_bytes)) as opened_image:
                return dominant_color_signal_from_image(opened_image)
        except Exception:
            LOGGER.debug("Failed to extract dominant color from query image", exc_info=True)
            return None, 0.0

    @staticmethod
    def _color_score_multiplier(query_color: Optional[str], item_color: Optional[str]) -> float:
        if not query_color or not item_color:
            return 1.0
        if query_color == item_color:
            return IMAGE_COLOR_EXACT_BOOST
        if query_color in NEUTRAL_COLOR_FAMILIES and item_color in NEUTRAL_COLOR_FAMILIES:
            if {query_color, item_color} == {"black", "white"}:
                return 0.97
            return IMAGE_COLOR_NEUTRAL_BOOST
        return IMAGE_COLOR_MISMATCH_PENALTY

    def _rerank_results_by_image_color(
        self,
        results: Sequence[SearchResult],
        query_color: Optional[str],
        query_color_confidence: float = 0.0,
    ) -> List[SearchResult]:
        if not query_color:
            return list(results)

        reranked: list[tuple[float, int, SearchResult]] = []
        for index, result in enumerate(results):
            metadata = dict(result.metadata or {})
            item_color = self._metadata_color_family(metadata)
            multiplier = self._color_score_multiplier(query_color, item_color)
            reranked.append((
                float(result.score) * multiplier,
                index,
                SearchResult(
                    item_id=result.item_id,
                    score=float(result.score) * multiplier,
                    metadata={
                        **metadata,
                        "image_query_color": query_color,
                        "image_query_color_confidence": round(query_color_confidence, 4),
                        "item_color_family": item_color or "",
                        "color_rerank_multiplier": round(multiplier, 4),
                    },
                ),
            ))

        reranked.sort(key=lambda item: (-item[0], item[1]))
        return [result for _, _, result in reranked]

    def _embedding_matrix_for_modality(self, modality: str) -> np.ndarray:
        normalized_modality = (modality or "multimodal").strip().lower()
        if normalized_modality in {"multimodal", "combined", "hybrid"}:
            matrix = self._embeddings
        elif normalized_modality == "text":
            matrix = self._text_embeddings
        elif normalized_modality == "image":
            matrix = self._image_embeddings
        else:
            raise ValueError("modality must be one of: multimodal, text, image")

        if matrix is None or matrix.size == 0:
            raise RuntimeError(f"{normalized_modality} embeddings are not initialized")
        if matrix.shape[0] != len(self.items):
            raise RuntimeError(
                f"{normalized_modality} embeddings do not match item count: "
                f"{matrix.shape[0]} != {len(self.items)}"
            )
        return matrix

    def _item_index_by_article_id(self) -> Dict[str, int]:
        lookup: Dict[str, int] = {}
        for index, item in enumerate(self.items):
            lookup[normalize_article_id(item.product_id)] = index
            if index < len(self.item_ids):
                lookup[normalize_article_id(self.item_ids[index])] = index
        return lookup

    def cross_similarity(
        self,
        article_ids: Sequence[Any],
        *,
        modality: str = "multimodal",
    ) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
        """Return pairwise cosine similarity for known catalog article ids."""

        requested_ids: list[str] = []
        seen: set[str] = set()
        for raw_article_id in article_ids:
            article_id = normalize_article_id(raw_article_id)
            if not article_id or article_id in seen:
                continue
            seen.add(article_id)
            requested_ids.append(article_id)

        matrix = self._embedding_matrix_for_modality(modality)
        index_by_id = self._item_index_by_article_id()
        found_ids: list[str] = []
        found_indices: list[int] = []
        missing_ids: list[str] = []
        for article_id in requested_ids:
            item_index = index_by_id.get(article_id)
            if item_index is None:
                missing_ids.append(article_id)
                continue
            found_ids.append(article_id)
            found_indices.append(item_index)

        if not found_indices:
            return found_ids, missing_ids, {}

        vectors = np.asarray(matrix[found_indices], dtype=np.float32).copy()
        self._normalize_matrix_inplace(vectors)
        scores = np.clip(vectors @ vectors.T, -1.0, 1.0)
        similarity: dict[str, dict[str, float]] = {}
        for row_index, source_id in enumerate(found_ids):
            similarity[source_id] = {}
            for column_index, target_id in enumerate(found_ids):
                if source_id == target_id:
                    continue
                similarity[source_id][target_id] = round(float(scores[row_index, column_index]), 6)
        return found_ids, missing_ids, similarity

    def _resolve_image_query_anchor(self, image_vec: np.ndarray, image_bytes: Optional[bytes]) -> Tuple[Optional[int], np.ndarray, bool]:
        image_scores = self._score_matrix(self._image_embeddings, image_vec)
        if image_bytes:
            cache_key = self.embedder._hash_image_bytes(image_bytes)
            anchor_idx = self._image_hash_to_item_index.get(cache_key)
            if anchor_idx is not None:
                return int(anchor_idx), image_scores, True
        if image_scores.size == 0 or not np.any(image_scores):
            return None, image_scores, False
        return int(np.argmax(image_scores)), image_scores, False

    def _search_image_mode(self, image_vec: np.ndarray, image_bytes: Optional[bytes], top_k: int) -> List[SearchResult]:
        anchor_idx, image_scores, exact_match = self._resolve_image_query_anchor(image_vec, image_bytes)
        if anchor_idx is None:
            fallback_scores = 0.75 * image_scores + 0.25 * self._score_matrix(self._embeddings, image_vec)
            return self._rank_results_from_scores(fallback_scores.astype(np.float32), top_k)

        hint_vec = self.embedder.embed_text(self._metadata_hint_text(self.items[anchor_idx]))
        if exact_match:
            combined_vec = self._weighted_combine_vectors([image_vec, hint_vec], [0.25, 0.75])
            image_weight, mm_weight, text_weight = 0.10, 0.20, 0.70
        else:
            combined_vec = self._weighted_combine_vectors([image_vec, hint_vec], [0.45, 0.55])
            image_weight, mm_weight, text_weight = 0.35, 0.20, 0.45

        multimodal_scores = self._score_matrix(self._embeddings, combined_vec)
        text_scores = self._score_matrix(self._text_embeddings, hint_vec)
        final_scores = (
            image_weight * image_scores
            + mm_weight * multimodal_scores
            + text_weight * text_scores
        ).astype(np.float32)
        return self._rank_results_from_scores(final_scores, top_k)

    def _build_index(self) -> None:
        # 엔진 내부 데이터셋으로부터 상품별 CLIP 임베딩을 생성해 기본 인덱스를 만든다.
        vectors: List[np.ndarray] = []
        text_vectors: List[np.ndarray] = []
        image_vectors: List[np.ndarray] = []
        self._image_hash_to_item_index = {}
        total_items = len(self.items)
        progress_interval = max(1, int(os.getenv("SEARCH_ENGINE_INDEX_PROGRESS_INTERVAL", "500")))
        started_at = time.perf_counter()
        LOGGER.info("Building %s search index for %d items", self.mode, total_items)
        for item_index, item in enumerate(self.items, start=1):
            text_value = item.description or item.name
            text_vector = self.embedder.embed_text(text_value) if text_value else np.zeros(self.dimension, dtype=np.float32)
            text_vectors.append(text_vector)
            image_vector = None
            image = self._load_item_image_for_embedding(item)
            try:
                if image is not None:
                    image_vector = self.embedder.embed_image(image)
                    item.metadata["_image_embedding"] = image_vector.astype(np.float32).tolist()
                    image_bytes = self._read_item_image_bytes(item)
                    if image_bytes:
                        self.embedder.register_image_bytes_embedding(image_bytes, image_vector)
                        self._image_hash_to_item_index[self.embedder._hash_image_bytes(image_bytes)] = len(image_vectors)
            finally:
                if image is not None and image is not item.image:
                    image.close()
            image_vectors.append(
                image_vector if image_vector is not None else np.zeros(self.dimension, dtype=np.float32)
            )
            vectors.append(self.embedder.combine_embeddings([text_vector, image_vector]))
            if item_index == total_items or item_index % progress_interval == 0:
                elapsed = max(time.perf_counter() - started_at, 0.001)
                LOGGER.info(
                    "Search index build progress: %d/%d items (%.1f%%, %.1f items/sec, %.1fs elapsed)",
                    item_index,
                    total_items,
                    (item_index / total_items) * 100.0 if total_items else 100.0,
                    item_index / elapsed,
                    elapsed,
                )

        if not vectors:
            raise ValueError("No items available to index")

        self._embeddings = np.vstack(vectors).astype(np.float32)
        self._text_embeddings = np.vstack(text_vectors).astype(np.float32)
        self._image_embeddings = np.vstack(image_vectors).astype(np.float32)
        self._normalize_matrix_inplace(self._embeddings)
        self._normalize_matrix_inplace(self._text_embeddings)
        self._normalize_matrix_inplace(self._image_embeddings)
        self.dimension = int(self._embeddings.shape[1])
        self.item_ids = [str(item.product_id) for item in self.items]
        self._image_item_indices = np.arange(len(self.items), dtype=np.int64)

        if faiss is not None:
            # Inner Product + L2 normalize 조합이라 cosine similarity 검색처럼 동작한다.
            self.index = self._build_inner_product_index(self._embeddings)
            self.text_index = self._build_inner_product_index(self._text_embeddings)
            self.image_index = self._build_inner_product_index(self._image_embeddings)
            LOGGER.info("Built FAISS HNSW index with %d items", len(self.items))
        else:
            self.index = self._build_inner_product_index(self._embeddings)
            self.text_index = self._build_inner_product_index(self._text_embeddings)
            self.image_index = self._build_inner_product_index(self._image_embeddings)
            LOGGER.warning("FAISS unavailable, using NumPy fallback index with %d items", len(self.items))
        self._is_built = True

    def build_index(
        self,
        embeddings: np.ndarray,
        item_ids: Optional[Sequence[Any]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        # app.py가 외부에서 만든 임베딩을 넘기는 레거시 경로와의 호환용 빌더다.
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim < 2:
            raise ValueError("embeddings must be at least a 2D array")
        if vectors.shape[0] == 0:
            raise ValueError("embeddings must not be empty")

        metadata_list = list(metadatas) if metadatas is not None else [{} for _ in range(vectors.shape[0])]
        ids = list(item_ids) if item_ids is not None else list(range(vectors.shape[0]))
        if len(ids) != vectors.shape[0] or len(metadata_list) != vectors.shape[0]:
            raise ValueError("embeddings, item_ids, and metadatas must have the same length")

        normalized_rows: List[np.ndarray] = []
        for row, metadata in zip(vectors, metadata_list):
            modality = "image" if str((metadata or {}).get("search_type", "")).lower() == "image" else "text"
            normalized_rows.append(self.embedder.project_external_embedding(row, modality=modality))

        self._embeddings = np.vstack(normalized_rows).astype(np.float32)
        self._normalize_matrix_inplace(self._embeddings)
        self._text_embeddings = self._embeddings.copy()
        self._image_embeddings = self._embeddings.copy()
        self.dimension = int(self._embeddings.shape[1])

        self.items = []
        self.item_ids = []
        for idx, (item_id, metadata) in enumerate(zip(ids, metadata_list)):
            payload = dict(metadata or {})
            product_id = str(payload.get("product_id", item_id))
            name = str(payload.get("name", product_id))
            price = float(payload.get("price", 0.0))
            description = str(payload.get("description", payload.get("prod_name", name)))
            self.items.append(
                SearchItem(
                    product_id=product_id,
                    name=name,
                    price=price,
                    description=description,
                    image=None,
                    image_path=str(payload.get("image_path", "")) or None,
                    metadata=payload,
                )
            )
            self.item_ids.append(str(item_id))

        self._image_item_indices = np.arange(len(self.items), dtype=np.int64)
        self._image_hash_to_item_index = {}
        self.index = self._build_inner_product_index(self._embeddings)
        self.text_index = self._build_inner_product_index(self._text_embeddings)
        self.image_index = self._build_inner_product_index(self._image_embeddings)
        self._is_built = True
        LOGGER.info("Built compatibility index with %d items", len(self.items))

    @staticmethod
    def _normalize_matrix_inplace(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms

    def _search_from_vector(self, query_vec: np.ndarray, top_k: int) -> List[SearchResult]:
        if self.index is None:
            raise RuntimeError("Search index is not initialized")

        # 인덱스와 쿼리를 모두 정규화해 inner product 검색이 안정적으로 되도록 한다.
        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        if faiss is not None:
            faiss.normalize_L2(query_vec)
        else:
            self._normalize_matrix_inplace(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results: List[SearchResult] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[idx]
            item_id = self.item_ids[idx] if idx < len(self.item_ids) else str(item.product_id)
            results.append(SearchResult(item_id=str(item_id), score=float(score), metadata=dict(item.metadata)))
        return results

    def _prepare_query_vector(self, query_vec: np.ndarray, modality: str) -> np.ndarray:
        # 외부 임베딩의 shape이 달라도 현재 인덱스 차원에 맞는 단일 쿼리 벡터로 변환한다.
        vector = np.asarray(query_vec, dtype=np.float32)
        if vector.size == 0:
            return np.zeros(self.dimension, dtype=np.float32)
        if vector.ndim == 1 and vector.shape[0] == self.dimension:
            return self.embedder._normalize(vector)
        if vector.ndim >= 2 and vector.shape[-1] == self.dimension:
            flattened = vector.reshape(-1, self.dimension)
            return self.embedder._normalize(flattened.mean(axis=0))
        return self.embedder.project_external_embedding(vector, modality=modality)

    def search(
        self,
        query: Optional[str] = None,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        top_k: Optional[int] = None,
        use_cache: bool = True,
        query_type: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        text_embedding: Optional[np.ndarray] = None,
        image_embedding: Optional[np.ndarray] = None,
    ) -> Any:
        # 1) 외부 임베딩 호환 모드: app.py가 직접 만든 벡터를 받아 검색
        if query_type is not None or embedding is not None or text_embedding is not None or image_embedding is not None:
            top_k = max(1, int(top_k or self.top_k_default))
            if query_type == "hybrid":
                query_vec = self.embedder.combine_embeddings(
                    [
                        self._prepare_query_vector(text_embedding, "text") if text_embedding is not None else None,
                        self._prepare_query_vector(image_embedding, "image") if image_embedding is not None else None,
                    ]
                )
            elif query_type == "image":
                source = image_embedding if image_embedding is not None else embedding
                query_vec = self._prepare_query_vector(source, "image")
            else:
                source = text_embedding if text_embedding is not None else embedding
                query_vec = self._prepare_query_vector(source, "text")

            if query_vec.size == 0 or not np.any(query_vec):
                return []
            return self._search_from_vector(query_vec, top_k)

        # 2) self-contained 모드: query/image를 받아 이 파일 내부에서 CLIP 임베딩까지 수행
        if self.index is None:
            raise RuntimeError("Search index is not initialized")

        top_k = max(1, int(top_k or self.top_k_default))
        image_query_color, image_query_color_confidence = self._image_color_signal(image, image_bytes)
        color_rerank_enabled = (
            bool(image_query_color)
            and image_query_color_confidence >= IMAGE_COLOR_RERANK_MIN_CONFIDENCE
        )
        search_top_k = top_k
        if color_rerank_enabled and len(self.items) > top_k:
            search_top_k = min(
                len(self.items),
                max(top_k, IMAGE_COLOR_RERANK_MIN_POOL, top_k * IMAGE_COLOR_RERANK_MULTIPLIER),
            )
        started = time.perf_counter()
        prepared_query = self._expand_query_terms(query or "")
        query_vec, search_type = self.embedder.embed_query(
            text=prepared_query,
            image=image,
            image_bytes=image_bytes,
            use_cache=use_cache,
        )
        if not np.any(query_vec):
            return {
                "search_type": search_type,
                "results": [],
                "latency_ms": 0.0,
                "total_count": 0,
            }

        if search_type == "image":
            vector_results = self._search_image_mode(query_vec, image_bytes=image_bytes, top_k=search_top_k)
        else:
            vector_results = self._search_from_vector(query_vec, search_top_k)
        if color_rerank_enabled:
            vector_results = self._rerank_results_by_image_color(
                vector_results,
                image_query_color,
                image_query_color_confidence,
            )[:top_k]
        latency_ms = (time.perf_counter() - started) * 1000.0

        results: List[Dict[str, Any]] = []
        for hit in vector_results:
            meta = hit.metadata or {}
            results.append(
                {
                    "product_id": str(meta.get("product_id", hit.item_id)),
                    "name": str(meta.get("name", "")),
                    "score": round(float(hit.score), 10),
                    "price": round(float(meta.get("price", 0.0)), 10),
                    "image_query_color": meta.get("image_query_color", ""),
                    "image_query_color_confidence": meta.get("image_query_color_confidence", 0.0),
                    "item_color_family": meta.get("item_color_family", ""),
                    "color_rerank_multiplier": meta.get("color_rerank_multiplier", 1.0),
                }
            )

        return {
            "search_type": search_type,
            "results": results,
            "latency_ms": round(latency_ms, 3),
            "total_count": len(results),
            "image_query_color": image_query_color or "",
            "image_query_color_confidence": round(image_query_color_confidence, 4),
            "color_rerank_applied": color_rerank_enabled,
        }

    def __len__(self) -> int:
        return len(self.items)

    def save_index(self, index_path: str, metadata_path: str) -> None:
        if self.index is None:
            raise RuntimeError("Index not initialized")
        if faiss is not None:
            faiss.write_index(self.index, index_path)
        else:
            np.savez_compressed(index_path + ".npz", vectors=getattr(self.index, "vectors", None))

        aux_path = Path(metadata_path).with_suffix(Path(metadata_path).suffix + ".aux.npz")
        np.savez_compressed(
            aux_path,
            multimodal_embeddings=self._embeddings,
            text_embeddings=self._text_embeddings,
            image_embeddings=self._image_embeddings,
        )

        payload = {
            "mode": self.mode,
            "dimension": self.dimension,
            "index_format": SEARCH_INDEX_FORMAT,
            "build_config": self._artifact_build_config(),
            "aux_vectors_file": aux_path.name,
            "items": [
                {
                    "product_id": item.product_id,
                    "name": item.name,
                    "price": item.price,
                    "description": item.description,
                    "image_path": item.image_path,
                    "metadata": self._metadata_for_artifact(item),
                }
                for item in self.items
            ],
        }
        Path(metadata_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _artifact_build_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "clip_model_name": self.embedder.model_name,
            "item_count": len(self.items),
            "source_articles": self._source_articles_build_config(),
        }
        if self.mode == "dev":
            config["dev_sample_size"] = self.dev_sample_size
            config["random_seed"] = self.random_seed
        return config

    def _source_articles_build_config(self) -> Dict[str, Any]:
        if self._source_articles_path is None:
            return {}
        try:
            source_path = Path(self._source_articles_path)
            stat = source_path.stat()
        except OSError:
            return {}
        return {
            "path": str(source_path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    @staticmethod
    def _metadata_for_artifact(item: SearchItem) -> Dict[str, Any]:
        metadata = dict(item.metadata or {})
        metadata.pop("_image_base64", None)
        return metadata

    @classmethod
    def load_from_artifacts(
        cls,
        index_path: str,
        metadata_path: str,
        mode: str = "production",
        data_root: Optional[str] = None,
        clip_model_name: str = CLIP_MODEL_NAME,
    ) -> "MultimodalSearchEngine":
        obj = cls.__new__(cls)
        obj.mode = mode
        obj.data_root = cls._resolve_runtime_data_root(data_root)
        obj.top_k_default = DEFAULT_TOP_K
        obj.random_seed = int(os.getenv("SEARCH_ENGINE_RANDOM_SEED", str(DEFAULT_RANDOM_SEED)))
        obj.dev_sample_size = int(os.getenv("SEARCH_ENGINE_DEV_SAMPLE_SIZE", str(DEFAULT_DEV_SAMPLE_SIZE)))
        obj.embedder = OpenAIClipEmbedder(model_name=clip_model_name)
        obj.dimension = int(obj.embedder.dim or 512)
        obj.text_index = None
        obj.image_index = None
        obj._text_embeddings = None
        obj._image_embeddings = None
        obj._image_item_indices = np.empty((0,), dtype=np.int64)
        obj._image_hash_to_item_index = {}
        obj._aux_vectors_path = None
        obj._source_articles_path = None
        if faiss is not None:
            obj.index = faiss.read_index(index_path)
        else:
            data = np.load(index_path + ".npz")
            obj.index = _NumpyInnerProductIndex(int(data["vectors"].shape[1]))
            obj.index.add(data["vectors"])

        meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        obj.items = []
        for item in meta.get("items", []):
            item_metadata = dict(item.get("metadata", {}))
            image_base64 = str(item.get("image_base64", "")).strip()
            if image_base64:
                item_metadata["_image_base64"] = image_base64
            search_item = SearchItem(
                product_id=str(item.get("product_id", "")),
                name=str(item.get("name", "")),
                price=float(item.get("price", 0.0)),
                description=str(item.get("description", "")),
                # Keep cached image bytes in metadata and avoid eagerly materializing
                # every PIL image during startup. This keeps dev/prod artifact loads
                # lighter and prevents unnecessary memory spikes.
                image=None,
                image_path=str(item.get("image_path", "")) or None,
                metadata=item_metadata,
            )
            obj.items.append(search_item)
        aux_file = str(meta.get("aux_vectors_file", "")).strip()
        aux_path = Path(metadata_path).with_name(aux_file) if aux_file else Path(metadata_path).with_suffix(Path(metadata_path).suffix + ".aux.npz")
        if aux_path.exists():
            try:
                aux = np.load(aux_path)
                obj._embeddings = np.asarray(aux["multimodal_embeddings"], dtype=np.float32)
                obj._text_embeddings = np.asarray(aux["text_embeddings"], dtype=np.float32)
                obj._image_embeddings = np.asarray(aux["image_embeddings"], dtype=np.float32)
                obj.text_index = obj._build_inner_product_index(obj._text_embeddings)
                obj.image_index = obj._build_inner_product_index(obj._image_embeddings)
                obj._image_item_indices = np.arange(len(obj.items), dtype=np.int64)
                obj._aux_vectors_path = aux_path
            except Exception:
                obj._embeddings = None
                obj._text_embeddings = None
                obj._image_embeddings = None
        else:
            obj._embeddings = None
        obj._is_built = True
        obj.item_ids = [str(item.product_id) for item in obj.items]
        for idx, item in enumerate(obj.items):
            image_vector = None
            if obj._image_embeddings is not None and idx < len(obj._image_embeddings):
                image_vector = np.asarray(obj._image_embeddings[idx], dtype=np.float32)
            else:
                embedding_values = item.metadata.get("_image_embedding")
                if embedding_values is not None:
                    try:
                        image_vector = np.asarray(embedding_values, dtype=np.float32)
                    except Exception:
                        image_vector = None
            if image_vector is None or image_vector.size == 0 or not np.any(image_vector):
                continue
            image_bytes = obj._read_item_image_bytes(item)
            if not image_bytes:
                continue
            try:
                embedding = obj.embedder._normalize(image_vector)
                obj.embedder.register_image_bytes_embedding(image_bytes, embedding)
                obj._image_hash_to_item_index[obj.embedder._hash_image_bytes(image_bytes)] = idx
            except Exception:
                continue
        if obj._text_embeddings is None or obj._image_embeddings is None:
            text_vectors: List[np.ndarray] = []
            image_vectors: List[np.ndarray] = []
            multimodal_vectors: List[np.ndarray] = []
            for item in obj.items:
                text_vector = obj.embedder.embed_text(item.description or item.name)
                image_values = item.metadata.get("_image_embedding")
                image_vector = (
                    np.asarray(image_values, dtype=np.float32)
                    if image_values is not None
                    else np.zeros(obj.dimension, dtype=np.float32)
                )
                image_vector = obj.embedder._normalize(image_vector)
                text_vectors.append(text_vector)
                image_vectors.append(image_vector)
                multimodal_vectors.append(obj.embedder.combine_embeddings([text_vector, image_vector]))
            obj._text_embeddings = np.vstack(text_vectors).astype(np.float32)
            obj._image_embeddings = np.vstack(image_vectors).astype(np.float32)
            obj._embeddings = np.vstack(multimodal_vectors).astype(np.float32)
            obj._normalize_matrix_inplace(obj._text_embeddings)
            obj._normalize_matrix_inplace(obj._image_embeddings)
            obj._normalize_matrix_inplace(obj._embeddings)
            obj.text_index = obj._build_inner_product_index(obj._text_embeddings)
            obj.image_index = obj._build_inner_product_index(obj._image_embeddings)
            obj._image_item_indices = np.arange(len(obj.items), dtype=np.int64)
        return obj

    @classmethod
    def artifact_paths_for_mode(
        cls,
        mode: str,
        data_root: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> tuple[Path, Path]:
        resolved_root = cls._resolve_runtime_data_root(data_root)
        cache_root = Path(cache_dir) if cache_dir is not None else resolved_root.parent / "faiss_index"
        index_name, meta_name = SEARCH_INDEX_FILENAMES.get(mode, SEARCH_INDEX_FILENAMES["production"])
        return cache_root / index_name, cache_root / meta_name

    @classmethod
    def load_cached_or_build(
        cls,
        mode: str = "production",
        data_root: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        clip_model_name: str = CLIP_MODEL_NAME,
    ) -> "MultimodalSearchEngine":
        index_path, meta_path = cls.artifact_paths_for_mode(mode=mode, data_root=data_root, cache_dir=cache_dir)
        if index_path.exists() and meta_path.exists() and cls._cached_artifacts_are_current(meta_path, mode, clip_model_name):
            try:
                LOGGER.info("Loading cached %s search artifacts from %s", mode, index_path)
                return cls.load_from_artifacts(
                    str(index_path),
                    str(meta_path),
                    mode=mode,
                    data_root=data_root,
                    clip_model_name=clip_model_name,
                )
            except Exception as exc:
                LOGGER.warning("Failed to load cached %s artifacts from %s: %s", mode, index_path, exc)
        elif index_path.exists() and meta_path.exists():
            LOGGER.info("Cached %s search artifacts are stale. Rebuilding %s artifacts.", mode, SEARCH_INDEX_FORMAT)

        LOGGER.info("No reusable %s search artifacts were found. Building index once and saving cache.", mode)
        engine = cls(mode=mode, data_root=data_root, clip_model_name=clip_model_name)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        engine.save_index(str(index_path), str(meta_path))
        LOGGER.info("Saved %s search artifacts to %s", mode, index_path)
        return engine

    @staticmethod
    def _cached_artifacts_are_current(meta_path: Path, mode: str, clip_model_name: str = CLIP_MODEL_NAME) -> bool:
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed to read cached search metadata %s: %s", meta_path, exc)
            return False
        if str(payload.get("index_format", "")) not in SUPPORTED_SEARCH_INDEX_FORMATS:
            return False
        normalized_mode = (mode or "").lower().strip()
        if str(payload.get("mode", "")).lower() != normalized_mode:
            return False

        build_config = payload.get("build_config", {})
        if not isinstance(build_config, dict):
            return False
        if str(build_config.get("clip_model_name", "")) != clip_model_name:
            return False
        payload_items = payload.get("items", [])
        if not isinstance(payload_items, list):
            return False
        try:
            cached_item_count = int(build_config.get("item_count", len(payload_items)))
        except (TypeError, ValueError):
            return False
        if cached_item_count != len(payload_items):
            return False
        if normalized_mode in {"dev", "production"}:
            source_articles = build_config.get("source_articles", {})
            if not MultimodalSearchEngine._cached_source_articles_are_current(source_articles):
                return False
        if normalized_mode != "dev":
            return True

        expected_sample_size = int(os.getenv("SEARCH_ENGINE_DEV_SAMPLE_SIZE", str(DEFAULT_DEV_SAMPLE_SIZE)))
        expected_random_seed = int(os.getenv("SEARCH_ENGINE_RANDOM_SEED", str(DEFAULT_RANDOM_SEED)))
        try:
            cached_sample_size = int(build_config.get("dev_sample_size", -1))
            cached_random_seed = int(build_config.get("random_seed", -1))
        except (TypeError, ValueError):
            return False
        return (
            cached_sample_size == expected_sample_size
            and cached_random_seed == expected_random_seed
            and cached_item_count >= expected_sample_size
        )

    @staticmethod
    def _cached_source_articles_are_current(source_config: Any) -> bool:
        if not isinstance(source_config, dict):
            return False
        source_path = str(source_config.get("path", "")).strip()
        if not source_path:
            return False
        try:
            stat = Path(source_path).stat()
            cached_size = int(source_config.get("size", -1))
            cached_mtime_ns = int(source_config.get("mtime_ns", -1))
        except (OSError, TypeError, ValueError):
            return False
        return cached_size == int(stat.st_size) and cached_mtime_ns == int(stat.st_mtime_ns)

    def find_item(self, product_id: str) -> Optional[SearchItem]:
        needle = normalize_article_id(product_id)
        for item in self.items:
            if normalize_article_id(item.product_id) == needle:
                return item
        return None

def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _sample_queries(mode: str) -> Iterable[str]:
    if mode == "production":
        return ("dress", "black jeans", "blue jacket")
    return ("white shirt", "running shorts", "canvas bag")


if __name__ == "__main__":
    _configure_logging()
    selected_mode = os.getenv("MODE", "test")
    LOGGER.info("Starting standalone search engine in %s mode", selected_mode)
    configured_data_root = os.getenv("SEARCH_ENGINE_DATA_ROOT") or os.getenv("DATA_ROOT")
    engine = MultimodalSearchEngine.load_cached_or_build(mode=selected_mode, data_root=configured_data_root)
    print(f"[search_engine] mode={selected_mode}")
    print(f"[search_engine] data_root={engine.data_root}")
    print(f"[search_engine] items={len(engine.items)}")
    print(f"[search_engine] dimension={engine.dimension}")
    for sample_query in _sample_queries(selected_mode):
        result = engine.search(query=sample_query, top_k=3)
        print(f"[search_engine] query={sample_query!r} -> {json.dumps(result, ensure_ascii=False)}")

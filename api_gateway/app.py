"""
API Gateway — port 8000

엔드포인트:
  POST /api/search             search-engine 프록시
  POST /api/personalized-search search 후보를 rec-models로 개인화 재정렬
  GET  /api/recommend          Redis 세션 붙여서 rec-models 프록시
  POST /api/events             Redis에 클릭/구매 이벤트 저장
  GET  /api/features/{user_id} Redis 유저 피처 조회
  GET  /api/images/{article_id} 상품 이미지 반환
  POST /api/onboarding         LLM 기반 콜드 스타트 페르소나 생성 (기능 C)
  POST /api/budget-set         예산 기반 패션 세트 추천 (기능 D)
  GET  /health
"""

import csv
import hashlib
import json
import logging
import os
import re
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from feature_store import RedisFeatureStore
from persona_registry import (
    DEFAULT_PERSONA_SCORES,
    PERSONA_KEYWORD_RULES,
    PERSONA_KEYS,
    PERSONA_SESSION_INTERESTS,
)

# ── 서비스 URL (docker-compose 서비스명 또는 환경변수로 오버라이드) ──
SEARCH_URL = os.getenv("SEARCH_ENGINE_URL", "http://search-engine:8002")
REC_URL = os.getenv("REC_MODELS_URL", "http://rec-models:8003")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_COOLDOWN_SECONDS = int(os.getenv("GEMINI_COOLDOWN_SECONDS", "600"))
GEMINI_COOLDOWN_KEY = "service:gemini:cooldown"

DEFAULT_IMAGE_ROOT = Path("/app/data/raw/images")
LOCAL_IMAGE_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "images"
USB_IMAGE_ROOT = Path("D:/imagedata")
IMAGE_ROOT = Path(os.getenv("IMAGE_ROOT", str(DEFAULT_IMAGE_ROOT)))
if not IMAGE_ROOT.exists():
    for candidate in (LOCAL_IMAGE_ROOT, USB_IMAGE_ROOT):
        if candidate.exists():
            IMAGE_ROOT = candidate
            break

ARTICLES_PATH = Path("/app/data/processed/articles_feature.csv")
ARTICLE_PRICE_MAP_PATH = Path("/app/data/processed/article_price_map.csv")
LOCAL_ARTICLE_PRICE_MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "article_price_map.csv"
RAW_TRANSACTIONS_PATH = Path("/app/data/raw/transactions_train.csv")
LOCAL_RAW_TRANSACTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "transactions_train.csv"
# item_features_{test,dev,prod}.csv — avg_price 컬럼 포함
_ITEM_FEATURES_CANDIDATES = [
    Path("/app/data/processed/item_features_test.csv"),
    Path("/app/data/processed/item_features_dev.csv"),
    Path("/app/data/processed/item_features.csv"),
]
# H&M 정규화 가격 → KRW 환산 계수 (중앙값 0.025 ≈ 25,000원 기준)
PRICE_KRW_FACTOR = 1_000_000

feature_store: RedisFeatureStore
# article_id → {name, brand, category, color, product_type, price}
article_meta: dict[str, dict] = {}


def _brand_label(department_name: str | None) -> str:
    department = (department_name or "").strip()
    return f"H&M · {department}" if department else "H&M"


def _load_feature_prices(meta: dict[str, dict]) -> int:
    loaded_count = 0
    for path in _ITEM_FEATURES_CANDIDATES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                aid = row.get("article_id", "").strip()
                if aid not in meta or meta[aid]["price"]:
                    continue
                try:
                    raw = float(row.get("avg_price", 0) or 0)
                except (ValueError, TypeError):
                    raw = 0
                if raw > 0:
                    meta[aid]["price"] = int(raw * PRICE_KRW_FACTOR)
                    meta[aid]["price_source"] = "item_features"
                    loaded_count += 1
    return loaded_count


def _load_price_map_prices(meta: dict[str, dict]) -> int:
    price_map_path = next(
        (path for path in (ARTICLE_PRICE_MAP_PATH, LOCAL_ARTICLE_PRICE_MAP_PATH) if path.exists()),
        None,
    )
    if price_map_path is None:
        return 0

    loaded_count = 0
    with price_map_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row.get("article_id", "").strip()
            if aid not in meta or meta[aid]["price"]:
                continue
            try:
                raw = float(row.get("avg_price", row.get("price_mean", 0)) or 0)
            except (ValueError, TypeError):
                raw = 0
            if raw > 0:
                meta[aid]["price"] = int(raw * PRICE_KRW_FACTOR)
                meta[aid]["price_source"] = "article_price_map"
                loaded_count += 1
    return loaded_count


def _raw_transaction_backfill_enabled() -> bool:
    return os.getenv("ENABLE_RAW_TRANSACTION_PRICE_BACKFILL", "0").strip().lower() in {"1", "true", "yes", "on"}


def _transaction_price_path() -> Path | None:
    if not _raw_transaction_backfill_enabled():
        logging.info(
            "Raw transaction price backfill disabled. Run data_pipeline/build_article_price_map.py "
            "to create article_price_map.csv for startup-safe price coverage."
        )
        return None
    configured_path = Path(os.getenv("TRANSACTIONS_PATH", "")).expanduser() if os.getenv("TRANSACTIONS_PATH") else None
    candidates = [path for path in (configured_path, RAW_TRANSACTIONS_PATH, LOCAL_RAW_TRANSACTIONS_PATH) if path]
    return next((path for path in candidates if path.exists()), None)


def _load_transaction_prices(meta: dict[str, dict]) -> int:
    transaction_path = _transaction_price_path()
    if transaction_path is None:
        return 0

    missing_ids = {aid for aid, item in meta.items() if not item.get("price")}
    if not missing_ids:
        return 0

    price_sums: dict[str, float] = {}
    price_counts: dict[str, int] = {}
    with transaction_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row.get("article_id", "").strip()
            if aid not in missing_ids:
                continue
            try:
                price = float(row.get("price", 0) or 0)
            except (ValueError, TypeError):
                continue
            if price <= 0:
                continue
            price_sums[aid] = price_sums.get(aid, 0.0) + price
            price_counts[aid] = price_counts.get(aid, 0) + 1

    for aid, price_sum in price_sums.items():
        count = price_counts.get(aid, 0)
        if count > 0:
            meta[aid]["price"] = int((price_sum / count) * PRICE_KRW_FACTOR)
            meta[aid]["price_source"] = "transactions"

    logging.info(
        "Loaded transaction prices for %d articles from %s",
        len(price_sums),
        transaction_path,
    )
    return len(price_sums)


def _load_article_meta() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not ARTICLES_PATH.exists():
        return meta
    with ARTICLES_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row.get("article_id", "").strip()
            if aid:
                department_name = row.get("department_name", "")
                section_name = row.get("section_name", "")
                meta[aid] = {
                    "name": row.get("prod_name", ""),
                    "brand": _brand_label(department_name),
                    "category": row.get("category", ""),
                    "main_category": row.get("main_category", ""),
                    "color": row.get("color", ""),
                    "product_type": row.get("product_type_name", ""),
                    "product_group": row.get("product_group_name", ""),
                    "department_name": department_name,
                    "section_name": section_name,
                    "garment_group": row.get("garment_group_name", ""),
                    "price": 0,
                    "price_source": "",
                }

    price_map_count = _load_price_map_prices(meta)
    feature_price_count = _load_feature_prices(meta)
    transaction_price_count = _load_transaction_prices(meta)
    priced_count = sum(1 for item in meta.values() if item.get("price"))
    logging.info(
        "Article metadata loaded: total=%d priced=%d price_map_prices=%d item_feature_prices=%d transaction_prices=%d coverage=%.1f%%",
        len(meta),
        priced_count,
        price_map_count,
        feature_price_count,
        transaction_price_count,
        (priced_count * 100 / len(meta)) if meta else 0.0,
    )

    return meta


_outfit_slot_index: dict[str, list[str]] = {}  # slot -> [article_id, ...]


def _outfit_slot(meta: dict) -> str:
    """아이템의 outfit slot을 반환한다 (top/bottom/outer/dress/accessory/shoes/other)."""
    group = str(meta.get("product_group", "")).lower()
    ptype = str(meta.get("product_type", "")).lower()
    if "lower body" in group:
        return "bottom"
    if "full body" in group:
        return "dress"
    if "upper body" in group:
        if any(k in ptype for k in ("coat", "jacket", "blazer", "cardigan", "waistcoat")):
            return "outer"
        return "top"
    if "shoe" in group:
        return "shoes"
    if group in ("accessories", "bags", "socks & tights"):
        return "accessory"
    if "underwear" in group or "nightwear" in group:
        return "underwear"
    return "other"


def _build_slot_index(meta: dict[str, dict]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for aid, item_meta in meta.items():
        slot = _outfit_slot(item_meta)
        if slot == "other":
            continue
        index.setdefault(slot, []).append(aid)
    return index


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feature_store, article_meta, _outfit_slot_index
    feature_store = RedisFeatureStore(host=REDIS_HOST, port=REDIS_PORT)
    article_meta = _load_article_meta()
    _outfit_slot_index = _build_slot_index(article_meta)
    yield


app = FastAPI(title="API Gateway", lifespan=lifespan)


def image_path_for_article(article_id: str) -> Path:
    normalized_id = article_id.strip()
    if not normalized_id.isdigit() or len(normalized_id) < 3:
        raise HTTPException(status_code=400, detail="Invalid article_id")
    return IMAGE_ROOT / normalized_id[:3] / f"{normalized_id}.jpg"


def image_url_for_article(article_id: str) -> str:
    normalized_id = article_id.strip()
    if not normalized_id or not normalized_id.isdigit():
        return ""
    return f"/api/images/{normalized_id}"


COLOR_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "black": ("black", "검정", "검은", "블랙"),
    "white": ("white", "흰색", "하얀", "화이트"),
    "blue": ("blue", "파랑", "파란", "블루", "네이비"),
    "red": ("red", "빨강", "빨간", "레드"),
    "pink": ("pink", "핑크", "분홍"),
    "grey": ("grey", "gray", "회색", "그레이"),
    "beige": ("beige", "베이지"),
    "green": ("green", "초록", "그린", "카키"),
    "yellow": ("yellow", "노랑", "옐로"),
}

COLOR_INTENT_TERMS: dict[str, str] = {
    "black": "black",
    "white": "white",
    "blue": "blue navy",
    "red": "red",
    "pink": "pink",
    "grey": "grey gray",
    "beige": "beige",
    "green": "green khaki",
    "yellow": "yellow",
}

TARGET_AUDIENCE_TO_MAIN_CATEGORY: dict[str, set[str]] = {
    "all": set(),
    "women": {"Ladieswear", "Divided"},
    "men": {"Menswear"},
    "kids": {"Baby/Children"},
}

TARGET_AUDIENCE_EXCLUDED_MAIN_CATEGORY: dict[str, set[str]] = {
    "women": {"Menswear", "Baby/Children"},
    "men": {"Ladieswear", "Divided", "Baby/Children"},
    "kids": {"Ladieswear", "Menswear", "Divided", "Sport"},
}

TARGET_AUDIENCE_TOKEN_MARKERS: dict[str, set[str]] = {
    "women": {"ladies", "ladieswear", "lady", "women", "womens", "woman", "female"},
    "men": {"men", "mens", "menswear", "man", "male"},
    "kids": {"baby", "children", "child", "kids", "kid", "boy", "boys", "girl", "girls", "toddler"},
}
OUTFIT_ITEM_FIT_WEIGHT = 0.45
OUTFIT_COMPATIBILITY_WEIGHT = 0.25
OUTFIT_QUERY_MATCH_WEIGHT = 0.20
OUTFIT_BUDGET_FIT_WEIGHT = 0.10

TARGET_AUDIENCE_TO_INTEREST: dict[str, str] = {
    "women": "Ladieswear",
    "men": "Menswear",
    "kids": "Kids",
}


def _normalize_target_audience(target_audience: str | None) -> str:
    target = (target_audience or "all").strip().lower()
    return target if target in TARGET_AUDIENCE_TO_MAIN_CATEGORY else "all"


def _target_audience_tokens(item: dict) -> set[str]:
    text = " ".join(
        str(item.get(key, "") or "")
        for key in (
            "main_category",
            "department_name",
            "section_name",
            "garment_group",
            "garment_group_name",
            "brand",
        )
    )
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _tokens_match_audience(tokens: set[str], target_audience: str) -> bool:
    markers = TARGET_AUDIENCE_TOKEN_MARKERS.get(target_audience, set())
    return any(token in markers for token in tokens)


def _item_matches_target_audience(item: dict, target_audience: str | None) -> bool:
    target = _normalize_target_audience(target_audience)
    if target == "all":
        return True

    main_category = str(item.get("main_category", "") or "").strip()
    if main_category in TARGET_AUDIENCE_TO_MAIN_CATEGORY[target]:
        return True
    if main_category in TARGET_AUDIENCE_EXCLUDED_MAIN_CATEGORY.get(target, set()):
        return False
    return _tokens_match_audience(_target_audience_tokens(item), target)


def _apply_target_audience_filter(
    items: list[dict],
    target_audience: str | None,
    *,
    min_results: int = 1,
    warn_below: int | None = None,
) -> list[dict]:
    target = _normalize_target_audience(target_audience)
    if target == "all":
        return items
    filtered = [item for item in items if _item_matches_target_audience(item, target)]
    threshold = min_results if warn_below is None else warn_below
    if len(filtered) < threshold:
        logging.info(
            "Target audience filter reduced results below requested minimum: target=%s before=%d after=%d min=%d",
            target,
            len(items),
            len(filtered),
            threshold,
        )
    return filtered


def _target_audience_interest(target_audience: str | None) -> str | None:
    return TARGET_AUDIENCE_TO_INTEREST.get(_normalize_target_audience(target_audience))


DEFAULT_PRICE_BY_PRODUCT: tuple[tuple[tuple[str, ...], int], ...] = (
    (("coat", "jacket", "blazer", "parka"), 79000),
    (("dress", "jumpsuit", "playsuit"), 59000),
    (("jeans", "trousers", "pants", "slacks", "chino", "jogger", "cargo"), 49000),
    (("skirt", "shorts"), 39000),
    (("shirt", "blouse", "sweater", "cardigan", "hoodie", "sweatshirt"), 39000),
    (("sneakers", "shoes", "boots", "sandals", "slippers", "pumps", "heels"), 69000),
    (("bag", "backpack"), 49000),
    (("swimsuit", "bikini"), 29000),
    (("bra", "underwear", "panties"), 19000),
    (("pyjama", "pajama", "robe", "night gown"), 39000),
    (("t-shirt", "tee", "top", "vest"), 25000),
)


def _estimated_price_for_meta(meta: dict, item: dict | None = None) -> int:
    item = item or {}
    haystack = " ".join(
        str(value or "").lower()
        for value in (
            meta.get("category"),
            meta.get("product_type"),
            meta.get("name"),
            item.get("category"),
            item.get("product_type"),
            item.get("name"),
            item.get("title"),
        )
    )
    for terms, price in DEFAULT_PRICE_BY_PRODUCT:
        if any(term in haystack for term in terms):
            return price
    return 39000


def _display_price(meta: dict, item: dict | None = None) -> tuple[int, bool]:
    raw_price = meta.get("price", 0)
    try:
        price = int(float(raw_price or 0))
    except (TypeError, ValueError):
        price = 0
    if price > 0:
        return price, False
    return _estimated_price_for_meta(meta, item), True


def _price_source_label(meta: dict, price_estimated: bool) -> str:
    if price_estimated:
        return "estimated"
    return str(meta.get("price_source") or "observed")


def _term_matches_text(text: str, term: str) -> bool:
    normalized_term = " ".join(str(term or "").lower().split())
    if not normalized_term:
        return False
    if any(ord(char) > 127 for char in normalized_term):
        return normalized_term in text

    pattern = re.escape(normalized_term).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None


def _any_term_matches_text(text: str, terms: tuple[str, ...]) -> bool:
    return any(_term_matches_text(text, term) for term in terms)


PRODUCT_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "dress": ("dress", "one-piece", "one piece", "원피스", "드레스"),
    "t-shirt": ("t-shirt", "tee", "t shirt", "티셔츠", "반팔"),
    "sweatshirt": ("sweatshirt", "sweat shirt", "맨투맨", "스웨트셔츠"),
    "sweater": ("sweater", "knit", "knitwear", "jumper", "pullover", "니트", "스웨터"),
    "cardigan": ("cardigan", "가디건"),
    "shirt": ("shirt", "셔츠"),
    "blouse": ("blouse", "블라우스"),
    "hoodie": ("hoodie", "hood", "후드", "후디"),
    "jacket": ("jacket", "자켓", "재킷"),
    "blazer": ("blazer", "블레이저"),
    "coat": ("coat", "코트"),
    "skirt": ("skirt", "스커트", "치마"),
    "trousers": (
        "trousers",
        "pants",
        "slacks",
        "slack",
        "chino",
        "chinos",
        "jogger",
        "joggers",
        "cargo pants",
        "cargo",
        "sweatpants",
        "바지",
        "팬츠",
        "슬랙스",
        "치노",
        "조거",
        "조거팬츠",
        "카고",
        "카고팬츠",
        "트레이닝바지",
        "와이드팬츠",
    ),
    "shorts": ("shorts", "short pants", "반바지", "쇼츠", "숏팬츠"),
    "jeans": ("jeans", "denim", "청바지", "데님"),
    "sneakers": ("sneakers", "운동화", "스니커즈"),
    "boots": ("boots", "boot", "부츠"),
    "sandals": ("sandals", "sandal", "heeled sandals", "flip flop", "flip-flop", "쪼리", "샌들", "플립플롭"),
    "slippers": ("slippers", "slipper", "slider", "slides", "슬리퍼", "실내화"),
    "heels": ("heels", "heel", "pumps", "pump", "wedge", "heeled sandals", "힐", "구두", "펌프스", "웨지"),
    "flats": ("flats", "flat shoe", "flat shoes", "ballerinas", "ballerina", "loafer", "loafers", "플랫", "로퍼", "발레리나"),
    "shoes": ("shoes", "shoe", "신발"),
    "vest-top": ("vest top", "tank top", "tanktop", "sleeveless top", "waistcoat", "나시", "민소매", "탱크탑", "조끼", "베스트"),
    "polo": ("polo", "polo shirt", "폴로", "폴로티", "카라티"),
    "bodysuit": ("bodysuit", "body suit", "바디수트"),
    "jumpsuit": ("jumpsuit", "playsuit", "점프수트", "플레이수트"),
    "dungarees": ("dungarees", "overall", "overalls", "멜빵바지", "오버롤"),
    "garment-set": ("garment set", "set", "two-piece", "two piece", "세트", "투피스"),
    "swimwear": ("swimwear", "swimsuit", "bikini", "bikini top", "rash guard", "rashguard", "수영복", "비키니", "래시가드"),
    "underwear": ("underwear", "panties", "briefs", "boxers", "속옷", "팬티", "드로즈", "브리프"),
    "bra": ("bra", "bralette", "브라", "브래지어"),
    "pyjama": ("pyjama", "pajama", "nightwear", "night gown", "nightgown", "sleepwear", "잠옷", "파자마", "나이트가운"),
    "robe": ("robe", "bathrobe", "가운", "로브"),
    "bag": (
        "bag",
        "backpack",
        "cross-body bag",
        "crossbody",
        "tote",
        "tote bag",
        "shoulder bag",
        "bumbag",
        "weekend bag",
        "가방",
        "백",
        "백팩",
        "크로스백",
        "토트백",
        "숄더백",
        "힙색",
    ),
    "hat": (
        "hat",
        "cap",
        "beanie",
        "bucket hat",
        "straw hat",
        "strawhat",
        "sunhat",
        "snapback",
        "fedora",
        "모자",
        "캡",
        "비니",
        "버킷햇",
        "벙거지",
        "볼캡",
        "야구모자",
        "스냅백",
        "페도라",
    ),
    "socks": ("socks", "sock", "ankle sock", "anklesock", "sportsock", "양말", "삭스"),
    "tights": ("tights", "legging", "leggings", "leggings/tights", "stockings", "스타킹", "타이즈", "타이츠", "레깅스"),
    "scarf": ("scarf", "scarves", "muffler", "스카프", "머플러"),
    "belt": ("belt", "벨트"),
    "gloves": ("gloves", "glove", "장갑"),
    "sunglasses": ("sunglasses", "shades", "선글라스"),
    "jewelry": ("jewelry", "jewellery", "주얼리", "쥬얼리", "장신구"),
    "earrings": ("earring", "earrings", "귀걸이", "이어링"),
    "necklace": ("necklace", "목걸이"),
    "ring": ("ring", "rings", "반지"),
    "bracelet": ("bracelet", "bracelets", "팔찌"),
    "watch": ("watch", "watches", "시계"),
    "hair-accessory": (
        "hair clip",
        "hair string",
        "hair ties",
        "hair band",
        "hairband",
        "alice band",
        "headband",
        "헤어밴드",
        "머리띠",
        "헤어핀",
        "머리핀",
        "헤어끈",
        "머리끈",
        "헤어악세사리",
        "헤어액세서리",
    ),
    "wallet": ("wallet", "지갑"),
    "umbrella": ("umbrella", "우산"),
    "tie": ("tie", "necktie", "넥타이"),
    "accessory": ("accessory", "accessories", "액세서리", "엑세서리", "악세사리", "잡화"),
}

PRODUCT_MATCH_TERMS: dict[str, tuple[str, ...]] = {
    "dress": ("dress",),
    "t-shirt": ("t-shirt", "tee"),
    "sweatshirt": ("sweatshirt", "sweat shirt"),
    "sweater": ("sweater", "knit", "knitted", "jumper", "pullover", "cardigan"),
    "cardigan": ("cardigan",),
    "shirt": ("shirt",),
    "blouse": ("blouse",),
    "hoodie": ("hoodie", "hooded", "hood"),
    "jacket": ("jacket", "blazer", "parka"),
    "blazer": ("blazer",),
    "coat": ("coat",),
    "skirt": ("skirt",),
    "trousers": (
        "trousers",
        "pants",
        "slacks",
        "slack",
        "chino",
        "chinos",
        "jogger",
        "joggers",
        "cargo pants",
        "cargo",
        "sweatpants",
        "outdoor trousers",
    ),
    "shorts": ("shorts",),
    "jeans": ("jeans", "denim"),
    "sneakers": ("sneakers", "trainer"),
    "boots": ("boots", "boot"),
    "sandals": ("sandals", "sandal", "heeled sandals", "flip flop", "flip-flop"),
    "slippers": ("slippers", "slipper", "slider", "slides"),
    "heels": ("heels", "heel", "pumps", "pump", "wedge", "heeled sandals"),
    "flats": ("flat shoe", "flat shoes", "ballerinas", "ballerina", "loafer", "loafers", "car shoe", "espadrille"),
    "shoes": (
        "shoes",
        "shoe",
        "sneakers",
        "boots",
        "sandals",
        "slippers",
        "pumps",
        "heels",
        "flat shoe",
        "ballerinas",
    ),
    "vest-top": ("vest top", "tanktop", "tank top", "waistcoat", "outdoor waistcoat", "tailored waistcoat"),
    "polo": ("polo shirt", "polo"),
    "bodysuit": ("bodysuit", "body suit"),
    "jumpsuit": ("jumpsuit/playsuit", "jumpsuit", "playsuit"),
    "dungarees": ("dungarees", "overall", "overalls"),
    "garment-set": ("garment set",),
    "swimwear": ("swimwear", "swimsuit", "swimwear bottom", "swimwear top", "swimwear set", "bikini top", "bikini", "sarong"),
    "underwear": ("underwear bottom", "underwear body", "underwear set", "kids underwear top", "briefs", "boxers", "panties"),
    "bra": ("bra", "bralette"),
    "pyjama": ("pyjama set", "pyjama jumpsuit/playsuit", "pyjama bottom", "night gown", "nightgown", "sleeping sack"),
    "robe": ("robe",),
    "bag": ("bag", "backpack", "cross-body bag", "crossbody", "tote bag", "shoulder bag", "bumbag", "weekend/gym bag"),
    "hat": (
        "hat/beanie",
        "hat/brim",
        "hat",
        "beanie",
        "cap/peaked",
        "cap",
        "bucket hat",
        "straw hat",
        "strawhat",
        "sunhat",
        "felt hat",
        "p-cap",
        "peak cap",
        "snowcap",
    ),
    "socks": ("socks", "sock", "ankle sock", "anklesock", "sportsock"),
    "tights": ("leggings/tights", "underwear tights", "leggings", "legging", "tights", "stockings"),
    "scarf": ("scarf",),
    "belt": ("belt",),
    "gloves": ("gloves", "glove"),
    "sunglasses": ("sunglasses",),
    "jewelry": ("jewelry", "jewellery", "earring", "earrings", "necklace", "ring", "bracelet", "watch"),
    "earrings": ("earring", "earrings"),
    "necklace": ("necklace",),
    "ring": ("ring",),
    "bracelet": ("bracelet",),
    "watch": ("watch",),
    "hair-accessory": ("hair/alice band", "hair clip", "hair string", "hair ties", "hairband", "alice band", "headband"),
    "wallet": ("wallet",),
    "umbrella": ("umbrella",),
    "tie": ("tie",),
    "accessory": (
        "accessories",
        "accessories set",
        "other accessories",
        "bag",
        "backpack",
        "cross-body bag",
        "tote bag",
        "shoulder bag",
        "bumbag",
        "hat/beanie",
        "hat/brim",
        "cap/peaked",
        "bucket hat",
        "straw hat",
        "felt hat",
        "scarf",
        "belt",
        "gloves",
        "sunglasses",
        "earring",
        "earrings",
        "necklace",
        "ring",
        "bracelet",
        "watch",
        "hair/alice band",
        "hair clip",
        "hair string",
        "hair ties",
        "wallet",
        "umbrella",
        "tie",
    ),
}

PRODUCT_INTENT_TERMS: dict[str, tuple[str, ...]] = {
    "dress": ("dress", "one-piece dress"),
    "t-shirt": ("t-shirt", "tee", "short sleeve top"),
    "sweatshirt": ("sweatshirt", "crewneck sweatshirt"),
    "sweater": ("sweater", "knitwear", "jumper", "pullover"),
    "cardigan": ("cardigan", "knit cardigan"),
    "shirt": ("shirt", "button shirt"),
    "blouse": ("blouse",),
    "hoodie": ("hoodie", "hooded sweatshirt"),
    "jacket": ("jacket", "outerwear"),
    "blazer": ("blazer", "tailored jacket"),
    "coat": ("coat", "outerwear"),
    "skirt": ("skirt",),
    "trousers": ("trousers", "pants", "slacks", "chinos", "joggers", "cargo pants"),
    "shorts": ("shorts", "short pants"),
    "jeans": ("jeans", "denim pants"),
    "sneakers": ("sneakers", "trainers"),
    "boots": ("boots",),
    "sandals": ("sandals", "flip flop", "heeled sandals"),
    "slippers": ("slippers", "slides"),
    "heels": ("heels", "pumps", "wedge heels"),
    "flats": ("flat shoes", "ballerinas", "loafers"),
    "shoes": ("shoes",),
    "vest-top": ("vest top", "tank top", "sleeveless top"),
    "polo": ("polo shirt",),
    "bodysuit": ("bodysuit",),
    "jumpsuit": ("jumpsuit", "playsuit"),
    "dungarees": ("dungarees", "overalls"),
    "garment-set": ("garment set", "two-piece set"),
    "swimwear": ("swimwear", "swimsuit", "bikini"),
    "underwear": ("underwear", "panties", "briefs"),
    "bra": ("bra", "bralette"),
    "pyjama": ("pyjama", "pajama", "nightwear", "sleepwear"),
    "robe": ("robe", "bathrobe"),
    "bag": ("bag", "backpack", "tote bag", "shoulder bag", "cross-body bag"),
    "hat": ("hat", "cap", "beanie", "bucket hat", "straw hat"),
    "socks": ("socks", "ankle socks"),
    "tights": ("tights", "leggings", "stockings"),
    "scarf": ("scarf", "muffler"),
    "belt": ("belt",),
    "gloves": ("gloves",),
    "sunglasses": ("sunglasses",),
    "jewelry": ("jewelry", "earrings", "necklace", "ring", "bracelet"),
    "earrings": ("earrings", "earring"),
    "necklace": ("necklace",),
    "ring": ("ring",),
    "bracelet": ("bracelet",),
    "watch": ("watch",),
    "hair-accessory": ("hair accessory", "hair clip", "headband", "hair ties"),
    "wallet": ("wallet",),
    "umbrella": ("umbrella",),
    "tie": ("tie", "necktie"),
    "accessory": ("accessories", "fashion accessories"),
}

PRODUCT_DISPLAY_LABELS: dict[str, str] = {
    "dress": "Dress",
    "t-shirt": "T-shirt",
    "sweatshirt": "Sweatshirt",
    "sweater": "Sweater",
    "cardigan": "Cardigan",
    "shirt": "Shirt",
    "blouse": "Blouse",
    "hoodie": "Hoodie",
    "jacket": "Jacket",
    "blazer": "Blazer",
    "coat": "Coat",
    "skirt": "Skirt",
    "trousers": "Trousers",
    "shorts": "Shorts",
    "jeans": "Jeans",
    "sneakers": "Sneakers",
    "boots": "Boots",
    "sandals": "Sandals",
    "slippers": "Slippers",
    "heels": "Heels",
    "flats": "Flats",
    "shoes": "Shoes",
    "vest-top": "Vest top",
    "polo": "Polo shirt",
    "bodysuit": "Bodysuit",
    "jumpsuit": "Jumpsuit/Playsuit",
    "dungarees": "Dungarees",
    "garment-set": "Garment Set",
    "swimwear": "Swimwear",
    "underwear": "Underwear",
    "bra": "Bra",
    "pyjama": "Pyjama",
    "robe": "Robe",
    "bag": "Bag",
    "hat": "Hat",
    "socks": "Socks",
    "tights": "Tights",
    "scarf": "Scarf",
    "belt": "Belt",
    "gloves": "Gloves",
    "sunglasses": "Sunglasses",
    "jewelry": "Jewelry",
    "earrings": "Earrings",
    "necklace": "Necklace",
    "ring": "Ring",
    "bracelet": "Bracelet",
    "watch": "Watch",
    "hair-accessory": "Hair accessory",
    "wallet": "Wallet",
    "umbrella": "Umbrella",
    "tie": "Tie",
    "accessory": "Accessory",
}

PRODUCT_EXCLUDE_TERMS: dict[str, tuple[str, ...]] = {
    "shirt": ("t-shirt", "t shirt", "tee", "sweatshirt", "hoodie", "hooded"),
    "sneakers": ("boots", "boot", "heels", "sandals", "slippers"),
    "boots": ("bootcut",),
    "hoodie": (),
    "socks": ("sock runner", "sock sneaker", "sockboot", "sock boot", "sneakers", "shoes", "boots"),
    "shorts": ("shorttop", "short-sleeved", "short sleeve"),
    "bra": ("bracelet",),
    "robe": ("wardrobe",),
}

PRODUCT_DOMINANCE_RULES: dict[str, tuple[str, ...]] = {
    "t-shirt": ("shirt",),
    "cardigan": ("sweater",),
    "hoodie": ("sweatshirt", "sweater"),
    "blazer": ("jacket",),
    "jeans": ("trousers",),
    "shorts": ("trousers",),
    "sneakers": ("shoes",),
    "boots": ("shoes",),
    "sandals": ("shoes",),
    "slippers": ("shoes",),
    "heels": ("shoes", "sandals"),
    "flats": ("shoes",),
    "bra": ("underwear",),
    "robe": ("pyjama",),
    "dungarees": ("trousers",),
    "bag": ("accessory",),
    "hat": ("accessory",),
    "socks": ("accessory",),
    "tights": ("accessory",),
    "scarf": ("accessory",),
    "belt": ("accessory",),
    "gloves": ("accessory",),
    "sunglasses": ("accessory",),
    "jewelry": ("accessory",),
    "earrings": ("jewelry", "accessory"),
    "necklace": ("jewelry", "accessory"),
    "ring": ("jewelry", "accessory"),
    "bracelet": ("jewelry", "accessory"),
    "watch": ("jewelry", "accessory"),
    "hair-accessory": ("accessory",),
    "wallet": ("accessory",),
    "umbrella": ("accessory",),
    "tie": ("accessory",),
}

COMPLEMENT_PRODUCT_GROUPS: dict[str, tuple[str, ...]] = {
    "dress": ("jacket", "coat", "sweater", "cardigan", "boots", "heels", "sneakers", "bag"),
    "t-shirt": ("trousers", "jeans", "shorts", "skirt", "jacket", "sneakers", "bag", "hat"),
    "shirt": ("trousers", "jeans", "shorts", "skirt", "jacket", "blazer", "sneakers", "bag", "hat"),
    "hoodie": ("trousers", "jeans", "sneakers", "jacket", "bag", "hat"),
    "cardigan": ("t-shirt", "shirt", "dress", "jeans", "skirt"),
    "jacket": ("t-shirt", "shirt", "trousers", "jeans", "skirt", "dress"),
    "blazer": ("shirt", "blouse", "trousers", "jeans", "skirt", "dress"),
    "coat": ("t-shirt", "shirt", "sweater", "trousers", "jeans", "skirt", "dress", "scarf"),
    "skirt": ("t-shirt", "shirt", "sweater", "jacket", "sneakers"),
    "trousers": ("t-shirt", "shirt", "sweater", "jacket", "sneakers", "belt"),
    "shorts": ("t-shirt", "shirt", "sneakers", "sandals"),
    "jeans": ("t-shirt", "shirt", "sweater", "jacket", "sneakers"),
    "sneakers": ("t-shirt", "hoodie", "trousers", "jeans"),
    "sandals": ("dress", "skirt", "shorts", "swimwear"),
    "slippers": ("pyjama", "robe", "shorts"),
    "heels": ("dress", "skirt", "blazer", "trousers"),
    "flats": ("dress", "skirt", "trousers", "jeans"),
    "vest-top": ("shorts", "jeans", "skirt", "cardigan"),
    "polo": ("trousers", "shorts", "jeans", "sneakers"),
    "swimwear": ("sandals", "hat", "sunglasses", "bag"),
    "underwear": ("robe", "pyjama"),
    "bra": ("underwear", "robe"),
    "pyjama": ("robe", "slippers"),
    "robe": ("pyjama", "slippers"),
    "bag": ("dress", "jacket", "t-shirt", "shirt"),
    "hat": ("hoodie", "jacket", "coat", "t-shirt", "sweater"),
    "socks": ("sneakers", "boots", "trousers", "jeans"),
    "tights": ("skirt", "dress", "boots"),
    "scarf": ("coat", "jacket", "sweater"),
    "belt": ("trousers", "jeans", "dress", "shirt"),
    "gloves": ("coat", "jacket", "scarf"),
    "sunglasses": ("dress", "t-shirt", "shirt", "hat"),
    "jewelry": ("dress", "blouse", "shirt"),
    "accessory": ("dress", "jacket", "t-shirt", "shirt", "bag", "hat"),
}

COLOR_COMPLEMENTS: dict[str, tuple[str, ...]] = {
    "black": ("black", "white", "grey", "beige", "blue"),
    "white": ("white", "black", "blue", "grey", "beige"),
    "blue": ("blue", "white", "black", "grey"),
    "red": ("red", "black", "white", "grey"),
    "pink": ("pink", "white", "grey", "beige"),
    "grey": ("grey", "black", "white", "blue"),
    "beige": ("beige", "white", "black", "blue"),
    "green": ("green", "black", "white", "beige"),
    "yellow": ("yellow", "white", "blue", "black"),
}

PRODUCT_GROUP_PRIORITY: dict[str, int] = {
    "blazer": 92,
    "jacket": 90,
    "coat": 88,
    "shorts": 87,
    "jeans": 86,
    "trousers": 85,
    "skirt": 84,
    "slippers": 83,
    "sandals": 83,
    "heels": 83,
    "flats": 83,
    "sneakers": 82,
    "boots": 82,
    "hat": 80,
    "bag": 78,
    "sunglasses": 77,
    "scarf": 76,
    "belt": 75,
    "gloves": 74,
    "jewelry": 73,
    "hoodie": 72,
    "sweatshirt": 71,
    "cardigan": 71,
    "sweater": 70,
    "blouse": 69,
    "shirt": 68,
    "polo": 67,
    "t-shirt": 66,
    "vest-top": 65,
    "dress": 64,
    "socks": 63,
    "tights": 62,
    "shoes": 60,
    "swimwear": 59,
    "jumpsuit": 59,
    "bodysuit": 58,
    "dungarees": 58,
    "earrings": 58,
    "necklace": 58,
    "ring": 58,
    "bracelet": 58,
    "watch": 58,
    "garment-set": 57,
    "hair-accessory": 57,
    "underwear": 56,
    "bra": 56,
    "pyjama": 56,
    "robe": 56,
    "wallet": 56,
    "umbrella": 56,
    "tie": 55,
    "accessory": 50,
}


def _derive_query_constraints(*texts: str | None) -> dict[str, set[str]]:
    normalized = " ".join(text or "" for text in texts).lower()
    colors = {
        color
        for color, terms in COLOR_QUERY_TERMS.items()
        if _any_term_matches_text(normalized, terms)
    }
    products = {
        product
        for product, terms in PRODUCT_QUERY_TERMS.items()
        if _any_term_matches_text(normalized, terms)
    }
    for specific_product, generic_products in PRODUCT_DOMINANCE_RULES.items():
        if specific_product in products:
            products.difference_update(generic_products)
    return {"colors": colors, "products": products}


def _item_product_text(item: dict) -> str:
    return " ".join(
        str(value or "")
        for value in (
            item.get("name"),
            item.get("title"),
            item.get("category"),
            item.get("product_type"),
            item.get("product_group"),
            item.get("graphical_appearance"),
            item.get("detail_desc"),
        )
    ).lower()


def _item_matches_product(item: dict, product: str) -> bool:
    text = _item_product_text(item)
    terms = PRODUCT_MATCH_TERMS.get(product, (product,))
    if not _any_term_matches_text(text, terms):
        return False

    exclude_terms = PRODUCT_EXCLUDE_TERMS.get(product, ())
    return not _any_term_matches_text(text, exclude_terms)


def _query_constraint_matches(items: list[dict], constraints: dict[str, set[str]]) -> list[dict]:
    if not constraints["products"] and not constraints["colors"]:
        return []
    return [item for item in items if _item_matches_query_constraints(item, constraints)]


def _prioritize_query_constraint_matches(items: list[dict], constraints: dict[str, set[str]]) -> list[dict]:
    if not constraints["products"] and not constraints["colors"]:
        return items

    matching = _query_constraint_matches(items, constraints)
    if not matching:
        return items

    matching_ids = {str(item.get("product_id", item.get("article_id", ""))) for item in matching}
    non_matching = [
        item
        for item in items
        if str(item.get("product_id", item.get("article_id", ""))) not in matching_ids
    ]
    return matching + non_matching


def _matched_product_label(item: dict, constraints: dict[str, set[str]]) -> str | None:
    matched_products = [
        product
        for product in constraints["products"]
        if _item_matches_product(item, product)
    ]
    if not matched_products:
        return None
    best_product = max(matched_products, key=lambda product: PRODUCT_GROUP_PRIORITY.get(product, 0))
    return PRODUCT_DISPLAY_LABELS.get(best_product)


def _apply_query_product_labels(items: list[dict], constraints: dict[str, set[str]]) -> list[dict]:
    if not constraints["products"]:
        return items

    labeled_items: list[dict] = []
    for item in items:
        label = _matched_product_label(item, constraints)
        if not label:
            labeled_items.append(item)
            continue

        labeled_items.append({
            **item,
            "category": label,
            "product_type": label,
            "matched_product_type": label,
        })
    return labeled_items


def _item_matches_query_constraints(item: dict, constraints: dict[str, set[str]]) -> bool:
    colors = constraints["colors"]
    products = constraints["products"]
    item_color = str(item.get("color", "")).lower()

    if colors and not any(color in item_color for color in colors):
        return False
    if products and not any(_item_matches_product(item, product) for product in products):
        return False
    return True


def _item_matches_query_color(item: dict, constraints: dict[str, set[str]]) -> bool:
    colors = constraints["colors"]
    if not colors:
        return True
    item_color = str(item.get("color", "")).lower()
    return any(color in item_color for color in colors)


def _item_matches_query_product(item: dict, constraints: dict[str, set[str]]) -> bool:
    products = constraints["products"]
    if not products:
        return True
    return any(_item_matches_product(item, product) for product in products)


def _catalog_item_for_article(article_id: str, meta: dict, *, score: float) -> dict:
    price, price_estimated = _display_price(meta, None)
    return {
        "product_id": article_id,
        "name": meta.get("name") or article_id,
        "score": score,
        "price": price,
        "image_url": image_url_for_article(article_id),
        "category": meta.get("category", ""),
        "color": meta.get("color", ""),
        "product_type": meta.get("product_type", ""),
        "brand": meta.get("brand") or "H&M",
        "main_category": meta.get("main_category", ""),
        "department_name": meta.get("department_name", ""),
        "section_name": meta.get("section_name", ""),
        "garment_group": meta.get("garment_group", ""),
        "price_estimated": price_estimated,
        "price_source": _price_source_label(meta, price_estimated),
        "reason": "catalog_constraint_backfill",
        "catalog_backfill": True,
    }


def _catalog_constraint_candidates(
    constraints: dict[str, set[str]],
    *,
    existing_ids: set[str],
    limit: int,
    require_color: bool,
    target_audience: str | None = None,
) -> list[dict]:
    if not constraints["products"] or limit <= 0:
        return []

    effective_constraints = {
        "products": set(constraints["products"]),
        "colors": set(constraints["colors"]) if require_color else set(),
    }
    matches: list[dict] = []
    for article_id, meta in article_meta.items():
        if article_id in existing_ids:
            continue
        candidate = {
            "product_id": article_id,
            "name": meta.get("name", ""),
            "category": meta.get("category", ""),
            "product_type": meta.get("product_type", ""),
            "color": meta.get("color", ""),
            "main_category": meta.get("main_category", ""),
            "department_name": meta.get("department_name", ""),
            "section_name": meta.get("section_name", ""),
            "garment_group": meta.get("garment_group", ""),
        }
        if not _item_matches_target_audience(candidate, target_audience):
            continue
        if not _item_matches_query_constraints(candidate, effective_constraints):
            continue

        product_priority = max(
            (PRODUCT_GROUP_PRIORITY.get(product, 0) for product in constraints["products"] if _item_matches_product(candidate, product)),
            default=0,
        )
        color_bonus = 8 if constraints["colors"] and _item_matches_query_color(candidate, constraints) else 0
        score = 0.25 + (product_priority / 1000) + (color_bonus / 1000)
        matches.append(_catalog_item_for_article(article_id, meta, score=score))
        existing_ids.add(article_id)
        if len(matches) >= limit:
            break
    return matches


def _with_catalog_constraint_backfill(
    items: list[dict],
    constraints: dict[str, set[str]],
    *,
    min_results: int,
    target_audience: str | None = None,
) -> list[dict]:
    if not constraints["products"] or min_results <= 0:
        return items

    existing_ids = {str(item.get("product_id", item.get("article_id", ""))) for item in items}
    target_filtered_items = _apply_target_audience_filter(items, target_audience)
    matching_count = len(_query_constraint_matches(target_filtered_items, constraints))
    missing_count = max(0, min_results - matching_count)
    if missing_count <= 0:
        return items

    backfill_items = _catalog_constraint_candidates(
        constraints,
        existing_ids=existing_ids,
        limit=missing_count,
        require_color=bool(constraints["colors"]),
        target_audience=target_audience,
    )
    if len(backfill_items) < missing_count and constraints["colors"]:
        backfill_items.extend(
            _catalog_constraint_candidates(
                constraints,
                existing_ids=existing_ids,
                limit=missing_count - len(backfill_items),
                require_color=False,
                target_audience=target_audience,
            )
        )
    return items + backfill_items


def _item_matches_any_product_group(item: dict, product_groups: set[str]) -> bool:
    if not product_groups:
        return True
    return any(_item_matches_product(item, product) for product in product_groups)


def _item_matches_complement_color(item: dict, colors: set[str]) -> bool:
    if not colors:
        return True
    allowed_colors = set(colors)
    for color in colors:
        allowed_colors.update(COLOR_COMPLEMENTS.get(color, ()))
    item_color = str(item.get("color", "")).lower()
    return any(color in item_color for color in allowed_colors)


def _complement_groups_for_constraints(constraints: dict[str, set[str]]) -> set[str]:
    groups: set[str] = set()
    for product in constraints["products"]:
        groups.update(COMPLEMENT_PRODUCT_GROUPS.get(product, ()))
    return groups


def _product_group_priority(item: dict, product_groups: set[str]) -> int:
    best = 0
    for product in product_groups:
        if _item_matches_product(item, product):
            best = max(best, PRODUCT_GROUP_PRIORITY.get(product, 50))
    return best


def _blend_personalized_with_search_intent(
    search_results: list[dict],
    personalized_results: list[dict],
    *,
    query: str,
    translated_query: str | None,
    top_n: int,
    personalization_weight: float | None = None,
) -> list[dict]:
    """Keep lexical search intent dominant; use personalization as a tie-breaker."""

    constraints = _derive_query_constraints(query, translated_query)
    constrained_results = _query_constraint_matches(search_results, constraints)
    if len(constrained_results) >= max(3, min(top_n, len(search_results))):
        search_results = constrained_results

    slider_value = max(0.0, min(float(personalization_weight if personalization_weight is not None else 0.7), 1.0))
    personal_blend = 0.08 + (0.32 * slider_value)
    search_blend = 1.0 - personal_blend
    personalized_by_id = {
        str(item.get("product_id", "")): item
        for item in personalized_results
        if item.get("product_id") is not None
    }
    max_search_score = max((float(item.get("score") or 0.0) for item in search_results), default=1.0) or 1.0
    max_personal_score = max((float(item.get("score") or 0.0) for item in personalized_results), default=1.0) or 1.0

    candidates: list[tuple[float, int, dict]] = []
    for index, search_item in enumerate(search_results):
        product_id = str(search_item.get("product_id", ""))
        personalized_item = personalized_by_id.get(product_id, {})
        search_score = float(search_item.get("score") or 0.0) / max_search_score
        personal_score = float(personalized_item.get("score") or 0.0) / max_personal_score
        matches_query = _item_matches_query_constraints(search_item, constraints)
        intent_bonus = 0.25 if matches_query else 0.0
        combined_score = (search_blend * search_score) + (personal_blend * personal_score) + intent_bonus
        merged_item = {
            **search_item,
            "score": combined_score,
            "search_score": search_item.get("score"),
            "personalized_score": personalized_item.get("score"),
            "reason": personalized_item.get("reason", search_item.get("reason", "search_intent_match")),
            "is_exploration": bool(personalized_item.get("is_exploration", False)),
        }
        candidates.append((combined_score, index, merged_item))

    full_matching_candidates = [
        candidate
        for candidate in candidates
        if _item_matches_query_constraints(candidate[2], constraints)
    ]
    color_matching_candidates = [
        candidate
        for candidate in candidates
        if _item_matches_query_color(candidate[2], constraints)
    ]
    if len(full_matching_candidates) >= max(3, top_n // 2):
        pool = full_matching_candidates
    elif constraints["colors"] and len(color_matching_candidates) >= top_n:
        pool = color_matching_candidates
    else:
        pool = candidates
    ranked = sorted(pool, key=lambda candidate: (-candidate[0], candidate[1]))
    return [
        {**item, "rank": rank}
        for rank, (_, _, item) in enumerate(ranked[:top_n], 1)
    ]


# ── 요청/응답 스키마 ──────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = ""
    image_base64: str | None = None
    top_k: int = 10


class PersonalizedSearchRequest(SearchRequest):
    user_id: str
    top_n: int = 10
    persona_hint: str | None = None
    personalization_weight: float | None = None
    target_audience: str | None = None


class EventRequest(BaseModel):
    user_id: str
    article_id: str | None = None
    item_id: str | None = None  # frontend 호환 (article_id 우선)
    event_type: str  # "click" | "view" | "cart" | "purchase" | "search"
    category: str | None = None
    query_text: str | None = None


class ExplainResultItem(BaseModel):
    id: int | str
    title: str = ""
    brand: str = ""
    price: str = ""


class ExplainResultsRequest(BaseModel):
    user_id: str
    query: str = ""
    persona: str | None = None
    target_audience: str | None = None
    items: list[ExplainResultItem]


class OnboardingRequest(BaseModel):
    user_id: str
    description: str  # 자유 입력 (예: "미니멀한 스타일 좋아하는 20대 여성입니다")
    style_choices: list[str] = []  # 선택지 (예: ["casual", "minimal", "sporty"])
    budget_range: str | None = None  # "low" | "mid" | "high"
    target_audience: str | None = None


VALID_PERSONAS = PERSONA_KEYS


def _normalize_persona_scores(persona_scores: dict[str, int | float]) -> dict[str, int]:
    filtered = {
        key: max(0, int(value))
        for key, value in persona_scores.items()
        if key in VALID_PERSONAS and isinstance(value, (int, float))
    }
    total = sum(filtered.values())
    if total == 0:
        return dict(DEFAULT_PERSONA_SCORES)

    sorted_keys = sorted(filtered, key=filtered.get, reverse=True)
    normalized = {key: round(filtered[key] * 100 / total) for key in sorted_keys}
    diff = 100 - sum(normalized.values())
    normalized[sorted_keys[0]] += diff
    return normalized


def _fallback_persona_scores(req: OnboardingRequest) -> dict[str, int]:
    """Deterministic local persona inference for quota/network fallback."""

    text = " ".join([req.description, " ".join(req.style_choices), req.budget_range or ""]).lower()
    scores = {persona: 0 for persona in VALID_PERSONAS}

    for persona, keywords in PERSONA_KEYWORD_RULES.items():
        scores[persona] += sum(18 for keyword in keywords if keyword in text)

    if req.budget_range == "low":
        scores["value"] += 30
    elif req.budget_range == "mid":
        scores["practical"] += 18
        scores["careful"] += 12
    elif req.budget_range == "high":
        scores["brand_loyal"] += 18
        scores["trendsetter"] += 12

    for style in req.style_choices:
        style_key = style.lower()
        if style_key in {"minimal", "classic", "casual"}:
            scores["practical"] += 14
            scores["careful"] += 8
        elif style_key == "street":
            scores["trendsetter"] += 18
            scores["impulse"] += 8
        elif style_key == "sporty":
            scores["practical"] += 12
            scores["trendsetter"] += 8
        elif style_key == "feminine":
            scores["trendsetter"] += 8
            scores["color_focus"] += 8

    return _normalize_persona_scores(scores)


QUERY_INTEREST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Ladieswear": (
        "women", "woman", "ladies", "dress", "skirt", "blouse", "jacket", "outer", "coat",
        "여성", "여자", "원피스", "스커트", "블라우스", "자켓", "재킷", "아우터", "코트",
    ),
    "Menswear": (
        "men", "man", "mens", "shirt", "suit", "jacket", "outer", "coat",
        "남성", "남자", "셔츠", "정장", "자켓", "재킷", "아우터", "코트",
    ),
    "Divided": (
        "denim", "jeans", "street", "casual", "청바지", "데님", "스트릿", "캐주얼",
    ),
    "Sport": (
        "sport", "sports", "active", "training", "스포츠", "운동", "트레이닝",
    ),
    "Kids": (
        "kids", "baby", "child", "키즈", "아동", "아이", "베이비",
    ),
    "Lingeries/Tights": (
        "lingerie", "tights", "underwear", "속옷", "타이츠", "스타킹",
    ),
}
QUERY_INTEREST_CATEGORIES = tuple(QUERY_INTEREST_KEYWORDS.keys())


def _infer_session_interest_from_query_keywords(query_text: str | None) -> dict[str, int]:
    if not query_text:
        return {}

    normalized_query = query_text.lower()
    inferred: dict[str, int] = {}
    for category, keywords in QUERY_INTEREST_KEYWORDS.items():
        if any(keyword.lower() in normalized_query for keyword in keywords):
            inferred[category] = inferred.get(category, 0) + 2
    return inferred


# ── 유틸: LLM 호출 ───────────────────────────────────────────

def _gemini_cooldown_ttl() -> int:
    try:
        ttl = feature_store.r.ttl(GEMINI_COOLDOWN_KEY)
    except Exception:
        return -2
    return int(ttl or -2)


def _gemini_available() -> bool:
    return bool(GEMINI_API_KEY) and _gemini_cooldown_ttl() <= 0


def _mark_gemini_cooldown(status_code: int) -> None:
    if status_code != 429:
        return
    try:
        feature_store.r.set(
            GEMINI_COOLDOWN_KEY,
            "quota_or_rate_limited",
            ex=GEMINI_COOLDOWN_SECONDS,
        )
    except Exception:
        pass
    logging.warning("Gemini quota/rate limit detected. Cooling down for %s seconds.", GEMINI_COOLDOWN_SECONDS)


async def _call_gemini(prompt: str, json_mode: bool = False, temperature: float | None = None) -> str:
    """Gemini Flash API 호출. GEMINI_API_KEY 미설정 시 빈 문자열 반환.

    json_mode=True이면 JSON 외 출력을 차단해 파싱 안정성을 높인다.
    """
    if not GEMINI_API_KEY:
        return ""
    cooldown_ttl = _gemini_cooldown_ttl()
    if cooldown_ttl > 0:
        logging.info("Skipping Gemini call during cooldown. ttl=%s", cooldown_ttl)
        return ""

    generation_config: dict = {"temperature": temperature if temperature is not None else (0.7 if not json_mode else 0.1)}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": generation_config,
            },
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            _mark_gemini_cooldown(e.response.status_code)
            raise
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _parse_gemini_reasons(llm_text: str, expected_count: int) -> list[str]:
    payload = json.loads(llm_text)
    raw_reasons = payload.get("reasons", [])
    if not isinstance(raw_reasons, list):
        logging.warning("Gemini reasons payload did not contain a list: %s", type(raw_reasons).__name__)
        return []

    reasons: list[str] = []
    for raw_reason in raw_reasons[:expected_count]:
        if isinstance(raw_reason, str):
            reason = raw_reason
        elif isinstance(raw_reason, dict):
            reason = str(raw_reason.get("reason") or raw_reason.get("text") or "")
        else:
            reason = ""
        reasons.append(" ".join(reason.split()))
    return reasons


def _coerce_interest_score(value: object) -> int:
    try:
        numeric_value = int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(numeric_value, 3))


def _parse_query_interest_payload(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}

    raw_interest = payload.get("interest", payload)
    if not isinstance(raw_interest, dict):
        return {}

    normalized_by_key = {str(key).strip().lower(): value for key, value in raw_interest.items()}
    inferred: dict[str, int] = {}
    for category in QUERY_INTEREST_CATEGORIES:
        score = _coerce_interest_score(normalized_by_key.get(category.lower()))
        if score > 0:
            inferred[category] = score
    return inferred


async def _infer_session_interest_from_query(query_text: str | None) -> dict[str, int]:
    normalized_query = (query_text or "").strip()
    if not normalized_query:
        return {}

    cached_interest = feature_store.get_query_interest_cache(normalized_query)
    if cached_interest is not None:
        return {
            category: _coerce_interest_score(score)
            for category, score in cached_interest.items()
            if category in QUERY_INTEREST_CATEGORIES and _coerce_interest_score(score) > 0
        }

    inferred_interest = _infer_session_interest_from_query_keywords(normalized_query)

    feature_store.set_query_interest_cache(normalized_query, inferred_interest)
    return inferred_interest


# ── 엔드포인트 ────────────────────────────────────────────────

def _has_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


async def _translate_to_english(query: str) -> str:
    """한국어 패션 검색어를 영어로 번역. 실패 시 원문 반환."""
    if not _gemini_available():
        return query
    prompt = (
        f"Translate this Korean fashion search query to English. "
        f"Return only the translated English text, nothing else.\n\nQuery: {query}"
    )
    try:
        result = await _call_gemini(prompt)
        return result.strip() if result.strip() else query
    except Exception:
        return query


SEARCH_INTENT_TERM_LIMIT = 12
SEARCH_INTENT_AVOID_LIMIT = 10
QUERY_RECOMMEND_INTEREST_MULTIPLIER = 3.0
SITUATIONAL_SEARCH_INTENTS: tuple[tuple[tuple[str, ...], dict[str, object]], ...] = (
    (
        ("피방", "pc방", "피씨방", "피시방", "겜방", "게임방", "게임하러", "pc bang", "pc cafe", "gaming"),
        {
            "intent_label": "편한 PC방 캐주얼룩",
            "preferred_terms": (
                "hoodie", "hooded sweatshirt", "sweatshirt", "joggers", "sweatpants",
                "relaxed trousers", "t-shirt", "sneakers", "comfortable casual", "streetwear",
            ),
            "avoid_terms": ("dress", "skirt", "blouse", "lingerie", "baby", "formal", "heels"),
            "session_interest": {"Divided": 3, "Sport": 2},
        },
    ),
    (
        ("후드", "후디", "hoodie", "hood"),
        {
            "intent_label": "후드 캐주얼룩",
            "preferred_terms": ("hoodie", "hooded sweatshirt", "sweatshirt", "casual", "streetwear"),
            "avoid_terms": ("dress", "skirt", "lingerie", "baby", "formal"),
            "session_interest": {"Divided": 2, "Sport": 1},
        },
    ),
    (
        ("츄리닝", "추리닝", "트레이닝복", "조거", "sweatpants", "jogger", "track pants"),
        {
            "intent_label": "트레이닝 캐주얼룩",
            "preferred_terms": ("joggers", "sweatpants", "track pants", "training pants", "sneakers", "sport casual"),
            "avoid_terms": ("dress", "skirt", "blouse", "lingerie", "baby", "formal"),
            "session_interest": {"Sport": 3, "Divided": 2},
        },
    ),
    (
        ("편한", "편하게", "꾸안꾸", "데일리", "daily", "comfortable", "casual"),
        {
            "intent_label": "편한 데일리룩",
            "preferred_terms": ("comfortable casual", "relaxed fit", "basic", "t-shirt", "sweatshirt", "trousers", "sneakers"),
            "avoid_terms": ("lingerie", "baby", "formal"),
            "session_interest": {"Divided": 2},
        },
    ),
    (
        ("데이트", "여자친구", "남자친구", "애인", "연인", "소개팅", "첫만남", "first date", "date outfit"),
        {
            "intent_label": "데이트/소개팅 스마트 캐주얼룩",
            "preferred_terms": (
                "shirt", "blouse", "knit", "sweater", "cardigan", "clean",
                "smart casual", "trousers", "slacks", "dress", "loafers",
            ),
            "avoid_terms": (
                "sweatpants", "joggers", "hoodie", "training pants", "sportswear",
                "lingerie", "baby", "dirty", "worn",
            ),
            "session_interest": {"Ladieswear": 3, "Menswear": 3, "Divided": 1},
        },
    ),
    (
        ("여친", "여자친구만날", "여자친구 만날", "여자친구 보러", "여자친구랑", "남친", "남자친구 만날"),
        {
            "intent_label": "연인 만나는 날 단정한 캐주얼룩",
            "preferred_terms": (
                "shirt", "knit", "sweater", "cardigan", "trousers", "slacks",
                "clean", "minimal", "smart casual", "loafers", "jacket",
            ),
            "avoid_terms": ("sweatpants", "joggers", "training pants", "lingerie", "baby", "formal suit"),
            "session_interest": {"Menswear": 3, "Ladieswear": 2, "Divided": 1},
        },
    ),
    (
        ("도서관", "공부", "스터디", "팀플", "과제", "library", "study"),
        {
            "intent_label": "도서관/스터디 깔끔한 데일리룩",
            "preferred_terms": (
                "cardigan", "knit", "sweater", "shirt", "t-shirt", "trousers",
                "jeans", "comfortable casual", "minimal", "clean", "sneakers",
            ),
            "avoid_terms": ("party", "heels", "lingerie", "baby", "swimwear", "formal suit"),
            "session_interest": {"Divided": 2, "Ladieswear": 1, "Menswear": 1},
        },
    ),
    (
        ("카페", "브런치", "친구랑", "친구들과", "약속", "cafe", "brunch"),
        {
            "intent_label": "카페/친구 약속 캐주얼룩",
            "preferred_terms": (
                "cardigan", "shirt", "blouse", "knit", "sweater", "jeans",
                "trousers", "clean", "casual", "sneakers",
            ),
            "avoid_terms": ("lingerie", "baby", "swimwear", "training pants"),
            "session_interest": {"Divided": 2, "Ladieswear": 2, "Menswear": 1},
        },
    ),
    (
        ("출근", "회사", "오피스", "회의", "면접", "인턴", "office", "work", "interview"),
        {
            "intent_label": "출근/면접 단정한 오피스룩",
            "preferred_terms": (
                "shirt", "blouse", "blazer", "jacket", "trousers", "slacks",
                "skirt", "coat", "clean", "minimal", "formal",
            ),
            "avoid_terms": ("hoodie", "sweatpants", "joggers", "training pants", "lingerie", "baby", "swimwear"),
            "session_interest": {"Ladieswear": 3, "Menswear": 3},
        },
    ),
    (
        ("비오는", "비 오는", "비올때", "비 올때", "장마", "rain", "rainy"),
        {
            "intent_label": "비 오는 날 실용 캐주얼룩",
            "preferred_terms": (
                "jacket", "coat", "parka", "hooded", "dark", "trousers",
                "boots", "sneakers", "waterproof", "practical",
            ),
            "avoid_terms": ("white trousers", "heels", "sandals", "suede", "lingerie", "baby"),
            "session_interest": {"Divided": 2, "Menswear": 1, "Ladieswear": 1},
        },
    ),
    (
        ("여행", "공항", "기차", "버스", "장거리", "travel", "airport"),
        {
            "intent_label": "여행/이동 편한 캐주얼룩",
            "preferred_terms": (
                "comfortable casual", "relaxed fit", "hoodie", "sweatshirt",
                "joggers", "trousers", "t-shirt", "sneakers", "jacket",
            ),
            "avoid_terms": ("heels", "formal suit", "lingerie", "baby"),
            "session_interest": {"Divided": 2, "Sport": 1},
        },
    ),
    (
        ("운동", "헬스", "러닝", "산책", "조깅", "workout", "running", "gym"),
        {
            "intent_label": "운동/산책 스포티룩",
            "preferred_terms": (
                "sportswear", "training", "leggings", "t-shirt", "tank top",
                "sneakers", "joggers", "sweatshirt", "functional",
            ),
            "avoid_terms": ("dress", "skirt", "blouse", "heels", "formal", "lingerie", "baby"),
            "session_interest": {"Sport": 3, "Divided": 1},
        },
    ),
)


def _normalize_term_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in value:
        term = str(raw_term).strip()
        normalized = term.lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _merge_search_interest(*interests: object) -> dict[str, int]:
    merged: dict[str, int] = {}
    for interest in interests:
        if not isinstance(interest, dict):
            continue
        parsed = _parse_query_interest_payload({"interest": interest})
        for category, score in parsed.items():
            merged[category] = max(merged.get(category, 0), score)
    return merged


def _fallback_search_intent(query: str | None, translated_query: str | None = None) -> dict[str, object]:
    normalized_query = " ".join((query or "").split())
    lowered = normalized_query.lower()
    preferred_terms: list[str] = []
    avoid_terms: list[str] = []
    session_interests: list[dict[str, int]] = []
    intent_label = "패션 검색"

    for keywords, intent in SITUATIONAL_SEARCH_INTENTS:
        if any(keyword in lowered or keyword in normalized_query for keyword in keywords):
            intent_label = str(intent.get("intent_label") or intent_label)
            preferred_terms.extend(_normalize_term_list(intent.get("preferred_terms"), SEARCH_INTENT_TERM_LIMIT))
            avoid_terms.extend(_normalize_term_list(intent.get("avoid_terms"), SEARCH_INTENT_AVOID_LIMIT))
            raw_interest = intent.get("session_interest", {})
            if isinstance(raw_interest, dict):
                session_interests.append(raw_interest)

    fallback_interest = _infer_session_interest_from_query_keywords(normalized_query)
    session_interest = _merge_search_interest(*session_interests, fallback_interest)
    base_query = (translated_query or normalized_query).strip()
    if preferred_terms:
        search_query = f"{base_query}. Keywords: {' '.join(dict.fromkeys(preferred_terms))}"
    else:
        search_query = base_query

    return {
        "intent_label": intent_label,
        "translated_query": translated_query or (base_query if base_query != normalized_query else None),
        "search_query": search_query,
        "preferred_terms": _normalize_term_list(preferred_terms, SEARCH_INTENT_TERM_LIMIT),
        "avoid_terms": _normalize_term_list(avoid_terms, SEARCH_INTENT_AVOID_LIMIT),
        "session_interest": session_interest,
        "source": "fallback",
    }


def _local_product_search_intent(query: str, constraints: dict[str, set[str]]) -> dict[str, object] | None:
    if not constraints["products"]:
        return None

    sorted_products = sorted(
        constraints["products"],
        key=lambda product: PRODUCT_GROUP_PRIORITY.get(product, 0),
        reverse=True,
    )
    product_terms: list[str] = []
    for product in sorted_products:
        product_terms.extend(PRODUCT_INTENT_TERMS.get(product, (PRODUCT_DISPLAY_LABELS.get(product, product),)))

    color_terms: list[str] = []
    for color in sorted(constraints["colors"]):
        color_terms.extend(COLOR_INTENT_TERMS.get(color, color).split())

    preferred_terms = _normalize_term_list(product_terms, SEARCH_INTENT_TERM_LIMIT)
    avoid_terms: list[str] = []
    for product in sorted_products:
        avoid_terms.extend(PRODUCT_EXCLUDE_TERMS.get(product, ()))

    base_terms = list(dict.fromkeys([*color_terms, *preferred_terms]))
    translated_query = " ".join(base_terms) if base_terms else query
    search_query = translated_query
    if preferred_terms:
        search_query = f"{translated_query}. Keywords: {' '.join(preferred_terms)}"

    return {
        "intent_label": query or "패션 검색",
        "translated_query": translated_query,
        "search_query": search_query,
        "preferred_terms": preferred_terms,
        "avoid_terms": _normalize_term_list(avoid_terms, SEARCH_INTENT_AVOID_LIMIT),
        "session_interest": _infer_session_interest_from_query_keywords(query),
        "source": "local_product_intent",
    }


def _parse_search_intent_payload(payload: object, query: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        return _fallback_search_intent(query)

    preferred_terms = _normalize_term_list(payload.get("preferred_terms"), SEARCH_INTENT_TERM_LIMIT)
    avoid_terms = _normalize_term_list(payload.get("avoid_terms"), SEARCH_INTENT_AVOID_LIMIT)
    translated_query = str(payload.get("translated_query") or "").strip() or None
    raw_search_query = str(payload.get("search_query") or "").strip()
    if not raw_search_query:
        raw_search_query = translated_query or query
    if preferred_terms and "keywords:" not in raw_search_query.lower():
        raw_search_query = f"{raw_search_query}. Keywords: {' '.join(preferred_terms)}"

    fallback = _fallback_search_intent(query, translated_query=translated_query)
    session_interest = _merge_search_interest(payload.get("session_interest"), fallback.get("session_interest"))
    return {
        "intent_label": str(payload.get("intent_label") or fallback.get("intent_label") or "패션 검색").strip(),
        "translated_query": translated_query,
        "search_query": raw_search_query,
        "preferred_terms": preferred_terms or fallback.get("preferred_terms", []),
        "avoid_terms": avoid_terms or fallback.get("avoid_terms", []),
        "session_interest": session_interest,
        "source": "llm",
    }


async def _infer_search_intent(query: str | None) -> dict[str, object]:
    normalized_query = " ".join((query or "").split())
    if not normalized_query:
        return _fallback_search_intent(query)

    local_constraints = _derive_query_constraints(normalized_query)
    local_intent = _local_product_search_intent(normalized_query, local_constraints)
    if local_intent is not None:
        try:
            feature_store.set_search_intent_cache(normalized_query, local_intent, fallback=True)
        except Exception as exc:
            logging.debug("Failed to cache local search intent: %s", exc)
        return local_intent

    cached_intent = feature_store.get_search_intent_cache(normalized_query)
    if cached_intent is not None:
        return cached_intent

    if not _gemini_available():
        fallback = _fallback_search_intent(normalized_query)
        feature_store.set_search_intent_cache(normalized_query, fallback, fallback=True)
        return fallback

    prompt = (
        "You are a fashion search intent parser for an H&M-like product catalog.\n"
        "Convert the user's natural-language fashion query into retrieval-friendly English.\n"
        "Prefer concrete product terms that exist in apparel catalogs. For situational Korean queries, infer the outfit context.\n"
        "Example: '피방갈때 입을 옷' means comfortable casual PC cafe/gaming outfit, usually hoodie, sweatshirt, joggers, sweatpants, sneakers, t-shirt.\n"
        "Return only JSON with these keys:\n"
        "- translated_query: concise English translation\n"
        "- search_query: English query for vector search, including concrete product/style words\n"
        "- preferred_terms: up to 12 product/style terms to boost\n"
        "- avoid_terms: up to 10 terms to demote if they conflict with the situation\n"
        "- intent_label: short Korean label for UI/debug\n"
        "- session_interest: scores 0-3 using only Ladieswear, Menswear, Divided, Sport, Kids, Lingeries/Tights\n\n"
        f"User query: {normalized_query}\n"
    )
    try:
        llm_text = await _call_gemini(prompt, json_mode=True)
        parsed = json.loads(llm_text)
        intent = _parse_search_intent_payload(parsed, normalized_query)
        feature_store.set_search_intent_cache(normalized_query, intent, fallback=False)
        return intent
    except Exception:
        translated = await _translate_to_english(normalized_query) if _has_korean(normalized_query) else normalized_query
        fallback = _fallback_search_intent(normalized_query, translated_query=translated)
        feature_store.set_search_intent_cache(normalized_query, fallback, fallback=True)
        return fallback


def _text_for_intent_match(item: dict) -> str:
    fields = (
        item.get("name"),
        item.get("brand"),
        item.get("category"),
        item.get("color"),
        item.get("product_type"),
        item.get("main_category"),
        item.get("product_group"),
    )
    return " ".join(str(field or "") for field in fields).lower()


def _apply_search_intent_preferences(items: list[dict], intent: dict[str, object]) -> list[dict]:
    preferred_terms = [term.lower() for term in _normalize_term_list(intent.get("preferred_terms"), SEARCH_INTENT_TERM_LIMIT)]
    avoid_terms = [term.lower() for term in _normalize_term_list(intent.get("avoid_terms"), SEARCH_INTENT_AVOID_LIMIT)]
    if not preferred_terms and not avoid_terms:
        return items

    adjusted: list[dict] = []
    for position, item in enumerate(items):
        text = _text_for_intent_match(item)
        preferred_hits = sum(1 for term in preferred_terms if term in text)
        avoid_hits = sum(1 for term in avoid_terms if term in text)
        intent_boost = (0.04 * preferred_hits) - (0.07 * avoid_hits)
        ranked_item = {**item}
        ranked_item["intent_boost"] = round(intent_boost, 4)
        ranked_item["_intent_rank_score"] = float(item.get("score", item.get("similarity", 0.0)) or 0.0) + intent_boost
        ranked_item["_original_position"] = position
        adjusted.append(ranked_item)

    adjusted.sort(key=lambda item: (item["_intent_rank_score"], -item["_original_position"]), reverse=True)
    for item in adjusted:
        item.pop("_intent_rank_score", None)
        item.pop("_original_position", None)
    return adjusted


RECOMMENDATION_REASON_FALLBACK_PREFIX = "기본 추천 근거입니다."
MODEL_REASON_LABELS = {
    "query_intent_match": "검색어에 담긴 품목 의도와 직접 맞아서 우선 추천했습니다.",
    "search_intent_match": "검색어 의도와 상품 카테고리가 잘 맞아서 추천했습니다.",
    "cold_start_popularity": "아직 클릭 데이터가 적어서 비슷한 사용자들에게 반응이 좋았던 인기 상품을 우선 추천했습니다.",
    "session_interest_match": "방금 설정한 관심사와 상품 카테고리가 잘 맞아서 추천했습니다.",
    "recent_click_similarity": "최근 클릭한 상품과 스타일이나 카테고리가 가까워서 함께 보기 좋은 후보입니다.",
    "ranking_score": "페르소나, 세션 관심사, 상품 특성을 종합했을 때 점수가 높게 나온 상품입니다.",
    "new_item_boost": "새로운 후보도 탐색할 수 있도록 노출한 상품입니다.",
    "bandit_reward_exploration": "최근 사용자 반응이 좋아지고 있어 탐색 후보로 함께 추천했습니다.",
    "mab_exploration": "취향 범위를 넓히기 위해 탐색 후보로 함께 노출했습니다.",
    "coverage_exploration": "비슷한 후보만 반복되지 않도록 다른 스타일 신호를 함께 반영했습니다.",
}


def _fallback_recommendation_reason(item: dict, *, limited: bool = True) -> str:
    reason_key = str(item.get("reason") or "")
    base_reason = MODEL_REASON_LABELS.get(
        reason_key,
        "추천 모델이 사용자 맥락과 상품 특성을 종합해 후보로 선택한 상품입니다.",
    )
    category = str(item.get("category") or item.get("product_type") or "").strip()
    color = str(item.get("color") or "").strip()
    detail_parts = []
    if color:
        detail_parts.append(f"{color} 컬러")
    if category:
        detail_parts.append(f"{category} 상품")
    detail = ""
    if detail_parts:
        detail = f" {' '.join(detail_parts)}이라 현재 코디 후보로 활용하기 좋습니다."
    prefix = f"{RECOMMENDATION_REASON_FALLBACK_PREFIX} " if limited else ""
    return f"{prefix}{base_reason}{detail}"


def _attach_local_reason_text(item: dict, *, limited: bool = False) -> dict:
    if item.get("reason_text"):
        return item
    enriched_item = {**item}
    enriched_item["reason_text"] = _fallback_recommendation_reason(enriched_item, limited=limited)
    enriched_item["reason_source"] = "local_reason_map"
    return enriched_item


def _enrich_search_results(items: list[dict]) -> list[dict]:
    enriched_results = []
    for item in items:
        pid = str(item.get("product_id", ""))
        meta = article_meta.get(pid, {})
        price, price_estimated = _display_price(meta, item)
        enriched_results.append({
            **item,
            "name": meta.get("name") or item.get("name") or pid,
            "brand": meta.get("brand") or "H&M",
            "category": meta.get("category", item.get("category", "")),
            "main_category": meta.get("main_category") or item.get("main_category", ""),
            "color": meta.get("color", ""),
            "product_type": meta.get("product_type", ""),
            "department_name": meta.get("department_name", item.get("department_name", "")),
            "section_name": meta.get("section_name", item.get("section_name", "")),
            "garment_group": meta.get("garment_group", item.get("garment_group", "")),
            "price": price,
            "price_estimated": price_estimated,
            "price_source": _price_source_label(meta, price_estimated),
            "image_url": image_url_for_article(pid),
        })
    return enriched_results


def _enrich_recommendation_results(items: list[dict]) -> list[dict]:
    enriched_results = []
    for rank, item in enumerate(items, 1):
        pid = str(item.get("product_id", ""))
        meta = article_meta.get(pid, {})
        price, price_estimated = _display_price(meta, item)
        enriched_results.append({
            **item,
            "rank": rank,
            "name": meta.get("name") or pid,
            "brand": meta.get("brand") or "H&M",
            "category": meta.get("category", ""),
            "main_category": meta.get("main_category") or item.get("main_category", ""),
            "color": meta.get("color", ""),
            "product_type": meta.get("product_type", ""),
            "department_name": meta.get("department_name", item.get("department_name", "")),
            "section_name": meta.get("section_name", item.get("section_name", "")),
            "garment_group": meta.get("garment_group", item.get("garment_group", "")),
            "price": price,
            "price_estimated": price_estimated,
            "price_source": _price_source_label(meta, price_estimated),
            "image_url": image_url_for_article(pid),
        })
    return enriched_results


def _top_persona_from_scores(persona_scores: dict[str, int] | None) -> str | None:
    if not persona_scores:
        return None
    return max(persona_scores, key=persona_scores.get)


@app.post("/api/search")
async def search(req: SearchRequest):
    """search-engine으로 검색 요청을 프록시한다.

    텍스트 쿼리는 Gemini/fallback으로 검색 의도를 확장한 뒤 CLIP에 전달한다.
    결과에 name/category/color/price enrichment를 적용한다.
    """
    requested_top_k = max(1, int(req.top_k))
    search_payload = req.model_dump()
    search_intent: dict[str, object] | None = None
    search_constraints: dict[str, set[str]] | None = None

    if req.query:
        search_intent = await _infer_search_intent(req.query)
        search_payload["query"] = search_intent.get("search_query") or req.query
        search_constraints = _derive_query_constraints(req.query, search_intent.get("translated_query"))
        if search_constraints["products"] or search_constraints["colors"]:
            search_payload["top_k"] = max(requested_top_k, SEARCH_INTENT_CANDIDATE_POOL)

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(f"{SEARCH_URL}/search", json=search_payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"search-engine 연결 실패: {e}")

    result = resp.json()

    enriched_results = _enrich_search_results(result.get("results", []))
    if search_intent:
        effective_constraints = search_constraints or _derive_query_constraints(req.query, search_intent.get("translated_query"))
        enriched_results = _apply_search_intent_preferences(enriched_results, search_intent)
        enriched_results = _with_catalog_constraint_backfill(
            enriched_results,
            effective_constraints,
            min_results=requested_top_k,
        )
        enriched_results = _prioritize_query_constraint_matches(
            enriched_results,
            effective_constraints,
        )
        enriched_results = _apply_query_product_labels(
            enriched_results,
            effective_constraints,
        )
    result["results"] = enriched_results[:requested_top_k]
    result["total_count"] = len(result["results"])

    if search_intent:
        result["original_query"] = req.query
        result["translated_query"] = search_intent.get("translated_query")
        result["expanded_query"] = search_intent.get("search_query")
        result["search_intent"] = {
            "intent_label": search_intent.get("intent_label"),
            "preferred_terms": search_intent.get("preferred_terms", []),
            "avoid_terms": search_intent.get("avoid_terms", []),
            "source": search_intent.get("source"),
        }

    return result


SEARCH_INTENT_CANDIDATE_POOL = 150
PERSONALIZED_SEARCH_CANDIDATE_POOL = 150


@app.post("/api/personalized-search")
async def personalized_search(req: PersonalizedSearchRequest):
    """Search broadly, then return both similarity order and personalized order."""

    top_n = max(1, min(int(req.top_n), 100))
    target_audience = _normalize_target_audience(req.target_audience)
    search_top_k = max(top_n, int(req.top_k), PERSONALIZED_SEARCH_CANDIDATE_POOL)
    search_intent = await _infer_search_intent(req.query) if req.query else None
    translated_query = (
        str(search_intent.get("translated_query") or "").strip() or None
        if search_intent
        else None
    )
    search_constraints = (
        _derive_query_constraints(req.query, search_intent.get("translated_query"))
        if search_intent
        else {"colors": set(), "products": set()}
    )
    if search_constraints["products"] or search_constraints["colors"]:
        search_top_k = max(search_top_k, SEARCH_INTENT_CANDIDATE_POOL)
    search_payload = {
        "query": search_intent.get("search_query") if search_intent else req.query,
        "image_base64": req.image_base64,
        "top_k": search_top_k,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            search_resp = await client.post(f"{SEARCH_URL}/search", json=search_payload)
            search_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"search-engine 연결 실패: {e}")

    search_result = search_resp.json()
    enriched_search_results = _enrich_search_results(search_result.get("results", []))
    if search_intent:
        enriched_search_results = _apply_search_intent_preferences(enriched_search_results, search_intent)
        enriched_search_results = _with_catalog_constraint_backfill(
            enriched_search_results,
            search_constraints,
            min_results=search_top_k,
            target_audience=target_audience,
        )
        enriched_search_results = _prioritize_query_constraint_matches(
            enriched_search_results,
            search_constraints,
        )
        enriched_search_results = _apply_query_product_labels(enriched_search_results, search_constraints)
    enriched_search_results = _apply_target_audience_filter(
        enriched_search_results,
        target_audience,
        min_results=top_n,
    )
    rerank_search_results = enriched_search_results
    explicit_matches = _query_constraint_matches(enriched_search_results, search_constraints)
    if len(explicit_matches) >= top_n:
        rerank_search_results = explicit_matches

    inferred_interest = (
        _merge_search_interest(search_intent.get("session_interest")) if search_intent else {}
    ) or await _infer_session_interest_from_query(req.query)
    target_interest = _target_audience_interest(target_audience)
    if target_interest:
        inferred_interest[target_interest] = max(inferred_interest.get(target_interest, 0), 10)
    if inferred_interest:
        interest = feature_store.get_session_interest(req.user_id)
        for category, score in inferred_interest.items():
            interest[category] = interest.get(category, 0) + score
        feature_store.set_session_interest(req.user_id, interest)
        feature_store.invalidate_recommendation_cache(req.user_id)

    features = feature_store.get_user_features(req.user_id)
    raw_persona_scores = features.get("persona_scores", {})
    persona_scores = _normalize_persona_scores(raw_persona_scores) if raw_persona_scores else {}
    persona_hint = req.persona_hint or _top_persona_from_scores(persona_scores)
    rerank_payload = {
        "user_id": req.user_id,
        "top_n": top_n,
        "search_candidates": [
            {
                "product_id": str(item.get("product_id", "")),
                "score": item.get("score", item.get("similarity")),
            }
            for item in rerank_search_results
            if item.get("product_id") is not None
        ],
        "recent_clicks": features["recent_clicks"],
        "session_interest": features["session_interest"] or None,
        "persona_hint": persona_hint,
        "persona_scores": persona_scores or None,
        "preferred_terms": search_intent.get("preferred_terms", []) if search_intent else [],
        "avoid_terms": search_intent.get("avoid_terms", []) if search_intent else [],
        "include_recommendation_candidates": False,
        "recommendation_candidate_pool_size": len(rerank_search_results) or PERSONALIZED_SEARCH_CANDIDATE_POOL,
        "personalization_weight": req.personalization_weight,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            rerank_resp = await client.post(f"{REC_URL}/rerank-candidates", json=rerank_payload)
            rerank_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"rec-models 연결 실패: {e}")

    personalized_result = rerank_resp.json()
    raw_personalized_results = _enrich_recommendation_results(personalized_result.get("recommendations", []))
    personalized_results = _blend_personalized_with_search_intent(
        rerank_search_results,
        raw_personalized_results,
        query=req.query,
        translated_query=translated_query,
        top_n=top_n,
        personalization_weight=req.personalization_weight,
    )
    personalized_results = [
        _attach_local_reason_text(item, limited=False)
        for item in personalized_results
    ]
    pipeline_latency = personalized_result.get("pipeline_latency", {})

    response = {
        "search_results": enriched_search_results[:top_n],
        "personalized_results": personalized_results,
        "search_latency_ms": search_result.get("latency_ms", search_result.get("total_ms")),
        "personalized_latency": pipeline_latency,
        "candidate_summary": personalized_result.get("candidate_summary", {}),
        "session_interest": features["session_interest"],
        "persona": personalized_result.get("persona", persona_hint or "personalized"),
        "persona_scores": persona_scores,
        "target_audience": target_audience,
        "candidate_pool_size": len(rerank_search_results),
        "explicit_candidate_count": len(explicit_matches),
    }
    if search_intent:
        response["original_query"] = req.query
        response["translated_query"] = search_intent.get("translated_query")
        response["expanded_query"] = search_intent.get("search_query")
        response["search_intent"] = {
            "intent_label": search_intent.get("intent_label"),
            "preferred_terms": search_intent.get("preferred_terms", []),
            "avoid_terms": search_intent.get("avoid_terms", []),
            "source": search_intent.get("source"),
        }
    return response


RECOMMEND_CACHE_TTL = 300  # 5분


def _weight_cache_suffix(weight_params: dict[str, object]) -> str:
    active_weights = {key: value for key, value in weight_params.items() if value is not None}
    if not active_weights:
        return ""
    encoded = json.dumps(active_weights, sort_keys=True, separators=(",", ":"))
    return f":{encoded}"


@app.get("/api/recommend")
async def recommend(
    user_id: str = Query(...),
    top_n: int = Query(10),
    persona_hint: str | None = Query(None),
    query_text: str | None = Query(None),
    personalization_weight: float | None = Query(None, ge=0.0, le=5.0),
    price_weight: float | None = Query(None, ge=0.0, le=5.0),
    popularity_weight: float | None = Query(None, ge=0.0, le=5.0),
    diversity_weight: float | None = Query(None, ge=0.0, le=5.0),
    freshness_weight: float | None = Query(None, ge=0.0, le=5.0),
    exploration_weight: float | None = Query(None, ge=0.0, le=5.0),
    long_tail_weight: float | None = Query(None, ge=0.0, le=5.0),
    include_reasons: bool = Query(False),
):
    """Redis 세션 데이터를 붙여 rec-models로 추천 요청을 프록시한다."""
    features = feature_store.get_user_features(user_id)
    click_count = features["click_count"]
    raw_persona_scores = features.get("persona_scores", {})
    persona_scores = _normalize_persona_scores(raw_persona_scores) if raw_persona_scores else {}
    normalized_query_text = " ".join((query_text or "").split()) or None
    query_search_intent: dict[str, object] | None = None
    effective_session_interest = dict(features["session_interest"] or {})

    if normalized_query_text:
        query_search_intent = await _infer_search_intent(normalized_query_text)
        intent_interest = _merge_search_interest(query_search_intent.get("session_interest"))
        updated_interest = False
        for category, score in intent_interest.items():
            serving_score = float(score) * QUERY_RECOMMEND_INTEREST_MULTIPLIER
            current_score = float(effective_session_interest.get(category, 0.0) or 0.0)
            if serving_score > current_score:
                effective_session_interest[category] = serving_score
                updated_interest = True
        if updated_interest:
            feature_store.set_session_interest(user_id, effective_session_interest)
            feature_store.invalidate_recommendation_cache(user_id)

    preferred_terms = (
        _normalize_term_list(query_search_intent.get("preferred_terms"), SEARCH_INTENT_TERM_LIMIT)
        if query_search_intent
        else []
    )
    avoid_terms = (
        _normalize_term_list(query_search_intent.get("avoid_terms"), SEARCH_INTENT_AVOID_LIMIT)
        if query_search_intent
        else []
    )
    weight_params = {
        "persona_hint": persona_hint,
        "persona_scores": persona_scores or None,
        "personalization_weight": personalization_weight,
        "price_weight": price_weight,
        "popularity_weight": popularity_weight,
        "diversity_weight": diversity_weight,
        "freshness_weight": freshness_weight,
        "exploration_weight": exploration_weight,
        "long_tail_weight": long_tail_weight,
    }
    cache_params = {
        **weight_params,
        "query_text": normalized_query_text,
        "preferred_terms": preferred_terms or None,
        "avoid_terms": avoid_terms or None,
    }

    # 캐시 키: include_reasons 여부에 따라 별도 키 사용
    reasons_suffix = ":reasons" if include_reasons else ""
    cache_key = f"cache:recommend:{user_id}:{top_n}:{click_count}{_weight_cache_suffix(cache_params)}{reasons_suffix}"
    cached = feature_store.r.get(cache_key)
    if cached:
        return json.loads(cached)

    params = {
        "user_id": user_id,
        "top_n": top_n,
        "recent_clicks": ",".join(features["recent_clicks"]),
        "click_count": click_count,
        "session_interest": json.dumps(effective_session_interest) if effective_session_interest else None,
        "persona_scores": json.dumps(persona_scores) if persona_scores else None,
    }
    if preferred_terms:
        params["preferred_terms"] = json.dumps(preferred_terms, ensure_ascii=False)
    if avoid_terms:
        params["avoid_terms"] = json.dumps(avoid_terms, ensure_ascii=False)
    params.update({
        key: value
        for key, value in weight_params.items()
        if value is not None and key != "persona_scores"
    })

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{REC_URL}/recommend", params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"rec-models 연결 실패: {e}")

    result = resp.json()
    if query_search_intent:
        result["original_query"] = normalized_query_text
        result["translated_query"] = query_search_intent.get("translated_query")
        result["expanded_query"] = query_search_intent.get("search_query")
        result["search_intent"] = {
            "intent_label": query_search_intent.get("intent_label"),
            "preferred_terms": query_search_intent.get("preferred_terms", []),
            "avoid_terms": query_search_intent.get("avoid_terms", []),
            "source": query_search_intent.get("source"),
        }
        result["query_session_interest"] = {
            category: effective_session_interest.get(category)
            for category in _merge_search_interest(query_search_intent.get("session_interest"))
        }

    # pipeline_latency 중첩 구조를 최상위로 풀어서 프론트가 바로 쓸 수 있게 함
    pl = result.pop("pipeline_latency", {})
    result.update(pl)

    # 상품명·카테고리를 articles_feature.csv에서 보강
    enriched = []
    for i, item in enumerate(result.get("recommendations", []), 1):
        pid = str(item.get("product_id", ""))
        meta = article_meta.get(pid, {})
        price, price_estimated = _display_price(meta, item)
        enriched_item = {
            **item,
            "rank": i,
            "name": meta.get("name") or pid,
            "brand": meta.get("brand") or "H&M",
            "category": meta.get("category", ""),
            "main_category": meta.get("main_category", ""),
            "color": meta.get("color", ""),
            "product_type": meta.get("product_type", ""),
            "price": price,
            "price_estimated": price_estimated,
            "price_source": _price_source_label(meta, price_estimated),
            "image_url": image_url_for_article(pid),
        }
        enriched.append(_attach_local_reason_text(enriched_item, limited=False))
    result["recommendations"] = enriched

    # B: LLM 추천 이유 생성 (include_reasons=True이고 API 키 있을 때만)
    reasons_generated = False
    if include_reasons and _gemini_available():
        items_desc = "\n".join(
            f"{item['rank']}. {item['name']} | category={item['category']} | "
            f"product_type={item['product_type']} | color={item['color']} | "
            f"price={int(item.get('price') or 0):,}원 | reason_key={item.get('reason', '')} | "
            f"score={item.get('score', 0):.3f}"
            for item in enriched
        )
        prompt = (
            "패션 쇼핑 앱의 추천 이유를 한국어로 작성하세요.\n"
            "상품마다 서로 다른 관점을 사용하세요: 검색 의도, 색감, 실루엣 인상, 활용 상황, 가격대, 취향 신호 중 하나를 골라 섞어 쓰세요.\n"
            "모든 문장을 같은 구조로 시작하지 마세요. 특히 '이 상품은 ... 페르소나에게 어울리는' 식 반복을 피하세요.\n"
            "상품 정보에 없는 소재, 핏, 기장, 브랜드 서사는 단정하지 마세요.\n"
            "각 이유는 자연스러운 1~2문장, 90자 안팎으로 쓰세요.\n"
            "입력 순서와 개수를 유지하고 reasons 배열만 반환하세요.\n\n"
            f"사용자 페르소나: {persona_hint or result.get('persona') or '개인화'}\n"
            f"검색어/맥락: {normalized_query_text or '최근 사용자 취향'}\n\n"
            f"상품 목록:\n{items_desc}\n\n"
            'JSON 형식: {"reasons":["이유1","이유2"]}'
        )
        try:
            llm_text = await _call_gemini(prompt, json_mode=True, temperature=0.55)
            reasons = _parse_gemini_reasons(llm_text, len(result["recommendations"]))
            for i, item in enumerate(result["recommendations"]):
                item["reason_text"] = reasons[i] if i < len(reasons) else ""
                item["reason_source"] = "gemini" if item["reason_text"] else "model"
            reasons_generated = any(item.get("reason_text") for item in result["recommendations"])
        except Exception:
            logging.exception("Gemini recommendation reasons failed. Using fallback reason_text.")

    if include_reasons and not reasons_generated:
        for item in result["recommendations"]:
            item["reason_text"] = _fallback_recommendation_reason(item, limited=True)
            item["reason_source"] = "fallback_token_limit"

    feature_store.r.set(cache_key, json.dumps(result), ex=RECOMMEND_CACHE_TTL)
    return result


@app.post("/api/explain-results")
async def explain_results(req: ExplainResultsRequest):
    """Generate AI reasons for the results already shown on the search page."""

    if not req.items:
        raise HTTPException(status_code=400, detail="설명할 검색 결과가 없습니다.")

    limited_items = req.items[:10]
    enriched_items = []
    for index, item in enumerate(limited_items, 1):
        normalized_id = str(item.id).strip()
        if normalized_id.isdigit():
            normalized_id = normalized_id.zfill(10)
        meta = article_meta.get(normalized_id, {})
        price, price_estimated = _display_price(meta, item.model_dump())
        enriched_items.append({
            "id": item.id,
            "rank": index,
            "name": meta.get("name") or item.title or normalized_id,
            "brand": meta.get("brand") or item.brand or "H&M",
            "category": meta.get("category", ""),
            "main_category": meta.get("main_category", ""),
            "color": meta.get("color", ""),
            "product_type": meta.get("product_type", ""),
            "price": price,
            "price_estimated": price_estimated,
            "price_source": _price_source_label(meta, price_estimated),
        })

    target_label = {
        "all": "전체 고객",
        "women": "여성 고객",
        "men": "남성 고객",
        "kids": "키즈 고객",
    }.get(_normalize_target_audience(req.target_audience), "전체 고객")

    if _gemini_available():
        items_desc = "\n".join(
            f"{item['rank']}. id={item['id']} | {item['name']} | "
            f"category={item['category']} | product_type={item['product_type']} | "
            f"color={item['color']} | brand={item['brand']} | "
            f"price={int(item['price']):,}원 | price_source={item['price_source']}"
            for item in enriched_items
        )
        prompt = (
            "패션 검색 결과 카드에 들어갈 추천 이유를 한국어로 작성하세요.\n"
            "각 상품마다 다른 관점을 선택해 자연스럽게 설명하세요: 검색어와의 연결, 색감, 카테고리 적합성, 가격대, 코디 활용 상황, 취향 신호.\n"
            "모든 이유를 같은 문장 구조로 시작하지 마세요. '이 상품은 ... 페르소나에게 어울리는' 같은 반복 표현은 금지합니다.\n"
            "상품 정보에 없는 소재, 핏, 기장, 디자인 특징은 단정하지 마세요.\n"
            "각 이유는 1~2문장으로 쓰되, 사람에게 말하듯 구체적이고 덜 템플릿처럼 작성하세요.\n"
            "입력된 상품 수와 같은 개수의 reasons 배열만 반환하세요.\n\n"
            f"검색어: {req.query or '없음'}\n"
            f"페르소나: {req.persona or '개인화'}\n\n"
            f"쇼핑 대상: {target_label}\n\n"
            f"상품 목록:\n{items_desc}\n\n"
            'JSON 형식: {"reasons":["이유1","이유2"]}'
        )
        try:
            llm_text = await _call_gemini(prompt, json_mode=True, temperature=0.6)
            reasons = _parse_gemini_reasons(llm_text, len(enriched_items))
            if sum(1 for reason in reasons if reason) < len(enriched_items):
                logging.info(
                    "Gemini returned partial result explanations: expected=%d got=%d",
                    len(enriched_items),
                    sum(1 for reason in reasons if reason),
                )
        except Exception:
            logging.exception("Gemini result explanations failed. Using local fallback reasons.")
            reasons = []
    else:
        logging.info("Gemini unavailable for result explanations. Using local fallback reasons.")
        reasons = []

    explanations = []
    for index, item in enumerate(enriched_items):
        category = item["category"] or item["product_type"] or "상품"
        color = item["color"] or "기본 색상"
        price_label = f"{int(item['price']):,}원대"
        estimated_note = "추정 가격 기준으로도 " if item.get("price_estimated") else ""
        fallback = (
            f"{item['name']}은 {color} 계열의 {category}라서 '{req.query or '현재 검색'}'에서 보인 색상과 품목 의도에 직접 맞습니다. "
            f"{estimated_note}{price_label}로 예산형 조합에 넣기 쉽고, {target_label} 기준의 현재 페르소나 선호와도 무난하게 연결됩니다."
        )
        reason = reasons[index] if index < len(reasons) and reasons[index] else fallback
        explanations.append({
            "id": item["id"],
            "reason": reason,
            "reason_source": "gemini" if index < len(reasons) and reasons[index] else "local_fallback",
        })

    return {"items": explanations}


@app.post("/api/events")
async def events(req: EventRequest):
    """클릭/구매 이벤트를 Redis에 저장하고 rec-models 세션도 업데이트한다."""
    effective_id = req.article_id or req.item_id

    # Redis 업데이트
    feature_store.r.incr("ct:event_count")
    if req.event_type in ("click", "view", "cart", "purchase") and effective_id:
        feature_store.push_click(req.user_id, effective_id)

    interest_changed = False
    if req.category:
        interest = feature_store.get_session_interest(req.user_id)
        interest[req.category] = interest.get(req.category, 0) + 1
        feature_store.set_session_interest(req.user_id, interest)
        interest_changed = True

    inferred_interest = {}
    if req.event_type == "search":
        inferred_interest = await _infer_session_interest_from_query(req.query_text)
    if inferred_interest:
        interest = feature_store.get_session_interest(req.user_id)
        for category, score in inferred_interest.items():
            interest[category] = interest.get(category, 0) + score
        feature_store.set_session_interest(req.user_id, interest)
        interest_changed = True

    if interest_changed:
        feature_store.invalidate_recommendation_cache(req.user_id)

    # rec-models 세션 업데이트 (실패해도 이벤트 저장은 성공으로 처리)
    if effective_id:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                await client.post(
                    f"{REC_URL}/session/update",
                    json={
                        "user_id": req.user_id,
                        "item_id": effective_id,
                        "event": req.event_type,
                    },
                )
            except httpx.RequestError:
                pass  # rec-models가 아직 없어도 게이트웨이는 정상 응답

    return {"status": "ok"}


@app.post("/api/onboarding")
async def onboarding(req: OnboardingRequest):
    """C-1: LLM으로 9개 페르소나 일치도 계산.

    유저 자유 입력 + 선택지를 LLM에 보내 9개 페르소나와의 일치도(%)를 반환한다.
    Redis에는 아직 저장하지 않는다.
    프론트엔드가 결과를 블록으로 보여주고 유저가 하나를 선택하면
    /api/onboarding/select를 호출해 확정한다.
    """
    style_text = ", ".join(req.style_choices) if req.style_choices else "없음"
    budget_text = {"low": "저가", "mid": "중간 가격대", "high": "고가"}.get(req.budget_range or "", "무관")
    target_audience = _normalize_target_audience(req.target_audience)

    prompt = (
        f"사용자가 아래와 같이 패션 취향을 설명했습니다.\n\n"
        f"자유 입력: {req.description}\n"
        f"선호 스타일: {style_text}\n"
        f"예산 범위: {budget_text}\n"
        f"쇼핑 대상: {target_audience}\n\n"
        f"아래 9개 패션 페르소나 중 이 사용자에게 해당하는 것을 골라 퍼센티지를 배분해주세요.\n\n"
        f"규칙 — 먼저 입력에서 페르소나와 연결되는 신호가 몇 개인지 파악하세요:\n"
        f"[신호 1개] 예: '파란색만 좋아', '할인 상품만 산다'\n"
        f"  → 해당 페르소나 70~85%, 나머지는 practical·careful·trendsetter 중 2~3개에 각각 5~15%씩 배분\n"
        f"  → 보조 페르소나 하나가 20% 이상이 되면 안 됩니다\n"
        f"[신호 2~3개] 예: '가성비 중시하고 붉은색 선호'\n"
        f"  → 가장 강한 신호 40~50%, 나머지 신호들이 나머지를 나눔. 무관한 페르소나는 0%\n"
        f"[신호 4개 이상 또는 모호] 예: '다양한 스타일을 즐기는 편'\n"
        f"  → 관련 페르소나들에 고르게 배분\n"
        f"- 합계는 반드시 100입니다.\n\n"
        f"페르소나 설명:\n"
        f"- trendsetter: 새로운 트렌드에 민감하고 다양한 스타일을 시도함\n"
        f"- practical: 실용적이고 목적 지향적인 구매, 기본 아이템 선호\n"
        f"- value: 가성비를 중시하고 세일/할인 상품을 적극 탐색\n"
        f"- brand_loyal: 특정 브랜드나 스타일에 반복적으로 집중\n"
        f"- impulse: 충동적으로 빠르게 구매 결정\n"
        f"- careful: 신중하게 오래 탐색하고 구매 전환율이 낮음\n"
        f"- repeat_stable: 동일한 상품이나 카테고리를 반복 구매\n"
        f"- color_focus: 특정 색상(예: 검정, 흰색, 파랑 등)을 기준으로 탐색, 색상 언급이 핵심 신호\n"
        f"- category_focus: 특정 카테고리(예: 아우터, 운동복 등)에만 집중\n\n"
        f"반드시 아래 JSON 키 이름 그대로, 숫자만 채워서 응답하세요 (합계 100):\n"
        f'{{"trendsetter": ?, "practical": ?, "value": ?, "brand_loyal": ?, '
        f'"impulse": ?, "careful": ?, "repeat_stable": ?, "color_focus": ?, "category_focus": ?}}'
    )

    # 같은 입력이면 캐시에서 바로 반환 (불필요한 Gemini 재호출 방지)
    cache_input = f"{req.description.strip().lower()}|{'|'.join(sorted(req.style_choices))}|{req.budget_range or ''}|{target_audience}"
    onboarding_cache_key = f"cache:onboarding:{hashlib.sha256(cache_input.encode('utf-8')).hexdigest()}"
    cached = feature_store.r.get(onboarding_cache_key)
    if cached:
        cached_result = json.loads(cached)
        feature_store.r.set(f"onboarding_scores:{req.user_id}", json.dumps(cached_result["persona_scores"]), ex=600)
        return cached_result

    if _gemini_available():
        try:
            llm_text = await _call_gemini(prompt, json_mode=True)
            normalized = _normalize_persona_scores(json.loads(llm_text))
        except httpx.HTTPStatusError as e:
            logging.warning("Gemini onboarding failed with status=%s. Using local fallback.", e.response.status_code)
            normalized = _fallback_persona_scores(req)
        except Exception:
            logging.exception("Gemini onboarding failed. Using local fallback.")
            normalized = _fallback_persona_scores(req)
    else:
        normalized = _fallback_persona_scores(req)

    result = {"persona_scores": normalized}
    feature_store.r.set(onboarding_cache_key, json.dumps(result), ex=3600)
    # select 호출 시 혼합에 쓸 점수를 임시 저장 (10분)
    feature_store.r.set(f"onboarding_scores:{req.user_id}", json.dumps(normalized), ex=600)
    return result


class PersonaSelectRequest(BaseModel):
    user_id: str
    persona: str  # 9개 중 유저가 선택한 페르소나
    persona_scores: dict[str, int | float] | None = None
    target_audience: str | None = None


@app.post("/api/onboarding/select")
async def onboarding_select(req: PersonaSelectRequest):
    """C-2: 유저가 선택한 페르소나를 Redis에 저장.

    프론트엔드에서 /api/onboarding 결과를 보고 유저가 고른 페르소나를
    session_interest로 변환해 Redis에 저장한다.
    다음 /api/recommend 호출 시 즉시 반영된다.
    """
    if req.persona not in VALID_PERSONAS:
        raise HTTPException(status_code=400, detail=f"알 수 없는 페르소나: {req.persona}")

    # 페르소나 → 카테고리 관심도 매핑
    persona_to_interest = PERSONA_SESSION_INTERESTS

    # 프론트가 보낸 최신 분석 점수를 우선 사용하고, 없으면 임시 Redis 캐시를 사용한다.
    selected_only_scores = {req.persona: 100}
    stored_scores = selected_only_scores
    if req.persona_scores:
        stored_scores = _normalize_persona_scores(req.persona_scores)

    stored_raw = feature_store.r.get(f"onboarding_scores:{req.user_id}")
    if stored_raw and not req.persona_scores:
        stored_scores = _normalize_persona_scores(json.loads(stored_raw))

    blended: dict[str, float] = {}
    for persona, weight in stored_scores.items():
        if weight <= 0 or persona not in persona_to_interest:
            continue
        for category, score in persona_to_interest[persona].items():
            blended[category] = blended.get(category, 0) + score * (weight / 100.0)
    session_interest = {key: round(value) for key, value in blended.items() if round(value) > 0}
    if not session_interest:
        session_interest = dict(persona_to_interest[req.persona])
    target_interest = _target_audience_interest(req.target_audience)
    if target_interest:
        session_interest[target_interest] = max(session_interest.get(target_interest, 0), 10)

    feature_store.set_persona_scores(req.user_id, stored_scores)
    feature_store.r.delete(f"onboarding_scores:{req.user_id}")
    feature_store.set_session_interest(req.user_id, session_interest)
    feature_store.invalidate_recommendation_cache(req.user_id)
    return {
        "status": "ok",
        "persona": req.persona,
        "persona_scores": stored_scores,
        "session_interest": session_interest,
    }


@app.post("/api/budget-set")
async def budget_set(
    user_id: str = Query(...),
    budget: int = Query(..., description="총 예산 (원)"),
    set_count: int = Query(3, description="구성할 세트 수"),
    query: str | None = Query(None, description="현재 검색어"),
    target_audience: str | None = Query(None, description="all | women | men | kids"),
):
    """D: 예산 기반 패션 세트 추천.

    rec-models 또는 search-engine 후보를 예산 내 아이템으로 필터링한 뒤,
    search-engine의 /cross-similarity API에서 받은 CLIP 기반 유사도를 조합 점수에 반영한다.
    search-engine이 준비되지 않았거나 유사도 계산에 실패하면 score 기반 그리디로 대체한다.
    """
    features = feature_store.get_user_features(user_id)
    raw_persona_scores = features.get("persona_scores", {})
    persona_scores = _normalize_persona_scores(raw_persona_scores) if raw_persona_scores else {}
    session_interest = dict(features["session_interest"]) if features["session_interest"] else {}
    target_audience = _normalize_target_audience(target_audience)
    target_interest = _target_audience_interest(target_audience)
    if target_interest:
        session_interest[target_interest] = max(session_interest.get(target_interest, 0), 10)
    inferred_interest = await _infer_session_interest_from_query(query)
    if inferred_interest:
        for category, score in inferred_interest.items():
            session_interest[category] = session_interest.get(category, 0) + score

    candidates: list[dict] = []
    anchor_candidates: list[dict] = []
    complement_candidates: list[dict] = []
    query_constraints = _derive_query_constraints(query)
    async with httpx.AsyncClient(timeout=10.0) as client:
        if query and query.strip():
            translated_query: str | None = None
            search_query = query.strip()
            if _has_korean(search_query) and _gemini_available():
                translated_query = await _translate_to_english(search_query)
                search_query = translated_query

            try:
                search_resp = await client.post(
                    f"{SEARCH_URL}/search",
                    json={"query": search_query, "top_k": SEARCH_INTENT_CANDIDATE_POOL},
                )
                search_resp.raise_for_status()
                search_results = _enrich_search_results(search_resp.json().get("results", []))
                search_results = _apply_target_audience_filter(search_results, target_audience, min_results=6)
                constraints = _derive_query_constraints(query, translated_query)
                query_constraints = constraints
                search_results = _apply_query_product_labels(search_results, constraints)
                constrained_results = [
                    item for item in search_results if _item_matches_query_constraints(item, constraints)
                ]
                product_results = [
                    item for item in search_results if _item_matches_query_product(item, constraints)
                ]
                color_results = [
                    item for item in search_results if _item_matches_query_color(item, constraints)
                ]
                anchor_candidates = constrained_results or product_results or color_results[:10] or search_results[:10]

                complement_groups = _complement_groups_for_constraints(constraints)
                complement_candidates = [
                    item
                    for item in search_results
                    if item not in anchor_candidates
                    and _item_matches_any_product_group(item, complement_groups)
                    and _item_matches_complement_color(item, constraints["colors"])
                ]
                if len(complement_candidates) < 6:
                    complement_candidates.extend(
                        item
                        for item in search_results
                        if item not in anchor_candidates
                        and item not in complement_candidates
                        and _item_matches_any_product_group(item, complement_groups)
                    )
                if len(complement_candidates) < 6:
                    complement_candidates.extend(
                        item
                        for item in search_results
                        if item not in anchor_candidates and item not in complement_candidates
                    )
                if complement_groups:
                    complement_candidates = sorted(
                        complement_candidates,
                        key=lambda item: (
                            _product_group_priority(item, complement_groups),
                            1 if _item_matches_complement_color(item, constraints["colors"]) else 0,
                            float(item.get("score") or 0),
                        ),
                        reverse=True,
                    )
                candidates = anchor_candidates + complement_candidates
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"search-engine 연결 실패: {e}")
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=str(e))

        if not candidates:
            params = {
                "user_id": user_id,
                "top_n": 50,
                "recent_clicks": ",".join(features["recent_clicks"]),
                "click_count": features["click_count"],
                "session_interest": json.dumps(session_interest) if session_interest else None,
                "persona_scores": json.dumps(persona_scores) if persona_scores else None,
            }
            try:
                rec_resp = await client.get(f"{REC_URL}/recommend", params=params)
                rec_resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=str(e))
            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"rec-models 연결 실패: {e}")
            candidates = rec_resp.json().get("recommendations", [])
            candidates = _apply_target_audience_filter(
                _enrich_recommendation_results(candidates),
                target_audience,
                min_results=6,
            )
            anchor_candidates = candidates
            complement_candidates = candidates

    # article_meta에서 가격 정보 보강 후 예산 내 필터
    # article_meta["price"]는 item_features CSV의 avg_price * PRICE_KRW_FACTOR (KRW)
    DEFAULT_PRICE = 25000  # 데이터 없는 상품의 폴백 (H&M KRW 중앙가)
    affordable = []
    for item in candidates:
        pid = str(item.get("product_id", ""))
        meta = article_meta.get(pid, {})
        price_int = meta.get("price", 0) or DEFAULT_PRICE
        if price_int <= budget:
            affordable.append({
                **item,
                **meta,
                "price_int": price_int,
                "article_id": pid,
                "image_url": image_url_for_article(pid),
            })

    if len(affordable) < 2:
        raise HTTPException(status_code=400, detail="예산 내 추천 가능한 상품이 부족합니다")

    # search-engine cross-similarity 호출 (미구현 시 빈 행렬로 대체)
    article_ids = [c["article_id"] for c in affordable[:20]]
    sim_matrix: dict = {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            sim_resp = await client.post(
                f"{SEARCH_URL}/cross-similarity",
                json={"article_ids": article_ids},
            )
            sim_resp.raise_for_status()
            sim_matrix = sim_resp.json().get("similarity", {})
        except httpx.HTTPStatusError as e:
            logging.info(
                "cross-similarity unavailable status=%s. Falling back to score-only budget set ranking.",
                e.response.status_code,
            )
        except httpx.RequestError as e:
            logging.info("cross-similarity request failed: %s. Falling back to score-only budget set ranking.", e)

    anchor_ids = {str(item.get("product_id", item.get("article_id", ""))) for item in anchor_candidates}
    complement_ids = {str(item.get("product_id", item.get("article_id", ""))) for item in complement_candidates}
    sets = _build_outfit_sets(
        affordable,
        sim_matrix,
        budget,
        set_count,
        anchor_ids=anchor_ids,
        complement_ids=complement_ids,
        query_constraints=query_constraints,
        target_audience=target_audience,
    )
    return {"sets": sets, "budget": budget, "set_count": len(sets)}


_OUTFIT_TEMPLATES = [
    ["top", "bottom", "outer", "accessory", "shoes"],
    ["top", "bottom", "outer", "shoes"],
    ["top", "bottom", "outer", "accessory"],
    ["top", "bottom", "accessory", "shoes"],
    ["top", "bottom", "accessory"],
    ["top", "bottom", "shoes"],
    ["top", "bottom"],
    ["dress", "outer", "accessory", "shoes"],
    ["dress", "outer", "accessory"],
    ["dress", "accessory", "shoes"],
    ["dress", "accessory"],
]


def _clamp_unit_score(value: float, *, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)):
        return minimum
    return max(minimum, min(float(value), 1.0))


def _build_outfit_sets(
    candidates: list[dict],
    sim_matrix: dict,
    budget: int,
    count: int,
    *,
    anchor_ids: set[str] | None = None,
    complement_ids: set[str] | None = None,
    query_constraints: dict[str, set[str]] | None = None,
    target_audience: str | None = None,
) -> list[list[dict]]:
    """Slot 기반 outfit 세트 조합.

    후보 풀을 outfit slot(top/bottom/outer/dress/accessory/shoes)으로 분류한 뒤
    템플릿에 따라 예산 내 세트를 구성한다. 후보 풀에 없는 슬롯은 전체 카탈로그에서 보충한다.
    """
    DEFAULT_PRICE = 25000
    target_audience = _normalize_target_audience(target_audience)
    anchor_ids = anchor_ids or set()
    complement_ids = complement_ids or set()
    raw_query_constraints = query_constraints or {}
    query_constraints = {
        "colors": set(raw_query_constraints.get("colors", set())),
        "products": set(raw_query_constraints.get("products", set())),
    }

    def to_catalog_item(aid: str, score: float = 0.12) -> dict:
        meta = article_meta.get(aid, {})
        price = int(meta.get("price") or 0) or DEFAULT_PRICE
        return {
            "product_id": aid,
            "article_id": aid,
            "name": meta.get("name", ""),
            "score": score,
            "price_int": price,
            "category": meta.get("category", ""),
            "main_category": meta.get("main_category", ""),
            "color": meta.get("color", ""),
            "product_type": meta.get("product_type", ""),
            "product_group": meta.get("product_group", ""),
            "department_name": meta.get("department_name", ""),
            "section_name": meta.get("section_name", ""),
            "garment_group": meta.get("garment_group", ""),
            "brand": meta.get("brand", ""),
            "image_url": image_url_for_article(aid),
            "price_estimated": not bool(meta.get("price")),
        }

    def normalized_raw_score(item: dict, max_raw_score: float) -> float:
        raw_score = float(item.get("score") or 0.0)
        if raw_score <= 0 or max_raw_score <= 0:
            return 0.0
        return _clamp_unit_score(raw_score / max_raw_score)

    def item_fit_score(item: dict, max_raw_score: float) -> float:
        aid = str(item.get("article_id") or item.get("product_id", ""))
        raw_score = float(item.get("score") or 0.0)
        score = 0.12

        if raw_score > 0:
            score = 0.35 + (0.35 * normalized_raw_score(item, max_raw_score))

        if aid in anchor_ids:
            score += 0.12
        elif aid in complement_ids:
            score += 0.08

        if query_constraints["products"] and _item_matches_query_product(item, query_constraints):
            score += 0.10
        if query_constraints["colors"] and _item_matches_query_color(item, query_constraints):
            score += 0.05
        if _item_matches_target_audience(item, target_audience):
            score += 0.03

        return round(_clamp_unit_score(score, minimum=0.05), 6)

    def pairwise_compatibility(outfit: list[dict]) -> float:
        values: list[float] = []
        for left_index, left in enumerate(outfit):
            left_id = str(left.get("article_id") or left.get("product_id", ""))
            for right in outfit[left_index + 1:]:
                right_id = str(right.get("article_id") or right.get("product_id", ""))
                raw_value = (
                    sim_matrix.get(left_id, {}).get(right_id)
                    if isinstance(sim_matrix.get(left_id), dict)
                    else None
                )
                if raw_value is None and isinstance(sim_matrix.get(right_id), dict):
                    raw_value = sim_matrix.get(right_id, {}).get(left_id)
                if raw_value is None:
                    continue
                values.append(_clamp_unit_score(float(raw_value)))

        if values:
            return sum(values) / len(values)

        item_scores = [float(item.get("item_score") or 0.0) for item in outfit]
        return (sum(item_scores) / len(item_scores)) * 0.7 if item_scores else 0.0

    def query_match_score(outfit: list[dict]) -> float:
        has_product = bool(query_constraints["products"])
        has_color = bool(query_constraints["colors"])
        if not has_product and not has_color:
            return 0.65

        scores: list[float] = []
        for item in outfit:
            total = 0
            matched = 0
            if has_product:
                total += 1
                matched += 1 if _item_matches_query_product(item, query_constraints) else 0
            if has_color:
                total += 1
                matched += 1 if _item_matches_query_color(item, query_constraints) else 0
            if total:
                scores.append(matched / total)
        return sum(scores) / len(scores) if scores else 0.0

    def budget_fit_score(total_price: int) -> float:
        if budget <= 0 or total_price <= 0:
            return 0.0
        usage = min(total_price / budget, 1.0)
        if usage <= 0.85:
            return usage / 0.85
        return max(0.7, 1.0 - ((usage - 0.85) / 0.15) * 0.3)

    def score_outfit(outfit: list[dict], total_price: int) -> list[dict]:
        item_scores = [float(item.get("item_score") or 0.0) for item in outfit]
        average_item_score = sum(item_scores) / len(item_scores) if item_scores else 0.0
        compatibility = pairwise_compatibility(outfit)
        query_score = query_match_score(outfit)
        budget_score = budget_fit_score(total_price)
        set_score = (
            (OUTFIT_ITEM_FIT_WEIGHT * average_item_score)
            + (OUTFIT_COMPATIBILITY_WEIGHT * compatibility)
            + (OUTFIT_QUERY_MATCH_WEIGHT * query_score)
            + (OUTFIT_BUDGET_FIT_WEIGHT * budget_score)
        )
        set_score = round(_clamp_unit_score(set_score, minimum=0.01), 6)

        return [
            {
                **item,
                "score": set_score,
                "set_score": set_score,
                "item_score": round(float(item.get("item_score") or 0.0), 6),
                "compatibility_score": round(compatibility, 6),
                "query_match_score": round(query_score, 6),
                "budget_fit_score": round(budget_score, 6),
                "set_total_price": total_price,
            }
            for item in outfit
        ]

    # 후보를 slot별로 분류
    slot_pools: dict[str, list[dict]] = {}
    for item in sorted(candidates, key=lambda x: float(x.get("score") or 0), reverse=True):
        aid = str(item.get("article_id") or item.get("product_id", ""))
        if not aid:
            continue
        meta = article_meta.get(aid, {})
        slot = _outfit_slot(meta)
        candidate_item = {**item, "article_id": aid}
        if not _item_matches_target_audience(candidate_item, target_audience):
            continue
        slot_pools.setdefault(slot, []).append(candidate_item)

    # 부족한 슬롯은 전체 카탈로그에서 보충 (target_audience 필터 적용)
    needed_slots = {"top", "bottom", "outer", "dress", "accessory", "shoes"}
    for slot in needed_slots:
        if len(slot_pools.get(slot, [])) < 3 and slot in _outfit_slot_index:
            existing_ids = {item["article_id"] for item in slot_pools.get(slot, [])}
            for aid in _outfit_slot_index[slot]:
                if len(slot_pools.get(slot, [])) >= 15:
                    break
                if aid in existing_ids:
                    continue
                catalog_item = to_catalog_item(aid)
                if not _item_matches_target_audience(catalog_item, target_audience):
                    continue
                slot_pools.setdefault(slot, []).append(catalog_item)

    all_slot_items = [item for pool in slot_pools.values() for item in pool]
    max_raw_score = max((float(item.get("score") or 0.0) for item in all_slot_items), default=0.0)
    for slot, pool in slot_pools.items():
        for item in pool:
            item["item_score"] = item_fit_score(item, max_raw_score)
        pool.sort(key=lambda item: float(item.get("item_score") or 0.0), reverse=True)

    sets: list[list[dict]] = []
    used_ids: set[str] = set()

    for template in _OUTFIT_TEMPLATES:
        if len(sets) >= count:
            break
        # 이 템플릿에 필요한 슬롯 후보가 있는지 확인
        if not all(slot_pools.get(slot) for slot in template):
            continue

        outfit: list[dict] = []
        cost = 0
        for slot in template:
            for item in slot_pools.get(slot, []):
                aid = item["article_id"]
                if aid in used_ids:
                    continue
                price = item.get("price_int", DEFAULT_PRICE)
                if cost + price > budget:
                    continue
                outfit.append(item)
                cost += price
                break  # 슬롯당 1개

        if len(outfit) >= 2:
            sets.append(score_outfit(outfit, cost))
            for item in outfit:
                used_ids.add(item["article_id"])

    sets.sort(key=lambda outfit: float(outfit[0].get("set_score") or 0.0), reverse=True)
    return sets


@app.get("/api/images/{article_id}")
async def get_image(article_id: str):
    """Return a local H&M product image by article id."""
    image_path = image_path_for_article(article_id)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        image_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/features/{user_id}")
async def get_features(user_id: str):
    """Redis에 저장된 유저 피처를 반환한다."""
    return feature_store.get_user_features(user_id)


@app.get("/health")
async def health():
    try:
        feature_store.r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

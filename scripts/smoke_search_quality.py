#!/usr/bin/env python3
"""Smoke-test search intent quality against a running API gateway."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request


API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000").rstrip("/")
SMOKE_USER_ID = os.getenv("SMOKE_USER_ID", "user_1024")

CASES = [
    {
        "query": "존나힙한후디",
        "top_k": 5,
        "expected_terms": ("hoodie", "hood"),
        "blocked_terms": ("t-shirt", "shirt", "jacket", "boots", "sneakers"),
    },
    {
        "query": "검정 후드티",
        "top_k": 5,
        "expected_terms": ("hoodie", "hood"),
        "blocked_terms": ("t-shirt", "shirt", "boots", "sneakers"),
    },
    {
        "query": "맨투맨",
        "top_k": 5,
        "expected_terms": ("sweatshirt", "sweater"),
        "blocked_terms": ("boots", "sneakers", "dress"),
    },
    {
        "query": "부츠",
        "top_k": 5,
        "expected_terms": ("boots",),
        "blocked_terms": ("hoodie", "t-shirt", "shirt"),
    },
    {
        "query": "스니커즈",
        "top_k": 5,
        "expected_terms": ("sneakers", "shoes"),
        "blocked_terms": ("hoodie", "t-shirt", "shirt"),
    },
    {
        "query": "셔츠",
        "top_k": 5,
        "expected_terms": ("shirt",),
        "blocked_terms": ("hoodie", "sneakers", "boots"),
    },
    {
        "query": "청바지",
        "top_k": 5,
        "expected_terms": ("jeans", "denim"),
        "blocked_terms": ("shorts", "skirt", "dress", "sneakers"),
    },
    {
        "query": "슬랙스",
        "top_k": 5,
        "expected_terms": ("trousers", "pants", "slacks"),
        "blocked_terms": ("jeans", "shorts", "skirt", "dress", "sneakers"),
    },
    {
        "query": "반바지",
        "top_k": 5,
        "expected_terms": ("shorts",),
        "blocked_terms": ("trousers", "jeans", "skirt", "dress", "sneakers"),
    },
    {
        "query": "레깅스",
        "top_k": 5,
        "expected_terms": ("leggings", "tights"),
        "blocked_terms": ("jeans", "shorts", "dress", "sneakers"),
    },
    {
        "query": "힙한 모자",
        "top_k": 5,
        "expected_terms": ("hat", "beanie", "cap"),
        "blocked_terms": ("socks", "t-shirt", "sweater", "sneakers"),
    },
    {
        "query": "검정 모자",
        "top_k": 5,
        "expected_terms": ("hat", "beanie", "cap"),
        "blocked_terms": ("socks", "t-shirt", "sweater", "sneakers"),
    },
    {
        "query": "양말",
        "top_k": 5,
        "expected_terms": ("socks", "sock", "tights"),
        "blocked_terms": ("hat", "beanie", "cap", "t-shirt", "sneakers", "boots"),
    },
    {
        "query": "슬리퍼",
        "top_k": 5,
        "expected_terms": ("slippers", "slipper"),
        "blocked_terms": ("sneakers", "boots", "t-shirt", "dress"),
    },
    {
        "query": "샌들",
        "top_k": 5,
        "expected_terms": ("sandals", "sandal", "flip flop"),
        "blocked_terms": ("sneakers", "boots", "t-shirt", "dress"),
    },
    {
        "query": "힐",
        "top_k": 5,
        "expected_terms": ("heels", "heel", "pumps", "wedge"),
        "blocked_terms": ("sneakers", "boots", "t-shirt", "dress"),
    },
    {
        "query": "로퍼",
        "top_k": 5,
        "expected_terms": ("flats", "flat shoe", "ballerinas", "loafer"),
        "blocked_terms": ("sneakers", "boots", "t-shirt", "dress"),
    },
    {
        "query": "가디건",
        "top_k": 5,
        "expected_terms": ("cardigan",),
        "blocked_terms": ("hoodie", "t-shirt", "sneakers", "boots"),
    },
    {
        "query": "블레이저",
        "top_k": 5,
        "expected_terms": ("blazer",),
        "blocked_terms": ("hoodie", "t-shirt", "sneakers", "boots"),
    },
    {
        "query": "나시",
        "top_k": 5,
        "expected_terms": ("vest top", "tank", "waistcoat"),
        "blocked_terms": ("hoodie", "sneakers", "boots", "jeans"),
    },
    {
        "query": "폴로티",
        "top_k": 5,
        "expected_terms": ("polo",),
        "blocked_terms": ("hoodie", "sneakers", "boots", "jeans"),
    },
    {
        "query": "수영복",
        "top_k": 5,
        "expected_terms": ("swimwear", "swimsuit", "bikini"),
        "blocked_terms": ("hoodie", "sneakers", "boots", "jeans"),
    },
    {
        "query": "브라",
        "top_k": 5,
        "expected_terms": ("bra",),
        "blocked_terms": ("bracelet", "sneakers", "boots", "jeans"),
    },
    {
        "query": "잠옷",
        "top_k": 5,
        "expected_terms": ("pyjama", "pajama", "night gown"),
        "blocked_terms": ("sneakers", "boots", "jeans", "blazer"),
    },
    {
        "query": "가운",
        "top_k": 5,
        "expected_terms": ("robe",),
        "blocked_terms": ("wardrobe", "sneakers", "boots", "jeans"),
    },
    {
        "query": "점프수트",
        "top_k": 5,
        "expected_terms": ("jumpsuit", "playsuit"),
        "blocked_terms": ("sneakers", "boots", "jeans", "t-shirt"),
    },
    {
        "query": "바디수트",
        "top_k": 5,
        "expected_terms": ("bodysuit", "body suit"),
        "blocked_terms": ("sneakers", "boots", "jeans", "blazer"),
    },
    {
        "query": "멜빵바지",
        "top_k": 5,
        "expected_terms": ("dungarees", "overall"),
        "blocked_terms": ("sneakers", "boots", "t-shirt", "dress"),
    },
    {
        "query": "투피스 세트",
        "top_k": 5,
        "expected_terms": ("garment set", "set"),
        "blocked_terms": ("sneakers", "boots", "t-shirt", "jeans"),
    },
    {
        "query": "선글라스",
        "top_k": 5,
        "expected_terms": ("sunglasses",),
        "blocked_terms": ("socks", "t-shirt", "sweater", "boots"),
    },
    {
        "query": "벨트",
        "top_k": 5,
        "expected_terms": ("belt",),
        "blocked_terms": ("socks", "t-shirt", "sweater", "boots"),
    },
    {
        "query": "스카프",
        "top_k": 5,
        "expected_terms": ("scarf",),
        "blocked_terms": ("socks", "t-shirt", "sneakers", "boots"),
    },
    {
        "query": "귀걸이",
        "top_k": 5,
        "expected_terms": ("earring", "earrings"),
        "blocked_terms": ("socks", "t-shirt", "sneakers", "boots"),
    },
    {
        "query": "액세서리",
        "top_k": 5,
        "expected_terms": ("accessory", "accessories"),
        "blocked_terms": ("t-shirt", "sweater", "trousers", "sneakers", "boots"),
    },
]


def post_json(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_GATEWAY_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (ConnectionResetError, socket.timeout, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"request failed after retries: {last_error}")


def item_text(item: dict) -> str:
    fields = (
        item.get("name"),
        item.get("category"),
        item.get("product_type"),
        item.get("matched_product_type"),
        item.get("brand"),
    )
    return " ".join(str(field or "") for field in fields).lower()


def validate_items(case: dict, items: list[dict], label: str) -> list[str]:
    errors: list[str] = []
    if len(items) < case["top_k"]:
        errors.append(f"{case['query']} {label}: expected {case['top_k']} results, got {len(items)}")
        return errors

    expected_terms = tuple(term.lower() for term in case["expected_terms"])
    blocked_terms = tuple(term.lower() for term in case["blocked_terms"])
    for rank, item in enumerate(items[:case["top_k"]], 1):
        text = item_text(item)
        if not any(term in text for term in expected_terms):
            errors.append(f"{case['query']} {label} rank {rank}: expected {expected_terms}, got {text!r}")
        if any(term in text for term in blocked_terms):
            errors.append(f"{case['query']} {label} rank {rank}: blocked term in {text!r}")
    return errors


def validate_case(case: dict) -> list[str]:
    payload = {
        "query": case["query"],
        "image_base64": None,
        "top_k": case["top_k"],
    }
    result = post_json("/api/search", payload)
    errors: list[str] = []
    errors.extend(validate_items(case, result.get("results", []), "search"))

    personalized_payload = {
        **payload,
        "top_k": max(case["top_k"], 150),
        "top_n": case["top_k"],
        "user_id": SMOKE_USER_ID,
        "personalization_weight": 0.7,
    }
    personalized = post_json("/api/personalized-search", personalized_payload)
    errors.extend(validate_items(case, personalized.get("personalized_results", []), "personalized"))
    if personalized.get("explicit_candidate_count", 0) <= 0:
        errors.append(f"{case['query']} personalized: expected explicit product candidates")
    return errors


def main() -> int:
    all_errors: list[str] = []
    for case in CASES:
        try:
            all_errors.extend(validate_case(case))
        except (RuntimeError, urllib.error.URLError, TimeoutError, ConnectionResetError, json.JSONDecodeError) as exc:
            all_errors.append(f"{case['query']}: request failed: {exc}")

    if all_errors:
        print("Search quality smoke test failed:")
        for error in all_errors:
            print(f"- {error}")
        return 1

    print(f"Search quality smoke test passed for {len(CASES)} cases against {API_GATEWAY_URL}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

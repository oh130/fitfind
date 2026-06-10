from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pydantic")
pytest.importorskip("redis")

sys.path.insert(0, str(Path("api_gateway").resolve()))

import app
from app import _build_outfit_sets


def test_budget_sets_assign_positive_set_and_item_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    article_meta = {
        "0000000001": {
            "name": "Oxford shirt",
            "product_group": "Garment Upper body",
            "product_type": "Shirt",
            "category": "Shirt",
            "main_category": "Menswear",
            "department_name": "Men Shirts",
            "section_name": "Men",
            "color": "White",
            "price": 30000,
        },
        "0000000002": {
            "name": "Chino trousers",
            "product_group": "Garment Lower body",
            "product_type": "Trousers",
            "category": "Trousers",
            "main_category": "Menswear",
            "department_name": "Men Trousers",
            "section_name": "Men",
            "color": "Black",
            "price": 40000,
        },
        "0000000003": {
            "name": "Canvas shoes",
            "product_group": "Shoes",
            "product_type": "Sneakers",
            "category": "Sneakers",
            "main_category": "Menswear",
            "department_name": "Men Shoes",
            "section_name": "Men",
            "color": "Black",
            "price": 20000,
        },
        "0000000004": {
            "name": "Ladies blouse",
            "product_group": "Garment Upper body",
            "product_type": "Blouse",
            "category": "Blouse",
            "main_category": "Ladieswear",
            "department_name": "Ladies Blouses",
            "section_name": "Women",
            "color": "White",
            "price": 25000,
        },
    }
    monkeypatch.setattr(app, "article_meta", article_meta)
    monkeypatch.setattr(app, "_outfit_slot_index", {"shoes": ["0000000003"]})

    candidates = [
        {
            "product_id": "0000000001",
            "article_id": "0000000001",
            "score": 0.72,
            "price_int": 30000,
            **article_meta["0000000001"],
        },
        {
            "product_id": "0000000002",
            "article_id": "0000000002",
            "score": 0.54,
            "price_int": 40000,
            **article_meta["0000000002"],
        },
        {
            "product_id": "0000000004",
            "article_id": "0000000004",
            "score": 0.99,
            "price_int": 25000,
            **article_meta["0000000004"],
        },
    ]

    sets = _build_outfit_sets(
        candidates,
        {
            "0000000001": {"0000000002": 0.76, "0000000003": 0.42},
            "0000000002": {"0000000003": 0.39},
        },
        budget=120000,
        count=1,
        anchor_ids={"0000000001"},
        complement_ids={"0000000002"},
        query_constraints={"products": {"shirt"}, "colors": set()},
        target_audience="men",
    )

    assert sets
    outfit = sets[0]
    assert all(item["score"] > 0 for item in outfit)
    assert all(item["set_score"] == outfit[0]["set_score"] for item in outfit)
    assert all(item["item_score"] > 0 for item in outfit)
    assert outfit[0]["set_total_price"] == 90000
    assert "0000000004" not in {item["article_id"] for item in outfit}

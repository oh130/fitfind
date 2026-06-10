from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pydantic")
pytest.importorskip("redis")

sys.path.insert(0, str(Path("api_gateway").resolve()))

from app import _apply_target_audience_filter, _item_matches_target_audience


def test_target_audience_filter_does_not_fall_back_to_other_audiences() -> None:
    items = [
        {"product_id": "1", "main_category": "Ladieswear"},
        {"product_id": "2", "main_category": "Menswear"},
    ]

    assert _apply_target_audience_filter(items, "men", min_results=5) == [items[1]]


def test_target_audience_filter_uses_department_for_sport_items() -> None:
    mens_sport = {
        "main_category": "Sport",
        "department_name": "Men Sport Woven",
        "section_name": "Men H&M Sport",
    }
    ladies_sport = {
        "main_category": "Sport",
        "department_name": "Ladies Sport Bottoms",
        "section_name": "Ladies H&M Sport",
    }

    assert _item_matches_target_audience(mens_sport, "men")
    assert not _item_matches_target_audience(ladies_sport, "men")

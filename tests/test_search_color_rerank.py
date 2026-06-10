from __future__ import annotations

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image, ImageDraw

from search_engine.search_engine import (
    MultimodalSearchEngine,
    SearchResult,
    dominant_color_signal_from_image,
    dominant_color_family_from_image,
    normalize_color_family,
)


def test_dominant_color_ignores_white_product_background() -> None:
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 25, 85, 105), fill="black")

    assert dominant_color_family_from_image(image) == "black"
    family, confidence = dominant_color_signal_from_image(image)
    assert family == "black"
    assert confidence >= 0.35


def test_dominant_color_signal_reports_low_confidence_for_balanced_colors() -> None:
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 59, 59), fill="black")
    draw.rectangle((60, 0, 119, 59), fill="red")
    draw.rectangle((0, 60, 59, 119), fill="blue")
    draw.rectangle((60, 60, 119, 119), fill="green")

    _family, confidence = dominant_color_signal_from_image(image)

    assert confidence < 0.35


def test_color_family_normalizes_catalog_color_names() -> None:
    assert normalize_color_family("Dark Grey") == "gray"
    assert normalize_color_family("Off White") == "white"
    assert normalize_color_family("Black") == "black"


def test_image_color_rerank_boosts_matching_color() -> None:
    engine = object.__new__(MultimodalSearchEngine)
    results = [
        SearchResult("white", 0.95, {"colour_group_name": "White"}),
        SearchResult("black", 0.92, {"colour_group_name": "Black"}),
        SearchResult("gray", 0.91, {"colour_group_name": "Grey"}),
    ]

    reranked = engine._rerank_results_by_image_color(results, "black", 0.62)

    assert [result.item_id for result in reranked] == ["black", "gray", "white"]
    assert reranked[0].metadata["image_query_color"] == "black"
    assert reranked[0].metadata["image_query_color_confidence"] == 0.62
    assert reranked[0].metadata["color_rerank_multiplier"] > 1

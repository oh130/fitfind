#!/usr/bin/env python3
"""Build article-level observed price means from H&M transactions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_price_map(transactions_path: Path, output_path: Path) -> None:
    price_sums: dict[str, float] = {}
    price_counts: dict[str, int] = {}

    with transactions_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                continue
            try:
                price = float(row.get("price", 0) or 0)
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            price_sums[article_id] = price_sums.get(article_id, 0.0) + price
            price_counts[article_id] = price_counts.get(article_id, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["article_id", "avg_price", "transaction_count"])
        writer.writeheader()
        for article_id in sorted(price_sums):
            count = price_counts[article_id]
            writer.writerow({
                "article_id": article_id,
                "avg_price": f"{price_sums[article_id] / count:.10f}",
                "transaction_count": count,
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transactions",
        type=Path,
        default=Path("data/raw/transactions_train.csv"),
        help="Path to transactions_train.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/article_price_map.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_price_map(args.transactions, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reads the Olist CSVs from data/raw/ and checks they are the files we expect."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reviews.config import ROOT

KAGGLE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"

# Columns parsed as dates, per table.
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
    "items": ["shipping_limit_date"],
}


class DataError(RuntimeError):
    pass


def raw_dir(cfg: dict) -> Path:
    return ROOT / cfg["data"]["raw_dir"]


def check_files(cfg: dict) -> list[str]:
    """Names of the files that are missing from data/raw/."""
    folder = raw_dir(cfg)
    return [f for f in cfg["data"]["files"].values() if not (folder / f).exists()]


def load_raw(cfg: dict) -> dict[str, pd.DataFrame]:
    missing = check_files(cfg)
    if missing:
        raise DataError(
            f"Missing from {raw_dir(cfg)}: {', '.join(missing)}\n"
            f"Download the dataset from {KAGGLE_URL} and extract the CSVs there."
        )

    tables: dict[str, pd.DataFrame] = {}
    expected = cfg["data"].get("expected_rows", {})
    for name, fname in cfg["data"]["files"].items():
        # Review text contains line breaks inside quotes; the C parser handles them.
        df = pd.read_csv(raw_dir(cfg) / fname, encoding="utf-8")
        for col in DATE_COLUMNS.get(name, []):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        want = expected.get(name)
        if want is not None and len(df) != want:
            raise DataError(
                f"{fname} has {len(df):,} rows, expected {want:,}. "
                "The download may be partial or a different release."
            )
        tables[name] = df
    return tables

"""Draws the hand labelling sample and stores the labels I give it."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from reviews.config import ROOT

LABEL_DIR = ROOT / "data" / "labels"
SAMPLE_FILE = LABEL_DIR / "sample.csv"
LABEL_FILE = LABEL_DIR / "labels.csv"
LABEL_COLUMNS = ["review_id", "themes", "unsure", "note", "labelled_at"]


def clean_text(s: str) -> str:
    return " ".join(str(s).split())


def reviews_with_text(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per review that has written text. A few review ids appear on
    more than one order with the same text; they are kept once."""
    r = raw["reviews"]
    r = r[r.review_comment_message.notna()].copy()
    r["text"] = r.review_comment_message.map(clean_text)
    r = r[r.text.str.len() > 0]
    return r.drop_duplicates("review_id")[["review_id", "review_score", "text"]]


def draw_sample(pool: pd.DataFrame, per_score: int, dev_share: float, seed: int) -> pd.DataFrame:
    parts = []
    for score, grp in pool.groupby("review_score"):
        take = grp.sample(min(per_score, len(grp)), random_state=seed).copy()
        n_dev = round(len(take) * dev_share)
        take["split"] = ["dev"] * n_dev + ["test"] * (len(take) - n_dev)
        parts.append(take)
    sample = pd.concat(parts)
    # interleave ratings so labelling never runs through 100 five star reviews in a row
    sample = sample.sample(frac=1, random_state=seed)
    order = {"dev": 0, "test": 1}
    sample = sample.sort_values("split", key=lambda s: s.map(order), kind="stable")
    return sample.reset_index(drop=True)


def load_labels() -> pd.DataFrame:
    if not LABEL_FILE.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.read_csv(LABEL_FILE, dtype=str, keep_default_na=False)


def save_label(review_id: str, themes: list[str], unsure: bool, note: str) -> int:
    """Writes one label, replacing any earlier one for the same review.
    The file is rewritten through a temp file so a crash cannot corrupt it."""
    df = load_labels()
    df = df[df.review_id != review_id]
    row = {
        "review_id": review_id,
        "themes": ";".join(sorted(themes)),
        "unsure": "1" if unsure else "0",
        "note": note.strip(),
        "labelled_at": datetime.now().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LABEL_FILE.with_suffix(".tmp")
    df.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, LABEL_FILE)
    return len(df)

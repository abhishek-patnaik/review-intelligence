"""Scores the classifier against labels: per theme precision, recall and F1,
plus whole review measures. The test split is only meant to be scored once."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from reviews import labels as lab


def classify_many(clf, texts: list[str], workers: int, progress: bool = True,
                  report_every: int = 0, started: float | None = None) -> list[list[str]]:
    import time
    from concurrent.futures import as_completed

    out: list[list[str] | None] = [None] * len(texts)
    started = started or time.time()
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_safe, clf, t): i for i, t in enumerate(texts)}
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is None:
                failures += 1
                res = []
            out[futures[fut]] = res
            if progress and (n % 25 == 0 or n == len(texts)):
                print(f"  {n}/{len(texts)}", flush=True)
            if report_every and (n % report_every == 0 or n == len(texts)):
                el = time.time() - started
                eta = el / n * (len(texts) - n)
                print(f"  {n:,}/{len(texts):,}  elapsed {el / 60:.0f} min  about {eta / 60:.0f} min left"
                      f"{f'  ({failures} failed, will retry next run)' if failures else ''}", flush=True)
    return out  # type: ignore[return-value]


def _safe(clf, text):
    """One failed request (a timeout, say) should not stop an hours long run.
    Failures are not cached, so the next run retries them."""
    try:
        return clf.classify(text)
    except Exception:
        return None


def to_matrix(theme_lists, themes: list[str]) -> np.ndarray:
    m = np.zeros((len(theme_lists), len(themes)), dtype=int)
    idx = {t: k for k, t in enumerate(themes)}
    for r, ts in enumerate(theme_lists):
        for t in ts:
            if t in idx:
                m[r, idx[t]] = 1
    return m


def score(truth, pred, themes: list[str]) -> tuple[pd.DataFrame, dict]:
    y, p = to_matrix(truth, themes), to_matrix(pred, themes)
    rows = []
    for k, t in enumerate(themes):
        tp = int((y[:, k] & p[:, k]).sum())
        fp = int(((1 - y[:, k]) & p[:, k]).sum())
        fn = int((y[:, k] & (1 - p[:, k])).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if tp + fp + fn else np.nan)
        rows.append({"theme": t, "support": tp + fn, "predicted": tp + fp,
                     "precision": prec, "recall": rec, "f1": f1})
    table = pd.DataFrame(rows)
    tp, fp, fn = (y & p).sum(), ((1 - y) & p).sum(), (y & (1 - p)).sum()
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    has_y, has_p = y.any(1), p.any(1)
    summary = {
        "reviews": len(truth),
        "exact_match": float((y == p).all(1).mean()),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(2 * micro_p * micro_r / (micro_p + micro_r)) if tp else 0.0,
        "macro_f1": float(table.loc[table.support >= 5, "f1"].mean()),
        "complaint_detection_accuracy": float((has_y == has_p).mean()),
    }
    return table, summary


def agreement(a, b, themes: list[str]) -> dict:
    """How often two labellers agree, as exact match and Cohen's kappa per theme."""
    x, y = to_matrix(a, themes), to_matrix(b, themes)
    kappas = []
    for k in range(len(themes)):
        po = (x[:, k] == y[:, k]).mean()
        pe = x[:, k].mean() * y[:, k].mean() + (1 - x[:, k].mean()) * (1 - y[:, k].mean())
        if pe < 1:
            kappas.append((po - pe) / (1 - pe))
    return {"reviews": len(a), "exact_match": float((x == y).all(1).mean()),
            "mean_kappa": float(np.mean(kappas))}


def load_truth(source: str) -> pd.DataFrame:
    sample = pd.read_csv(lab.SAMPLE_FILE, dtype={"review_id": str}, keep_default_na=False)
    f = lab.LABEL_DIR / ("reference_labels.csv" if source == "working" else "labels.csv")
    truth = pd.read_csv(f, dtype=str, keep_default_na=False)[["review_id", "themes"]]
    return sample.merge(truth, on="review_id")

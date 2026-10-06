"""Exports everything the explorer site needs into one small JSON file.
The site runs no model: every number is computed here from the pipeline output."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from reviews import size

EXAMPLES_PER_THEME = 24


def build(cfg: dict, raw: dict, themes_by_text: dict, out_path, translate=None) -> dict:
    themes = list(cfg["themes"])
    orders = size.order_facts(raw)
    reviews = size.review_table(raw, themes_by_text)
    table, summ, t = size.theme_impact(reviews, orders, themes, orders.order_purchase_timestamp.max())
    rating = size.rating_cost(t, themes, summ["no_complaint_avg_stars"])
    table = table.merge(rating, left_on="theme", right_index=True)
    valid = size.validation(t)
    conc = size.concentration(t)
    fix = size.fix_list(table.drop(columns="rating_cost"), rating, cfg)

    # category x theme rates for the 14 categories with the most written reviews
    t["cat"] = t.category.fillna("unknown")
    top_cats = t[t.cat != "unknown"].cat.value_counts().head(14).index
    cat_rows = []
    for c in top_cats:
        g = t[t.cat == c]
        cat_rows.append({"category": c.replace("_", " "), "reviews": int(len(g)),
                         "complaint_rate": float(g.complaint.mean()),
                         "avg_stars": float(g.review_score.mean()),
                         "rates": {th: float(g[th].mean()) for th in themes}})

    # monthly share of written reviews that complain, overall and for delivery
    t["month"] = t.review_creation_date.dt.to_period("M").astype(str)
    m = t.groupby("month").agg(reviews=("complaint", "size"), complaint=("complaint", "mean"),
                               not_delivered=("not_delivered", "mean"), late=("late_delivery", "mean"))
    m = m[m.reviews >= 300].reset_index()

    # example reviews: ones with exactly this theme, short enough to read, fixed seed
    rng = np.random.default_rng(cfg["labels"]["random_seed"])
    ex = {}
    single = t[t[themes].sum(axis=1) == 1]
    for th in themes + ["none"]:
        pool = t[~t.complaint] if th == "none" else single[single[th]]
        pool = pool[pool.text.str.len().between(25, 240)].drop_duplicates("text")
        pick = pool.iloc[rng.permutation(len(pool))[:EXAMPLES_PER_THEME]]
        ex[th] = [{"pt": r.text, "stars": int(r.review_score), "category": str(r.cat).replace("_", " "),
                   "value": round(float(r.order_value), 2) if pd.notna(r.order_value) else None}
                  for r in pick.itertuples()]
    if translate:
        flat = [e for lst in ex.values() for e in lst]
        for e, en in zip(flat, translate([e["pt"] for e in flat])):
            e["en"] = en

    evals = {}
    for v in ("v1", "v2"):
        f = out_path.parent.parent / "outputs" / "tables" / f"eval_{cfg['llm']['model'].replace(':', '_')}_{v}_dev_working.json"
        if f.exists():
            evals[v] = json.loads(f.read_text())

    data = {
        "summary": {**{k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                       for k, v in summ.items()}, **conc},
        "model": cfg["llm"]["model"], "prompt_version": cfg["llm"]["prompt_version"],
        "themes": [{"id": th, "definition": cfg["themes"][th]} for th in themes],
        "impact": json.loads(table.to_json(orient="records")),
        "validation": json.loads(valid.to_json(orient="records")),
        "fix": json.loads(fix.to_json(orient="records")),
        "categories": cat_rows,
        "monthly": json.loads(m.to_json(orient="records")),
        "examples": ex,
        "evals": evals,
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return data

"""Layer 3: links each complaint theme to what happened to the order and the
customer, so themes can be ranked by business impact instead of by count."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reviews.labels import clean_text


def order_facts(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per order: value, delivery timing, customer and whether that
    customer ordered again afterwards."""
    o = raw["orders"].merge(raw["customers"][["customer_id", "customer_unique_id"]], on="customer_id")
    items = raw["items"].groupby("order_id").agg(
        order_value=("price", "sum"), freight=("freight_value", "sum"),
        units=("order_item_id", "count"), sellers=("seller_id", "nunique"))
    cat = (raw["items"].merge(raw["products"][["product_id", "product_category_name"]], on="product_id", how="left")
           .merge(raw["categories"], on="product_category_name", how="left"))
    cat["category"] = cat.product_category_name_english.fillna(cat.product_category_name).fillna("unknown")
    main_cat = cat.sort_values("price", ascending=False).drop_duplicates("order_id").set_index("order_id")
    o = o.join(items, on="order_id").join(main_cat[["category", "seller_id"]], on="order_id")
    o["delivered_late"] = o.order_delivered_customer_date > o.order_estimated_delivery_date
    o["days_late"] = (o.order_delivered_customer_date - o.order_estimated_delivery_date).dt.days
    o["never_delivered"] = o.order_delivered_customer_date.isna()
    # repeat purchase: does the same person place another order after this one?
    o = o.sort_values("order_purchase_timestamp")
    o["next_order"] = o.groupby("customer_unique_id").order_purchase_timestamp.shift(-1)
    o["bought_again"] = o.next_order.notna()
    return o


def review_table(raw: dict[str, pd.DataFrame], themes_by_text: dict[str, str | None]) -> pd.DataFrame:
    r = raw["reviews"].copy()
    r["text"] = r.review_comment_message.map(lambda s: clean_text(s) if isinstance(s, str) else "")
    r["has_text"] = r.text.str.len() > 0
    r["themes"] = r.text.map(themes_by_text)
    return r


def theme_impact(reviews: pd.DataFrame, orders: pd.DataFrame, themes: list[str], data_end) -> tuple[pd.DataFrame, dict]:
    """Per theme: volume, rating, order value, delivery facts and repeat purchase,
    compared against reviews with text that contain no complaint."""
    t = reviews[reviews.has_text & reviews.themes.notna()].merge(orders, on="order_id", how="left")
    # repeat purchase needs time to happen: only orders at least 180 days before the data ends
    t["can_repeat"] = t.order_purchase_timestamp <= data_end - pd.Timedelta(days=180)
    flags = pd.DataFrame({th: t.themes.str.split(";").map(lambda xs: th in xs) for th in themes})
    t = pd.concat([t, flags], axis=1)
    t["complaint"] = flags.any(axis=1)
    base = t[~t.complaint]
    base_rep = base[base.can_repeat]
    rows = []
    for th in themes:
        g = t[t[th]]
        rep = g[g.can_repeat]
        p, n = rep.bought_again.mean() if len(rep) else np.nan, len(rep)
        rows.append({
            "theme": th,
            "reviews": len(g),
            "share_of_complaints": len(g) / t.complaint.sum(),
            "avg_stars": g.review_score.mean(),
            "stars_vs_no_complaint": g.review_score.mean() - base.review_score.mean(),
            "one_star_share": (g.review_score == 1).mean(),
            "order_value": g.order_value.sum(),
            "median_order_value": g.order_value.median(),
            "delivered_late_share": g.delivered_late.mean(),
            "never_delivered_share": g.never_delivered.mean(),
            "repeat_rate": p,
            "repeat_n": n,
            "repeat_ci_low": wilson(p, n)[0] if n else np.nan,
            "repeat_ci_high": wilson(p, n)[1] if n else np.nan,
        })
    table = pd.DataFrame(rows).sort_values("order_value", ascending=False)
    summary = {
        "reviews_total": len(reviews),
        "reviews_with_text": int(reviews.has_text.sum()),
        "reviews_classified": len(t),
        "complaint_reviews": int(t.complaint.sum()),
        "complaint_share": float(t.complaint.mean()),
        "multi_theme_share": float((flags.sum(axis=1) > 1)[t.complaint].mean()),
        "no_complaint_avg_stars": float(base.review_score.mean()),
        "no_complaint_repeat_rate": float(base_rep.bought_again.mean()),
        "no_complaint_repeat_n": len(base_rep),
        "no_complaint_repeat_ci_low": wilson(base_rep.bought_again.mean(), len(base_rep))[0],
        "no_complaint_repeat_ci_high": wilson(base_rep.bought_again.mean(), len(base_rep))[1],
        "complaint_order_value": float(t.loc[t.complaint, "order_value"].sum()),
        "all_order_value": float(orders.order_value.sum()),
    }
    return table, summary, t


def validation(t: pd.DataFrame) -> pd.DataFrame:
    """An outside check on the classifier that needs no labels: reviews the model
    calls late or not delivered should line up with the delivery dates Olist recorded."""
    groups = {
        "Tagged late delivery": t[t.late_delivery],
        "Tagged not delivered": t[t.not_delivered],
        "Any other complaint": t[t.complaint & ~t.late_delivery & ~t.not_delivered],
        "No complaint": t[~t.complaint],
    }
    rows = []
    for name, g in groups.items():
        rows.append({"group": name, "reviews": len(g),
                     "delivered_after_estimate": g.delivered_late.mean(),
                     "no_delivery_date": g.never_delivered.mean(),
                     "median_days_late_when_late": g.loc[g.delivered_late, "days_late"].median()})
    return pd.DataFrame(rows)


def by_segment(t: pd.DataFrame, col: str, theme: str, min_reviews: int = 150, top: int = 10) -> pd.DataFrame:
    """Complaint rate for one theme by category or seller, among segments with enough reviews."""
    g = t[t[col] != "unknown"].groupby(col).agg(reviews=(theme, "size"), hits=(theme, "sum"))
    g = g[g.reviews >= min_reviews]
    g["rate"] = g.hits / g.reviews
    return g.sort_values("rate", ascending=False).head(top).reset_index()


def concentration(t: pd.DataFrame, col: str = "seller_id", min_reviews: int = 30) -> dict:
    """Do some sellers complain far more than others? Among sellers with enough
    written reviews, compares the worst 10% by complaint rate with the rest.
    Ranking by rate, not by count, so big sellers are not flagged for being big."""
    g = t.groupby(col).agg(reviews=("complaint", "size"), complaints=("complaint", "sum"))
    g = g[g.reviews >= min_reviews]
    g["rate"] = g.complaints / g.reviews
    k = max(1, int(round(0.10 * len(g))))
    worst, rest = g.nlargest(k, "rate"), g.drop(g.nlargest(k, "rate").index)
    return {"sellers_rated": len(g), "min_reviews": min_reviews, "worst_sellers": k,
            "worst_rate": float(worst.complaints.sum() / worst.reviews.sum()),
            "rest_rate": float(rest.complaints.sum() / rest.reviews.sum()),
            "worst_share_of_complaints": float(worst.complaints.sum() / g.complaints.sum()),
            "worst_share_of_reviews": float(worst.reviews.sum() / g.reviews.sum())}


def rating_cost(t: pd.DataFrame, themes: list[str], base_stars: float) -> pd.Series:
    """Stars each theme takes off the average rating of reviews with text.
    A review's shortfall against the no complaint average is split equally
    between the themes it mentions, so overlapping themes are not double counted."""
    flags = t[themes].astype(int)
    n = flags.sum(axis=1).replace(0, np.nan)
    shortfall = (base_stars - t.review_score).clip(lower=0) / n
    return (flags.mul(shortfall, axis=0).sum() / len(t)).rename("rating_cost")


def fix_list(table: pd.DataFrame, rating: pd.Series, cfg: dict) -> pd.DataFrame:
    """Ranks themes by the order value they touch, with the rating damage alongside.
    Repeat purchase is left out of the ranking: it showed no reliable effect."""
    f = table.merge(rating, left_on="theme", right_index=True)
    f = f[f.theme != "other_complaint"].sort_values("order_value", ascending=False)
    f["owner"] = f.theme.map(cfg["size"]["owner"])
    f["action"] = f.theme.map(cfg["size"]["action"])
    return f[["theme", "owner", "reviews", "order_value", "avg_stars", "rating_cost", "action"]].reset_index(drop=True)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)

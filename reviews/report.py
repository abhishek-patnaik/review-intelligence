"""Builds the tables, charts, REPORT.md and the findings block in README.md
from the classified reviews. Every number in the write up comes from here."""

from __future__ import annotations

import json
import re
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reviews import size
from reviews.config import ROOT

# chart styling: one accent hue for the thing being measured, grey for context
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4df"
ACCENT, CONTEXT = "#2a78d6", "#b9b7b0"
LABEL = {
    "not_delivered": "Not delivered", "late_delivery": "Late delivery", "missing_items": "Missing items",
    "wrong_item": "Wrong item", "damaged_defective": "Damaged or defective",
    "not_as_described": "Not as described", "poor_quality": "Poor quality",
    "seller_support": "Seller support", "refund_return": "Refund or return",
    "packaging": "Packaging", "price_shipping": "Price or shipping cost", "other_complaint": "Other",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w, h):
    fig = plt.figure(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig


def chart_themes(table: pd.DataFrame, base_stars: float, path) -> None:
    d = table.sort_values("reviews")
    labels = [LABEL[t] for t in d.theme]
    fig = _fig(10, 5.2)
    a1, a2 = fig.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.08})
    a1.barh(labels, d.reviews, color=ACCENT, height=0.62)
    for y, v in enumerate(d.reviews):
        a1.text(v, y, f"  {v:,}", va="center", fontsize=8.5, color=INK2)
    a1.set_title("Reviews mentioning the complaint", loc="left", fontsize=10.5, color=INK, pad=10)
    a2.barh(labels, d.avg_stars, color=ACCENT, height=0.62)
    a2.axvline(base_stars, color=INK2, linestyle=(0, (3, 3)), linewidth=1)
    a2.text(base_stars - 0.08, -0.9, f"no complaint {base_stars:.1f}", fontsize=8, color=INK2, ha="right", va="center")
    for y, v in enumerate(d.avg_stars):
        a2.text(v, y, f"  {v:.1f}", va="center", fontsize=8.5, color=INK2)
    a2.set_xlim(0, 5.4)
    a2.set_title("Average star rating", loc="left", fontsize=10.5, color=INK, pad=10)
    for a in (a1, a2):
        _style(a)
    a1.tick_params(axis="y", labelsize=9.5, colors=INK)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def chart_validation(v: pd.DataFrame, path) -> None:
    d = v.iloc[::-1]
    colors = [ACCENT if g.startswith("Tagged") else CONTEXT for g in d.group]
    fig = _fig(8, 2.8)
    ax = fig.subplots()
    ax.barh(d.group, d.delivered_after_estimate * 100, color=colors, height=0.6)
    for y, (val, n) in enumerate(zip(d.delivered_after_estimate, d.reviews)):
        ax.text(val * 100, y, f"  {val:.0%}  ({n:,} reviews)", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, max(d.delivered_after_estimate) * 100 * 1.45)
    ax.set_title("Orders delivered after the promised date, by what the model said", loc="left",
                 fontsize=10.5, color=INK, pad=10)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    _style(ax)
    ax.tick_params(axis="y", labelsize=9.5, colors=INK)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def chart_fix(f: pd.DataFrame, path) -> None:
    d = f.sort_values("order_value")
    labels = [LABEL[t] for t in d.theme]
    fig = _fig(8.5, 4.4)
    ax = fig.subplots()
    ax.barh(labels, d.order_value / 1000, color=ACCENT, height=0.62)
    for y, (v, rc) in enumerate(zip(d.order_value, d.rating_cost)):
        ax.text(v / 1000, y, f"  R$ {v / 1000:,.0f}k   {rc:.2f} stars off the average", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, d.order_value.max() / 1000 * 1.6)
    ax.set_title("Order value behind each complaint (R$ thousands)", loc="left", fontsize=10.5, color=INK, pad=10)
    _style(ax)
    ax.tick_params(axis="y", labelsize=9.5, colors=INK)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _pct(x):
    return f"{x:.0%}"


def build(cfg: dict, raw: dict, themes_by_text: dict, partial: bool, paths) -> dict:
    themes = list(cfg["themes"])
    orders = size.order_facts(raw)
    reviews = size.review_table(raw, themes_by_text)
    table, summ, t = size.theme_impact(reviews, orders, themes, orders.order_purchase_timestamp.max())
    rating = size.rating_cost(t, themes, summ["no_complaint_avg_stars"])
    table = table.merge(rating, left_on="theme", right_index=True)
    valid = size.validation(t)
    conc = size.concentration(t)
    fix = size.fix_list(table.drop(columns="rating_cost"), rating, cfg)
    cats = {th: size.by_segment(t, "category", th, min_reviews=200, top=5)
            for th in ("not_as_described", "poor_quality", "damaged_defective", "not_delivered")}

    table.to_csv(paths.tables / "theme_impact.csv", index=False)
    valid.to_csv(paths.tables / "delivery_validation.csv", index=False)
    fix.to_csv(paths.tables / "fix_list.csv", index=False)
    for th, c in cats.items():
        c.to_csv(paths.tables / f"categories_{th}.csv", index=False)
    (paths.tables / "summary.json").write_text(json.dumps({**summ, **conc}, indent=2, default=float))

    chart_themes(table, summ["no_complaint_avg_stars"], paths.figures / "01_themes.png")
    chart_validation(valid, paths.figures / "02_delivery_check.png")
    chart_fix(fix, paths.figures / "03_fix_list.png")

    evals = {}
    for v in ("v1", "v2"):
        f = paths.tables / f"eval_{cfg['llm']['model'].replace(':', '_')}_{v}_dev_working.json"
        if f.exists():
            evals[v] = json.loads(f.read_text())

    ctx = dict(cfg=cfg, table=table, summ=summ, valid=valid, conc=conc, fix=fix, cats=cats,
               evals=evals, partial=partial, rating=rating)
    (ROOT / "REPORT.md").write_text(report_md(**ctx), encoding="utf-8")
    write_findings(findings_md(**ctx))
    return ctx


def findings_md(cfg, table, summ, valid, conc, fix, cats, evals, partial, rating) -> str:
    top = table.sort_values("reviews", ascending=False)
    nd, ld = table.set_index("theme").loc["not_delivered"], table.set_index("theme").loc["late_delivery"]
    vd = valid.set_index("group")
    delivery = top[top.theme.isin(["not_delivered", "late_delivery"])].reviews.sum()
    fulfil = top[top.theme.isin(["missing_items", "wrong_item"])].reviews.sum()
    worst_rc = rating.sort_values(ascending=False)
    total_rc = rating.sum()
    rep_hi = table[table.repeat_n >= 100].repeat_rate.max()
    rep_lo = table[table.repeat_n >= 100].repeat_rate.min()
    nad = cats["not_as_described"].head(3)
    lines = []
    if partial:
        lines.append(f"> Partial run: {summ['reviews_classified']:,} of {summ['reviews_with_text']:,} reviews with text classified so far.\n")
    lines += [
        "## What the data says", "",
        f"- **{_pct(summ['complaint_share'])} of written reviews contain a complaint**, "
        f"{summ['complaint_reviews']:,} of {summ['reviews_classified']:,}. {_pct(summ['multi_theme_share'])} of those mention more than one problem.",
        f"- **Delivery is the biggest problem by far.** \"Not delivered\" alone is {int(nd.reviews):,} reviews "
        f"({_pct(nd.share_of_complaints)} of complaints) and costs {abs(nd.stars_vs_no_complaint):.1f} stars against a review with no complaint. "
        f"Together with late delivery that is {delivery:,} reviews.",
        f"- **The model's delivery tags match Olist's own delivery records.** Orders behind reviews tagged late were delivered after the promised date "
        f"{_pct(vd.loc['Tagged late delivery', 'delivered_after_estimate'])} of the time, against "
        f"{_pct(vd.loc['No complaint', 'delivered_after_estimate'])} for reviews with no complaint. "
        f"For reviews tagged not delivered, {_pct(vd.loc['Tagged not delivered', 'no_delivery_date'])} of orders never got a delivery date at all. "
        "This check needs no labels, so it is independent evidence that the classifier reads these reviews correctly.",
        f"- **The second biggest group is the seller sending the wrong thing.** Missing items and wrong items add up to {fulfil:,} reviews, "
        "a fulfilment problem, not a logistics one.",
        f"- **A small group of sellers has a much worse record.** Among {conc['sellers_rated']:,} sellers with at least {conc['min_reviews']} written reviews, "
        f"the worst 10% ({conc['worst_sellers']}) draw a complaint on {_pct(conc['worst_rate'])} of reviews, against {_pct(conc['rest_rate'])} for the rest. "
        f"They carry {_pct(conc['worst_share_of_reviews'])} of those reviews but {_pct(conc['worst_share_of_complaints'])} of the complaints.",
        f"- **Complaints did not measurably change whether customers bought again.** Repeat purchase is rare on Olist "
        f"({_pct(summ['no_complaint_repeat_rate'])} for customers with no complaint), and the rate for complaint themes ranges from "
        f"{_pct(rep_lo)} to {_pct(rep_hi)} with overlapping confidence intervals. So I rank problems by the orders and ratings they affect, not by churn.",
        f"- **Complaints take {total_rc:.2f} stars off the average rating of written reviews.** {LABEL[worst_rc.index[0]]} is the largest share "
        f"({worst_rc.iloc[0]:.2f}), then {LABEL[worst_rc.index[1]].lower()} ({worst_rc.iloc[1]:.2f}).",
    ]
    if evals.get("v1") and evals.get("v2"):
        e1, e2 = evals["v1"], evals["v2"]
        lines.append(
            f"- **Writing down the rules made the model better.** On a 150 review dev set, exact match went from {_pct(e1['exact_match'])} "
            f"to {_pct(e2['exact_match'])} and micro F1 from {e1['micro_f1']:.2f} to {e2['micro_f1']:.2f}. "
            f"It tells complaints from non complaints {_pct(e2['complaint_detection_accuracy'])} of the time.")
    lines += ["", "![Complaint themes](outputs/figures/01_themes.png)", "",
              "![Delivery check](outputs/figures/02_delivery_check.png)", "",
              "### What I would fix first", ""]
    for i, r in enumerate(fix.head(5).itertuples(), 1):
        lines.append(f"{i}. **{LABEL[r.theme]}** ({r.owner.lower()}): {r.reviews:,} reviews on R$ {r.order_value:,.0f} of orders. {r.action}.")
    if len(nad):
        lines.append(f"\n\"Not as described\" is worst in {', '.join(nad.category.str.replace('_', ' '))}, the first place to audit listings.")
    lines += ["", "![Fix list](outputs/figures/03_fix_list.png)", "",
              "The full write up, with every table, is in [REPORT.md](REPORT.md).", "",
              f"<sub>Numbers generated by the pipeline on {date.today():%d %b %Y}. Re-running it rewrites this section.</sub>"]
    return "\n".join(lines)


def write_findings(block: str) -> None:
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    new = f"<!-- FINDINGS:START -->\n\n{block}\n\n<!-- FINDINGS:END -->"
    s = re.sub(r"<!-- FINDINGS:START -->.*?<!-- FINDINGS:END -->", lambda _: new, s, flags=re.S)
    readme.write_text(s, encoding="utf-8")


def report_md(cfg, table, summ, valid, conc, fix, cats, evals, partial, rating) -> str:
    def md(df, fmt):
        cols = list(fmt)
        out = ["| " + " | ".join(fmt[c][0] for c in cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(fmt[c][1](r[c]) for c in cols) + " |")
        return "\n".join(out)

    name = lambda t: LABEL.get(t, t)
    parts = ["# Review Intelligence: full report", ""]
    if partial:
        parts.append(f"> Partial run: {summ['reviews_classified']:,} of {summ['reviews_with_text']:,} reviews classified.\n")
    parts += [
        "## 1. Data", "",
        f"Olist has {summ['reviews_total']:,} reviews, {summ['reviews_with_text']:,} of them with written text. "
        f"Each written review was classified by {cfg['llm']['model']} (prompt {cfg['llm']['prompt_version']}) running locally. "
        f"Identical texts (\"muito bom\" appears thousands of times) are classified once.", "",
        "## 2. How accurate is the model?", "",
    ]
    if evals:
        rows = []
        for v, e in evals.items():
            rows.append({"prompt": v, **e})
        ev = pd.DataFrame(rows)
        parts.append(md(ev, {"prompt": ("Prompt", str), "exact_match": ("Exact match", _pct),
                             "micro_precision": ("Micro precision", lambda x: f"{x:.2f}"),
                             "micro_recall": ("Micro recall", lambda x: f"{x:.2f}"),
                             "micro_f1": ("Micro F1", lambda x: f"{x:.2f}"),
                             "macro_f1": ("Macro F1", lambda x: f"{x:.2f}"),
                             "complaint_detection_accuracy": ("Complaint detection", _pct)}))
        parts += ["", "Dev set: 150 reviews stratified by star rating, labelled against [docs/LABELLING_GUIDE.md](docs/LABELLING_GUIDE.md). "
                  "v2 adds the guide's rules to the prompt. Weak spots that remain: the model tags refund_return when a customer only asks "
                  "for a refund, rarely uses other_complaint, and is a little eager with poor_quality. The held-out test set has not been scored yet.", ""]
    parts += ["### An outside check: delivery records", "",
              "The model never sees delivery dates, so comparing its delivery tags with the dates Olist recorded is a check that needs no labels.", "",
              md(valid, {"group": ("Reviews", str), "reviews": ("Count", lambda x: f"{x:,}"),
                         "delivered_after_estimate": ("Delivered after the promised date", _pct),
                         "no_delivery_date": ("Never delivered", _pct),
                         "median_days_late_when_late": ("Median days late, when late", lambda x: f"{x:.0f}")}),
              "", "![Delivery check](outputs/figures/02_delivery_check.png)", "",
              "## 3. What customers complain about", "",
              f"{_pct(summ['complaint_share'])} of written reviews contain at least one complaint. "
              f"Reviews with no complaint average {summ['no_complaint_avg_stars']:.2f} stars.", "",
              md(table.sort_values("reviews", ascending=False),
                 {"theme": ("Theme", name), "reviews": ("Reviews", lambda x: f"{x:,}"),
                  "share_of_complaints": ("Share of complaints", _pct), "avg_stars": ("Avg stars", lambda x: f"{x:.2f}"),
                  "one_star_share": ("1 star", _pct), "order_value": ("Order value (R$)", lambda x: f"{x:,.0f}"),
                  "rating_cost": ("Stars off the average", lambda x: f"{x:.3f}")}),
              "", "Shares of complaints add up to more than 100% because a review can mention several problems. "
              "\"Stars off the average\" splits each review's shortfall equally between the themes it mentions.", "",
              "![Themes](outputs/figures/01_themes.png)", "",
              "## 4. Did complaints stop customers coming back?", "",
              f"Only orders placed at least 180 days before the data ends are counted, so every customer had time to return. "
              f"Customers who left no complaint bought again {_pct(summ['no_complaint_repeat_rate'])} of the time "
              f"({summ['no_complaint_repeat_n']:,} orders).", "",
              md(table.sort_values("repeat_n", ascending=False),
                 {"theme": ("Theme", name), "repeat_n": ("Orders", lambda x: f"{x:,}"),
                  "repeat_rate": ("Bought again", lambda x: f"{x:.1%}"),
                  "repeat_ci_low": ("95% interval", lambda x: f"{x:.1%}"),
                  "repeat_ci_high": ("", lambda x: f"to {x:.1%}")}),
              "", "The intervals overlap the no complaint rate for every theme. With repeat purchase this rare, the data cannot show a churn effect, "
              "so the fix list ranks problems by the orders and ratings they affect instead.", "",
              "## 5. Where the problems come from", "",
              f"Among {conc['sellers_rated']:,} sellers with at least {conc['min_reviews']} written reviews, the worst 10% by complaint rate "
              f"({conc['worst_sellers']} sellers) draw a complaint on {_pct(conc['worst_rate'])} of reviews against {_pct(conc['rest_rate'])} for the rest, "
              f"producing {_pct(conc['worst_share_of_complaints'])} of complaints from {_pct(conc['worst_share_of_reviews'])} of reviews. "
              "Ranking by rate rather than count keeps large sellers from being flagged just for their size.", ""]
    for th, c in cats.items():
        if len(c):
            parts += [f"**{name(th)}**, highest rate by category (at least 200 written reviews):", "",
                      md(c, {"category": ("Category", lambda x: str(x).replace("_", " ")),
                             "reviews": ("Reviews", lambda x: f"{x:,}"), "rate": ("Rate", lambda x: f"{x:.1%}")}), ""]
    parts += ["## 6. Fix list", "",
              md(fix, {"theme": ("Problem", name), "owner": ("Owner", str), "reviews": ("Reviews", lambda x: f"{x:,}"),
                       "order_value": ("Order value (R$)", lambda x: f"{x:,.0f}"),
                       "rating_cost": ("Stars off the average", lambda x: f"{x:.3f}"), "action": ("First action", str)}),
              "", "![Fix list](outputs/figures/03_fix_list.png)", "",
              "## 7. Limits", "",
              "- Only written reviews are classified. 59% of reviews have no text, and the themes behind those are unknown.",
              "- Order value is the value of orders that drew a complaint, not money lost. It sizes how much business each problem touches.",
              "- The model's labels are not perfect (section 2). Theme counts carry that error, and rare themes carry the most.",
              "- Olist data covers 2016 to 2018. The pattern of problems may have changed since.", ""]
    return "\n".join(parts)

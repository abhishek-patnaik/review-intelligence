import numpy as np
import pandas as pd

from reviews import size

THEMES = ["a", "b"]


def _t(rows):
    t = pd.DataFrame(rows, columns=["review_score", "a", "b", "seller_id"])
    t["complaint"] = t.a | t.b
    return t


def test_rating_cost_splits_shared_reviews():
    t = _t([(1, True, True, "s"), (5, False, False, "s")])
    rc = size.rating_cost(t, THEMES, base_stars=5.0)
    # one review 4 stars short, shared by two themes, over two reviews
    assert rc["a"] == rc["b"] == 1.0
    assert rc.sum() == 2.0


def test_rating_cost_ignores_reviews_above_baseline():
    t = _t([(5, True, False, "s")])
    assert size.rating_cost(t, THEMES, base_stars=4.0)["a"] == 0


def test_concentration_ranks_by_rate_not_size():
    rows = [(5, False, False, "big")] * 90 + [(1, True, False, "big")] * 10       # 10% rate, big
    rows += [(1, True, False, "small")] * 20 + [(5, False, False, "small")] * 20  # 50% rate, small
    rows += [(5, False, False, f"s{i}") for i in range(8) for _ in range(30)]     # clean sellers
    c = size.concentration(_t(rows), min_reviews=30)
    assert c["worst_sellers"] == 1
    assert c["worst_rate"] == 0.5


def test_wilson_interval_contains_estimate():
    lo, hi = size.wilson(0.05, 400)
    assert lo < 0.05 < hi
    assert np.isnan(size.wilson(0.0, 0)[0])

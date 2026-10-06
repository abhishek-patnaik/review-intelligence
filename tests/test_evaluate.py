import math

from reviews.evaluate import agreement, score

THEMES = ["a", "b", "c"]


def test_perfect_prediction():
    truth = [["a"], ["b", "c"], []]
    table, summary = score(truth, truth, THEMES)
    assert summary["exact_match"] == 1.0
    assert summary["micro_f1"] == 1.0
    assert (table.f1 == 1.0).all()


def test_precision_and_recall_counts():
    truth = [["a"], ["a"], [], ["b"]]
    pred = [["a"], [], ["a"], ["b"]]
    table, summary = score(truth, pred, THEMES)
    a = table.set_index("theme").loc["a"]
    assert a.precision == 0.5 and a.recall == 0.5
    assert summary["exact_match"] == 0.5
    # theme c never appears in truth or prediction: undefined, not zero
    assert math.isnan(table.set_index("theme").loc["c"].f1)


def test_complaint_detection_ignores_which_theme():
    truth = [["a"], []]
    pred = [["b"], []]
    _, summary = score(truth, pred, THEMES)
    assert summary["complaint_detection_accuracy"] == 1.0


def test_agreement_identical_labellers():
    x = [["a"], ["b"], [], ["a", "c"]]
    assert agreement(x, x, THEMES)["exact_match"] == 1.0

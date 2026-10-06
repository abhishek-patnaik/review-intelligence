import pandas as pd

from reviews import labels
from reviews.translate import split_sentences


def _pool():
    rows = []
    for score in range(1, 6):
        for k in range(300):
            rows.append({"review_id": f"r{score}_{k}", "review_score": score, "text": f"texto {k}"})
    return pd.DataFrame(rows)


def test_sample_is_stratified_and_split():
    s = labels.draw_sample(_pool(), per_score=100, dev_share=0.3, seed=42)
    assert len(s) == 500
    assert s.review_score.value_counts().eq(100).all()
    assert (s.split == "dev").sum() == 150
    # dev rows come first, so labelling starts with the set used to tune the prompt
    assert (s.split.iloc[:150] == "dev").all()
    assert s.review_id.is_unique


def test_sample_is_reproducible():
    a = labels.draw_sample(_pool(), 100, 0.3, 42)
    b = labels.draw_sample(_pool(), 100, 0.3, 42)
    assert a.review_id.tolist() == b.review_id.tolist()


def test_save_label_replaces_earlier_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "LABEL_DIR", tmp_path)
    monkeypatch.setattr(labels, "LABEL_FILE", tmp_path / "labels.csv")
    labels.save_label("r1", ["wrong_item"], False, "")
    n = labels.save_label("r1", ["not_as_described", "late_delivery"], True, "fake perfume")
    df = labels.load_labels()
    assert n == 1
    assert df.loc[0, "themes"] == "late_delivery;not_as_described"
    assert df.loc[0, "unsure"] == "1"


def test_split_sentences():
    assert split_sentences("Chegou rápido. Produto bom!  Recomendo") == [
        "Chegou rápido.", "Produto bom!", "Recomendo"]

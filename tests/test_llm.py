from reviews.llm import parse

ALLOWED = {"late_delivery": "", "wrong_item": "", "packaging": ""}


def test_parse_keeps_only_allowed_themes():
    assert parse('{"themes": ["wrong_item", "made_up", "late_delivery"]}', ALLOWED) == [
        "late_delivery", "wrong_item"]


def test_parse_deduplicates():
    assert parse('{"themes": ["packaging", "packaging"]}', ALLOWED) == ["packaging"]


def test_parse_bad_json_means_no_themes():
    assert parse("not json", ALLOWED) == []
    assert parse('["late_delivery"]', ALLOWED) == []

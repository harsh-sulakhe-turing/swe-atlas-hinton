import copy
from autoqc.seed import seed_bad_negative, seed_factual


def _neg(id_, title):
    return {"id": id_, "title": title,
            "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def _pos(id_):
    return {"id": id_, "title": "1.1: States X",
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def test_seed_rewrites_first_negative_and_reports_id():
    items = [_pos("a" * 32), _neg("b" * 32, "2.1: Claims that bytes bodies fail")]
    original = copy.deepcopy(items)
    mutated, changed_id = seed_bad_negative(items)
    assert changed_id == "b" * 32
    assert items == original  # input not mutated
    neg = [it for it in mutated if "negative" in it["annotations"]["type"]][0]
    assert "Does not claim" in neg["title"]
    assert "Claims that" not in neg["title"]


def test_seed_returns_none_when_no_negative():
    items = [_pos("a" * 32)]
    mutated, changed_id = seed_bad_negative(items)
    assert changed_id is None


def test_seed_factual_injects_nonexistent_symbol_into_positive():
    items = [{"id": "1.1", "title": "States the retry count is 3",
              "annotations": {"type": "positive"}},
             {"id": "2.1", "title": "Claims X", "annotations": {"type": "negative"}}]
    original = copy.deepcopy(items)
    mutated, mid = seed_factual(items)
    assert mid == "1.1"
    assert "nonexistent_autoqc_symbol" in mutated[0]["title"]
    assert items[0]["title"] == "States the retry count is 3"  # input untouched
    assert items == original  # input not mutated


def test_seed_factual_no_positive_returns_none():
    mutated, mid = seed_factual([{"id": "2.1", "title": "n", "annotations": {"type": "negative"}}])
    assert mid is None


def test_seed_factual_skips_negative_picks_first_positive():
    items = [{"id": "2.1", "title": "Claims X", "annotations": {"type": "negative"}},
             {"id": "1.1", "title": "States retry count is 3", "annotations": {"type": "positive"}}]
    mutated, mid = seed_factual(items)
    assert mid == "1.1"
    assert "nonexistent_autoqc_symbol" in mutated[1]["title"]
    assert mutated[0]["title"] == "Claims X"  # negative untouched

import copy
from autoqc.seed import seed_wildcard


def _pos(id_, title="1.1: States the port"):
    return {"id": id_, "title": title,
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(id_):
    return {"id": id_, "title": "2.1: Claims that X",
            "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_seed_wildcard_appends_hatch_to_first_positive():
    items = [_neg("n"), _pos("p", "1.1: States the port")]
    original = copy.deepcopy(items)
    mutated, changed = seed_wildcard(items)
    assert changed == "p"
    assert items == original  # input untouched
    pos = [it for it in mutated if "positive" in it["annotations"]["type"]][0]
    assert "or similar" in pos["title"]


def test_seed_wildcard_none_when_no_positive():
    mutated, changed = seed_wildcard([_neg("n")])
    assert changed is None

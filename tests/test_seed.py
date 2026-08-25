import copy
from autoqc.seed import seed_bad_negative
from autoqc.semantic.checks import NegativeSemanticsCheck
from autoqc.semantic.engine import run_check
from autoqc.llm import FakeLLMClient
import types


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


def test_seeded_mutant_trips_q07_via_engine():
    items = [_neg("b" * 32, "2.1: Claims that bytes bodies fail")]
    mutated, _ = seed_bad_negative(items)

    def responder(messages):
        text = " ".join(m["content"] for m in messages)
        return {"passed": ("Does not claim" not in text), "evidence": []}

    r = run_check(NegativeSemanticsCheck(), types.SimpleNamespace(rubrics=mutated),
                  FakeLLMClient(responder), k=3)
    assert r.passed is False  # the seeded defect is caught

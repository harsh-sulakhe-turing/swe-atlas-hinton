import types
from autoqc.semantic.checks import NegativeSemanticsCheck, SEMANTIC_CHECKS
from autoqc.semantic.engine import run_check
from autoqc.llm import FakeLLMClient
from autoqc.model import Severity


def _bundle(items):
    return types.SimpleNamespace(rubrics=items)


def _pos(idx, id_):
    return {"id": id_, "title": f"{idx}: States X",
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(idx, id_, title):
    return {"id": id_, "title": title,
            "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_units_are_only_negatives():
    c = NegativeSemanticsCheck()
    b = _bundle([_pos("1.1", "a" * 32),
                 _neg("2.1", "b" * 32, "2.1: Claims that X is false")])
    units = c.units(b)
    assert len(units) == 1 and units[0].key == "b" * 32


def test_registry_contains_q07():
    assert any(c.id == "Q07" for c in SEMANTIC_CHECKS)
    assert NegativeSemanticsCheck().severity is Severity.REJECT


def test_parse_reads_passed_and_evidence():
    j = NegativeSemanticsCheck().parse({"passed": True, "evidence": ["ok"], "reason": "r"})
    assert j.passed is True and j.evidence == ["ok"] and j.reason == "r"


def test_prompts_include_the_title():
    c = NegativeSemanticsCheck()
    u = c.units(_bundle([_neg("2.1", "b" * 32, "2.1: Does not claim bytes fail")]))[0]
    pm = c.proposer_messages(None, u)
    am = c.adversary_messages(None, u, True)
    assert any("Does not claim bytes fail" in m["content"] for m in pm)
    assert any("Does not claim bytes fail" in m["content"] for m in am)
    # adversary stance flips on agg_passed
    am_pass = " ".join(m["content"] for m in c.adversary_messages(None, u, True))
    am_fail = " ".join(m["content"] for m in c.adversary_messages(None, u, False))
    assert am_pass != am_fail


def test_end_to_end_flags_bad_negative_with_fake_llm():
    # A well-phrased negative -> proposers pass; a "Does not claim" one -> proposers fail.
    def responder(messages):
        text = " ".join(m["content"] for m in messages)
        bad = "Does not claim" in text
        # proposer: passed=not bad ; adversary agrees (no overturn)
        return {"passed": (not bad), "evidence": []}
    good = _bundle([_neg("2.1", "b" * 32, "2.1: Claims that bytes bodies fail")])
    bad = _bundle([_neg("2.1", "c" * 32, "2.1: Does not claim that bytes bodies fail")])
    r_good = run_check(NegativeSemanticsCheck(), good, FakeLLMClient(responder), k=3)
    r_bad = run_check(NegativeSemanticsCheck(), bad, FakeLLMClient(responder), k=3)
    assert r_good.passed is True
    assert r_bad.passed is False and "c" * 32 in r_bad.detail

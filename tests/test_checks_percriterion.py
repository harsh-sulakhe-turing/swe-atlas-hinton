from autoqc.model import Severity
from autoqc.agent.checks import Q01, Q02, Q05, SEMANTIC_CHECKS, _positives


def _pos(i):
    return {"id": i, "title": "1.1: X", "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(i):
    return {"id": i, "title": "2.1: Claims X", "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_ids_and_severities():
    assert (Q01.id, Q01.severity) == ("Q01", Severity.REJECT)
    assert (Q02.id, Q02.severity) == ("Q02", Severity.REJECT)
    assert (Q05.id, Q05.severity) == ("Q05", Severity.REJECT)
    assert all(c in SEMANTIC_CHECKS for c in (Q01, Q02, Q05))


def test_q01_q02_scope_all_criteria():
    items = [_pos("a"), _neg("b")]
    assert {c["id"] for c in Q01.scope(items)} == {"a", "b"}
    assert {c["id"] for c in Q02.scope(items)} == {"a", "b"}


def test_q05_scope_positives_only():
    assert [c["id"] for c in Q05.scope([_pos("a"), _neg("b")])] == ["a"]
    assert [c["id"] for c in _positives([_pos("a"), _neg("b")])] == ["a"]


def test_guidance_is_specific():
    assert "independently" in Q01.guidance.lower()          # atomicity
    assert "thorough" in Q02.guidance.lower() or "subjective" in Q02.guidance.lower()  # binary
    assert "prompt" in Q05.guidance.lower()                 # unrequested scope refs the prompt

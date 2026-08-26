import types
from autoqc.model import Stage, Severity
from autoqc.agent.deterministic import check_q09, check_q12_size, DETERMINISTIC_CHECKS


def _b(items):
    return types.SimpleNamespace(rubrics=items, prompt="q")


def _pos(i):
    return {"id": i, "title": "1.1: X", "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(i):
    return {"id": i, "title": "2.1: Claims X", "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_q09_passes_with_negative():
    r = check_q09(_b([_pos("a"), _neg("b")]))
    assert r.id == "Q09" and r.stage is Stage.SEMANTIC and r.severity is Severity.WARN
    assert r.passed is True


def test_q09_fails_without_negative():
    r = check_q09(_b([_pos("a")]))
    assert r.passed is False and "negative" in r.detail.lower()


def test_q09_none_safe():
    assert check_q09(types.SimpleNamespace(rubrics=None, prompt="q")).passed is False


def test_q12_passes_at_or_below_limit():
    r = check_q12_size(_b([_pos(str(i)) for i in range(18)]))
    assert r.id == "Q12" and r.passed is True


def test_q12_fails_above_limit():
    r = check_q12_size(_b([_pos(str(i)) for i in range(19)]))
    assert r.passed is False and r.severity is Severity.WARN and "19" in r.detail


def test_deterministic_registry():
    assert DETERMINISTIC_CHECKS == [check_q09, check_q12_size]

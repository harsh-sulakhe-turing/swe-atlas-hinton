from pathlib import Path
from autoqc.agent.engine import adjudicate_factual, run_factual
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient
from autoqc.model import Stage, Severity


def _f(cid, passed, ev):
    return {"check_id": "Q06", "criterion_id": cid, "passed": passed, "evidence": ev}


def test_adjudicate_agree_pass():
    a = adjudicate_factual([_f("1", True, ["a.go:1"])], [_f("1", True, ["a.go:9"])], {"1"})
    assert a["1"] == {"passed": True, "needs_human": False}


def test_adjudicate_agree_reject_same_file_is_hard_reject():
    a = adjudicate_factual([_f("1", False, ["a.go:1"])], [_f("1", False, ["a.go:40"])], {"1"})
    assert a["1"] == {"passed": False, "needs_human": False}


def test_adjudicate_reject_different_files_needs_human():
    a = adjudicate_factual([_f("1", False, ["a.go:1"])], [_f("1", False, ["b.go:1"])], {"1"})
    assert a["1"]["needs_human"] is True


def test_adjudicate_disagreement_needs_human():
    a = adjudicate_factual([_f("1", True, ["a.go:1"])], [_f("1", False, ["a.go:1"])], {"1"})
    assert a["1"]["needs_human"] is True


def test_adjudicate_missing_round_needs_human():
    a = adjudicate_factual([_f("1", True, ["a.go:1"])], [], {"1"})
    assert a["1"]["needs_human"] is True


def _bundle_with(criteria):
    class B:
        prompt = "p"; base_commit = "abc"; repository = "r"
        rubrics = criteria
    return B()


def _responder_all(passed, ev):
    def r(messages, tools):
        # single-turn submit for every criterion mentioned in the user context
        import re
        user = next(m["content"] for m in messages if m["role"] == "user")
        ids = re.findall(r"criterion_id=(\S+)", user)
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {
            "findings": [_f(i, passed, ev) for i in ids]}}]}
    return r


def test_run_factual_confirmed_reject(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    res = run_factual(b, FakeLLMClient(_responder_all(False, ["x.go:3"])), ctx)
    assert res.id == "Q06" and res.stage is Stage.FACTUAL and res.severity is Severity.REJECT
    assert res.passed is False and res.needs_human is False  # -> not_sound


def test_run_factual_all_pass(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    res = run_factual(b, FakeLLMClient(_responder_all(True, ["x.go:3"])), ctx)
    assert res.passed is True and res.needs_human is False


def test_run_factual_no_criteria_passes(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    res = run_factual(_bundle_with([]), FakeLLMClient(_responder_all(True, ["x:1"])), ctx)
    assert res.passed is True and res.needs_human is False

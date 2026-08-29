import types
from autoqc.model import Severity
from autoqc.agent import checks as C
from autoqc.agent.engine import run_check
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient

def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)

def _pos(i, title="1.1: States X"):
    return {"id": i * 32, "title": title,
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}

def test_al01_and_q13_registered():
    ids = {c.id: c for c in C.SEMANTIC_CHECKS}
    assert ids["AL01"].severity is Severity.WARN and ids["AL01"].unit_mode == "bundle"
    assert ids["Q13"].severity is Severity.REJECT and ids["Q13"].unit_mode == "rubric"

def test_al01_fail_mismatch(tmp_path):
    AL01 = {c.id: c for c in C.SEMANTIC_CHECKS}["AL01"]
    b = types.SimpleNamespace(rubrics=[], prompt="about aiohttp pooling",
                              answer="about JAX autodiff", instruction="about aiohttp pooling")
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "AL01", "criterion_id": "bundle", "passed": False,
             "evidence": ["answer is about a different task"]}]}}]}
    r = run_check(AL01, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.passed is False

def test_q13_reject_no_exploration_criterion(tmp_path):
    Q13 = {c.id: c for c in C.SEMANTIC_CHECKS}["Q13"]
    b = types.SimpleNamespace(rubrics=[_pos("a")], prompt="How does X work?")
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "Q13", "criterion_id": "rubric", "passed": False,
             "evidence": ["no criterion verifies codebase exploration"]}]}}]}
    r = run_check(Q13, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.passed is False

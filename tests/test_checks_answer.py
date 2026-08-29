import types
from autoqc.model import Severity
from autoqc.agent import checks as C
from autoqc.agent.engine import run_check
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient

def _b(answer):
    return types.SimpleNamespace(rubrics=[], prompt="How does X work?", answer=answer, instruction="i")

def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)

def _fake(check_id, passed, ev):
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": check_id, "criterion_id": "answer", "passed": passed, "evidence": [ev]}]}}]}
    return FakeLLMClient(responder)

def test_answer_checks_registered_with_severities():
    ids = {c.id: c for c in C.SEMANTIC_CHECKS}
    assert ids["A01"].severity is Severity.WARN
    assert ids["A04"].severity is Severity.REJECT
    for cid in ("A01", "A02", "A03", "A04", "A05"):
        assert ids[cid].unit_mode == "answer" and ids[cid].role_kind == "prose"

def test_a04_reject_no_evidence_shown(tmp_path):
    A04 = {c.id: c for c in C.SEMANTIC_CHECKS}["A04"]
    r = run_check(A04, _b("I ran the repro and it reused the socket."),
                  _fake("A04", False, "claims a run but shows no command/output"), _ctx(tmp_path), k=3)
    assert r.passed is False

def test_a02_pass_continuous_narrative(tmp_path):
    A02 = {c.id: c for c in C.SEMANTIC_CHECKS}["A02"]
    r = run_check(A02, _b("I started by ... then I traced ... which showed ..."),
                  _fake("A02", True, "continuous narrative, no headers"), _ctx(tmp_path), k=3)
    assert r.passed is True

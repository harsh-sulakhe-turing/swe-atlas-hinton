import types
from autoqc.model import Severity
from autoqc.agent import checks as C
from autoqc.agent.engine import run_check
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient

def _b(prompt):
    return types.SimpleNamespace(rubrics=[], prompt=prompt, answer="a", instruction="i")

def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)

def _fake(check_id, passed, ev):
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": check_id, "criterion_id": "prompt", "passed": passed, "evidence": [ev]}]}}]}
    return FakeLLMClient(responder)

def test_p02_and_p03_registered_as_reject():
    ids = {c.id: c for c in C.SEMANTIC_CHECKS}
    assert ids["P02"].severity is Severity.REJECT and ids["P02"].unit_mode == "prompt"
    assert ids["P03"].severity is Severity.REJECT and ids["P03"].role_kind == "prose"

def test_p02_fail_multi_goal(tmp_path):
    P02 = {c.id: c for c in C.SEMANTIC_CHECKS}["P02"]
    r = run_check(P02, _b("Do A. Also refactor B. Also benchmark C."),
                  _fake("P02", False, "three independent goals"), _ctx(tmp_path), k=3)
    assert r.passed is False

def test_p03_pass_natural(tmp_path):
    P03 = {c.id: c for c in C.SEMANTIC_CHECKS}["P03"]
    r = run_check(P03, _b("Our aiohttp pool keeps reopening sockets on small file bodies; why?"),
                  _fake("P03", True, "reads as a real developer question"), _ctx(tmp_path), k=3)
    assert r.passed is True

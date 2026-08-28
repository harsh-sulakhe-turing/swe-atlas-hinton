import types
from autoqc.model import Severity
from autoqc.agent.checks import (SemanticCheck, _answer_unit, proposer_context,
                                 prose_proposer_role)
from autoqc.agent.engine import run_check
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient

ANSWER_CHECK = SemanticCheck(id="A01", name="Investigation-first", severity=Severity.WARN,
                             scope=_answer_unit, guidance="opens by investigating.",
                             unit_mode="answer", role_kind="prose")

def _b(answer):
    return types.SimpleNamespace(rubrics=[], prompt="How does X work?",
                                 answer=answer, instruction="i")

def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)

def test_prose_role_is_document_oriented():
    assert "rubric" not in prose_proposer_role().system_prompt.lower()

def test_answer_context_injects_answer_text():
    b = _b("I started by exploring the repo, then traced the close.")
    pc = proposer_context(b, ANSWER_CHECK, _answer_unit(b.rubrics))
    assert "I started by exploring" in pc and 'criterion_id="answer"' in pc

def test_run_check_answer_mode_pass(tmp_path):
    b = _b("I started by exploring the repo ...")
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "A01", "criterion_id": "answer", "passed": True, "evidence": ["investigates first"]}]}}]}
    r = run_check(ANSWER_CHECK, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.id == "A01" and r.passed is True and r.needs_human is False

def test_run_check_answer_mode_fail(tmp_path):
    b = _b("The answer is: conn.close() in the cancel handler.")
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "A01", "criterion_id": "answer", "passed": False, "evidence": ["conclusion-first lede"]}]}}]}
    r = run_check(ANSWER_CHECK, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.passed is False and "answer" in r.detail

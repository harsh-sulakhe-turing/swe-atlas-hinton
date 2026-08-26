import types
from pathlib import Path
import pytest
from autoqc.model import Severity, Stage
from autoqc.agent.checks import SemanticCheck, proposer_context, adversary_context, _full_rubric_block
from autoqc.agent.engine import run_check
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient


def _all(items):
    return [it for it in items if isinstance(it, dict)]


RUBRIC_CHECK = SemanticCheck(id="Q04", name="Prompt coverage", severity=Severity.REJECT,
                             scope=_all, guidance="every obligation must be covered.",
                             unit_mode="rubric")


def _bundle():
    items = [{"id": "a" * 32, "title": "1.1: States X",
              "annotations": {"type": "positive hli verifier", "importance": "must have"}}]
    return types.SimpleNamespace(rubrics=items, prompt="How does X work?")


def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)


def test_rubric_context_asks_for_single_rubric_finding():
    b = _bundle()
    pc = proposer_context(b, RUBRIC_CHECK, RUBRIC_CHECK.scope(b.rubrics))
    assert "rubric" in pc.lower() and "How does X work?" in pc  # whole-rubric + prompt shown


def test_run_check_rubric_mode_pass(tmp_path):
    b = _bundle()
    def responder(messages, tools):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "Q04", "criterion_id": "rubric", "passed": True, "evidence": ["all covered"]}]}}]}
    r = run_check(RUBRIC_CHECK, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.id == "Q04" and r.passed is True and r.needs_human is False


def test_run_check_rubric_mode_fail(tmp_path):
    b = _bundle()
    def responder(messages, tools):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "Q04", "criterion_id": "rubric", "passed": False, "evidence": ["obligation Y uncovered"]}]}}]}
    r = run_check(RUBRIC_CHECK, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.passed is False and "rubric" in r.detail


@pytest.mark.parametrize("bad_ann", [None, [], "negative", 42])
def test_full_rubric_block_never_raises_on_nondict_annotations(bad_ann):
    # never-raise: a criterion with annotations that isn't a dict must not crash
    # the whole-rubric context builder (the per-rubric checks Q04/Q08/Q10/Q11 path).
    items = [{"id": "a" * 32, "title": "1.1: States X", "annotations": bad_ann}]
    b = types.SimpleNamespace(rubrics=items, prompt="How does X work?")
    block = _full_rubric_block(b)  # must not raise
    assert "criterion_id=" in block
    # a non-dict annotation can never be read as negative -> defaults to pos
    assert "[pos]" in block


def test_run_check_rubric_mode_empty_scope_passes(tmp_path):
    empty_neg_check = SemanticCheck(id="Q08", name="disc neg", severity=Severity.WARN,
                                    scope=lambda items: [], guidance="g", unit_mode="rubric")
    r = run_check(empty_neg_check, _bundle(), FakeLLMClient(lambda m, t: {"tool_calls": []}), _ctx(tmp_path), k=3)
    assert r.passed is True and r.needs_human is False

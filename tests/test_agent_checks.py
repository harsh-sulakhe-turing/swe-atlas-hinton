import types
from autoqc.model import Severity
from autoqc.agent.checks import (SemanticCheck, Q07, Q03, Q01, Q02, Q05, SEMANTIC_CHECKS,
                                 proposer_role, adversary_role,
                                 proposer_context, adversary_context)


def _bundle(items, prompt="the question"):
    return types.SimpleNamespace(rubrics=items, prompt=prompt)


def _pos(id_, title="1.1: States X"):
    return {"id": id_, "title": title,
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(id_, title="2.1: Claims that X"):
    return {"id": id_, "title": title,
            "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_registry_and_severities():
    assert [c.id for c in SEMANTIC_CHECKS] == ["Q07", "Q03", "Q01", "Q02", "Q05"]
    assert all(c.severity is Severity.REJECT for c in [Q07, Q03, Q01, Q02, Q05])


def test_q07_scope_is_negatives_only():
    items = [_pos("a"), _neg("b")]
    got = Q07.scope(items)
    assert [c["id"] for c in got] == ["b"]


def test_q03_scope_is_all_criteria():
    items = [_pos("a"), _neg("b")]
    got = Q03.scope(items)
    assert {c["id"] for c in got} == {"a", "b"}


def test_roles_are_distinct():
    p, a = proposer_role(), adversary_role()
    assert "adversarial" in a.system_prompt.lower()
    assert p.system_prompt != a.system_prompt
    assert any(t.name == "submit_findings" for t in p.tools)


def test_proposer_context_lists_criteria_and_guidance():
    b = _bundle([_neg("b", "2.1: Does not claim X")])
    ctx = proposer_context(b, Q07, Q07.scope(b.rubrics))
    assert "b" in ctx and "Does not claim X" in ctx
    assert "Q07" in ctx


def test_q03_guidance_has_eg_guardrail():
    # the interchangeable e.g./such as guardrail must be in the prompt guidance
    assert "such as" in Q03.guidance.lower() or "e.g." in Q03.guidance.lower()
    assert "interchangeable" in Q03.guidance.lower()


def test_adversary_context_states_aggregate():
    b = _bundle([_neg("b")])
    agg = {"b": {"passed": False, "split": False, "evidence": []}}
    ctx = adversary_context(b, Q07, Q07.scope(b.rubrics), agg)
    assert "b" in ctx and ("fail" in ctx.lower() or "reject" in ctx.lower())

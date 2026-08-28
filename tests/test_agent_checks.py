import types
from autoqc.model import Severity
from autoqc.agent.checks import (SemanticCheck, Q07, Q03, Q01, Q02, Q05, Q04, Q08, Q10, Q11, P02, P03, A01, A02, A03, A04, A05, SEMANTIC_CHECKS,
                                 proposer_role, adversary_role,
                                 proposer_context, adversary_context,
                                 Q06, factual_role, factual_context)


def _bundle(items, prompt="the question"):
    return types.SimpleNamespace(rubrics=items, prompt=prompt)


def _pos(id_, title="1.1: States X"):
    return {"id": id_, "title": title,
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(id_, title="2.1: Claims that X"):
    return {"id": id_, "title": title,
            "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_registry_and_severities():
    assert set(c.id for c in SEMANTIC_CHECKS) == {"Q01", "Q02", "Q03", "Q04", "Q05", "Q07", "Q08", "Q10", "Q11", "P02", "P03", "A01", "A02", "A03", "A04", "A05"}
    assert all(c.severity is Severity.REJECT for c in [Q07, Q03, Q01, Q02, Q05])
    assert all(c.severity is Severity.REJECT for c in [Q04])
    assert all(c.severity is Severity.REJECT for c in [P02, P03])
    assert all(c.severity is Severity.WARN for c in [Q08, Q10, Q11])
    assert all(c.severity is Severity.REJECT for c in [A04])
    assert all(c.severity is Severity.WARN for c in [A01, A02, A03, A05])


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


def test_q06_is_reject_percriterion():
    assert Q06.id == "Q06" and Q06.severity is Severity.REJECT
    assert Q06.unit_mode == "criterion"


def test_q06_not_in_semantic_checks():
    from autoqc.agent.checks import SEMANTIC_CHECKS
    assert Q06 not in SEMANTIC_CHECKS


def test_factual_role_has_run_bash():
    names = {t.name for t in factual_role().tools}
    assert "run_bash" in names and "submit_findings" in names


def test_factual_context_mentions_base_commit_and_criteria():
    class B:
        prompt = "Explain the retry logic."
        base_commit = "eea1d62f0438f75075d9feb2c022a86083e618b2"
        repository = "cosi-project/runtime"
    crit = [{"id": "1.1", "title": "States the default retry count is 3"}]
    ctx = factual_context(B(), crit)
    assert "eea1d62f" in ctx and "1.1" in ctx and "/testbed" in ctx
    assert "cosi-project/runtime" in ctx


def test_text_roles_have_no_read_tools():
    for role in (proposer_role(), adversary_role()):
        names = {t.name for t in role.tools}
        assert names == {"submit_findings"}, names


def test_factual_role_still_has_run_bash():
    names = {t.name for t in factual_role().tools}
    assert "run_bash" in names and "submit_findings" in names

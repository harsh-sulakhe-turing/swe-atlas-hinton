import types
from pathlib import Path
from autoqc.model import Stage, Severity
from autoqc.agent.engine import run_check, run_semantic
from autoqc.agent.engine import (proposer_pass, adversary_pass, finalize_check,
                                 _check_prep, TEXT_MAX_TURNS, aggregate)
from autoqc.agent.checks import Q07
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient


def _bundle(neg_ids, prompt="q"):
    items = [{"id": i, "title": f"2.{n+1}: Claims that X",
              "annotations": {"type": "negative hli verifier", "importance": "must have"}}
             for n, i in enumerate(neg_ids)]
    return types.SimpleNamespace(rubrics=items, prompt=prompt)


def _ctx(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)


def _is_adversary(messages):
    return "adversarial" in messages[0]["content"].lower()


def _submit(findings):
    return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": findings}}]}


def _f(cid, passed):
    return {"check_id": "Q07", "criterion_id": cid, "passed": passed, "evidence": ["ev"]}


def test_run_check_unanimous_pass(tmp_path):
    b = _bundle(["n1"])
    def responder(messages, tools):
        return _submit([_f("n1", True)])  # proposers + adversary all pass
    r = run_check(Q07, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.id == "Q07" and r.stage is Stage.SEMANTIC and r.severity is Severity.REJECT
    assert r.passed is True and r.needs_human is False


def test_run_check_unanimous_reject(tmp_path):
    b = _bundle(["n1"])
    def responder(messages, tools):
        return _submit([_f("n1", False)])
    r = run_check(Q07, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.passed is False and r.needs_human is False and "n1" in r.detail


def test_run_check_adversary_overturn_needs_human(tmp_path):
    b = _bundle(["n1"])
    def responder(messages, tools):
        if _is_adversary(messages):
            return _submit([_f("n1", True)])   # adversary defends the reject
        return _submit([_f("n1", False)])       # proposers reject
    r = run_check(Q07, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.passed is False and r.needs_human is True  # disputed reject


def test_run_check_split_needs_human(tmp_path):
    b = _bundle(["n1"])
    calls = {"n": 0}
    def responder(messages, tools):
        if _is_adversary(messages):
            return _submit([_f("n1", True)])
        calls["n"] += 1
        return _submit([_f("n1", calls["n"] % 2 == 0)])  # T/F/T across 3 proposer passes
    r = run_check(Q07, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.needs_human is True


def test_run_check_empty_scope_passes(tmp_path):
    b = types.SimpleNamespace(rubrics=[{"id": "p", "title": "1.1: X",
        "annotations": {"type": "positive hli verifier", "importance": "must have"}}], prompt="q")
    r = run_check(Q07, b, FakeLLMClient(lambda m, t: _submit([])), _ctx(tmp_path), k=3)
    assert r.passed is True and r.needs_human is False  # no negatives -> nothing to judge


def test_failed_pass_causes_needs_human(tmp_path):
    b = _bundle(["n1"])
    class Boom(FakeLLMClient):
        def chat(self, messages, tools=None):
            raise RuntimeError("gateway down")
    r = run_check(Q07, b, Boom(lambda m, t: {}), _ctx(tmp_path), k=3)
    assert r.needs_human is True  # all passes failed -> abstain -> split

def test_votes_log_records_passes(tmp_path):
    b = _bundle(["n1"])
    log = []
    run_check(Q07, b, FakeLLMClient(lambda m, t: _submit([_f("n1", True)])), _ctx(tmp_path), k=3, votes_log=log)
    assert len([e for e in log if e["role"] == "proposer"]) == 3
    assert len([e for e in log if e["role"] == "adversary"]) == 1


def test_run_semantic_returns_one_result_per_check(tmp_path):
    b = _bundle(["n1"])
    results = run_semantic(b, FakeLLMClient(lambda m, t: _submit([])), _ctx(tmp_path), k=1, factual=False)
    ids = {r.id for r in results}
    assert {"Q07", "Q03", "Q09", "Q12"} <= ids


def test_text_max_turns_is_low():
    assert TEXT_MAX_TURNS == 3


def test_proposer_pass_returns_own_and_log(tmp_path):
    b = _bundle(["n1"])
    criteria, allowed = _check_prep(Q07, b)
    own, log = proposer_pass(Q07, b, FakeLLMClient(lambda m, t: _submit([_f("n1", False)])),
                             _ctx(tmp_path), criteria, allowed)
    assert log == {"check": "Q07", "role": "proposer", "ok": True,
                   "findings": [_f("n1", False)]}
    assert own == [_f("n1", False)]


def test_proposer_pass_never_submits_is_bounded_and_not_ok(tmp_path):
    # responder returns text, never calls submit -> bounded by TEXT_MAX_TURNS -> ok False
    b = _bundle(["n1"])
    criteria, allowed = _check_prep(Q07, b)
    own, log = proposer_pass(Q07, b, FakeLLMClient(lambda m, t: {"text": "hmm"}),
                             _ctx(tmp_path), criteria, allowed)
    assert log["ok"] is False and own == []


def test_finalize_check_matches_serial_reject(tmp_path):
    b = _bundle(["n1"])
    criteria, allowed = _check_prep(Q07, b)
    agg = aggregate([[_f("n1", False)], [_f("n1", False)], [_f("n1", False)]], allowed)
    r = finalize_check(Q07, agg, [_f("n1", False)])  # adversary agrees reject
    assert r.passed is False and r.needs_human is False and "n1" in r.detail


def test_run_semantic_includes_deterministic_checks(tmp_path):
    import types as _t
    from autoqc.agent.engine import run_semantic
    from autoqc.agent.tools import AgentContext
    from autoqc.llm import FakeLLMClient
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    b = _t.SimpleNamespace(rubrics=[{"id": "p", "title": "1.1: X",
        "annotations": {"type": "positive hli verifier", "importance": "must have"}}], prompt="q")
    results = run_semantic(b, FakeLLMClient(lambda m, t: {"tool_calls": [
        {"id": "s", "name": "submit_findings", "args": {"findings": []}}]}),
        AgentContext(bundle_dir=tmp_path), k=1, factual=False)
    ids = {r.id for r in results}
    assert {"Q09", "Q12"} <= ids  # deterministic checks present

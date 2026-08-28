import re
import types
from pathlib import Path
from autoqc.agent import engine as engine_module
from autoqc.agent.engine import run_check, run_checks_parallel, run_semantic
from autoqc.agent.checks import SEMANTIC_CHECKS
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient


def _bundle():
    # criteria that give several checks a non-empty scope (positives + negatives)
    items = [
        {"id": "a" * 32, "title": "1.1: States X",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}},
        {"id": "b" * 32, "title": "2.1: Claims that X is false",
         "annotations": {"type": "negative hli verifier", "importance": "must have"}},
    ]
    return types.SimpleNamespace(rubrics=items, prompt="explain X")


def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(exist_ok=True); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)


def _submit_all_pass(messages, tools):
    # submit passed=True for every criterion_id mentioned in the user context
    import re
    user = next(m["content"] for m in messages if m["role"] == "user")
    cm = re.search(r"Check (Q\d\d|P\d\d)", user)
    check_id = cm.group(1) if cm else "Q07"
    if 'criterion_id="rubric"' in user:
        ids = ["rubric"]
    elif 'criterion_id="prompt"' in user:
        ids = ["prompt"]
    elif 'criterion_id="answer"' in user:
        ids = ["answer"]
    elif 'criterion_id="bundle"' in user:
        ids = ["bundle"]
    else:
        ids = list(dict.fromkeys(re.findall(r"criterion_id=([0-9a-fA-F]{6,})", user))) or ["rubric"]
    return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
        {"check_id": check_id, "criterion_id": i, "passed": True, "evidence": ["ok"]} for i in ids]}}]}


def _verdicts(results):
    return {r.id: (r.passed, r.needs_human) for r in results}


def test_parallel_equals_serial(tmp_path):
    b = _bundle()
    client = FakeLLMClient(_submit_all_pass)
    serial = [run_check(c, b, client, _ctx(tmp_path), k=3) for c in SEMANTIC_CHECKS]
    parallel = run_checks_parallel(SEMANTIC_CHECKS, b, client, _ctx(tmp_path), k=3)
    assert [r.id for r in parallel] == [c.id for c in SEMANTIC_CHECKS]  # check order preserved
    assert _verdicts(parallel) == _verdicts(serial)


def test_parallel_is_stable_across_runs(tmp_path):
    b = _bundle()
    client = FakeLLMClient(_submit_all_pass)
    runs = [_verdicts(run_checks_parallel(SEMANTIC_CHECKS, b, client, _ctx(tmp_path), k=3))
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_parallel_width_one_is_correct(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOQC_ENGINE_WORKERS", "1")
    b = _bundle()
    client = FakeLLMClient(_submit_all_pass)
    serial = [run_check(c, b, client, _ctx(tmp_path), k=3) for c in SEMANTIC_CHECKS]
    parallel = run_checks_parallel(SEMANTIC_CHECKS, b, client, _ctx(tmp_path), k=3)
    assert _verdicts(parallel) == _verdicts(serial)


def test_parallel_votes_log_complete_for_scoped_check(tmp_path):
    b = _bundle()
    log = []
    run_checks_parallel(SEMANTIC_CHECKS, b, FakeLLMClient(_submit_all_pass),
                        _ctx(tmp_path), k=3, votes_log=log)
    q03 = [e for e in log if e["check"] == "Q03"]  # Q03 scopes all criteria -> non-empty
    assert len([e for e in q03 if e["role"] == "proposer"]) == 3
    assert len([e for e in q03 if e["role"] == "adversary"]) == 1


def test_parallel_k_zero_equals_serial(tmp_path):
    b = _bundle()
    client = FakeLLMClient(_submit_all_pass)
    serial = [run_check(c, b, client, _ctx(tmp_path), k=0) for c in SEMANTIC_CHECKS]
    parallel = run_checks_parallel(SEMANTIC_CHECKS, b, client, _ctx(tmp_path), k=0)
    assert [r.id for r in parallel] == [c.id for c in SEMANTIC_CHECKS]  # no KeyError, order preserved
    assert _verdicts(parallel) == _verdicts(serial)


def test_parallel_isolates_one_failing_check(tmp_path, monkeypatch):
    # Raising inside FakeLLMClient.chat would be swallowed by run_agent (it
    # returns AgentResult(ok=False)), which never reaches the scheduler's
    # per-future except -- so isolation is exercised by making proposer_pass
    # itself raise for one check, driving the scheduler's own except path.
    b = _bundle()
    real_proposer_pass = engine_module.proposer_pass

    def flaky_proposer_pass(check, bundle, client, ctx, criteria, allowed):
        if check.id == "Q03":
            raise RuntimeError("boom on Q03 only")
        return real_proposer_pass(check, bundle, client, ctx, criteria, allowed)

    monkeypatch.setattr(engine_module, "proposer_pass", flaky_proposer_pass)
    results = run_checks_parallel(SEMANTIC_CHECKS, b, FakeLLMClient(_submit_all_pass),
                                  _ctx(tmp_path), k=3)
    v = _verdicts(results)
    assert v["Q03"][1] is True          # Q03's proposers all raised -> needs_human
    assert len(results) == len(SEMANTIC_CHECKS)  # run completed, nothing sunk
    for c in SEMANTIC_CHECKS:
        if c.id != "Q03":
            assert v[c.id][0] is True   # other checks still pass normally


def test_run_semantic_parallel_still_returns_all_checks(tmp_path):
    b = _bundle()
    results = run_semantic(b, FakeLLMClient(_submit_all_pass), _ctx(tmp_path), k=1, factual=False)
    ids = {r.id for r in results}
    assert {"Q07", "Q03", "Q09", "Q12"} <= ids


def test_engine_workers_bad_env_falls_back(monkeypatch):
    from autoqc.agent.engine import _engine_workers
    monkeypatch.setenv("AUTOQC_ENGINE_WORKERS", "not-a-number")
    assert _engine_workers() == 8
    monkeypatch.setenv("AUTOQC_ENGINE_WORKERS", "4")
    assert _engine_workers() == 4


def test_run_checks_parallel_survives_bad_workers_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOQC_ENGINE_WORKERS", "garbage")
    b = _bundle()
    results = run_checks_parallel(SEMANTIC_CHECKS, b, FakeLLMClient(_submit_all_pass),
                                  _ctx(tmp_path), k=3)
    assert len(results) == len(SEMANTIC_CHECKS)  # no crash despite unparseable env value


def _is_adversary(messages):
    return "adversarial" in messages[0]["content"].lower()


_Q07_REJECT_ID = "b" * 32  # negative criterion from _bundle(), scoped by Q07


def _submit_mixed(messages, tools):
    # Everything passes, EXCEPT Q07's negative criterion: proposers reject it,
    # and the adversary defends it (passed=True) -> overturn -> needs_human,
    # while the aggregate verdict (reject) stands. This makes the compared
    # verdict maps non-uniform (a reject + a needs_human), unlike the
    # all-pass responder used elsewhere in this file.
    user = next(m["content"] for m in messages if m["role"] == "user")
    cm = re.search(r"Check (Q\d\d)", user)
    check_id = cm.group(1) if cm else "Q07"
    if 'criterion_id="rubric"' in user:
        ids = ["rubric"]
    else:
        ids = list(dict.fromkeys(re.findall(r"criterion_id=([0-9a-fA-F]{6,})", user))) or ["rubric"]
    is_adv = _is_adversary(messages)
    findings = []
    for i in ids:
        if check_id == "Q07" and i == _Q07_REJECT_ID:
            passed = True if is_adv else False
        else:
            passed = True
        findings.append({"check_id": check_id, "criterion_id": i, "passed": passed, "evidence": ["ok"]})
    return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": findings}}]}


def test_parallel_equals_serial_mixed_verdicts(tmp_path):
    b = _bundle()
    client = FakeLLMClient(_submit_mixed)  # shared: responder is a pure function of messages
    serial = [run_check(c, b, client, _ctx(tmp_path), k=3) for c in SEMANTIC_CHECKS]
    parallel = run_checks_parallel(SEMANTIC_CHECKS, b, client, _ctx(tmp_path), k=3)
    assert _verdicts(parallel) == _verdicts(serial)
    # sanity: the verdict map is genuinely non-uniform, so this test actually
    # exercises a scheduler that must agree with serial beyond "all pass".
    verdicts = _verdicts(parallel)
    assert any(nh for (_p, nh) in verdicts.values())
    assert any(not p for (p, _nh) in verdicts.values())

import types
from pathlib import Path
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
    cm = re.search(r"Check (Q\d\d)", user)
    check_id = cm.group(1) if cm else "Q07"
    if 'criterion_id="rubric"' in user:
        ids = ["rubric"]
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


def test_parallel_isolates_one_failing_check(tmp_path):
    b = _bundle()
    def responder(messages, tools):
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "Check Q03" in user:
            raise RuntimeError("boom on Q03 only")
        return _submit_all_pass(messages, tools)
    results = run_checks_parallel(SEMANTIC_CHECKS, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    v = _verdicts(results)
    assert v["Q03"][1] is True          # Q03 all passes failed -> needs_human
    assert len(results) == len(SEMANTIC_CHECKS)  # run completed, nothing sunk


def test_run_semantic_parallel_still_returns_all_checks(tmp_path):
    b = _bundle()
    results = run_semantic(b, FakeLLMClient(_submit_all_pass), _ctx(tmp_path), k=1, factual=False)
    ids = {r.id for r in results}
    assert {"Q07", "Q03", "Q09", "Q12"} <= ids

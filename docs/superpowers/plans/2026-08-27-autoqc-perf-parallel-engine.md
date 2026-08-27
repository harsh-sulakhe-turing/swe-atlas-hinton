# Parallel Semantic Engine + Single-Turn Text Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the semantic engine's single-task latency from ~10 min toward ~1–2 min by running the text-check agent passes over one bounded thread pool and making each text pass a single gateway call.

**Architecture:** Give the text roles only `submit_findings` with a low turn cap (one call per pass). Split `run_check` into reusable leaves (`proposer_pass`, `adversary_pass`, `finalize_check`), keep `run_check` as a serial wrapper, and add a flat bounded `ThreadPoolExecutor` scheduler in `run_semantic` that submits all proposer passes at once and launches each check's adversary as its proposers finish. Verdicts are unchanged (aggregation is order-independent).

**Tech Stack:** Python 3.10 (default `python3`), pytest, `concurrent.futures` threads, stdlib only. Reuses `run_agent`, `aggregate`, `adjudicate`, `FakeLLMClient`.

**Spec:** `docs/superpowers/specs/2026-08-27-autoqc-perf-parallel-engine-design.md`

## Global Constraints

- **Test runner:** always `python3 -m pytest`. `pyproject` sets `pythonpath=["."]`.
- **Whole suite stays offline & deterministic** — no test invokes Docker, the network, or a real gateway. Text-check tests drive `FakeLLMClient`.
- **Verdicts must not change:** the parallel `run_semantic` must produce the **same `CheckResult`s** (same `passed`/`needs_human` per check) as the serial path over the same responder. `aggregate`/`adjudicate` logic is reused verbatim, not reimplemented.
- **Do not break `run_check`'s signature** `run_check(check, bundle, client, ctx, k=3, votes_log=None) -> CheckResult` — `scripts/calibrate_clean.py` drives it directly, and existing tests call it.
- **Do not change** the deterministic checks (Q09/Q12), the Q06 factual stage, or `run_semantic`'s signature `run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None, factual=True)`.
- **Concurrency knob:** `W = int(os.environ.get("AUTOQC_ENGINE_WORKERS", "8"))`. **Text turn cap:** `TEXT_MAX_TURNS = 3`.
- **Threads only** — the gateway client is stdlib `urllib`, stateless per call, already shared across threads by `calibrate_clean`. No async, no locks: pool workers run pure pass functions with no shared writes; all scheduler bookkeeping happens on the main thread.

---

## File Structure

- **Modify `autoqc/agent/tools.py`** — add `text_tools()`.
- **Modify `autoqc/agent/checks.py`** — `proposer_role`/`adversary_role` use `text_tools()`.
- **Modify `autoqc/agent/engine.py`** — add `TEXT_MAX_TURNS`, `_empty_scope_result`, `_check_prep`, `proposer_pass`, `adversary_pass`, `finalize_check`; rewrite `run_check` as a serial wrapper over the leaves; add `run_checks_parallel` and call it from `run_semantic`.
- **Tests:** extend `tests/test_agent_tools.py`, `tests/test_agent_checks.py`, `tests/test_engine_run.py`; new `tests/test_engine_parallel.py`.

---

## Task 1: Single-turn text tools + roles

**Files:**
- Modify: `autoqc/agent/tools.py`, `autoqc/agent/checks.py`
- Test: `tests/test_agent_tools.py`, `tests/test_agent_checks.py`

**Interfaces:**
- Consumes: `SUBMIT_FINDINGS`, `Tool` (already in tools.py); `Role` (runner.py).
- Produces: `text_tools() -> list[Tool]` returning exactly `[SUBMIT_FINDINGS]`. `proposer_role()`/`adversary_role()` return `Role`s whose `tools` is `text_tools()`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_agent_tools.py
from autoqc.agent.tools import text_tools

def test_text_tools_is_submit_only():
    names = [t.name for t in text_tools()]
    assert names == ["submit_findings"]
```

```python
# add to tests/test_agent_checks.py
from autoqc.agent.checks import proposer_role, adversary_role, factual_role

def test_text_roles_have_no_read_tools():
    for role in (proposer_role(), adversary_role()):
        names = {t.name for t in role.tools}
        assert names == {"submit_findings"}, names

def test_factual_role_still_has_run_bash():
    names = {t.name for t in factual_role().tools}
    assert "run_bash" in names and "submit_findings" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_tools.py -k text_tools tests/test_agent_checks.py -k "text_roles or factual_role_still" -v`
Expected: FAIL — `ImportError` for `text_tools`; role tools still include `read_bundle_file`/`list_dir`.

- [ ] **Step 3: Implement**

In `autoqc/agent/tools.py`, next to `default_tools`:

```python
def text_tools() -> list[Tool]:
    """Text checks preload the whole bundle into context, so the agent needs no
    read tools — only the terminal submit. Forces a single-turn answer."""
    return [SUBMIT_FINDINGS]
```

In `autoqc/agent/checks.py`, change the two role factories (import `text_tools` alongside the existing `from autoqc.agent.tools import default_tools, factual_tools`):

```python
from autoqc.agent.tools import default_tools, factual_tools, text_tools
```

```python
def proposer_role() -> Role:
    return Role(name="proposer", system_prompt=_PROPOSER_SYS, tools=text_tools())


def adversary_role() -> Role:
    return Role(name="adversary", system_prompt=_ADVERSARY_SYS, tools=text_tools())
```

(`factual_role()` is unchanged — it keeps `factual_tools()`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_tools.py tests/test_agent_checks.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS (existing `run_check`/`run_semantic` tests use responders that submit on turn 1, so removing read tools changes nothing for them).

- [ ] **Step 6: Commit**

```bash
git add autoqc/agent/tools.py autoqc/agent/checks.py tests/test_agent_tools.py tests/test_agent_checks.py
git commit -m "perf(engine): single-turn text roles (submit-only tools)"
```

---

## Task 2: Decompose run_check into leaves + low turn cap

**Files:**
- Modify: `autoqc/agent/engine.py`
- Test: `tests/test_engine_run.py`

**Interfaces:**
- Consumes: `run_agent` (runner.py); `proposer_role`, `adversary_role`, `proposer_context`, `adversary_context` (checks.py); `aggregate`, `adjudicate`, `_own` (already in engine.py); `CheckResult`, `Stage`.
- Produces (all in engine.py):
  - `TEXT_MAX_TURNS = 3`
  - `_empty_scope_result(check) -> CheckResult`
  - `_check_prep(check, bundle) -> tuple[list, set]` returning `(criteria, allowed)`; `criteria` is `check.scope(items)`, `allowed` is `{"rubric"}` for rubric-mode else `{c["id"] for c in criteria}`.
  - `proposer_pass(check, bundle, client, ctx, criteria, allowed) -> tuple[list, dict]` → `(own_findings, log_entry)`.
  - `adversary_pass(check, bundle, client, ctx, criteria, allowed, agg) -> tuple[list, dict]` → `(own_findings, log_entry)`.
  - `finalize_check(check, agg, adv_findings) -> CheckResult`.
  - `run_check(...)` unchanged in signature/behavior, now built from the leaves.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_engine_run.py
from autoqc.agent.engine import (proposer_pass, adversary_pass, finalize_check,
                                 _check_prep, TEXT_MAX_TURNS, aggregate)

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
```

(The existing `test_run_check_*` tests stay and must keep passing — they now exercise the serial wrapper.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_engine_run.py -k "text_max_turns or proposer_pass or finalize_check" -v`
Expected: FAIL — `ImportError` for the new names.

- [ ] **Step 3: Implement in `autoqc/agent/engine.py`**

Add `import os` at the top (next to `import re`). Add the constant and leaves, and rewrite `run_check`:

```python
TEXT_MAX_TURNS = 3


def _empty_scope_result(check) -> CheckResult:
    return CheckResult(id=check.id, name=check.name, stage=Stage.SEMANTIC,
                       severity=check.severity, passed=True)


def _check_prep(check, bundle):
    items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
    criteria = check.scope(items)
    allowed = ({"rubric"} if getattr(check, "unit_mode", "criterion") == "rubric"
               else {c["id"] for c in criteria})
    return criteria, allowed


def proposer_pass(check, bundle, client, ctx, criteria, allowed):
    res = run_agent(proposer_role(), proposer_context(bundle, check, criteria),
                    client, ctx, max_turns=TEXT_MAX_TURNS)
    log = {"check": check.id, "role": "proposer", "ok": res.ok, "findings": res.findings}
    return (_own(res.findings, check.id, allowed) if res.ok else []), log


def adversary_pass(check, bundle, client, ctx, criteria, allowed, agg):
    res = run_agent(adversary_role(), adversary_context(bundle, check, criteria, agg),
                    client, ctx, max_turns=TEXT_MAX_TURNS)
    log = {"check": check.id, "role": "adversary", "ok": res.ok, "findings": res.findings}
    return (_own(res.findings, check.id, allowed) if res.ok else []), log


def finalize_check(check, agg, adv_findings) -> CheckResult:
    adj = adjudicate(agg, adv_findings)
    passed = all(v["passed"] for v in adj.values()) if adj else True
    needs_human = any(v["needs_human"] for v in adj.values())
    problem_ids = [cid for cid, v in adj.items() if (not v["passed"]) or v["needs_human"]]
    evidence = []
    for v in agg.values():
        evidence.extend(v.get("evidence") or [])
    detail = "" if passed and not needs_human else "criteria needing attention: " + ", ".join(problem_ids)
    return CheckResult(id=check.id, name=check.name, stage=Stage.SEMANTIC, severity=check.severity,
                       passed=passed, needs_human=needs_human, evidence=evidence[:20], detail=detail)
```

Rewrite `run_check` as the serial wrapper (same signature, same behavior):

```python
def run_check(check, bundle, client, ctx, k=3, votes_log=None) -> CheckResult:
    criteria, allowed = _check_prep(check, bundle)
    if not criteria:
        return _empty_scope_result(check)
    finding_sets = []
    for _ in range(k):
        own, log = proposer_pass(check, bundle, client, ctx, criteria, allowed)
        if votes_log is not None:
            votes_log.append(log)
        finding_sets.append(own)
    agg = aggregate(finding_sets, allowed)
    adv_own, adv_log = adversary_pass(check, bundle, client, ctx, criteria, allowed, agg)
    if votes_log is not None:
        votes_log.append(adv_log)
    return finalize_check(check, agg, adv_own)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_engine_run.py -v`
Expected: PASS — the new leaf tests AND every existing `test_run_check_*`/`test_votes_log_records_passes` test (behavior of `run_check` is unchanged).

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add autoqc/agent/engine.py tests/test_engine_run.py
git commit -m "perf(engine): split run_check into passes/finalize leaves + text turn cap"
```

---

## Task 3: Flat bounded parallel scheduler

**Files:**
- Modify: `autoqc/agent/engine.py`
- Test: `tests/test_engine_parallel.py` (create)

**Interfaces:**
- Consumes: `_check_prep`, `_empty_scope_result`, `proposer_pass`, `adversary_pass`, `finalize_check`, `aggregate` (Task 2); `SEMANTIC_CHECKS`, `DETERMINISTIC_CHECKS`, `run_factual_stage`.
- Produces:
  - `run_checks_parallel(checks, bundle, client, ctx, k, votes_log=None) -> list[CheckResult]` — one result per check, **in `checks` order**; uses a `ThreadPoolExecutor(max_workers=W)`; per-check adversary launches when that check's `k` proposers complete; results are identical to serial `run_check` per check.
  - `run_semantic(...)` calls `run_checks_parallel` for the text checks (same signature, same tail: deterministic checks + factual stage).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine_parallel.py
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
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_engine_parallel.py -v`
Expected: FAIL — `ImportError` for `run_checks_parallel`.

- [ ] **Step 3: Implement in `autoqc/agent/engine.py`**

Add the imports at the top:

```python
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
```

Add the scheduler and rewire `run_semantic`:

```python
def run_checks_parallel(checks, bundle, client, ctx, k, votes_log=None) -> list:
    """Run every text check over one bounded thread pool. All proposer passes are
    submitted up front; each check's adversary is launched when its k proposers
    finish. Verdicts are identical to serial run_check (aggregate is order-free).
    Bookkeeping runs only on this (main) thread; pool workers are pure passes."""
    w = int(os.environ.get("AUTOQC_ENGINE_WORKERS", "8"))
    preps, results_by_id = {}, {}
    for check in checks:
        criteria, allowed = _check_prep(check, bundle)
        if not criteria:
            results_by_id[check.id] = _empty_scope_result(check)
        else:
            preps[check.id] = (check, criteria, allowed)

    prop_logs = {cid: [] for cid in preps}     # cid -> [log dict] (completion order)
    prop_sets = {cid: [] for cid in preps}     # cid -> [own findings]
    remaining = {cid: k for cid in preps}
    aggs, adv_logs = {}, {}

    with ThreadPoolExecutor(max_workers=max(1, w)) as pool:
        fut_meta, pending = {}, set()
        for cid, (check, criteria, allowed) in preps.items():
            for _ in range(k):
                fut = pool.submit(proposer_pass, check, bundle, client, ctx, criteria, allowed)
                fut_meta[fut] = ("proposer", cid)
                pending.add(fut)

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                kind, cid = fut_meta.pop(fut)
                check, criteria, allowed = preps[cid]
                if kind == "proposer":
                    try:
                        own, log = fut.result()
                    except Exception as e:  # never sink the run
                        own, log = [], {"check": cid, "role": "proposer", "ok": False,
                                        "findings": [], "error": repr(e)}
                    prop_sets[cid].append(own)
                    prop_logs[cid].append(log)
                    remaining[cid] -= 1
                    if remaining[cid] == 0:
                        agg = aggregate(prop_sets[cid], allowed)
                        aggs[cid] = agg
                        afut = pool.submit(adversary_pass, check, bundle, client, ctx,
                                           criteria, allowed, agg)
                        fut_meta[afut] = ("adversary", cid)
                        pending.add(afut)
                else:  # adversary
                    try:
                        adv_own, adv_log = fut.result()
                    except Exception as e:
                        adv_own, adv_log = [], {"check": cid, "role": "adversary", "ok": False,
                                               "findings": [], "error": repr(e)}
                    adv_logs[cid] = adv_log
                    results_by_id[cid] = finalize_check(check, aggs[cid], adv_own)

    if votes_log is not None:  # deterministic order: check order, proposers then adversary
        for check in checks:
            cid = check.id
            if cid in preps:
                votes_log.extend(prop_logs[cid])
                if cid in adv_logs:
                    votes_log.append(adv_logs[cid])

    return [results_by_id[check.id] for check in checks]


def run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None, factual=True):
    results = run_checks_parallel(checks, bundle, client, ctx, k, votes_log=votes_log)
    results += [fn(bundle) for fn in DETERMINISTIC_CHECKS]
    if factual:
        results.append(run_factual_stage(bundle, client, votes_log=votes_log))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_engine_parallel.py -v`
Expected: PASS (all seven).

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS — including the existing `test_run_semantic_returns_one_result_per_check` and `test_run_semantic_includes_deterministic_checks` (the parallel path yields the same results).

- [ ] **Step 6: Commit**

```bash
git add autoqc/agent/engine.py tests/test_engine_parallel.py
git commit -m "perf(engine): flat bounded thread pool scheduler for text checks"
```

---

## Self-Review

**Spec coverage:**
- §3.1 one pool + `AUTOQC_ENGINE_WORKERS` (default 8) → Task 3 (`run_checks_parallel`).
- §3.2 decompose `run_check` into leaves; keep serial wrapper → Task 2.
- §3.3 scheduler (submit all proposers, per-check barrier, deterministic order, no locks, per-future error isolation) → Task 3 (`run_checks_parallel` + `test_parallel_isolates_one_failing_check`).
- §4 single-turn text tools + `TEXT_MAX_TURNS=3` → Task 1 (`text_tools`, roles) + Task 2 (constant + passes use it; `test_proposer_pass_never_submits_is_bounded_and_not_ok`).
- §6 testing: equivalence (`test_parallel_equals_serial`), order-independence/stability (`test_parallel_is_stable_across_runs`), votes_log completeness (`test_parallel_votes_log_complete_for_scoped_check`), role tools (`test_text_roles_have_no_read_tools`), turn-cap bound (Task 2), error isolation (`test_parallel_isolates_one_failing_check`), width knob (`test_parallel_width_one_is_correct`), serial wrapper unchanged (existing `test_run_check_*`).
- §7 non-goals (model tiering, Q06 parallelism, build pool, Q06/text overlap) → not implemented, by design.
- §8 files touched → Tasks 1–3 match exactly.

**Placeholder scan:** no TBD/TODO; all code steps are complete and runnable; the width (8) and turn cap (3) are concrete constants.

**Type consistency:** `_check_prep -> (criteria, allowed)` consumed the same way in `run_check` (Task 2) and `run_checks_parallel` (Task 3); `proposer_pass`/`adversary_pass -> (own, log)` consistent between definition (Task 2) and scheduler use (Task 3); `finalize_check(check, agg, adv_findings) -> CheckResult` consistent; `run_checks_parallel(checks, bundle, client, ctx, k, votes_log=None)` consistent between Task 3 def and its `run_semantic` call. `text_tools()` (Task 1) consumed by roles that Task 2/3 passes invoke. `run_check`/`run_semantic` signatures unchanged, matching the Global Constraints.

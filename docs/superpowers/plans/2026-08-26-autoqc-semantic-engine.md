# AutoQC Semantic Engine (Q07 + Q03) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the AgentRunner primitive into a working semantic stage: an ensemble + adversary engine that runs `run_agent` passes and adjudicates them into `CheckResult`s, the first two checks (Q07 negative-semantics, Q03 wildcard), CLI wiring, and a first **live smoke run** against GLM 5.2.

**Architecture:** `agent/engine.py` has two pure functions — `aggregate` (majority + split across K proposer finding-sets) and `adjudicate` (split or adversary-overturn → `needs_human`) — plus `run_check` (K proposer passes + 1 asymmetric adversary pass via `run_agent`, validated against the rubric, rolled up to one `CheckResult`) and `run_semantic` (loop the checks). `agent/checks.py` defines the `SemanticCheck` (id, name, severity, criterion scope, prompt guidance), `Q07`/`Q03`, the proposer/adversary `Role` builders, and `SEMANTIC_CHECKS`. `cli.run` runs structural then semantic when a client is available (structural-only otherwise, preserving M1). Everything is `FakeLLMClient`-tested; the live smoke is a separate script the operator runs with `.env`.

**Tech Stack:** Python 3.11+ target (runs on 3.10). `pytest`. No third-party runtime deps. Every unit test uses `FakeLLMClient`.

**Spec:** `docs/superpowers/specs/2026-08-25-autoqc-agent-architecture.md` (§3 primitive, §4 contract, §5 ensemble/adversary/adjudication).

## Global Constraints

- Run every test with `python3 -m pytest`. **All unit tests use `FakeLLMClient`** — no test hits the network or needs keys. The live smoke (final section) is operator-run, not a unit test.
- Reuse `CheckResult`/`Severity`/`Stage`, `run_agent`/`Role`/`AgentResult`, `default_tools`/`AgentContext`/`validate_findings`, `default_client` — all already on `main`. Do not modify them.
- Each check → exactly one `CheckResult` (`id` = the check id, `stage=Stage.SEMANTIC`, severity from the check): `passed` = all its criteria passed; `needs_human` = any criterion split/overturned/uncertain.
- Adjudication: a criterion is `needs_human` when the ensemble is split (disagreement or an abstaining pass) OR the adversary overturns the aggregate. This feeds the C1-fixed `compute_verdict` (a disputed reject → `needs_human_review`).
- Robustness: a failed `run_agent` pass (`ok=False`) contributes no verdicts (its criteria abstain → `needs_human`); never raises.
- Q03 must NOT flag `e.g.`/`such as` with interchangeable items (118/124 good-set tasks use it) — only true escape hatches or non-interchangeable lists. This guardrail goes in the Q03 guidance string.
- `if git commit is blocked, leave files staged and report DONE_WITH_CONCERNS` — the controller commits.

---

## File Structure

- `autoqc/agent/engine.py` — `aggregate`, `adjudicate`, `run_check`, `run_semantic`.
- `autoqc/agent/checks.py` — `SemanticCheck`, scopes, `Q07`, `Q03`, `SEMANTIC_CHECKS`, `proposer_role`/`adversary_role`, `proposer_context`/`adversary_context`.
- `autoqc/cli.py` — MODIFIED: run structural + semantic; structural-only with no client.
- `tests/` — one module per new source file.
- `scripts/smoke_semantic.py` — operator-run live smoke (created in the final section).

---

### Task 1: engine pure core — `aggregate` + `adjudicate`

**Files:**
- Create: `autoqc/agent/engine.py`
- Test: `tests/test_engine_core.py`

**Interfaces:**
- Produces:
  - `aggregate(finding_sets: list[list[dict]], allowed_ids: set[str]) -> dict[str, dict]` — for each criterion id in `allowed_ids`, gather the `passed` votes across finding-sets (a finding matches by `criterion_id`); returns `{cid: {"passed": bool, "split": bool, "evidence": list[str]}}`. `passed` = strict majority of cast votes (`sum(votes)*2 > len(votes)`); `split` = votes disagree OR fewer votes than finding-sets (an abstaining pass) OR zero votes; zero votes → `passed=False, split=True`.
  - `adjudicate(agg: dict[str, dict], adversary_findings: list[dict]) -> dict[str, dict]` — `{cid: {"passed": bool, "needs_human": bool}}`; `overturn` = an adversary finding for `cid` whose `passed` differs from the aggregate; `needs_human = agg[cid]["split"] or overturn`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_core.py
from autoqc.agent.engine import aggregate, adjudicate


def _f(cid, passed, ev="e"):
    return {"check_id": "Q07", "criterion_id": cid, "passed": passed, "evidence": [ev]}


def test_aggregate_unanimous_pass():
    sets = [[_f("a", True)], [_f("a", True)], [_f("a", True)]]
    agg = aggregate(sets, {"a"})
    assert agg["a"]["passed"] is True and agg["a"]["split"] is False


def test_aggregate_unanimous_fail():
    sets = [[_f("a", False)], [_f("a", False)], [_f("a", False)]]
    agg = aggregate(sets, {"a"})
    assert agg["a"]["passed"] is False and agg["a"]["split"] is False


def test_aggregate_split_on_disagreement():
    sets = [[_f("a", True)], [_f("a", False)], [_f("a", True)]]
    agg = aggregate(sets, {"a"})
    assert agg["a"]["passed"] is True   # 2/3 majority
    assert agg["a"]["split"] is True    # not unanimous


def test_aggregate_abstain_counts_as_split():
    sets = [[_f("a", True)], [], [_f("a", True)]]  # one pass abstained
    agg = aggregate(sets, {"a"})
    assert agg["a"]["split"] is True


def test_aggregate_no_votes_is_fail_and_split():
    agg = aggregate([[], [], []], {"a"})
    assert agg["a"]["passed"] is False and agg["a"]["split"] is True


def test_aggregate_collects_evidence():
    sets = [[_f("a", False, "ev1")], [_f("a", False, "ev2")]]
    agg = aggregate(sets, {"a"})
    assert "ev1" in agg["a"]["evidence"] and "ev2" in agg["a"]["evidence"]


def test_adjudicate_clean_pass():
    agg = {"a": {"passed": True, "split": False, "evidence": []}}
    adj = adjudicate(agg, adversary_findings=[])
    assert adj["a"] == {"passed": True, "needs_human": False}


def test_adjudicate_split_needs_human():
    agg = {"a": {"passed": True, "split": True, "evidence": []}}
    adj = adjudicate(agg, [])
    assert adj["a"]["needs_human"] is True


def test_adjudicate_adversary_overturn_needs_human():
    agg = {"a": {"passed": False, "split": False, "evidence": []}}  # a reject
    adv = [{"check_id": "Q07", "criterion_id": "a", "passed": True, "evidence": ["actually fine"]}]
    adj = adjudicate(agg, adv)
    assert adj["a"]["passed"] is False and adj["a"]["needs_human"] is True  # disputed reject
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_engine_core.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'autoqc.agent.engine'`)

- [ ] **Step 3: Write `autoqc/agent/engine.py` (core only)**

```python
from __future__ import annotations


def aggregate(finding_sets, allowed_ids):
    out = {}
    n = len(finding_sets)
    for cid in allowed_ids:
        votes, evidence = [], []
        for fs in finding_sets:
            for f in fs:
                if f.get("criterion_id") == cid:
                    votes.append(bool(f.get("passed")))
                    evidence.extend(f.get("evidence") or [])
        if not votes:
            out[cid] = {"passed": False, "split": True, "evidence": evidence}
            continue
        passed = sum(votes) * 2 > len(votes)
        split = (len(set(votes)) > 1) or (len(votes) < n)
        out[cid] = {"passed": passed, "split": split, "evidence": evidence}
    return out


def adjudicate(agg, adversary_findings):
    adv = {f.get("criterion_id"): bool(f.get("passed"))
           for f in (adversary_findings or []) if f.get("criterion_id")}
    out = {}
    for cid, v in agg.items():
        overturn = cid in adv and adv[cid] != v["passed"]
        out[cid] = {"passed": v["passed"], "needs_human": bool(v["split"] or overturn)}
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_engine_core.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/engine.py tests/test_engine_core.py
git commit -m "feat(autoqc): engine core — aggregate + adjudicate"
```

---

### Task 2: the checks (Q07, Q03) + roles + prompts

**Files:**
- Create: `autoqc/agent/checks.py`
- Test: `tests/test_agent_checks.py`

**Interfaces:**
- Consumes: `Severity` (model); `Role`, `default_tools` (agent).
- Produces:
  - `SemanticCheck(id, name, severity, scope, guidance)` where `scope(items) -> list[dict]` selects the criteria this check judges.
  - `Q07` (severity REJECT, scope = negatives) and `Q03` (severity REJECT, scope = all criteria); `SEMANTIC_CHECKS = [Q07, Q03]`.
  - `proposer_role() -> Role`, `adversary_role() -> Role` (distinct system prompts; adversary's contains the word "adversarial").
  - `proposer_context(bundle, check, criteria) -> str` and `adversary_context(bundle, check, criteria, agg) -> str` — each lists the criteria by `criterion_id` + title and states the check guidance; the adversary context also states the current aggregate verdict per criterion.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_checks.py
import types
from autoqc.model import Severity
from autoqc.agent.checks import (SemanticCheck, Q07, Q03, SEMANTIC_CHECKS,
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
    assert [c.id for c in SEMANTIC_CHECKS] == ["Q07", "Q03"]
    assert Q07.severity is Severity.REJECT and Q03.severity is Severity.REJECT


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_checks.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'autoqc.agent.checks'`)

- [ ] **Step 3: Write `autoqc/agent/checks.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from autoqc.model import Severity
from autoqc.agent.runner import Role
from autoqc.agent.tools import default_tools


@dataclass
class SemanticCheck:
    id: str
    name: str
    severity: Severity
    scope: Callable          # (items) -> list[dict]
    guidance: str


def _is_dict_crit(it):
    return isinstance(it, dict) and it.get("id")


def _negatives(items):
    return [it for it in (items or [])
            if _is_dict_crit(it) and "negative" in str(
                it.get("annotations", {}).get("type", "") if isinstance(it.get("annotations"), dict) else "")]


def _all_criteria(items):
    return [it for it in (items or []) if _is_dict_crit(it)]


Q07 = SemanticCheck(
    id="Q07", name="Negative score-flip semantics", severity=Severity.REJECT, scope=_negatives,
    guidance=("A NEGATIVE criterion must state the FALSE assertion whose PRESENCE in an answer "
              "should fail it (for example 'Claims that every retry uses exponential backoff'). "
              "It VIOLATES this check if phrased as 'Does not claim...', as a required omission, "
              "or as the correct behavior. passed=true means correctly phrased."))

Q03 = SemanticCheck(
    id="Q03", name="No wildcard / escape hatch", severity=Severity.REJECT, scope=_all_criteria,
    guidance=("A criterion VIOLATES this check if it can be satisfied by open-ended wording: a "
              "bare trailing '*', 'or similar', 'or other', 'and so on', or an e.g./such-as list "
              "whose items are NOT interchangeable full answers. IMPORTANT: an e.g./such as list is "
              "FINE and common when every listed item is an interchangeable way to satisfy the same "
              "requirement — do NOT flag interchangeable examples. passed=true means no escape hatch."))

SEMANTIC_CHECKS = [Q07, Q03]

_PROPOSER_SYS = (
    "You are a strict, evidence-based QC reviewer of grading rubrics. Judge only the check and "
    "criteria you are given. You may read bundle files with the tools if helpful. Finish by "
    "calling submit_findings with exactly one finding per listed criterion; evidence must quote "
    "the criterion text you relied on.")

_ADVERSARY_SYS = (
    "You are an adversarial second reviewer of grading-rubric QC. For each criterion a prior "
    "review marked FAIL, argue whether it is actually acceptable (defend it). For each marked "
    "PASS, look for a violation the first reviewer missed (attack it). Finish by calling "
    "submit_findings with your verdict per criterion; passed=true means the criterion is fine.")


def proposer_role() -> Role:
    return Role(name="proposer", system_prompt=_PROPOSER_SYS, tools=default_tools())


def adversary_role() -> Role:
    return Role(name="adversary", system_prompt=_ADVERSARY_SYS, tools=default_tools())


def _criteria_block(criteria) -> str:
    return "\n".join(f"- criterion_id={c['id']}  title={c.get('title','')!r}" for c in criteria)


def proposer_context(bundle, check, criteria) -> str:
    return (f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            f"Judge each of these criteria (submit one finding per criterion, check_id={check.id}):\n"
            f"{_criteria_block(criteria)}\n\n"
            "passed=true if the criterion SATISFIES the check, passed=false if it VIOLATES it.")


def adversary_context(bundle, check, criteria, agg) -> str:
    lines = []
    for c in criteria:
        v = agg.get(c["id"], {})
        verdict = "PASS" if v.get("passed") else "FAIL (reject)"
        lines.append(f"- criterion_id={c['id']}  prior_verdict={verdict}  title={c.get('title','')!r}")
    return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            "A prior review produced these verdicts. Challenge them per the rules above, then "
            f"submit_findings with your verdict per criterion (check_id={check.id}):\n"
            + "\n".join(lines))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_checks.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/checks.py tests/test_agent_checks.py
git commit -m "feat(autoqc): Q07 + Q03 semantic checks, roles, prompts"
```

---

### Task 3: `run_check` + `run_semantic`

**Files:**
- Modify: `autoqc/agent/engine.py` (add `run_check`, `run_semantic`)
- Test: `tests/test_engine_run.py`

**Interfaces:**
- Consumes: `aggregate`/`adjudicate` (Task 1); `SemanticCheck`, `proposer_role`/`adversary_role`, `proposer_context`/`adversary_context` (Task 2); `run_agent` (runner); `validate_findings` (tools); `CheckResult`/`Stage` (model).
- Produces:
  - `run_check(check, bundle, client, ctx, k=3, votes_log=None) -> CheckResult` — if the check has no in-scope criteria, returns a passing `CheckResult`. Else: K proposer passes via `run_agent(proposer_role(), proposer_context(...), client, ctx)`; each pass's findings are filtered by `validate_findings(res.findings, allowed_ids)` and to `check_id == check.id`; a failed pass (`ok=False`) contributes `[]`. Aggregate → 1 adversary pass → adjudicate → roll up: `passed = all criteria passed`, `needs_human = any needs_human`, `detail` = uncertain/failing criterion ids, `evidence` = concatenated (cap 20). Append each pass's raw findings + role + ok to `votes_log` when provided.
  - `run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None) -> list[CheckResult]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_run.py
import types
from pathlib import Path
from autoqc.model import Stage, Severity
from autoqc.agent.engine import run_check, run_semantic
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
    results = run_semantic(b, FakeLLMClient(lambda m, t: _submit([])), _ctx(tmp_path), k=1)
    ids = {r.id for r in results}
    assert ids == {"Q07", "Q03"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_engine_run.py -v`
Expected: FAIL (`ImportError: cannot import name 'run_check'`)

- [ ] **Step 3: Add `run_check` + `run_semantic` to `autoqc/agent/engine.py`**

```python
# add these imports at the top of engine.py
from autoqc.model import CheckResult, Stage
from autoqc.agent.runner import run_agent
from autoqc.agent.checks import (SEMANTIC_CHECKS, proposer_role, adversary_role,
                                 proposer_context, adversary_context)
from autoqc.agent.tools import validate_findings


def _own(findings, check_id, allowed_ids):
    valid, _ = validate_findings(findings, allowed_ids)
    return [f for f in valid if f.get("check_id") == check_id]


def run_check(check, bundle, client, ctx, k=3, votes_log=None) -> CheckResult:
    items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
    criteria = check.scope(items)
    if not criteria:
        return CheckResult(id=check.id, name=check.name, stage=Stage.SEMANTIC,
                           severity=check.severity, passed=True)
    allowed = {c["id"] for c in criteria}

    finding_sets = []
    p_ctx = proposer_context(bundle, check, criteria)
    for _ in range(k):
        res = run_agent(proposer_role(), p_ctx, client, ctx)
        if votes_log is not None:
            votes_log.append({"check": check.id, "role": "proposer",
                              "ok": res.ok, "findings": res.findings})
        finding_sets.append(_own(res.findings, check.id, allowed) if res.ok else [])

    agg = aggregate(finding_sets, allowed)
    adv_res = run_agent(adversary_role(), adversary_context(bundle, check, criteria, agg), client, ctx)
    if votes_log is not None:
        votes_log.append({"check": check.id, "role": "adversary",
                          "ok": adv_res.ok, "findings": adv_res.findings})
    adv_findings = _own(adv_res.findings, check.id, allowed) if adv_res.ok else []

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


def run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None):
    return [run_check(c, bundle, client, ctx, k=k, votes_log=votes_log) for c in checks]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_engine_run.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add autoqc/agent/engine.py tests/test_engine_run.py
git commit -m "feat(autoqc): run_check + run_semantic (ensemble + adversary passes)"
```

---

### Task 4: pipeline wiring

**Files:**
- Modify: `autoqc/cli.py`
- Test: `tests/test_cli_semantic.py`

**Interfaces:**
- Consumes: `run_semantic` (engine); `AgentContext` (tools); `default_client` (llm); existing `load_bundle`/`run_structural`/`compute_verdict`/`to_record`/`to_markdown`.
- Produces: MODIFIED `run(bundle_dir, out_dir, llm=None, k=3) -> Verdict`. Always runs structural; resolves `client = llm if llm is not None else default_client()`; if the client is not None, appends `run_semantic(bundle, client, AgentContext(bundle_dir=Path(bundle_dir)), k=k)` to results; if None, semantic is skipped (M1 structural-only preserved). `main` unchanged in signature.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_semantic.py
import json
import types
from pathlib import Path
from autoqc.cli import run
from autoqc.llm import FakeLLMClient
from autoqc.model import Verdict


def _bundle(root: Path, neg_title):
    (root / "tests").mkdir(parents=True)
    (root / "tests/rubrics.json").write_text(json.dumps([
        {"id": "a" * 32, "title": "1.1: States X",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}},
        {"id": "b" * 32, "title": neg_title,
         "annotations": {"type": "negative hli verifier", "importance": "must have"}}]))
    (root / "tests/prompt.txt").write_text("q")
    (root / "solution").mkdir(); (root / "solution/answer.txt").write_text("a")
    (root / "environment").mkdir(); (root / "environment/Dockerfile").write_text("FROM x")
    (root / "task.toml").write_text('[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))


def _submit(findings):
    return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": findings}}]}


def test_semantic_runs_with_client_all_pass(tmp_path):
    b = tmp_path / "task"; b.mkdir()
    _bundle(b, "2.1: Claims that X fails")
    # every criterion passes both checks
    def responder(messages, tools):
        import re
        ids = re.findall(r"criterion_id=(\w+)", " ".join(m["content"] for m in messages))
        cid = "Q07" if "Q07" in " ".join(m["content"] for m in messages) else "Q03"
        return _submit([{"check_id": cid, "criterion_id": i, "passed": True, "evidence": ["ok"]}
                        for i in dict.fromkeys(ids)])
    v = run(b, tmp_path / "out", llm=FakeLLMClient(responder))
    rec = json.loads((tmp_path / "out/review_record.json").read_text())
    assert {"Q07", "Q03"} <= {r["id"] for r in rec["results"]}
    assert v is Verdict.SOUND


def test_semantic_reject_makes_not_sound(tmp_path):
    b = tmp_path / "task"; b.mkdir()
    _bundle(b, "2.1: Does not claim that X fails")  # a Q07 violation
    def responder(messages, tools):
        import re
        txt = " ".join(m["content"] for m in messages)
        cid = "Q07" if "Q07" in txt else "Q03"
        ids = list(dict.fromkeys(re.findall(r"criterion_id=(\w+)", txt)))
        # fail the negative on Q07, pass everything else
        return _submit([{"check_id": cid, "criterion_id": i,
                         "passed": not (cid == "Q07"), "evidence": ["x"]} for i in ids])
    v = run(b, tmp_path / "out", llm=FakeLLMClient(responder))
    assert v is Verdict.NOT_SOUND
    assert "Q07" in (tmp_path / "out/report.md").read_text()


def test_structural_only_when_no_client(tmp_path, monkeypatch):
    for var in ("EVAL_API_KEY", "EVAL_BASE_URL", "OPENAI_API_KEY", "OPENAI_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    b = tmp_path / "task"; b.mkdir()
    _bundle(b, "2.1: Does not claim that X fails")
    v = run(b, tmp_path / "out")  # no client, no env
    rec = json.loads((tmp_path / "out/review_record.json").read_text())
    assert not any(r["id"] in ("Q07", "Q03") for r in rec["results"])
    assert v is Verdict.SOUND
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli_semantic.py -v`
Expected: FAIL (`run()` has no `llm` parameter → TypeError)

- [ ] **Step 3: Modify `autoqc/cli.py`**

Add imports beside the existing ones:

```python
from autoqc.agent.engine import run_semantic
from autoqc.agent.tools import AgentContext
from autoqc.llm import default_client
```

Replace `run` with:

```python
def run(bundle_dir, out_dir, llm=None, k: int = 3) -> Verdict:
    bundle = load_bundle(Path(bundle_dir))
    results = run_structural(bundle)

    client = llm if llm is not None else default_client()
    if client is not None:
        ctx = AgentContext(bundle_dir=Path(bundle_dir))
        results.extend(run_semantic(bundle, client, ctx, k=k))

    verdict = compute_verdict(results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_record.json").write_text(
        json.dumps(to_record(bundle, results, verdict), indent=2))
    (out / "report.md").write_text(to_markdown(bundle, results, verdict))
    return verdict
```

`main` is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cli_semantic.py tests/test_cli_e2e.py -v`
Expected: PASS — the new tests AND the existing structural-only e2e tests (no client + cleared env).

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add autoqc/cli.py tests/test_cli_semantic.py
git commit -m "feat(autoqc): wire semantic stage (run_semantic) into the pipeline"
```

---

## Self-Review

**Spec coverage:**
- Ensemble aggregation + adjudication (split/overturn → needs_human) → Task 1. ✓
- Q07 + Q03 with the e.g.-guardrail in Q03 guidance → Task 2. ✓
- `run_check` (K proposer + 1 adversary via run_agent) + `run_semantic` + per-vote log + failed-pass robustness → Task 3. ✓
- Pipeline wiring, structural-only without a client → Task 4. ✓
- Live smoke → the operator section below.
- Deferred (correctly absent): remaining text checks (Q01/Q02/Q04/Q05/Q08–Q12), Q06 factual/run_bash, holistic single-pass optimization, record-schema unification.

**Placeholder scan:** none.

**Type consistency:** `aggregate(finding_sets, allowed_ids) -> {cid: {passed,split,evidence}}` and `adjudicate(agg, adversary_findings) -> {cid: {passed,needs_human}}` defined in Task 1, consumed in Task 3. `SemanticCheck.scope/guidance`, `proposer_role`/`adversary_role`, `proposer_context`/`adversary_context` defined in Task 2, consumed in Task 3. `run_semantic(bundle, client, ctx, checks, k, votes_log)` from Task 3 consumed in Task 4. `FakeLLMClient` responder is `(messages, tools)` throughout. `run(bundle_dir, out_dir, llm, k)` extends the M1 two-arg call.

---

## Live smoke run (operator-run, after the tasks pass)

Not a unit test — this makes real GLM 5.2 calls via `.env`, so the controller runs it, not a subagent.

- [ ] **Create `scripts/smoke_semantic.py`:**

```python
"""Live smoke: run the full AutoQC pipeline (structural + semantic) against the real
gateway on one bundle. Reads .env, prints verdict + semantic findings + token cost."""
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from autoqc.bundle import load_bundle
from autoqc.structural import run_structural
from autoqc.agent.engine import run_semantic
from autoqc.agent.tools import AgentContext
from autoqc.verdict import compute_verdict
from autoqc.llm import GatewayLLMClient

# load .env
for line in open(".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        import os
        os.environ.setdefault(k, v)

bundle_dir = Path(sys.argv[1])
bundle = load_bundle(bundle_dir)
client = GatewayLLMClient()
assert client.available(), ".env missing EVAL_API_KEY/EVAL_BASE_URL"
print(f"model={client.model}  bundle={bundle_dir.name}")

results = run_structural(bundle)
votes = []
results += run_semantic(bundle, client, AgentContext(bundle_dir=bundle_dir), k=3, votes_log=votes)
verdict = compute_verdict(results)

for r in results:
    if r.stage.value == "semantic":
        print(f"  {r.id} {r.name}: passed={r.passed} needs_human={r.needs_human}  {r.detail}")
print("VERDICT:", verdict.value)
print("semantic passes made:", len(votes))
```

- [ ] **Run it against one real internal sample** (e.g. the aiohttp bundle) and report the verdict, the Q07/Q03 per-criterion outcomes, whether they look sane, and the token cost. Use the findings to decide K, prompt tweaks, and whether the holistic-single-pass optimization is warranted (design doc §9).

---

## Notes for the next plans

- Q06 factual role + `run_bash` in the built container (the `tool.run` guard is already in place).
- Remaining text checks (Q01/Q02/Q04/Q05/Q08–Q12) — each is a new `SemanticCheck` entry + guidance.
- Per-vote log → persist into the review record (schema unification, retire unused `ReviewRecord`).
- Calibration: seeded-defect corpus + the internal 10 + curated public good-set; tune K / thresholds.

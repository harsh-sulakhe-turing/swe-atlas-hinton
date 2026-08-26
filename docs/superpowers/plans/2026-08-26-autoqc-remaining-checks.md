# AutoQC Remaining Text Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the nine remaining semantic checks (Q01, Q02, Q04, Q05, Q08, Q09, Q10, Q11, Q12) so AutoQC covers every text-level rubric-quality dimension, then live-calibrate the full set.

**Architecture:** Two are deterministic counts (Q09 = ≥1 negative; Q12 = ≤18 criteria) → pure functions, no LLM. The rest are agent checks reusing the engine: Q01/Q02/Q05 are per-criterion (existing model); Q04/Q08/Q10/Q11 are per-rubric, enabled by a small `unit_mode="rubric"` extension that submits one finding with `criterion_id="rubric"`. All new agent checks are just data (id, name, severity, scope, guidance) plus prompts; the engine, adjudication, and verdict are unchanged. Calibration's `CHECKS` is generalized so the live run measures false-reject across every check. Q12's semantic "non-redundant" half is deferred (only its size rule ships here).

**Tech Stack:** Python 3.11+ (runs on 3.10). `pytest`. No third-party runtime deps. Unit tests use `FakeLLMClient` / fabricated `CheckResult`s.

**Spec:** `docs/superpowers/specs/2026-08-25-autoqc-quality-rubric-design.md` §5 (Q01–Q12 definitions) and `2026-08-25-autoqc-agent-architecture.md` (engine/contract).

## Global Constraints

- Run every test with `python3 -m pytest`. All unit tests are offline.
- Reuse `run_check`/`run_semantic`/`aggregate`/`adjudicate` (engine), `Role`/`run_agent` (runner), `SemanticCheck`/`proposer_role`/`adversary_role`/`proposer_context`/`adversary_context` (checks), `CheckResult`/`Severity`/`Stage` (model). Extend, don't rewrite.
- Severities (from the rubric spec): Q01/Q02/Q04/Q05 = REJECT; Q08/Q09/Q10/Q11/Q12 = WARN.
- Each check → exactly one `CheckResult` (id = the check id, `stage=Stage.SEMANTIC`).
- Deterministic checks (Q09, Q12) run inside `run_semantic` (they're semantic-layer quality signals) so structural-only-when-no-client is preserved and existing e2e stays green.
- Q03's e.g.-guardrail principle applies to Q01/Q02 too: don't flag legitimate phrasing; guidance must be precise about what VIOLATES vs what's fine.
- `if git commit is blocked, leave files staged and report DONE_WITH_CONCERNS` — the controller commits.

---

## File Structure

- `autoqc/agent/deterministic.py` — `check_q09`, `check_q12_size`, `DETERMINISTIC_CHECKS`.
- `autoqc/agent/engine.py` — MODIFY: `run_semantic` also runs `DETERMINISTIC_CHECKS`; `run_check` handles `unit_mode="rubric"`.
- `autoqc/agent/checks.py` — MODIFY: `SemanticCheck` gains `unit_mode`; context builders handle rubric mode; add Q01/Q02/Q04/Q05/Q08/Q10/Q11; extend `SEMANTIC_CHECKS`.
- `autoqc/calibrate.py` — MODIFY: generalize `CHECKS` to all check ids.
- `tests/` — one module per change.

---

### Task 1: deterministic checks Q09 + Q12 (size)

**Files:**
- Create: `autoqc/agent/deterministic.py`
- Modify: `autoqc/agent/engine.py` (`run_semantic` runs deterministic checks)
- Test: `tests/test_deterministic_checks.py`

**Interfaces:**
- Produces: `check_q09(bundle) -> CheckResult` (id "Q09", WARN; `passed` = at least one negative criterion exists). `check_q12_size(bundle, limit=18) -> CheckResult` (id "Q12", WARN; `passed` = criterion count ≤ limit). `DETERMINISTIC_CHECKS = [check_q09, check_q12_size]`. Both None-safe. `run_semantic` appends `[fn(bundle) for fn in DETERMINISTIC_CHECKS]` after the agent checks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deterministic_checks.py
import types
from autoqc.model import Stage, Severity
from autoqc.agent.deterministic import check_q09, check_q12_size, DETERMINISTIC_CHECKS


def _b(items):
    return types.SimpleNamespace(rubrics=items, prompt="q")


def _pos(i):
    return {"id": i, "title": "1.1: X", "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(i):
    return {"id": i, "title": "2.1: Claims X", "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_q09_passes_with_negative():
    r = check_q09(_b([_pos("a"), _neg("b")]))
    assert r.id == "Q09" and r.stage is Stage.SEMANTIC and r.severity is Severity.WARN
    assert r.passed is True


def test_q09_fails_without_negative():
    r = check_q09(_b([_pos("a")]))
    assert r.passed is False and "negative" in r.detail.lower()


def test_q09_none_safe():
    assert check_q09(types.SimpleNamespace(rubrics=None, prompt="q")).passed is False


def test_q12_passes_at_or_below_limit():
    r = check_q12_size(_b([_pos(str(i)) for i in range(18)]))
    assert r.id == "Q12" and r.passed is True


def test_q12_fails_above_limit():
    r = check_q12_size(_b([_pos(str(i)) for i in range(19)]))
    assert r.passed is False and r.severity is Severity.WARN and "19" in r.detail


def test_deterministic_registry():
    assert DETERMINISTIC_CHECKS == [check_q09, check_q12_size]
```

Also add to `tests/test_engine_run.py` (append):

```python
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
        AgentContext(bundle_dir=tmp_path), k=1)
    ids = {r.id for r in results}
    assert {"Q09", "Q12"} <= ids  # deterministic checks present
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_deterministic_checks.py -v`
Expected: FAIL (`No module named 'autoqc.agent.deterministic'`)

- [ ] **Step 3: Write `autoqc/agent/deterministic.py`**

```python
from __future__ import annotations
from autoqc.model import CheckResult, Stage, Severity


def _criteria(bundle):
    items = getattr(bundle, "rubrics", None)
    return [it for it in items if isinstance(it, dict)] if isinstance(items, list) else []


def _is_negative(it):
    ann = it.get("annotations")
    return "negative" in str(ann.get("type", "") if isinstance(ann, dict) else "")


def check_q09(bundle) -> CheckResult:
    has_neg = any(_is_negative(it) for it in _criteria(bundle))
    return CheckResult(id="Q09", name="Negative present", stage=Stage.SEMANTIC,
                       severity=Severity.WARN, passed=has_neg,
                       detail="" if has_neg else "no negative criterion present")


def check_q12_size(bundle, limit: int = 18) -> CheckResult:
    n = len(_criteria(bundle))
    ok = n <= limit
    return CheckResult(id="Q12", name="Economical size", stage=Stage.SEMANTIC,
                       severity=Severity.WARN, passed=ok,
                       detail="" if ok else f"{n} criteria (>{limit}) — routes to human review")


DETERMINISTIC_CHECKS = [check_q09, check_q12_size]
```

- [ ] **Step 4: Wire into `run_semantic` in `autoqc/agent/engine.py`**

Add import near the other agent imports:

```python
from autoqc.agent.deterministic import DETERMINISTIC_CHECKS
```

Change `run_semantic` to append the deterministic results:

```python
def run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None):
    results = [run_check(c, bundle, client, ctx, k=k, votes_log=votes_log) for c in checks]
    results += [fn(bundle) for fn in DETERMINISTIC_CHECKS]
    return results
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_deterministic_checks.py tests/test_engine_run.py -v`
Expected: PASS

- [ ] **Step 6: Whole suite + commit**

Run: `python3 -m pytest -q` (expect green)

```bash
git add autoqc/agent/deterministic.py autoqc/agent/engine.py tests/test_deterministic_checks.py tests/test_engine_run.py
git commit -m "feat(autoqc): deterministic Q09 (negative present) + Q12 (size) checks"
```

---

### Task 2: per-rubric `unit_mode` extension

**Files:**
- Modify: `autoqc/agent/checks.py` (`SemanticCheck.unit_mode`; context builders)
- Modify: `autoqc/agent/engine.py` (`run_check` rubric-mode path)
- Test: `tests/test_rubric_mode.py`

**Interfaces:**
- `SemanticCheck` gains `unit_mode: str = "criterion"`.
- `proposer_context`/`adversary_context`: when `check.unit_mode == "rubric"`, the context shows the prompt + the FULL rubric (all criteria, labeled positive/negative) and instructs the agent to submit exactly ONE finding with `criterion_id="rubric"` judging the whole rubric for this check. (Criterion mode is unchanged.)
- `run_check`: when `check.unit_mode == "rubric"`, `allowed_ids = {"rubric"}`; still K proposer passes + 1 adversary pass; `aggregate`/`adjudicate` operate over the single `"rubric"` key; rolled up to one `CheckResult`. If `check.scope(items)` is empty (nothing to judge, e.g. Q08 with no negatives), return a passing `CheckResult` (as today).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rubric_mode.py
import types
from pathlib import Path
from autoqc.model import Severity, Stage
from autoqc.agent.checks import SemanticCheck, proposer_context, adversary_context
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


def test_run_check_rubric_mode_empty_scope_passes(tmp_path):
    empty_neg_check = SemanticCheck(id="Q08", name="disc neg", severity=Severity.WARN,
                                    scope=lambda items: [], guidance="g", unit_mode="rubric")
    r = run_check(empty_neg_check, _bundle(), FakeLLMClient(lambda m, t: {"tool_calls": []}), _ctx(tmp_path), k=3)
    assert r.passed is True and r.needs_human is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rubric_mode.py -v`
Expected: FAIL (`SemanticCheck.__init__() got an unexpected keyword argument 'unit_mode'`)

- [ ] **Step 3: Edit `autoqc/agent/checks.py`**

Add `unit_mode` to the dataclass (after `guidance`):

```python
    unit_mode: str = "criterion"
```

Replace `proposer_context` and `adversary_context` with unit-mode-aware versions:

```python
def _full_rubric_block(bundle) -> str:
    items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
    lines = []
    for it in items:
        if not isinstance(it, dict):
            continue
        typ = "neg" if "negative" in str(it.get("annotations", {}).get("type", "")) else "pos"
        lines.append(f"- [{typ}] criterion_id={it.get('id')}  title={it.get('title', '')!r}")
    return "\n".join(lines)


def proposer_context(bundle, check, criteria) -> str:
    if check.unit_mode == "rubric":
        return (f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
                f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
                f"The full rubric:\n{_full_rubric_block(bundle)}\n\n"
                "Judge the rubric AS A WHOLE for this check. Submit EXACTLY ONE finding with "
                f"check_id={check.id} and criterion_id=\"rubric\": passed=true if the rubric "
                "SATISFIES the check, passed=false if it VIOLATES it; evidence must justify it.")
    return (f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            f"Judge each of these criteria (submit one finding per criterion, check_id={check.id}):\n"
            f"{_criteria_block(criteria)}\n\n"
            "passed=true if the criterion SATISFIES the check, passed=false if it VIOLATES it.")


def adversary_context(bundle, check, criteria, agg) -> str:
    if check.unit_mode == "rubric":
        v = agg.get("rubric", {})
        verdict = "PASS" if v.get("passed") else "FAIL (reject)"
        return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
                f"The full rubric:\n{_full_rubric_block(bundle)}\n\n"
                f"A prior review judged the whole rubric: {verdict}. Challenge that per the rules "
                f"above, then submit EXACTLY ONE finding (check_id={check.id}, criterion_id=\"rubric\").")
    lines = []
    for c in criteria:
        v = agg.get(c["id"], {})
        verdict = "PASS" if v.get("passed") else "FAIL (reject)"
        lines.append(f"- criterion_id={c['id']}  prior_verdict={verdict}  title={c.get('title', '')!r}")
    return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            "A prior review produced these verdicts. Challenge them per the rules above, then "
            f"submit_findings with your verdict per criterion (check_id={check.id}):\n"
            + "\n".join(lines))
```

- [ ] **Step 4: Edit `run_check` in `autoqc/agent/engine.py`**

Replace the `allowed`/units setup so rubric mode uses the `"rubric"` key. After computing `criteria = check.scope(items)` and the empty-scope early return, add:

```python
    allowed = {"rubric"} if getattr(check, "unit_mode", "criterion") == "rubric" else {c["id"] for c in criteria}
```

(Use `allowed` for `_own(...)` and `aggregate(...)` exactly as before — the rest of `run_check` is unchanged, since `aggregate`/`adjudicate` already key on whatever ids are in `allowed`.)

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_rubric_mode.py tests/test_agent_checks.py tests/test_engine_run.py -v`
Expected: PASS (rubric-mode tests + the existing criterion-mode tests still green)

- [ ] **Step 6: Whole suite + commit**

Run: `python3 -m pytest -q`

```bash
git add autoqc/agent/checks.py autoqc/agent/engine.py tests/test_rubric_mode.py
git commit -m "feat(autoqc): per-rubric unit_mode for whole-rubric semantic checks"
```

---

### Task 3: per-criterion checks Q01, Q02, Q05

**Files:**
- Modify: `autoqc/agent/checks.py` (add Q01, Q02, Q05; extend `SEMANTIC_CHECKS`)
- Test: `tests/test_checks_percriterion.py`

**Interfaces:**
- Adds `Q01` (Atomicity, REJECT, scope=all criteria), `Q02` (Binary judgeability, REJECT, scope=all criteria), `Q05` (No unrequested scope, REJECT, scope=positives). All `unit_mode="criterion"` (default). Appended to `SEMANTIC_CHECKS` after Q03.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_checks_percriterion.py
from autoqc.model import Severity
from autoqc.agent.checks import Q01, Q02, Q05, SEMANTIC_CHECKS, _positives


def _pos(i):
    return {"id": i, "title": "1.1: X", "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(i):
    return {"id": i, "title": "2.1: Claims X", "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_ids_and_severities():
    assert (Q01.id, Q01.severity) == ("Q01", Severity.REJECT)
    assert (Q02.id, Q02.severity) == ("Q02", Severity.REJECT)
    assert (Q05.id, Q05.severity) == ("Q05", Severity.REJECT)
    assert all(c in SEMANTIC_CHECKS for c in (Q01, Q02, Q05))


def test_q01_q02_scope_all_criteria():
    items = [_pos("a"), _neg("b")]
    assert {c["id"] for c in Q01.scope(items)} == {"a", "b"}
    assert {c["id"] for c in Q02.scope(items)} == {"a", "b"}


def test_q05_scope_positives_only():
    assert [c["id"] for c in Q05.scope([_pos("a"), _neg("b")])] == ["a"]
    assert [c["id"] for c in _positives([_pos("a"), _neg("b")])] == ["a"]


def test_guidance_is_specific():
    assert "independently" in Q01.guidance.lower()          # atomicity
    assert "thorough" in Q02.guidance.lower() or "subjective" in Q02.guidance.lower()  # binary
    assert "prompt" in Q05.guidance.lower()                 # unrequested scope refs the prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_checks_percriterion.py -v`
Expected: FAIL (`cannot import name 'Q01'`)

- [ ] **Step 3: Add to `autoqc/agent/checks.py`** (after Q03, before `SEMANTIC_CHECKS`)

```python
def _positives(items):
    return [it for it in (items or [])
            if _is_dict_crit(it) and "positive" in str(
                it.get("annotations", {}).get("type", "") if isinstance(it.get("annotations"), dict) else "")]


Q01 = SemanticCheck(
    id="Q01", name="Atomicity", severity=Severity.REJECT, scope=_all_criteria,
    guidance=("A criterion VIOLATES this if it bundles TWO OR MORE independently gradable facts "
              "(each could be independently true or false and separately meaningful). A cause and "
              "its direct effect may stay together only when partial satisfaction would be "
              "meaningless. A coherent cluster answering one sub-question is acceptable. passed=true "
              "if the criterion tests one gradable proposition."))

Q02 = SemanticCheck(
    id="Q02", name="Binary judgeability", severity=Severity.REJECT, scope=_all_criteria,
    guidance=("A criterion VIOLATES this if a grader cannot decide it met/not-met from an answer's "
              "text alone: subjective quality words ('thorough', 'clearly', 'well', 'adequately', "
              "'good'), or an undefined completeness claim. Naming specific facts/values/paths is "
              "fine. passed=true if the criterion is objectively decidable."))

Q05 = SemanticCheck(
    id="Q05", name="No unrequested scope", severity=Severity.REJECT, scope=_positives,
    guidance=("A POSITIVE criterion VIOLATES this if it grades a fact the prompt neither requests "
              "nor requires as an indispensable causal link to answer it. Use the task prompt "
              "(shown above) to judge. passed=true if the criterion is requested by the prompt or is "
              "a necessary causal step toward the requested answer."))
```

Then extend the registry:

```python
SEMANTIC_CHECKS = [Q07, Q03, Q01, Q02, Q05]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_checks_percriterion.py tests/test_agent_checks.py -v`
Expected: PASS

- [ ] **Step 5: Whole suite + commit**

Run: `python3 -m pytest -q`

```bash
git add autoqc/agent/checks.py tests/test_checks_percriterion.py
git commit -m "feat(autoqc): per-criterion checks Q01 (atomicity), Q02 (judgeability), Q05 (scope)"
```

---

### Task 4: per-rubric checks Q04, Q08, Q10, Q11 + calibration generalization

**Files:**
- Modify: `autoqc/agent/checks.py` (add Q04, Q08, Q10, Q11 with `unit_mode="rubric"`; extend `SEMANTIC_CHECKS`)
- Modify: `autoqc/calibrate.py` (generalize `CHECKS`)
- Test: `tests/test_checks_perrubric.py`, `tests/test_calibrate_metrics.py` (extend)

**Interfaces:**
- Adds `Q04` (Prompt coverage, REJECT), `Q08` (Discriminating negatives, WARN), `Q10` (Empirical result graded, WARN), `Q11` (Not lookup-dominated, WARN) — all `unit_mode="rubric"`, `scope=_all_criteria` (used only as the non-empty gate; the rubric-mode context shows the whole rubric + prompt). Appended to `SEMANTIC_CHECKS`.
- `calibrate.CHECKS` becomes the full list of agent+deterministic check ids so `score_case`/`summarize` measure false-reject across every check.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_checks_perrubric.py
from autoqc.model import Severity
from autoqc.agent.checks import Q04, Q08, Q10, Q11, SEMANTIC_CHECKS


def test_ids_severities_and_rubric_mode():
    assert (Q04.id, Q04.severity, Q04.unit_mode) == ("Q04", Severity.REJECT, "rubric")
    assert (Q08.id, Q08.severity, Q08.unit_mode) == ("Q08", Severity.WARN, "rubric")
    assert (Q10.id, Q10.severity, Q10.unit_mode) == ("Q10", Severity.WARN, "rubric")
    assert (Q11.id, Q11.severity, Q11.unit_mode) == ("Q11", Severity.WARN, "rubric")
    assert all(c in SEMANTIC_CHECKS for c in (Q04, Q08, Q10, Q11))


def test_registry_full_set():
    ids = [c.id for c in SEMANTIC_CHECKS]
    assert set(ids) == {"Q01", "Q02", "Q03", "Q04", "Q05", "Q07", "Q08", "Q10", "Q11"}
    assert len(ids) == len(set(ids))  # no dupes


def test_guidance_specific():
    assert "obligation" in Q04.guidance.lower() or "cover" in Q04.guidance.lower()
    assert "inverse" in Q08.guidance.lower() or "redundant" in Q08.guidance.lower()
    assert "observed" in Q10.guidance.lower() or "empirical" in Q10.guidance.lower()
    assert "lookup" in Q11.guidance.lower() or "trivia" in Q11.guidance.lower()
```

Extend `tests/test_calibrate_metrics.py` (append):

```python
def test_checks_covers_full_set():
    from autoqc.calibrate import CHECKS
    assert set(CHECKS) >= {"Q01", "Q02", "Q03", "Q04", "Q05", "Q07", "Q08", "Q09", "Q10", "Q11", "Q12"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_checks_perrubric.py -v`
Expected: FAIL (`cannot import name 'Q04'`)

- [ ] **Step 3: Add to `autoqc/agent/checks.py`** (after Q05)

```python
Q04 = SemanticCheck(
    id="Q04", name="Prompt coverage", severity=Severity.REJECT, scope=_all_criteria, unit_mode="rubric",
    guidance=("The rubric VIOLATES this if any explicit obligation in the task prompt has NO positive "
              "criterion grading it (especially a central runtime result buried or absent). passed=true "
              "if every prompt obligation maps to at least one positive criterion. Evidence: name any "
              "uncovered obligation."))

Q08 = SemanticCheck(
    id="Q08", name="Discriminating negatives", severity=Severity.WARN, scope=_all_criteria, unit_mode="rubric",
    guidance=("Grading is all-or-nothing: a negative adds power only if NO positive already forces the "
              "correct version of the same fact. The rubric VIOLATES this (warn) if its negatives are "
              "mostly REDUNDANT inverses of positives. passed=true if at least one negative is "
              "discriminating — it catches a plausible wrong answer that no positive already forces. "
              "Evidence: cite a redundant negative, or the discriminating one."))

Q10 = SemanticCheck(
    id="Q10", name="Empirical result graded", severity=Severity.WARN, scope=_all_criteria, unit_mode="rubric",
    guidance=("If the task prompt requires running the software, the rubric VIOLATES this (warn) if NO "
              "positive grades the SPECIFIC observed result (a value, comparison, state transition, "
              "generated artifact, or error). A criterion that merely says the answer 'ran' or "
              "'reported empirical evidence' is insufficient. passed=true if a specific empirical "
              "result is graded, OR the prompt does not require running anything."))

Q11 = SemanticCheck(
    id="Q11", name="Not lookup-dominated", severity=Severity.WARN, scope=_all_criteria, unit_mode="rubric",
    guidance=("The rubric VIOLATES this (warn) if it is mostly trivia lookups (ports, constants, default "
              "values, file lists, single line numbers) with little causal or synthesis content. "
              "passed=true if the criteria as a whole require meaningful reasoning, not just lookups."))
```

Extend the registry:

```python
SEMANTIC_CHECKS = [Q07, Q03, Q01, Q02, Q05, Q04, Q08, Q10, Q11]
```

- [ ] **Step 4: Generalize `CHECKS` in `autoqc/calibrate.py`**

Replace `CHECKS = ("Q07", "Q03")` with the full set:

```python
CHECKS = ("Q01", "Q02", "Q03", "Q04", "Q05", "Q07", "Q08", "Q09", "Q10", "Q11", "Q12")
```

(No other change needed: `score_case` already iterates `CHECKS`; a check absent from a given `results` list scores `got_fail=False`, which is correct.)

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_checks_perrubric.py tests/test_calibrate_metrics.py tests/test_agent_checks.py -v`
Expected: PASS

- [ ] **Step 6: Whole suite + commit**

Run: `python3 -m pytest -q`

```bash
git add autoqc/agent/checks.py autoqc/calibrate.py tests/test_checks_perrubric.py tests/test_calibrate_metrics.py
git commit -m "feat(autoqc): per-rubric checks Q04/Q08/Q10/Q11 + full-set calibration CHECKS"
```

---

## Self-Review

**Spec coverage:** All nine remaining checks — Q09/Q12(size) deterministic (Task 1), the per-rubric framework (Task 2), Q01/Q02/Q05 (Task 3), Q04/Q08/Q10/Q11 (Task 4). Q12 semantic-redundancy deferred (noted). Severities match spec §5. ✓

**Placeholder scan:** none.

**Type consistency:** `unit_mode` added in Task 2, read by `proposer_context`/`adversary_context`/`run_check`; Task 3/4 checks set it. `_positives`/`_all_criteria` scopes consistent. `SEMANTIC_CHECKS` grown additively (final set of 9 agent checks). `DETERMINISTIC_CHECKS` consumed by `run_semantic`. `CHECKS` generalized; `score_case` unchanged (iterates `CHECKS`, absent-check → got_fail False). `FakeLLMClient` responder `(messages, tools)` throughout.

---

## Live calibration (operator-run, after the tasks pass)

- [ ] **Re-run the calibration over the full check set** using the existing `scripts/calibrate.py` on 2 internal bundles (b01, b03):
  `python3 scripts/calibrate.py <b01_dir> <b03_dir>`
- [ ] **Report:** the per-case table (now covering all checks via the generalized `CHECKS`), plus `recall` (still driven by the Q07/Q03 seeds), `false_reject_rate` (now the KEY number — do any of the 9 new checks wrongly fail the CLEAN bundles?), and `detection_accuracy`. Interpret: a good result is `false_reject_rate ≈ 0` on the clean cases across all checks. Any check that false-fires on a clean exemplar rubric is a prompt to tune (guidance too aggressive) — that's expected v1 iteration; note which checks fire and why.

## Notes for the next iterations

- Q12 semantic non-redundancy (two criteria grade the same fact) — deferred.
- Per-check seeds (q01_bad … q11_bad) so recall is measured for every check, not just Q07/Q03.
- Prompt tuning from the calibration (guidance strings are v1).
- Q06 factual/run_bash/container — the remaining non-text check.

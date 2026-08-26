# AutoQC Calibration Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether AutoQC's semantic checks actually work — build a labeled corpus (a clean bundle plus seeded single-defect variants), pure metric functions (per-check recall, false-reject rate, verdict accuracy), and a runner — then a **live calibration run** against GLM 5.2 that prints the metrics.

**Architecture:** `seed.py` gains `seed_wildcard` (a Q03 defect) alongside the existing `seed_bad_negative` (a Q07 defect). `calibrate.py` defines a `Case` (a bundle dir + its expected per-check failures), `build_corpus` (writes labeled clean + seeded variants of a base bundle), `score_case`/`summarize` (pure metrics over `CheckResult`s), and `run_corpus` (runs AutoQC per case). `scripts/calibrate.py` is the operator entry point that builds a corpus from real internal samples and runs it live. Everything except the live script is `FakeLLMClient`-tested.

**Tech Stack:** Python 3.11+ (runs on 3.10). `pytest`. No third-party runtime deps. Unit tests use `FakeLLMClient` or fabricated `CheckResult`s.

**Spec:** `docs/superpowers/specs/2026-08-25-autoqc-execution-harness-design.md` §8 (calibration & acceptance) and the testing strategy therein.

## Global Constraints

- Run every test with `python3 -m pytest`. **All unit tests are offline** (fabricated `CheckResult`s or `FakeLLMClient`); the live calibration is the operator script only.
- Reuse `seed_bad_negative` (seed.py), `run` (cli.py), `CheckResult`/`Verdict`/`Stage` (model), `load_bundle` — unchanged.
- Score at **check-verdict granularity** (did the expected check fail?), not by parsing criterion ids out of `detail` — robust and sufficient for recall/false-reject.
- `build_corpus` must never mutate the base bundle; it copies to a work dir.
- `if git commit is blocked, leave files staged and report DONE_WITH_CONCERNS` — the controller commits.

---

## File Structure

- `autoqc/seed.py` — MODIFY: add `seed_wildcard`.
- `autoqc/calibrate.py` — `Case`, `build_corpus`, `score_case`, `summarize`, `run_corpus`.
- `scripts/calibrate.py` — operator-run live calibration.
- `tests/` — one module per new/changed source.

---

### Task 1: `seed_wildcard` + labeled corpus builder

**Files:**
- Modify: `autoqc/seed.py`
- Create: `autoqc/calibrate.py`
- Test: `tests/test_seed_wildcard.py`, `tests/test_calibrate_corpus.py`

**Interfaces:**
- Produces:
  - `seed_wildcard(items) -> tuple[list[dict], str | None]` (seed.py) — deep-copies, appends `", or similar"` to the first positive criterion's title (a Q03 escape-hatch defect), returns `(mutated, mutated_id)`; `(mutated, None)` if no positive. Does not mutate input.
  - `Case(name: str, bundle_dir: Path, expected_flags: dict[str, set[str]])` (calibrate.py) — `expected_flags` maps a check id to the set of criterion ids expected to be flagged; empty dict = a clean case (nothing should fail). `Case.expected_not_sound` property = `bool(expected_flags)`.
  - `build_corpus(base_bundle_dir, work_dir) -> list[Case]` — writes three labeled variants into `work_dir`: `clean` (copy, `expected_flags={}`), `q07_bad` (copy with `seed_bad_negative` applied to `tests/rubrics.json`, `expected_flags={"Q07": {seeded_id}}`), `q03_bad` (copy with `seed_wildcard`, `expected_flags={"Q03": {seeded_id}}`). Returns the three `Case`s.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_seed_wildcard.py
import copy
from autoqc.seed import seed_wildcard


def _pos(id_, title="1.1: States the port"):
    return {"id": id_, "title": title,
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(id_):
    return {"id": id_, "title": "2.1: Claims that X",
            "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_seed_wildcard_appends_hatch_to_first_positive():
    items = [_neg("n"), _pos("p", "1.1: States the port")]
    original = copy.deepcopy(items)
    mutated, changed = seed_wildcard(items)
    assert changed == "p"
    assert items == original  # input untouched
    pos = [it for it in mutated if "positive" in it["annotations"]["type"]][0]
    assert "or similar" in pos["title"]


def test_seed_wildcard_none_when_no_positive():
    mutated, changed = seed_wildcard([_neg("n")])
    assert changed is None
```

```python
# tests/test_calibrate_corpus.py
import json
from pathlib import Path
from autoqc.calibrate import Case, build_corpus


def _write_bundle(root: Path):
    (root / "tests").mkdir(parents=True)
    (root / "tests/rubrics.json").write_text(json.dumps([
        {"id": "p" * 32, "title": "1.1: States the port",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}},
        {"id": "n" * 32, "title": "2.1: Claims that X fails",
         "annotations": {"type": "negative hli verifier", "importance": "must have"}}]))
    (root / "tests/prompt.txt").write_text("q")
    (root / "solution").mkdir(); (root / "solution/answer.txt").write_text("a")
    (root / "environment").mkdir(); (root / "environment/Dockerfile").write_text("FROM x")
    (root / "task.toml").write_text('[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))


def test_case_expected_not_sound():
    assert Case("c", Path("."), {}).expected_not_sound is False
    assert Case("b", Path("."), {"Q07": {"x"}}).expected_not_sound is True


def test_build_corpus_writes_three_labeled_variants(tmp_path):
    base = tmp_path / "base"; base.mkdir(); _write_bundle(base)
    cases = build_corpus(base, tmp_path / "work")
    by = {c.name: c for c in cases}
    assert set(by) == {"clean", "q07_bad", "q03_bad"}
    # base is untouched
    assert "or similar" not in (base / "tests/rubrics.json").read_text()
    assert "Does not claim" not in (base / "tests/rubrics.json").read_text()
    # clean is a faithful copy with no defect and no expected flags
    assert by["clean"].expected_flags == {}
    assert (by["clean"].bundle_dir / "tests/rubrics.json").exists()
    # q07_bad has the seeded negative + labels Q07
    assert "Does not claim" in (by["q07_bad"].bundle_dir / "tests/rubrics.json").read_text()
    assert set(by["q07_bad"].expected_flags) == {"Q07"}
    # q03_bad has the wildcard + labels Q03
    assert "or similar" in (by["q03_bad"].bundle_dir / "tests/rubrics.json").read_text()
    assert set(by["q03_bad"].expected_flags) == {"Q03"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_seed_wildcard.py tests/test_calibrate_corpus.py -v`
Expected: FAIL (`cannot import name 'seed_wildcard'` / `No module named 'autoqc.calibrate'`)

- [ ] **Step 3: Add `seed_wildcard` to `autoqc/seed.py`**

```python
def seed_wildcard(items):
    """Inject a Q03 defect: append an open escape hatch to the first positive
    criterion's title. Returns (mutated_items, mutated_id). Does not mutate input."""
    import copy
    mutated = copy.deepcopy(items)
    for it in mutated:
        if not isinstance(it, dict):
            continue
        ann = it.get("annotations")
        typ = ann.get("type", "") if isinstance(ann, dict) else ""
        if "positive" in str(typ):
            title = str(it.get("title", ""))
            it["title"] = title.rstrip(".") + ", or similar"
            return mutated, str(it.get("id")) if it.get("id") is not None else None
    return mutated, None
```

- [ ] **Step 4: Write `autoqc/calibrate.py` (corpus part)**

```python
from __future__ import annotations
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from autoqc.seed import seed_bad_negative, seed_wildcard


@dataclass
class Case:
    name: str
    bundle_dir: Path
    expected_flags: dict = field(default_factory=dict)  # check_id -> set(criterion_ids)

    @property
    def expected_not_sound(self) -> bool:
        return bool(self.expected_flags)


def _copy_bundle(base: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(base, dest)
    return dest


def _seed_into(bundle_dir: Path, seed_fn):
    rf = bundle_dir / "tests" / "rubrics.json"
    items = json.loads(rf.read_text())
    mutated, changed = seed_fn(items)
    rf.write_text(json.dumps(mutated, indent=2))
    return changed


def build_corpus(base_bundle_dir, work_dir) -> list[Case]:
    base = Path(base_bundle_dir)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    clean = _copy_bundle(base, work / "clean")
    q07 = _copy_bundle(base, work / "q07_bad")
    q07_id = _seed_into(q07, seed_bad_negative)
    q03 = _copy_bundle(base, work / "q03_bad")
    q03_id = _seed_into(q03, seed_wildcard)

    return [
        Case("clean", clean, {}),
        Case("q07_bad", q07, {"Q07": {q07_id}} if q07_id else {"Q07": set()}),
        Case("q03_bad", q03, {"Q03": {q03_id}} if q03_id else {"Q03": set()}),
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_seed_wildcard.py tests/test_calibrate_corpus.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add autoqc/seed.py autoqc/calibrate.py tests/test_seed_wildcard.py tests/test_calibrate_corpus.py
git commit -m "feat(autoqc): seed_wildcard + labeled calibration corpus builder"
```

---

### Task 2: metrics + corpus runner

**Files:**
- Modify: `autoqc/calibrate.py` (add `score_case`, `summarize`, `run_corpus`)
- Test: `tests/test_calibrate_metrics.py`

**Interfaces:**
- Consumes: `Case` (Task 1); `run` (cli); `CheckResult`/`Stage`/`Verdict` (model).
- Produces:
  - `score_case(case, results) -> dict` — from the full `results` list, for each of `("Q07", "Q03")` compute `expected_fail = check in case.expected_flags`, `got_fail = the CheckResult for that check exists and passed is False`, `correct = expected_fail == got_fail`. Also `verdict_correct` = whether the overall not-soundness matches `case.expected_not_sound` (a case is "not sound" if any expected check failed; compare to whether any reject-severity semantic check failed). Returns `{"name", "checks": {cid: {...}}, "verdict_correct": bool}`.
  - `summarize(scored) -> dict` — over the scored cases: `recall` = of all `(case, check)` pairs where `expected_fail`, the fraction with `got_fail`; `false_reject_rate` = of all `(case, check)` pairs where NOT `expected_fail`, the fraction with `got_fail` (a clean check wrongly failed); `verdict_accuracy` = fraction of cases with `verdict_correct`.
  - `run_corpus(cases, client, out_root, k=3) -> list[dict]` — for each case, calls `run(case.bundle_dir, out_root/case.name, llm=client, k=k)`, reads back the semantic `CheckResult`s from the written `review_record.json`, `score_case`s them, and returns the scored list (each augmented with the verdict).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate_metrics.py
import json
from pathlib import Path
from autoqc.model import CheckResult, Stage, Severity
from autoqc.calibrate import Case, score_case, summarize, run_corpus
from autoqc.llm import FakeLLMClient


def _cr(cid, passed):
    return CheckResult(id=cid, name=cid, stage=Stage.SEMANTIC, severity=Severity.REJECT, passed=passed)


def test_score_case_clean_all_pass():
    case = Case("clean", Path("."), {})
    s = score_case(case, [_cr("Q07", True), _cr("Q03", True)])
    assert s["checks"]["Q07"]["correct"] and s["checks"]["Q03"]["correct"]
    assert s["verdict_correct"] is True


def test_score_case_q07_caught():
    case = Case("q07_bad", Path("."), {"Q07": {"x"}})
    s = score_case(case, [_cr("Q07", False), _cr("Q03", True)])
    assert s["checks"]["Q07"]["got_fail"] is True and s["checks"]["Q07"]["correct"] is True
    assert s["checks"]["Q03"]["got_fail"] is False and s["checks"]["Q03"]["correct"] is True
    assert s["verdict_correct"] is True


def test_score_case_missed_defect():
    case = Case("q07_bad", Path("."), {"Q07": {"x"}})
    s = score_case(case, [_cr("Q07", True), _cr("Q03", True)])  # missed it
    assert s["checks"]["Q07"]["correct"] is False
    assert s["verdict_correct"] is False


def test_score_case_false_reject_on_clean():
    case = Case("clean", Path("."), {})
    s = score_case(case, [_cr("Q07", True), _cr("Q03", False)])  # wrongly failed Q03
    assert s["checks"]["Q03"]["correct"] is False
    assert s["verdict_correct"] is False


def test_summarize_metrics():
    scored = [
        {"name": "clean", "checks": {"Q07": {"expected_fail": False, "got_fail": False, "correct": True},
                                     "Q03": {"expected_fail": False, "got_fail": True, "correct": False}},
         "verdict_correct": False},
        {"name": "q07_bad", "checks": {"Q07": {"expected_fail": True, "got_fail": True, "correct": True},
                                       "Q03": {"expected_fail": False, "got_fail": False, "correct": True}},
         "verdict_correct": True},
    ]
    m = summarize(scored)
    assert m["recall"] == 1.0            # 1 expected-fail, caught
    assert m["false_reject_rate"] == 1 / 3  # 3 clean checks, 1 wrongly failed
    assert m["verdict_accuracy"] == 0.5


def test_run_corpus_reads_records_and_scores(tmp_path):
    # a bundle whose Q07 negative is bad; fake client fails Q07, passes Q03
    import json as _j
    b = tmp_path / "task"; (b / "tests").mkdir(parents=True)
    (b / "tests/rubrics.json").write_text(_j.dumps([
        {"id": "n" * 32, "title": "2.1: Does not claim that X",
         "annotations": {"type": "negative hli verifier", "importance": "must have"}}]))
    (b / "tests/prompt.txt").write_text("q")
    (b / "solution").mkdir(); (b / "solution/answer.txt").write_text("a")
    (b / "environment").mkdir(); (b / "environment/Dockerfile").write_text("FROM x")
    (b / "task.toml").write_text('[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))

    def responder(messages, tools):
        import re
        txt = " ".join(m["content"] for m in messages)
        cid = "Q07" if "Q07" in txt else "Q03"
        ids = list(dict.fromkeys(re.findall(r"criterion_id=(\w+)", txt)))
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": cid, "criterion_id": i, "passed": cid != "Q07", "evidence": ["x"]} for i in ids]}}]}

    case = Case("q07_bad", b, {"Q07": {"n" * 32}})
    scored = run_corpus([case], FakeLLMClient(responder), tmp_path / "out", k=1)
    assert scored[0]["checks"]["Q07"]["got_fail"] is True
    assert scored[0]["checks"]["Q07"]["correct"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_calibrate_metrics.py -v`
Expected: FAIL (`cannot import name 'score_case'`)

- [ ] **Step 3: Add to `autoqc/calibrate.py`**

```python
# add near the top with the other imports:
from autoqc.cli import run

CHECKS = ("Q07", "Q03")


def score_case(case, results) -> dict:
    by = {r.id: r for r in results}
    checks = {}
    any_expected_fail = bool(case.expected_flags)
    any_got_fail = False
    for cid in CHECKS:
        expected_fail = cid in case.expected_flags
        got_fail = (cid in by) and (by[cid].passed is False)
        any_got_fail = any_got_fail or got_fail
        checks[cid] = {"expected_fail": expected_fail, "got_fail": got_fail,
                       "correct": expected_fail == got_fail}
    return {"name": case.name, "checks": checks,
            "verdict_correct": any_got_fail == any_expected_fail}


def summarize(scored) -> dict:
    exp_total = exp_caught = clean_total = clean_failed = 0
    for s in scored:
        for cid, c in s["checks"].items():
            if c["expected_fail"]:
                exp_total += 1
                exp_caught += 1 if c["got_fail"] else 0
            else:
                clean_total += 1
                clean_failed += 1 if c["got_fail"] else 0
    vc = sum(1 for s in scored if s["verdict_correct"])
    return {
        "recall": (exp_caught / exp_total) if exp_total else None,
        "false_reject_rate": (clean_failed / clean_total) if clean_total else None,
        "verdict_accuracy": (vc / len(scored)) if scored else None,
        "n_cases": len(scored),
    }


def run_corpus(cases, client, out_root, k=3) -> list[dict]:
    from autoqc.model import CheckResult, Stage, Severity
    out_root = Path(out_root)
    scored = []
    for case in cases:
        verdict = run(case.bundle_dir, out_root / case.name, llm=client, k=k)
        rec = json.loads((out_root / case.name / "review_record.json").read_text())
        results = [CheckResult(id=r["id"], name=r["name"],
                               stage=Stage(r["stage"]), severity=Severity(r["severity"]),
                               passed=r["passed"], needs_human=r["needs_human"])
                   for r in rec["results"] if r["stage"] == "semantic"]
        s = score_case(case, results)
        s["verdict"] = verdict.value
        scored.append(s)
    return scored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_calibrate_metrics.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add autoqc/calibrate.py tests/test_calibrate_metrics.py
git commit -m "feat(autoqc): calibration metrics (recall, false-reject, verdict accuracy) + run_corpus"
```

---

## Self-Review

**Spec coverage (calibration slice):**
- Seeded-defect corpus (Q07 + Q03) + clean control → Task 1. ✓
- Metrics: per-check recall, false-reject rate, verdict accuracy → Task 2. ✓
- Corpus runner reading back the review record → Task 2. ✓
- Live calibration → operator section. ✓
- Deferred (correctly absent): boundary-guard cases beyond the clean control, the full 134-task corpus, stability/flip-rate ablation, misattribution metric, human-labeled oracle — all next iterations.

**Placeholder scan:** none.

**Type consistency:** `Case(name, bundle_dir, expected_flags)` + `.expected_not_sound` (Task 1) consumed by `score_case`/`run_corpus` (Task 2). `build_corpus -> list[Case]` (Task 1) feeds `run_corpus` (Task 2). `score_case -> dict` shape consumed by `summarize`. `run_corpus(cases, client, out_root, k)` calls the existing `run(bundle_dir, out_dir, llm, k)`. `FakeLLMClient` responder `(messages, tools)` throughout.

---

## Live calibration run (operator-run, after the tasks pass)

- [ ] **Create `scripts/calibrate.py`:**

```python
"""Live calibration: build a labeled corpus from a base bundle (clean + seeded
Q07/Q03 variants), run AutoQC against the real gateway, print per-case scores +
summary metrics. Operator-run (real model calls)."""
import os
import sys
from pathlib import Path
sys.path.insert(0, ".")
for line in open(".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)
from autoqc.calibrate import build_corpus, run_corpus, summarize
from autoqc.llm import GatewayLLMClient

bases = sys.argv[1:]  # one or more base bundle dirs
client = GatewayLLMClient()
assert client.available(), ".env missing EVAL_API_KEY / EVAL_BASE_URL"
work = Path("/tmp/autoqc_calib")
cases = []
for i, b in enumerate(bases):
    cases += build_corpus(Path(b), work / f"bundle{i}")
print(f"model={client.model}  cases={len(cases)}")
scored = run_corpus(cases, client, work / "out", k=3)
print(f"\n{'case':10} {'verdict':18} {'Q07':>8} {'Q03':>8}")
for s in scored:
    q07 = "OK" if s["checks"]["Q07"]["correct"] else "WRONG"
    q03 = "OK" if s["checks"]["Q03"]["correct"] else "WRONG"
    print(f"{s['name']:10} {s['verdict']:18} {q07:>8} {q03:>8}")
m = summarize(scored)
print(f"\nrecall={m['recall']}  false_reject_rate={m['false_reject_rate']}  "
      f"verdict_accuracy={m['verdict_accuracy']}  n={m['n_cases']}")
```

- [ ] **Run it against 1–2 internal samples** and report the per-case table + summary metrics (recall = did it catch the seeded Q07/Q03 defects; false_reject_rate = did it wrongly fail clean checks; verdict_accuracy). Interpret: a good result is recall≈1.0, false_reject_rate≈0. Use it to decide whether to expand the corpus, add the remaining checks, or tune prompts/K.

---

## Notes for the next iterations

- Expand the corpus: multiple base bundles, boundary guards (a legitimate interchangeable `e.g.` that must NOT flag), more seed types.
- Stability/flip-rate: run each case N times, measure verdict variance (the ensemble ablation from the spec).
- Misattribution metric: did a seeded defect get flagged by the wrong check?
- Add the remaining text checks (Q01/Q02/Q04/Q05/Q08–Q12), then Q06 factual/run_bash — each expands the corpus.

# AutoQC Foundation (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic foundation of AutoQC — load a Harbor task bundle, run the Stage 1 structural checks (S01–S08), compute a verdict, and emit both a structured review record and a markdown rework report — with no LLM dependency.

**Architecture:** A small Python package `autoqc`. `bundle.py` tolerantly loads a bundle (parsing what it can, recording errors instead of raising). `structural.py` runs eight pure-function checks that each return a `CheckResult`. `verdict.py` aggregates results into one of three verdicts. `report.py` renders the structured JSON record and the human markdown report. `cli.py` wires it end-to-end. Everything downstream (semantic engine, Q06) will produce the same `CheckResult` type, so this plan locks that interface.

**Tech Stack:** Python 3.11+ (stdlib `tomllib`, `json`, `re`, `dataclasses`, `pathlib`), `pytest` for tests. No third-party runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-08-25-autoqc-execution-harness-design.md` (and its companion `2026-08-25-autoqc-quality-rubric-design.md`, which defines S01–S08 and the §6 verdict).

## Global Constraints

- Rubric item schema (from the rubric spec §5): each item is `{ "id": <32 hex>, "title": "N.x: ...", "annotations": { "type": "positive hli verifier" | "negative hli verifier", "importance": "must have" } }`.
- ID format: `^[0-9a-f]{32}$`.
- Title numbering: `1.x` ⇒ positive, `2.x` ⇒ negative.
- Required bundle files: `tests/prompt.txt`, `tests/rubrics.json`, `solution/answer.txt`, `task.toml`, `environment/Dockerfile`.
- `task.toml` must declare `[metadata].repository` and a 40-char `[metadata].base_commit`.
- Severities: S01–S06 are `reject`; S07–S08 are `warn`.
- Verdict (§6): any reject-severity check fails → `not_sound`; else any warn fails or any result needs human → `needs_human_review`; else `sound`.
- Never raise on malformed input — record the problem as a failed `CheckResult`.

---

## File Structure

- `pyproject.toml` — package + pytest config.
- `autoqc/__init__.py` — package marker, version.
- `autoqc/model.py` — `Severity`, `Stage`, `Verdict` enums; `CheckResult`, `ReviewRecord` dataclasses.
- `autoqc/bundle.py` — `Bundle` dataclass + `load_bundle(root)`.
- `autoqc/structural.py` — `run_structural(bundle) -> list[CheckResult]` and the eight check functions.
- `autoqc/verdict.py` — `compute_verdict(results) -> Verdict`.
- `autoqc/report.py` — `to_record(...)`, `to_markdown(...)`.
- `autoqc/cli.py` — `main(argv)` entrypoint.
- `tests/` — one test module per source module.

---

### Task 1: Project skeleton and data model

**Files:**
- Create: `pyproject.toml`
- Create: `autoqc/__init__.py`
- Create: `autoqc/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `Severity(Enum){REJECT="reject", WARN="warn"}`; `Stage(Enum){STRUCTURAL="structural", SEMANTIC="semantic", FACTUAL="factual"}`; `Verdict(Enum){NOT_SOUND="not_sound", NEEDS_HUMAN_REVIEW="needs_human_review", SOUND="sound"}`. `CheckResult(id:str, name:str, stage:Stage, severity:Severity, passed:bool, needs_human:bool=False, evidence:list[str]=field(default_factory=list), detail:str="")`. `ReviewRecord(task_name:str, guideline_version:str, results:list[CheckResult], verdict:Verdict)`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "autoqc"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.setuptools.packages.find]
include = ["autoqc*"]
```

- [ ] **Step 2: Write `autoqc/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_model.py
from autoqc.model import Severity, Stage, Verdict, CheckResult, ReviewRecord


def test_checkresult_defaults():
    r = CheckResult(id="S01", name="Parses as JSON array",
                    stage=Stage.STRUCTURAL, severity=Severity.REJECT, passed=True)
    assert r.evidence == []
    assert r.needs_human is False
    assert r.detail == ""


def test_enum_values_are_report_strings():
    assert Severity.REJECT.value == "reject"
    assert Verdict.NEEDS_HUMAN_REVIEW.value == "needs_human_review"
    assert Stage.FACTUAL.value == "factual"


def test_review_record_holds_results():
    r = CheckResult(id="S05", name="Has a positive", stage=Stage.STRUCTURAL,
                    severity=Severity.REJECT, passed=False, detail="no positive criterion")
    rec = ReviewRecord(task_name="t", guideline_version="1.0.0",
                       results=[r], verdict=Verdict.NOT_SOUND)
    assert rec.results[0].passed is False
    assert rec.verdict is Verdict.NOT_SOUND
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.model'`

- [ ] **Step 5: Write `autoqc/model.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    REJECT = "reject"
    WARN = "warn"


class Stage(Enum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    FACTUAL = "factual"


class Verdict(Enum):
    NOT_SOUND = "not_sound"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    SOUND = "sound"


@dataclass
class CheckResult:
    id: str
    name: str
    stage: Stage
    severity: Severity
    passed: bool
    needs_human: bool = False
    evidence: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class ReviewRecord:
    task_name: str
    guideline_version: str
    results: list[CheckResult]
    verdict: Verdict
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml autoqc/__init__.py autoqc/model.py tests/test_model.py
git commit -m "feat(autoqc): project skeleton and CheckResult/Verdict model"
```

---

### Task 2: Bundle loader (tolerant)

**Files:**
- Create: `autoqc/bundle.py`
- Test: `tests/test_bundle.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Bundle(root:Path, rubrics_raw:str|None, rubrics:object|None, rubrics_error:str|None, prompt:str|None, answer:str|None, task_toml:dict|None, repository:str|None, base_commit:str|None, files_present:dict[str,bool])` and `load_bundle(root:Path) -> Bundle`. `rubrics` is the parsed JSON (any type) or `None` if unparseable; `rubrics_error` holds the parse message. Never raises for missing/broken files.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bundle.py
import json
from pathlib import Path
from autoqc.bundle import load_bundle


def _write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _good_bundle(root: Path):
    _write(root, "tests/prompt.txt", "How does X work?")
    _write(root, "tests/rubrics.json", json.dumps([
        {"id": "a" * 32, "title": "1.1: States X",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}}
    ]))
    _write(root, "solution/answer.txt", "X works via Y.")
    _write(root, "environment/Dockerfile", "FROM python:3.11")
    _write(root, "task.toml",
           '[metadata]\nrepository = "org/repo"\nbase_commit = "%s"\n' % ("a" * 40))


def test_loads_good_bundle(tmp_path):
    _good_bundle(tmp_path)
    b = load_bundle(tmp_path)
    assert b.rubrics_error is None
    assert isinstance(b.rubrics, list)
    assert b.repository == "org/repo"
    assert b.base_commit == "a" * 40
    assert b.prompt.startswith("How does")
    assert all(b.files_present.values())


def test_invalid_json_is_recorded_not_raised(tmp_path):
    _good_bundle(tmp_path)
    (tmp_path / "tests/rubrics.json").write_text("{ not json ]")
    b = load_bundle(tmp_path)
    assert b.rubrics is None
    assert b.rubrics_error is not None


def test_missing_files_flagged(tmp_path):
    # only prompt exists
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/prompt.txt").write_text("q")
    b = load_bundle(tmp_path)
    assert b.files_present["tests/prompt.txt"] is True
    assert b.files_present["environment/Dockerfile"] is False
    assert b.task_toml is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.bundle'`

- [ ] **Step 3: Write `autoqc/bundle.py`**

```python
from __future__ import annotations
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FILES = [
    "tests/prompt.txt",
    "tests/rubrics.json",
    "solution/answer.txt",
    "task.toml",
    "environment/Dockerfile",
]


@dataclass
class Bundle:
    root: Path
    rubrics_raw: str | None
    rubrics: object | None
    rubrics_error: str | None
    prompt: str | None
    answer: str | None
    task_toml: dict | None
    repository: str | None
    base_commit: str | None
    files_present: dict[str, bool]


def _read_text(p: Path) -> str | None:
    try:
        return p.read_text()
    except OSError:
        return None


def load_bundle(root: Path) -> Bundle:
    root = Path(root)
    files_present = {rel: (root / rel).is_file() for rel in REQUIRED_FILES}

    rubrics_raw = _read_text(root / "tests/rubrics.json")
    rubrics, rubrics_error = None, None
    if rubrics_raw is None:
        rubrics_error = "tests/rubrics.json is missing"
    else:
        try:
            rubrics = json.loads(rubrics_raw)
        except json.JSONDecodeError as e:
            rubrics_error = f"invalid JSON: {e}"

    task_toml, repository, base_commit = None, None, None
    raw_toml = _read_text(root / "task.toml")
    if raw_toml is not None:
        try:
            task_toml = tomllib.loads(raw_toml)
            meta = task_toml.get("metadata", {})
            repository = meta.get("repository")
            base_commit = meta.get("base_commit")
        except tomllib.TOMLDecodeError:
            task_toml = None

    return Bundle(
        root=root,
        rubrics_raw=rubrics_raw,
        rubrics=rubrics,
        rubrics_error=rubrics_error,
        prompt=_read_text(root / "tests/prompt.txt"),
        answer=_read_text(root / "solution/answer.txt"),
        task_toml=task_toml,
        repository=repository,
        base_commit=base_commit,
        files_present=files_present,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bundle.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/bundle.py tests/test_bundle.py
git commit -m "feat(autoqc): tolerant Harbor bundle loader"
```

---

### Task 3: Structural reject checks (S01–S06)

**Files:**
- Create: `autoqc/structural.py`
- Test: `tests/test_structural_reject.py`

**Interfaces:**
- Consumes: `Bundle` (Task 2); `CheckResult`, `Stage`, `Severity` (Task 1).
- Produces: `run_structural(bundle) -> list[CheckResult]` (this task implements S01–S06; S07–S08 added in Task 4). Individual helpers `_s01(bundle)` … `_s06(bundle)` each return a `CheckResult`. `ID_RE = re.compile(r"^[0-9a-f]{32}$")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_structural_reject.py
import json
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.structural import run_structural
from autoqc.model import Severity


def _bundle(tmp_path: Path, rubrics, with_files=True):
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/rubrics.json").write_text(
        rubrics if isinstance(rubrics, str) else json.dumps(rubrics))
    if with_files:
        (tmp_path / "tests/prompt.txt").write_text("q")
        (tmp_path / "solution").mkdir(exist_ok=True)
        (tmp_path / "solution/answer.txt").write_text("a")
        (tmp_path / "environment").mkdir(exist_ok=True)
        (tmp_path / "environment/Dockerfile").write_text("FROM x")
        (tmp_path / "task.toml").write_text(
            '[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))
    return load_bundle(tmp_path)


def _by_id(results):
    return {r.id: r for r in results}


def _positive(idx="1.1"):
    return {"id": "a" * 32, "title": f"{idx}: States X",
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def test_s01_invalid_json_fails(tmp_path):
    b = _bundle(tmp_path, "{ not json ]")
    r = _by_id(run_structural(b))["S01"]
    assert r.passed is False and r.severity is Severity.REJECT


def test_s01_non_array_fails(tmp_path):
    b = _bundle(tmp_path, {"not": "an array"})
    assert _by_id(run_structural(b))["S01"].passed is False


def test_s03_bad_id_fails(tmp_path):
    bad = _positive()
    bad["id"] = "XYZ"
    assert _by_id(run_structural(_bundle(tmp_path, [bad])))["S03"].passed is False


def test_s03_duplicate_id_fails(tmp_path):
    a, c = _positive("1.1"), _positive("1.2")  # same id "a"*32
    assert _by_id(run_structural(_bundle(tmp_path, [a, c])))["S03"].passed is False


def test_s04_type_number_mismatch_fails(tmp_path):
    item = _positive("2.1")  # negative number, positive type
    assert _by_id(run_structural(_bundle(tmp_path, [item])))["S04"].passed is False


def test_s05_no_positive_fails(tmp_path):
    neg = {"id": "b" * 32, "title": "2.1: Claims wrongly",
           "annotations": {"type": "negative hli verifier", "importance": "must have"}}
    assert _by_id(run_structural(_bundle(tmp_path, [neg])))["S05"].passed is False


def test_s06_missing_dockerfile_fails(tmp_path):
    b = _bundle(tmp_path, [_positive()], with_files=True)
    (tmp_path / "environment/Dockerfile").unlink()
    b = load_bundle(tmp_path)
    assert _by_id(run_structural(b))["S06"].passed is False


def test_all_pass_on_good_bundle(tmp_path):
    good = [_positive("1.1"),
            {"id": "b" * 32, "title": "2.1: Claims X is false",
             "annotations": {"type": "negative hli verifier", "importance": "must have"}}]
    results = _by_id(run_structural(_bundle(tmp_path, good)))
    for sid in ["S01", "S02", "S03", "S04", "S05", "S06"]:
        assert results[sid].passed is True, sid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_structural_reject.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.structural'`

- [ ] **Step 3: Write `autoqc/structural.py` (S01–S06 + partial runner)**

```python
from __future__ import annotations
import re
from autoqc.bundle import Bundle, REQUIRED_FILES
from autoqc.model import CheckResult, Stage, Severity

ID_RE = re.compile(r"^[0-9a-f]{32}$")
TITLE_RE = re.compile(r"^\s*([12])\.(\d+)\s*:")


def _ok(id, name, sev, detail="", evidence=None):
    return CheckResult(id=id, name=name, stage=Stage.STRUCTURAL, severity=sev,
                       passed=True, evidence=evidence or [], detail=detail)


def _fail(id, name, sev, detail, evidence=None):
    return CheckResult(id=id, name=name, stage=Stage.STRUCTURAL, severity=sev,
                       passed=False, evidence=evidence or [], detail=detail)


def _items(bundle: Bundle) -> list[dict]:
    return bundle.rubrics if isinstance(bundle.rubrics, list) else []


def _s01(b: Bundle) -> CheckResult:
    n, s = "Parses as JSON array", Severity.REJECT
    if b.rubrics_error is not None:
        return _fail("S01", n, s, f"rubrics.json does not parse: {b.rubrics_error}")
    if not isinstance(b.rubrics, list) or len(b.rubrics) == 0:
        return _fail("S01", n, s, "rubrics.json must be a non-empty JSON array")
    return _ok("S01", n, s)


def _s02(b: Bundle) -> CheckResult:
    n, s = "Item shape", Severity.REJECT
    for i, it in enumerate(_items(b)):
        if not isinstance(it, dict):
            return _fail("S02", n, s, f"item {i} is not an object")
        if "id" not in it or "title" not in it:
            return _fail("S02", n, s, f"item {i} missing id/title")
        ann = it.get("annotations")
        if not isinstance(ann, dict) or "type" not in ann or "importance" not in ann:
            return _fail("S02", n, s, f"item {i} missing annotations.type/importance")
    return _ok("S02", n, s)


def _s03(b: Bundle) -> CheckResult:
    n, s = "ID format & uniqueness", Severity.REJECT
    seen = set()
    for i, it in enumerate(_items(b)):
        iid = it.get("id") if isinstance(it, dict) else None
        if not isinstance(iid, str) or not ID_RE.match(iid):
            return _fail("S03", n, s, f"item {i} id {iid!r} is not 32 lowercase hex")
        if iid in seen:
            return _fail("S03", n, s, f"duplicate id {iid}")
        seen.add(iid)
    return _ok("S03", n, s)


def _s04(b: Bundle) -> CheckResult:
    n, s = "Type/number consistency", Severity.REJECT
    for it in _items(b):
        if not isinstance(it, dict):
            continue
        m = TITLE_RE.match(str(it.get("title", "")))
        if not m:
            return _fail("S04", n, s, f"title not numbered N.x: {it.get('title')!r}")
        num, typ = m.group(1), str(it.get("annotations", {}).get("type", ""))
        if num == "1" and "positive" not in typ:
            return _fail("S04", n, s, f"1.x must be positive: {it.get('title')!r}")
        if num == "2" and "negative" not in typ:
            return _fail("S04", n, s, f"2.x must be negative: {it.get('title')!r}")
    return _ok("S04", n, s)


def _s05(b: Bundle) -> CheckResult:
    n, s = "Has a positive", Severity.REJECT
    for it in _items(b):
        if isinstance(it, dict) and "positive" in str(it.get("annotations", {}).get("type", "")):
            return _ok("S05", n, s)
    return _fail("S05", n, s, "no positive (1.x) criterion present")


def _s06(b: Bundle) -> CheckResult:
    n, s = "Bundle completeness", Severity.REJECT
    missing = [f for f in REQUIRED_FILES if not b.files_present.get(f)]
    if missing:
        return _fail("S06", n, s, f"missing files: {', '.join(missing)}")
    if not b.repository:
        return _fail("S06", n, s, "task.toml missing [metadata].repository")
    if not (isinstance(b.base_commit, str) and len(b.base_commit) == 40):
        return _fail("S06", n, s, "task.toml base_commit must be 40 chars")
    return _ok("S06", n, s)


def run_structural(bundle: Bundle) -> list[CheckResult]:
    return [_s01(bundle), _s02(bundle), _s03(bundle),
            _s04(bundle), _s05(bundle), _s06(bundle)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_structural_reject.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/structural.py tests/test_structural_reject.py
git commit -m "feat(autoqc): structural reject checks S01-S06"
```

---

### Task 4: Structural warn checks (S07–S08)

**Files:**
- Modify: `autoqc/structural.py` (add `_s07`, `_s08`; append to `run_structural`)
- Test: `tests/test_structural_warn.py`

**Interfaces:**
- Consumes: same as Task 3.
- Produces: `run_structural` now returns S01–S08 (8 results). S07/S08 have `severity=Severity.WARN`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_structural_warn.py
import json
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.structural import run_structural
from autoqc.model import Severity


def _bundle(tmp_path: Path, rubrics):
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/rubrics.json").write_text(json.dumps(rubrics))
    (tmp_path / "tests/prompt.txt").write_text("q")
    (tmp_path / "solution").mkdir(exist_ok=True)
    (tmp_path / "solution/answer.txt").write_text("a")
    (tmp_path / "environment").mkdir(exist_ok=True)
    (tmp_path / "environment/Dockerfile").write_text("FROM x")
    (tmp_path / "task.toml").write_text(
        '[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))
    return load_bundle(tmp_path)


def _by_id(results):
    return {r.id: r for r in results}


def _item(idx, id_, typ):
    return {"id": id_, "title": f"{idx}: text",
            "annotations": {"type": typ, "importance": "must have"}}


def test_s07_bad_vocab_warns(tmp_path):
    bad = _item("1.1", "a" * 32, "positive verifier")  # wrong type string
    r = _by_id(run_structural(_bundle(tmp_path, [bad])))["S07"]
    assert r.passed is False and r.severity is Severity.WARN


def test_s08_numbering_gap_warns(tmp_path):
    items = [_item("1.1", "a" * 32, "positive hli verifier"),
             _item("1.3", "b" * 32, "positive hli verifier")]  # gap: no 1.2
    r = _by_id(run_structural(_bundle(tmp_path, items)))["S08"]
    assert r.passed is False and r.severity is Severity.WARN


def test_s07_s08_pass_on_clean(tmp_path):
    items = [_item("1.1", "a" * 32, "positive hli verifier"),
             _item("1.2", "b" * 32, "positive hli verifier"),
             _item("2.1", "c" * 32, "negative hli verifier")]
    res = _by_id(run_structural(_bundle(tmp_path, items)))
    assert res["S07"].passed is True and res["S08"].passed is True
    assert len(run_structural(_bundle(tmp_path, items))) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_structural_warn.py -v`
Expected: FAIL (`KeyError: 'S07'` — run_structural returns only S01–S06)

- [ ] **Step 3: Add `_s07`, `_s08` and extend `run_structural` in `autoqc/structural.py`**

```python
VALID_TYPES = {"positive hli verifier", "negative hli verifier"}


def _s07(b: Bundle) -> CheckResult:
    n, s = "Type vocabulary", Severity.WARN
    for it in _items(b):
        if not isinstance(it, dict):
            continue
        ann = it.get("annotations", {})
        if ann.get("type") not in VALID_TYPES:
            return _fail("S07", n, s, f"unexpected type {ann.get('type')!r}")
        if ann.get("importance") != "must have":
            return _fail("S07", n, s, f"unexpected importance {ann.get('importance')!r}")
    return _ok("S07", n, s)


def _s08(b: Bundle) -> CheckResult:
    n, s = "Sequential numbering", Severity.WARN
    nums = {"1": [], "2": []}
    for it in _items(b):
        if not isinstance(it, dict):
            continue
        m = TITLE_RE.match(str(it.get("title", "")))
        if m:
            nums[m.group(1)].append(int(m.group(2)))
    for prefix, seq in nums.items():
        if not seq:
            continue
        expected = list(range(1, len(seq) + 1))
        if sorted(seq) != expected:
            return _fail("S08", n, s,
                         f"{prefix}.x numbering not sequential: got {sorted(seq)}")
    return _ok("S08", n, s)
```

Then change the last line of `run_structural` to append S07 and S08:

```python
def run_structural(bundle: Bundle) -> list[CheckResult]:
    return [_s01(bundle), _s02(bundle), _s03(bundle),
            _s04(bundle), _s05(bundle), _s06(bundle),
            _s07(bundle), _s08(bundle)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_structural_warn.py tests/test_structural_reject.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add autoqc/structural.py tests/test_structural_warn.py
git commit -m "feat(autoqc): structural warn checks S07-S08"
```

---

### Task 5: Verdict aggregation

**Files:**
- Create: `autoqc/verdict.py`
- Test: `tests/test_verdict.py`

**Interfaces:**
- Consumes: `CheckResult`, `Severity`, `Verdict` (Task 1).
- Produces: `compute_verdict(results: list[CheckResult]) -> Verdict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verdict.py
from autoqc.model import CheckResult, Stage, Severity, Verdict
from autoqc.verdict import compute_verdict


def _r(sev, passed, needs_human=False):
    return CheckResult(id="x", name="n", stage=Stage.STRUCTURAL,
                       severity=sev, passed=passed, needs_human=needs_human)


def test_reject_failure_is_not_sound():
    results = [_r(Severity.REJECT, False), _r(Severity.WARN, True)]
    assert compute_verdict(results) is Verdict.NOT_SOUND


def test_warn_failure_is_needs_human():
    results = [_r(Severity.REJECT, True), _r(Severity.WARN, False)]
    assert compute_verdict(results) is Verdict.NEEDS_HUMAN_REVIEW


def test_needs_human_flag_routes_to_human():
    results = [_r(Severity.REJECT, True, needs_human=True)]
    assert compute_verdict(results) is Verdict.NEEDS_HUMAN_REVIEW


def test_all_pass_is_sound():
    results = [_r(Severity.REJECT, True), _r(Severity.WARN, True)]
    assert compute_verdict(results) is Verdict.SOUND


def test_reject_beats_warn():
    results = [_r(Severity.REJECT, False), _r(Severity.WARN, False)]
    assert compute_verdict(results) is Verdict.NOT_SOUND
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verdict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.verdict'`

- [ ] **Step 3: Write `autoqc/verdict.py`**

```python
from __future__ import annotations
from autoqc.model import CheckResult, Severity, Verdict


def compute_verdict(results: list[CheckResult]) -> Verdict:
    if any(r.severity is Severity.REJECT and not r.passed for r in results):
        return Verdict.NOT_SOUND
    warn_fail = any(r.severity is Severity.WARN and not r.passed for r in results)
    needs_human = any(r.needs_human for r in results)
    if warn_fail or needs_human:
        return Verdict.NEEDS_HUMAN_REVIEW
    return Verdict.SOUND
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verdict.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/verdict.py tests/test_verdict.py
git commit -m "feat(autoqc): verdict aggregation"
```

---

### Task 6: Structured review record

**Files:**
- Create: `autoqc/report.py`
- Test: `tests/test_report_record.py`

**Interfaces:**
- Consumes: `Bundle`, `CheckResult`, `ReviewRecord`, `Verdict`.
- Produces: `to_record(bundle: Bundle, results: list[CheckResult], verdict: Verdict, guideline_version: str = "1.0.0") -> dict` — a JSON-serializable dict with keys `guideline_version`, `task` (`{name, repository, base_commit}`), `verdict` (string), `results` (list of `{id, name, stage, severity, passed, needs_human, evidence, detail}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_record.py
import json
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.model import CheckResult, Stage, Severity, Verdict
from autoqc.report import to_record


def _min_bundle(tmp_path: Path):
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/rubrics.json").write_text("[]")
    (tmp_path / "task.toml").write_text(
        '[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))
    return load_bundle(tmp_path)


def test_record_is_json_serializable_and_shaped(tmp_path):
    b = _min_bundle(tmp_path)
    results = [CheckResult(id="S01", name="Parses as JSON array",
                           stage=Stage.STRUCTURAL, severity=Severity.REJECT,
                           passed=False, detail="bad json", evidence=["x"])]
    rec = to_record(b, results, Verdict.NOT_SOUND)
    s = json.dumps(rec)  # must not raise
    back = json.loads(s)
    assert back["verdict"] == "not_sound"
    assert back["task"]["repository"] == "o/r"
    assert back["results"][0]["severity"] == "reject"
    assert back["results"][0]["passed"] is False
    assert back["results"][0]["detail"] == "bad json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_record.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.report'`

- [ ] **Step 3: Write `autoqc/report.py` (record only)**

```python
from __future__ import annotations
from autoqc.bundle import Bundle
from autoqc.model import CheckResult, Verdict


def _result_dict(r: CheckResult) -> dict:
    return {
        "id": r.id, "name": r.name,
        "stage": r.stage.value, "severity": r.severity.value,
        "passed": r.passed, "needs_human": r.needs_human,
        "evidence": list(r.evidence), "detail": r.detail,
    }


def to_record(bundle: Bundle, results: list[CheckResult],
              verdict: Verdict, guideline_version: str = "1.0.0") -> dict:
    return {
        "guideline_version": guideline_version,
        "task": {
            "name": bundle.root.name,
            "repository": bundle.repository,
            "base_commit": bundle.base_commit,
        },
        "verdict": verdict.value,
        "results": [_result_dict(r) for r in results],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report_record.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add autoqc/report.py tests/test_report_record.py
git commit -m "feat(autoqc): structured review record"
```

---

### Task 7: Markdown rework report

**Files:**
- Modify: `autoqc/report.py` (add `to_markdown`)
- Test: `tests/test_report_markdown.py`

**Interfaces:**
- Consumes: same as Task 6.
- Produces: `to_markdown(bundle: Bundle, results: list[CheckResult], verdict: Verdict) -> str` — reject failures first (each with id, name, detail, evidence), then warn failures under a "Review these" heading. Passing checks are not listed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_markdown.py
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.model import CheckResult, Stage, Severity, Verdict
from autoqc.report import to_markdown


def _min_bundle(tmp_path: Path):
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/rubrics.json").write_text("[]")
    (tmp_path / "task.toml").write_text('[metadata]\nrepository="o/r"\n')
    return load_bundle(tmp_path)


def test_markdown_lists_rejects_first_then_warns(tmp_path):
    b = _min_bundle(tmp_path)
    results = [
        CheckResult(id="S07", name="Type vocabulary", stage=Stage.STRUCTURAL,
                    severity=Severity.WARN, passed=False, detail="odd type"),
        CheckResult(id="S01", name="Parses as JSON array", stage=Stage.STRUCTURAL,
                    severity=Severity.REJECT, passed=False, detail="bad json",
                    evidence=["line 1"]),
        CheckResult(id="S05", name="Has a positive", stage=Stage.STRUCTURAL,
                    severity=Severity.REJECT, passed=True),
    ]
    md = to_markdown(b, results, Verdict.NOT_SOUND)
    assert "not_sound" in md
    # reject appears before warn
    assert md.index("S01") < md.index("S07")
    assert "bad json" in md and "line 1" in md
    # passing check not shown
    assert "S05" not in md


def test_markdown_sound_has_no_issues_section(tmp_path):
    b = _min_bundle(tmp_path)
    results = [CheckResult(id="S01", name="Parses as JSON array",
                           stage=Stage.STRUCTURAL, severity=Severity.REJECT, passed=True)]
    md = to_markdown(b, results, Verdict.SOUND)
    assert "sound" in md
    assert "Must fix" not in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_markdown.py -v`
Expected: FAIL with `ImportError: cannot import name 'to_markdown'`

- [ ] **Step 3: Add `to_markdown` to `autoqc/report.py`**

```python
from autoqc.model import Severity  # add to existing imports


def to_markdown(bundle: Bundle, results: list[CheckResult], verdict: Verdict) -> str:
    lines = [f"# AutoQC report — {bundle.root.name}",
             "", f"**Verdict:** `{verdict.value}`", ""]
    rejects = [r for r in results if r.severity is Severity.REJECT and not r.passed]
    warns = [r for r in results if r.severity is Severity.WARN and not r.passed]

    if rejects:
        lines.append("## Must fix (reject)")
        for r in rejects:
            lines.append(f"- **{r.id} {r.name}** — {r.detail}")
            for e in r.evidence:
                lines.append(f"  - evidence: {e}")
        lines.append("")
    if warns:
        lines.append("## Review these (warn)")
        for r in warns:
            lines.append(f"- **{r.id} {r.name}** — {r.detail}")
        lines.append("")
    if not rejects and not warns:
        lines.append("No issues found.")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report_markdown.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/report.py tests/test_report_markdown.py
git commit -m "feat(autoqc): markdown rework report"
```

---

### Task 8: CLI wiring and end-to-end test

**Files:**
- Create: `autoqc/cli.py`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- Consumes: `load_bundle`, `run_structural`, `compute_verdict`, `to_record`, `to_markdown`.
- Produces: `run(bundle_dir: str | Path, out_dir: str | Path) -> Verdict` (writes `<out_dir>/review_record.json` and `<out_dir>/report.md`, returns the verdict) and `main(argv: list[str] | None = None) -> int` (exit code 0 for `sound`, 1 for `needs_human_review`, 2 for `not_sound`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_e2e.py
import json
from pathlib import Path
from autoqc.cli import run
from autoqc.model import Verdict


def _good(tmp_path: Path):
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/rubrics.json").write_text(json.dumps([
        {"id": "a" * 32, "title": "1.1: States X",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}},
        {"id": "b" * 32, "title": "2.1: Claims X is false",
         "annotations": {"type": "negative hli verifier", "importance": "must have"}},
    ]))
    (tmp_path / "tests/prompt.txt").write_text("q")
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution/answer.txt").write_text("a")
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment/Dockerfile").write_text("FROM x")
    (tmp_path / "task.toml").write_text(
        '[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))


def test_good_bundle_is_sound_and_writes_reports(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    _good(bundle)
    out = tmp_path / "out"
    verdict = run(bundle, out)
    assert verdict is Verdict.SOUND
    rec = json.loads((out / "review_record.json").read_text())
    assert rec["verdict"] == "sound"
    assert (out / "report.md").read_text().startswith("# AutoQC report")


def test_malformed_bundle_is_not_sound(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    (bundle / "tests").mkdir()
    (bundle / "tests/rubrics.json").write_text("{ not json ]")
    out = tmp_path / "out"
    verdict = run(bundle, out)
    assert verdict is Verdict.NOT_SOUND
    assert "Must fix" in (out / "report.md").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_e2e.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.cli'`

- [ ] **Step 3: Write `autoqc/cli.py`**

```python
from __future__ import annotations
import json
import sys
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.structural import run_structural
from autoqc.verdict import compute_verdict
from autoqc.report import to_record, to_markdown
from autoqc.model import Verdict

_EXIT = {Verdict.SOUND: 0, Verdict.NEEDS_HUMAN_REVIEW: 1, Verdict.NOT_SOUND: 2}


def run(bundle_dir, out_dir) -> Verdict:
    bundle = load_bundle(Path(bundle_dir))
    results = run_structural(bundle)
    verdict = compute_verdict(results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_record.json").write_text(
        json.dumps(to_record(bundle, results, verdict), indent=2))
    (out / "report.md").write_text(to_markdown(bundle, results, verdict))
    return verdict


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 1:
        print("usage: autoqc <bundle_dir> [out_dir]", file=sys.stderr)
        return 64
    bundle_dir = argv[0]
    out_dir = argv[1] if len(argv) > 1 else "autoqc_out"
    verdict = run(bundle_dir, out_dir)
    print(f"verdict: {verdict.value}  (reports in {out_dir})")
    return _EXIT[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full suite to verify everything passes**

Run: `pytest -v`
Expected: PASS (all tests across all modules)

- [ ] **Step 5: Smoke-test against a real internal sample (manual)**

Run: `python -m autoqc.cli <path-to-an-extracted-task-dir> /tmp/autoqc_smoke`
Expected: prints `verdict: needs_human_review` or `sound`; `/tmp/autoqc_smoke/report.md` and `review_record.json` exist. (A real internal sample passes structural; semantic warnings arrive in the next plan.)

- [ ] **Step 6: Commit**

```bash
git add autoqc/cli.py tests/test_cli_e2e.py
git commit -m "feat(autoqc): CLI wiring and end-to-end structural pipeline"
```

---

## Self-Review

**Spec coverage (foundation slice of the execution-harness spec):**
- Stage 1 S01–S08 → Tasks 3–4. ✓
- Verdict §6 → Task 5. ✓
- Structured record (`task_quality_spec.json`-shaped subset) → Task 6. ✓
- Markdown rework report (reject-first) → Task 7. ✓
- Never-raise-on-malformed → Task 2 (tolerant loader) + tested in Tasks 3, 8. ✓
- Bundle inputs (prompt/rubric/answer/toml/Dockerfile) → Task 2. ✓
- Deferred to next plans (correctly absent here): Stage 2a semantic engine, ensemble/adversary, Q06 container, seed harness, calibration metrics.

**Placeholder scan:** no TBD/TODO; every code and test step carries real content. ✓

**Type consistency:** `CheckResult` fields (`id, name, stage, severity, passed, needs_human, evidence, detail`) are defined in Task 1 and used unchanged in Tasks 3, 4, 6, 7. `run_structural`, `compute_verdict`, `to_record`, `to_markdown`, `load_bundle`, `run` signatures match across their producer and consumer tasks. `ID_RE`/`TITLE_RE` defined once in Task 3, reused in Task 4. ✓

---

## Notes for the next plan (Milestone 2 — not in scope here)

The semantic engine plan will add: `autoqc/llm.py` (gateway client, OpenAI-compatible, mirroring the bundle's `evaluate_answer.py` env pattern), `autoqc/semantic/engine.py` (K-vote ensemble + asymmetric adversary), `autoqc/semantic/checks.py` (Q03, Q07 first), and `autoqc/seed.py` (defect-injection corpus generator). It consumes the `CheckResult`/`Verdict` interface locked by this plan.

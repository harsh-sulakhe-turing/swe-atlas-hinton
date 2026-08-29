# AutoQC Phase 1.5 — Prompt & Answer Quality Rubric — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second quality rubric (13 checks) that gates the prompt, answer, alignment, hygiene, and codebase-exploration quality of a task bundle, reusing the shipped two-stage engine.

**Architecture:** Two deterministic checks (`P01`, `H01`) run in the no-LLM structural stage. Nine semantic checks (`P02`, `P03`, `A01`–`A05`, `AL01`, `Q13`) run through the existing proposer(K=3)+adversary ensemble, with new whole-document unit modes and a prose review role. Two grounded checks (`P04`, `A06`) share one `ContainerSession` with the existing `Q06` factual check. Verdict composition and reporting are unchanged — new `CheckResult`s flow through by severity.

**Tech Stack:** Python 3.11+ (stdlib only — `urllib`, `tomllib`, `subprocess`), pytest, Docker (grounded stage only), OpenAI-compatible gateway (OpenRouter, `z-ai/glm-5.2`).

**Spec:** `docs/superpowers/specs/2026-08-28-autoqc-phase1.5-prompt-answer-design.md`

## Global Constraints

- **Stdlib only** — no third-party runtime deps. The gateway client is `urllib`; TOML is `tomllib`.
- **Tests are offline** — every unit test drives `FakeLLMClient` or pure Python; no network, no Docker in unit tests (fake the container/runner). Matches the existing 122-test suite.
- **No heuristics feeding a semantic verdict** — regex/statistics may only implement the two genuinely-factual deterministic checks (`P01` marker match, `H01` file presence). Every prose-*quality* judgment (`P02`,`P03`,`A01`–`A06`,`AL01`,`Q13`) is decided by the LLM ensemble, never by a heuristic pre-filter.
- **New check IDs must be registered in `CHECK_IDS`** (`autoqc/agent/tools.py`) or `validate_findings` and the `submit_findings` enum reject them.
- **Severity → verdict** (unchanged, `autoqc/verdict.py`): undisputed `reject` fail → `not_sound`; disputed reject or any `warn` fail or any `needs_human` → `needs_human_review`; else `sound`.
- **Reject set:** `P01, P02, P03, P04, A04, A06, Q13`. **Warn set:** `A01, A02, A03, A05, AL01, H01`.
- **CheckResult stage:** deterministic checks use `Stage.STRUCTURAL`; text-semantic use `Stage.SEMANTIC`; grounded use `Stage.FACTUAL`.

---

### Task 1: Load `instruction.md` into the bundle

**Files:**
- Modify: `autoqc/bundle.py` (the `Bundle` dataclass + `load_bundle`)
- Test: `tests/test_bundle.py`

**Interfaces:**
- Produces: `Bundle.instruction: str | None` — the text of `instruction.md` at bundle root, or `None` if absent/unreadable. Consumed by `P01`, `AL01`, and the grounded contexts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bundle.py (append)
def test_load_bundle_reads_instruction_md(tmp_path):
    from autoqc.bundle import load_bundle
    (tmp_path / "instruction.md").write_text("the rendered prompt")
    b = load_bundle(tmp_path)
    assert b.instruction == "the rendered prompt"

def test_load_bundle_instruction_absent_is_none(tmp_path):
    from autoqc.bundle import load_bundle
    b = load_bundle(tmp_path)
    assert b.instruction is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bundle.py -k instruction -v`
Expected: FAIL — `Bundle.__init__() missing ... 'instruction'` / attribute error.

- [ ] **Step 3: Add the field and load it**

In `autoqc/bundle.py`, add to the `Bundle` dataclass (after `answer`):

```python
    instruction: str | None
```

In `load_bundle`, add to the `Bundle(...)` construction (after `answer=...`):

```python
        instruction=_read_text(root / "instruction.md"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_bundle.py -v`
Expected: PASS (all bundle tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add autoqc/bundle.py tests/test_bundle.py
git commit -m "feat(bundle): load instruction.md into Bundle.instruction"
```

---

### Task 2: Admit the new check IDs into the findings contract

**Files:**
- Modify: `autoqc/agent/tools.py:7-8` (`CHECK_IDS`, `ALLOWED_READ`)
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Produces: `CHECK_IDS` now contains `Q01`–`Q13`, `P01`–`P04`, `A01`–`A06`, `AL01`, `H01`. `validate_findings` and the `submit_findings` enum accept them. `ALLOWED_READ` includes `instruction.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_tools.py (append)
from autoqc.agent.tools import CHECK_IDS, validate_findings, ALLOWED_READ

def test_check_ids_include_phase15():
    for cid in ("Q13", "P01", "P04", "A01", "A06", "AL01", "H01"):
        assert cid in CHECK_IDS

def test_validate_findings_accepts_answer_unit():
    valid, problems = validate_findings(
        [{"check_id": "A01", "criterion_id": "answer", "passed": True, "evidence": ["ok"]}],
        {"answer"})
    assert len(valid) == 1 and problems == []

def test_instruction_is_readable():
    assert "instruction.md" in ALLOWED_READ
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_tools.py -k "phase15 or answer_unit or instruction" -v`
Expected: FAIL — `Q13`/`P01`/… not in `CHECK_IDS`.

- [ ] **Step 3: Extend the ID set and read allow-list**

In `autoqc/agent/tools.py`, replace lines 7-8:

```python
CHECK_IDS = ({f"Q{n:02d}" for n in range(1, 14)}
             | {"P01", "P02", "P03", "P04",
                "A01", "A02", "A03", "A04", "A05", "A06",
                "AL01", "H01"})
ALLOWED_READ = {"tests/prompt.txt", "tests/rubrics.json", "solution/answer.txt",
                "task.toml", "instruction.md"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/tools.py tests/test_agent_tools.py
git commit -m "feat(tools): register Phase 1.5 check IDs and allow reading instruction.md"
```

---

### Task 3: `P01` — instruction.md rendered, not the authoring placeholder (deterministic, reject)

**Files:**
- Create: `autoqc/text_deterministic.py`
- Test: `tests/test_text_deterministic.py`

**Interfaces:**
- Produces: `check_p01(bundle) -> CheckResult` (id `P01`, `Stage.STRUCTURAL`, `Severity.REJECT`). Fails when `bundle.instruction` is `None` or contains any known authoring-template marker.
- Consumes: `Bundle.instruction` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text_deterministic.py
import types
from autoqc.model import Stage, Severity
from autoqc.text_deterministic import check_p01

def _b(instruction):
    return types.SimpleNamespace(instruction=instruction, root=".")

def test_p01_rejects_placeholder_marker():
    r = check_p01(_b("...\n<question>\nDescribe the developer's realistic, multi-part "
                     "question here without telegraphing the measured result.\n</question>"))
    assert r.id == "P01" and r.stage is Stage.STRUCTURAL and r.severity is Severity.REJECT
    assert r.passed is False and "placeholder" in r.detail.lower()

def test_p01_rejects_missing_instruction():
    r = check_p01(_b(None))
    assert r.passed is False

def test_p01_passes_rendered_prompt():
    r = check_p01(_b("I have a service that POSTs file-like bodies with aiohttp and the "
                     "connection pool is misbehaving. Why does a zero-length file body ..."))
    assert r.passed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_text_deterministic.py -k p01 -v`
Expected: FAIL — module `autoqc.text_deterministic` does not exist.

- [ ] **Step 3: Implement `check_p01`**

```python
# autoqc/text_deterministic.py
from __future__ import annotations
from autoqc.model import CheckResult, Stage, Severity

# Verbatim fragments of the authoring template left in un-rendered instruction.md,
# taken from the Batch-1 placeholder bundles. A case-insensitive substring match is
# an exact-marker test, not a heuristic quality judgment.
PLACEHOLDER_MARKERS = (
    "describe the developer's realistic, multi-part question here",
    "without telegraphing the measured result",
)

def check_p01(bundle) -> CheckResult:
    text = getattr(bundle, "instruction", None)
    n, s = "Instruction rendered, not placeholder", Severity.REJECT
    if not isinstance(text, str) or not text.strip():
        return CheckResult(id="P01", name=n, stage=Stage.STRUCTURAL, severity=s,
                           passed=False, detail="instruction.md is missing or empty")
    low = text.lower()
    hit = next((m for m in PLACEHOLDER_MARKERS if m in low), None)
    if hit:
        return CheckResult(id="P01", name=n, stage=Stage.STRUCTURAL, severity=s,
                           passed=False,
                           detail=f"instruction.md still contains authoring placeholder text: {hit!r}",
                           evidence=[f"marker matched: {hit!r}"])
    return CheckResult(id="P01", name=n, stage=Stage.STRUCTURAL, severity=s, passed=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_text_deterministic.py -k p01 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/text_deterministic.py tests/test_text_deterministic.py
git commit -m "feat(P01): deterministic instruction.md placeholder check"
```

---

### Task 4: `H01` — no committed cache artifacts (deterministic, warn) + registry

**Files:**
- Modify: `autoqc/text_deterministic.py`
- Test: `tests/test_text_deterministic.py`

**Interfaces:**
- Produces: `check_h01(bundle) -> CheckResult` (id `H01`, `Stage.STRUCTURAL`, `Severity.WARN`). Fails when the bundle tree contains `__pycache__` or `*.pyc`.
- Produces: `TEXT_DETERMINISTIC_CHECKS = [check_p01, check_h01]`.
- Consumes: `Bundle.root` (a `pathlib.Path`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text_deterministic.py (append)
from pathlib import Path
from autoqc.text_deterministic import check_h01, TEXT_DETERMINISTIC_CHECKS, check_p01

def _broot(tmp_path):
    return types.SimpleNamespace(instruction="ok", root=tmp_path)

def test_h01_warns_on_pycache(tmp_path):
    (tmp_path / "tests" / "__pycache__").mkdir(parents=True)
    (tmp_path / "tests" / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    r = check_h01(_broot(tmp_path))
    assert r.id == "H01" and r.severity is Severity.WARN and r.passed is False
    assert "__pycache__" in r.detail

def test_h01_passes_clean_bundle(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "prompt.txt").write_text("q")
    assert check_h01(_broot(tmp_path)).passed is True

def test_text_deterministic_registry():
    assert TEXT_DETERMINISTIC_CHECKS == [check_p01, check_h01]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_text_deterministic.py -k "h01 or registry" -v`
Expected: FAIL — `check_h01` / `TEXT_DETERMINISTIC_CHECKS` undefined.

- [ ] **Step 3: Implement `check_h01` and the registry**

```python
# autoqc/text_deterministic.py (append)
from pathlib import Path

def check_h01(bundle) -> CheckResult:
    n, s = "No committed cache artifacts", Severity.WARN
    root = Path(getattr(bundle, "root", "."))
    hits = []
    try:
        hits = [str(p.relative_to(root)) for p in root.rglob("*")
                if p.name == "__pycache__" or p.suffix == ".pyc"]
    except OSError:
        pass
    if hits:
        return CheckResult(id="H01", name=n, stage=Stage.STRUCTURAL, severity=s,
                           passed=False, detail=f"cache artifacts committed: {', '.join(sorted(hits)[:5])}",
                           evidence=sorted(hits)[:5])
    return CheckResult(id="H01", name=n, stage=Stage.STRUCTURAL, severity=s, passed=True)

TEXT_DETERMINISTIC_CHECKS = [check_p01, check_h01]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_text_deterministic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/text_deterministic.py tests/test_text_deterministic.py
git commit -m "feat(H01): deterministic bundle-hygiene check + registry"
```

---

### Task 5: Wire deterministic text checks into the CLI (run without a key)

**Files:**
- Modify: `autoqc/cli.py:17-24` (the `run` function)
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- Consumes: `TEXT_DETERMINISTIC_CHECKS` (Task 4).
- Produces: `run(...)` now includes `P01`/`H01` results in every run, even when no LLM client is present.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_e2e.py (append)
def test_run_includes_text_deterministic_without_client(tmp_path):
    from autoqc.cli import run
    from autoqc.model import Verdict
    # a bundle whose instruction.md is a placeholder -> P01 rejects, no client needed
    (tmp_path / "instruction.md").write_text(
        "<question>\nDescribe the developer's realistic, multi-part question here.\n</question>")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "rubrics.json").write_text("[]")
    out = tmp_path / "out"
    v = run(tmp_path, out, llm=None, factual=False)
    import json
    rec = json.loads((out / "review_record.json").read_text())
    ids = {c["id"] for c in rec["results"]}
    assert "P01" in ids and "H01" in ids
    assert v is Verdict.NOT_SOUND  # P01 is an undisputed reject
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli_e2e.py -k text_deterministic -v`
Expected: FAIL — `P01` not in results.

- [ ] **Step 3: Call the deterministic text checks in `run`**

In `autoqc/cli.py`, add the import (top):

```python
from autoqc.text_deterministic import TEXT_DETERMINISTIC_CHECKS
```

In `run`, right after `results = run_structural(bundle)`:

```python
    results += [fn(bundle) for fn in TEXT_DETERMINISTIC_CHECKS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cli_e2e.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/cli.py tests/test_cli_e2e.py
git commit -m "feat(cli): run P01/H01 in the no-LLM structural stage"
```

---

### Task 6: Prose review infrastructure — whole-document roles, scopes, contexts, engine role selection

**Files:**
- Modify: `autoqc/agent/checks.py` (add `role_kind` to `SemanticCheck`; prose roles; whole-doc scopes; whole-doc context branches)
- Modify: `autoqc/agent/engine.py:97-108` (`proposer_pass`, `adversary_pass` — select role by `role_kind`)
- Test: `tests/test_prose_mode.py`

**Interfaces:**
- Produces (checks.py):
  - `SemanticCheck` gains `role_kind: str = "rubric"`.
  - `_prompt_unit(items) -> [{"id": "prompt"}]`, `_answer_unit(items) -> [{"id": "answer"}]`, `_bundle_unit(items) -> [{"id": "bundle"}]` (ignore rubric items; return one synthetic unit).
  - `prose_proposer_role() -> Role`, `prose_adversary_role() -> Role` (system prompts for document review; `tools=text_tools()`).
  - `proposer_context` / `adversary_context` handle `unit_mode in {"prompt","answer","bundle"}`, injecting the relevant document text and instructing exactly one finding with `criterion_id` equal to the unit id.
- Produces (engine.py): `proposer_pass`/`adversary_pass` pick `prose_proposer_role()`/`prose_adversary_role()` when `check.role_kind == "prose"`, else the existing rubric roles.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prose_mode.py
import types
from autoqc.model import Severity
from autoqc.agent.checks import (SemanticCheck, _answer_unit, proposer_context,
                                 prose_proposer_role)
from autoqc.agent.engine import run_check
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient

ANSWER_CHECK = SemanticCheck(id="A01", name="Investigation-first", severity=Severity.WARN,
                             scope=_answer_unit, guidance="opens by investigating.",
                             unit_mode="answer", role_kind="prose")

def _b(answer):
    return types.SimpleNamespace(rubrics=[], prompt="How does X work?",
                                 answer=answer, instruction="i")

def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)

def test_prose_role_is_document_oriented():
    assert "rubric" not in prose_proposer_role().system_prompt.lower()

def test_answer_context_injects_answer_text():
    b = _b("I started by exploring the repo, then traced the close.")
    pc = proposer_context(b, ANSWER_CHECK, _answer_unit(b.rubrics))
    assert "I started by exploring" in pc and 'criterion_id="answer"' in pc

def test_run_check_answer_mode_pass(tmp_path):
    b = _b("I started by exploring the repo ...")
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "A01", "criterion_id": "answer", "passed": True, "evidence": ["investigates first"]}]}}]}
    r = run_check(ANSWER_CHECK, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.id == "A01" and r.passed is True and r.needs_human is False

def test_run_check_answer_mode_fail(tmp_path):
    b = _b("The answer is: conn.close() in the cancel handler.")
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "A01", "criterion_id": "answer", "passed": False, "evidence": ["conclusion-first lede"]}]}}]}
    r = run_check(ANSWER_CHECK, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.passed is False and "answer" in r.detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prose_mode.py -v`
Expected: FAIL — `_answer_unit` / `prose_proposer_role` / `role_kind` undefined; role not selected.

- [ ] **Step 3: Add `role_kind`, scopes, prose roles, and context branches**

In `autoqc/agent/checks.py`, add `role_kind` to the dataclass:

```python
@dataclass
class SemanticCheck:
    id: str
    name: str
    severity: Severity
    scope: Callable          # (items) -> list[dict]
    guidance: str
    unit_mode: str = "criterion"
    role_kind: str = "rubric"
```

Add whole-document scopes (near the other scope helpers):

```python
def _prompt_unit(items): return [{"id": "prompt"}]
def _answer_unit(items): return [{"id": "answer"}]
def _bundle_unit(items): return [{"id": "bundle"}]
```

Add prose roles (near `proposer_role`/`adversary_role`):

```python
_PROSE_PROPOSER_SYS = (
    "You are a strict QC reviewer of a SWE task's authored text (its prompt or its "
    "reference answer). Judge only the one check and the one document you are given, "
    "from the text shown below. Finish by calling submit_findings with exactly one "
    "finding for the listed unit; evidence must quote the text you relied on.")

_PROSE_ADVERSARY_SYS = (
    "You are an adversarial second reviewer of authored SWE task text. If a prior "
    "review marked the document FAIL, argue whether it is actually acceptable; if PASS, "
    "look for a violation it missed. Finish by calling submit_findings with your verdict "
    "for the one unit; passed=true means the document is fine.")

def prose_proposer_role() -> Role:
    return Role(name="prose_proposer", system_prompt=_PROSE_PROPOSER_SYS, tools=text_tools())

def prose_adversary_role() -> Role:
    return Role(name="prose_adversary", system_prompt=_PROSE_ADVERSARY_SYS, tools=text_tools())
```

Add a helper that returns the document text for a unit mode, and branches in both context builders. In `proposer_context`, before the `if check.unit_mode == "rubric":` line, add:

```python
    if check.unit_mode in ("prompt", "answer", "bundle"):
        return _doc_context(bundle, check)
```

and define `_doc_context` (module-level):

```python
def _doc_text(bundle, unit_mode) -> str:
    if unit_mode == "prompt":
        return f"Task prompt (tests/prompt.txt):\n{getattr(bundle, 'prompt', '') or ''}"
    if unit_mode == "answer":
        return (f"Task prompt (for context):\n{getattr(bundle, 'prompt', '') or ''}\n\n"
                f"Reference answer (solution/answer.txt):\n{getattr(bundle, 'answer', '') or ''}")
    # bundle: alignment across all three files
    return (f"instruction.md:\n{getattr(bundle, 'instruction', '') or ''}\n\n"
            f"tests/prompt.txt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"solution/answer.txt:\n{getattr(bundle, 'answer', '') or ''}")

def _doc_context(bundle, check) -> str:
    unit = check.unit_mode
    return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            f"{_doc_text(bundle, unit)}\n\n"
            f"Judge this document AS A WHOLE for the check. Submit EXACTLY ONE finding "
            f"with check_id={check.id} and criterion_id=\"{unit}\": passed=true if it "
            f"SATISFIES the check, passed=false if it VIOLATES it; evidence must quote the text.")
```

In `adversary_context`, before its `if check.unit_mode == "rubric":` line, add:

```python
    if check.unit_mode in ("prompt", "answer", "bundle"):
        unit = check.unit_mode
        v = agg.get(unit, {})
        verdict = "PASS" if v.get("passed") else "FAIL"
        return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
                f"{_doc_text(bundle, unit)}\n\n"
                f"A prior review judged this document: {verdict}. Challenge that per the "
                f"rules above, then submit EXACTLY ONE finding (check_id={check.id}, "
                f"criterion_id=\"{unit}\").")
```

In `autoqc/agent/engine.py`, update `proposer_pass` and `adversary_pass` to select the role. Add near the imports:

```python
from autoqc.agent.checks import prose_proposer_role, prose_adversary_role
```

Replace the role calls:

```python
def proposer_pass(check, bundle, client, ctx, criteria, allowed):
    role = (prose_proposer_role() if getattr(check, "role_kind", "rubric") == "prose"
            else proposer_role())
    res = run_agent(role, proposer_context(bundle, check, criteria),
                    client, ctx, max_turns=TEXT_MAX_TURNS)
    log = {"check": check.id, "role": "proposer", "ok": res.ok, "findings": res.findings}
    return (_own(res.findings, check.id, allowed) if res.ok else []), log

def adversary_pass(check, bundle, client, ctx, criteria, allowed, agg):
    role = (prose_adversary_role() if getattr(check, "role_kind", "rubric") == "prose"
            else adversary_role())
    res = run_agent(role, adversary_context(bundle, check, criteria, agg),
                    client, ctx, max_turns=TEXT_MAX_TURNS)
    log = {"check": check.id, "role": "adversary", "ok": res.ok, "findings": res.findings}
    return (_own(res.findings, check.id, allowed) if res.ok else []), log
```

> Note: `_own` calls `validate_findings(findings, allowed)`; for whole-doc checks `allowed == {"answer"}` (etc.), so a finding with `criterion_id="answer"` validates via the existing `cid in allowed_criterion_ids` path — no engine `_check_prep` change is needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_prose_mode.py tests/test_rubric_mode.py tests/test_engine_core.py -v`
Expected: PASS (new prose tests + unchanged rubric/engine tests).

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/checks.py autoqc/agent/engine.py tests/test_prose_mode.py
git commit -m "feat(engine): whole-document prose checks (roles, scopes, contexts, role selection)"
```

---

### Task 7: Prompt text checks — `P02` (single-goal, reject) + `P03` (natural, reject)

**Files:**
- Modify: `autoqc/agent/checks.py` (add `P02`, `P03`; append to `SEMANTIC_CHECKS`)
- Test: `tests/test_checks_prompt.py`

**Interfaces:**
- Consumes: `_prompt_unit`, prose roles/contexts (Task 6); `SemanticCheck`.
- Produces: `P02`, `P03` `SemanticCheck`s registered in `SEMANTIC_CHECKS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_checks_prompt.py
import types
from autoqc.model import Severity
from autoqc.agent import checks as C
from autoqc.agent.engine import run_check
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient

def _b(prompt):
    return types.SimpleNamespace(rubrics=[], prompt=prompt, answer="a", instruction="i")

def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)

def _fake(check_id, passed, ev):
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": check_id, "criterion_id": "prompt", "passed": passed, "evidence": [ev]}]}}]}
    return FakeLLMClient(responder)

def test_p02_and_p03_registered_as_reject():
    ids = {c.id: c for c in C.SEMANTIC_CHECKS}
    assert ids["P02"].severity is Severity.REJECT and ids["P02"].unit_mode == "prompt"
    assert ids["P03"].severity is Severity.REJECT and ids["P03"].role_kind == "prose"

def test_p02_fail_multi_goal(tmp_path):
    P02 = {c.id: c for c in C.SEMANTIC_CHECKS}["P02"]
    r = run_check(P02, _b("Do A. Also refactor B. Also benchmark C."),
                  _fake("P02", False, "three independent goals"), _ctx(tmp_path), k=3)
    assert r.passed is False

def test_p03_pass_natural(tmp_path):
    P03 = {c.id: c for c in C.SEMANTIC_CHECKS}["P03"]
    r = run_check(P03, _b("Our aiohttp pool keeps reopening sockets on small file bodies; why?"),
                  _fake("P03", True, "reads as a real developer question"), _ctx(tmp_path), k=3)
    assert r.passed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_checks_prompt.py -v`
Expected: FAIL — `P02`/`P03` not in `SEMANTIC_CHECKS`.

- [ ] **Step 3: Define `P02`, `P03` and register them**

In `autoqc/agent/checks.py`, after the existing check definitions and before `SEMANTIC_CHECKS = [...]`:

```python
P02 = SemanticCheck(
    id="P02", name="Single coherent goal", severity=Severity.REJECT, scope=_prompt_unit,
    unit_mode="prompt", role_kind="prose",
    guidance=("The prompt VIOLATES this if it bundles two or more independent tasks/goals "
              "(each separately deliverable) into one request. A single question with several "
              "closely-related sub-parts that build to one answer is FINE. passed=true if the "
              "prompt pursues one coherent goal."))

P03 = SemanticCheck(
    id="P03", name="Natural conversational request", severity=Severity.REJECT, scope=_prompt_unit,
    unit_mode="prompt", role_kind="prose",
    guidance=("The prompt VIOLATES this if it reads as a rigid enumerated checklist / numbered "
              "instruction list rather than a natural developer question, OR if it telegraphs the "
              "measured result the answer is supposed to discover. Prose with a few inline "
              "sub-questions is FINE. passed=true if it reads as a natural, non-spoiling request."))
```

Update the registry line:

```python
SEMANTIC_CHECKS = [Q07, Q03, Q01, Q02, Q05, Q04, Q08, Q10, Q11, P02, P03]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_checks_prompt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/checks.py tests/test_checks_prompt.py
git commit -m "feat(P02,P03): prompt single-goal and naturalness checks"
```

---

### Task 8: Answer text checks — `A01`–`A05`

**Files:**
- Modify: `autoqc/agent/checks.py` (add `A01`–`A05`; append to `SEMANTIC_CHECKS`)
- Test: `tests/test_checks_answer.py`

**Interfaces:**
- Consumes: `_answer_unit`, prose roles/contexts (Task 6).
- Produces: `A01`,`A02`,`A03` (`warn`), `A04` (`reject`), `A05` (`warn`) registered in `SEMANTIC_CHECKS`, all `unit_mode="answer"`, `role_kind="prose"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_checks_answer.py
import types
from autoqc.model import Severity
from autoqc.agent import checks as C
from autoqc.agent.engine import run_check
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient

def _b(answer):
    return types.SimpleNamespace(rubrics=[], prompt="How does X work?", answer=answer, instruction="i")

def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)

def _fake(check_id, passed, ev):
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": check_id, "criterion_id": "answer", "passed": passed, "evidence": [ev]}]}}]}
    return FakeLLMClient(responder)

def test_answer_checks_registered_with_severities():
    ids = {c.id: c for c in C.SEMANTIC_CHECKS}
    assert ids["A01"].severity is Severity.WARN
    assert ids["A04"].severity is Severity.REJECT
    for cid in ("A01", "A02", "A03", "A04", "A05"):
        assert ids[cid].unit_mode == "answer" and ids[cid].role_kind == "prose"

def test_a04_reject_no_evidence_shown(tmp_path):
    A04 = {c.id: c for c in C.SEMANTIC_CHECKS}["A04"]
    r = run_check(A04, _b("I ran the repro and it reused the socket."),
                  _fake("A04", False, "claims a run but shows no command/output"), _ctx(tmp_path), k=3)
    assert r.passed is False

def test_a02_pass_continuous_narrative(tmp_path):
    A02 = {c.id: c for c in C.SEMANTIC_CHECKS}["A02"]
    r = run_check(A02, _b("I started by ... then I traced ... which showed ..."),
                  _fake("A02", True, "continuous narrative, no headers"), _ctx(tmp_path), k=3)
    assert r.passed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_checks_answer.py -v`
Expected: FAIL — `A01`–`A05` not registered.

- [ ] **Step 3: Define `A01`–`A05` and register them**

In `autoqc/agent/checks.py`:

```python
A01 = SemanticCheck(
    id="A01", name="Investigation-first opening", severity=Severity.WARN, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if it OPENS with the conclusion / a 'short answer' lede "
              "before any acknowledgment of investigation. A brief summary is fine AFTER an opening "
              "that acknowledges exploring the codebase. passed=true if it opens investigation-first."))

A02 = SemanticCheck(
    id="A02", name="Continuous narrative", severity=Severity.WARN, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if it is organized under bold or numbered SECTION HEADERS "
              "(e.g. '**Root cause**', '1. Summary') rather than continuous prose that interleaves "
              "reasoning and evidence. Inline code blocks and their output are fine. passed=true if "
              "it reads as continuous narrative."))

A03 = SemanticCheck(
    id="A03", name="First-person voice", severity=Severity.WARN, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if it is NOT written in sustained first-person ('I traced', "
              "'I confirmed'). An impersonal report voice ('The system does X') throughout violates "
              "it. passed=true if first-person narration is sustained."))

A04 = SemanticCheck(
    id="A04", name="Evidence shown inline", severity=Severity.REJECT, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if it CLAIMS an empirical result (ran a command, observed a "
              "value/state/error) but does NOT show the actual command AND its output inline. Pure "
              "code-reading conclusions with file:line citations do not need command output. "
              "passed=true if every claimed run shows command + output."))

A05 = SemanticCheck(
    id="A05", name="Bash-only method", severity=Severity.WARN, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if its investigation METHOD is creating/committing a "
              "script or source file as the deliverable, or modifying the repo, rather than "
              "read-only bash exploration (grep/cat/find/git plus temporary, cleaned-up probes). "
              "passed=true if the method is read-only bash investigation."))
```

Update the registry line:

```python
SEMANTIC_CHECKS = [Q07, Q03, Q01, Q02, Q05, Q04, Q08, Q10, Q11,
                   P02, P03, A01, A02, A03, A04, A05]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_checks_answer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/checks.py tests/test_checks_answer.py
git commit -m "feat(A01-A05): answer-prose quality checks"
```

---

### Task 9: `AL01` (alignment, warn) + `Q13` (rubric exploration coverage, reject)

**Files:**
- Modify: `autoqc/agent/checks.py` (add `AL01`, `Q13`; append to `SEMANTIC_CHECKS`)
- Test: `tests/test_checks_align_explore.py`

**Interfaces:**
- Consumes: `_bundle_unit` (Task 6); `_all_criteria` (existing).
- Produces: `AL01` (`unit_mode="bundle"`, `role_kind="prose"`, warn) and `Q13` (`unit_mode="rubric"`, rubric role, reject) in `SEMANTIC_CHECKS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_checks_align_explore.py
import types
from autoqc.model import Severity
from autoqc.agent import checks as C
from autoqc.agent.engine import run_check
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient

def _ctx(tmp_path):
    (tmp_path / "tests").mkdir(); (tmp_path / "tests/rubrics.json").write_text("[]")
    return AgentContext(bundle_dir=tmp_path)

def _pos(i, title="1.1: States X"):
    return {"id": i * 32, "title": title,
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}

def test_al01_and_q13_registered():
    ids = {c.id: c for c in C.SEMANTIC_CHECKS}
    assert ids["AL01"].severity is Severity.WARN and ids["AL01"].unit_mode == "bundle"
    assert ids["Q13"].severity is Severity.REJECT and ids["Q13"].unit_mode == "rubric"

def test_al01_fail_mismatch(tmp_path):
    AL01 = {c.id: c for c in C.SEMANTIC_CHECKS}["AL01"]
    b = types.SimpleNamespace(rubrics=[], prompt="about aiohttp pooling",
                              answer="about JAX autodiff", instruction="about aiohttp pooling")
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "AL01", "criterion_id": "bundle", "passed": False,
             "evidence": ["answer is about a different task"]}]}}]}
    r = run_check(AL01, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.passed is False

def test_q13_reject_no_exploration_criterion(tmp_path):
    Q13 = {c.id: c for c in C.SEMANTIC_CHECKS}["Q13"]
    b = types.SimpleNamespace(rubrics=[_pos("a")], prompt="How does X work?")
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "Q13", "criterion_id": "rubric", "passed": False,
             "evidence": ["no criterion verifies codebase exploration"]}]}}]}
    r = run_check(Q13, b, FakeLLMClient(responder), _ctx(tmp_path), k=3)
    assert r.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_checks_align_explore.py -v`
Expected: FAIL — `AL01`/`Q13` not registered.

- [ ] **Step 3: Define `AL01`, `Q13` and register them**

In `autoqc/agent/checks.py`:

```python
AL01 = SemanticCheck(
    id="AL01", name="Files describe one task", severity=Severity.WARN, scope=_bundle_unit,
    unit_mode="bundle", role_kind="prose",
    guidance=("VIOLATES this if instruction.md, tests/prompt.txt, and solution/answer.txt do not "
              "all describe the SAME task (different subject, repo, or question). Minor wording "
              "differences are fine. passed=true if all three correspond to one task."))

Q13 = SemanticCheck(
    id="Q13", name="Rubric verifies exploration", severity=Severity.REJECT, scope=_all_criteria,
    unit_mode="rubric",
    guidance=("The rubric VIOLATES this if NO must-have criterion verifies that the model actually "
              "EXPLORED the codebase — i.e. grades a repo-derived fact, path, mechanism, or observed "
              "runtime result that could only be produced by investigating the code, not by general "
              "knowledge. passed=true if at least one criterion forces demonstrated exploration."))
```

Update the registry line:

```python
SEMANTIC_CHECKS = [Q07, Q03, Q01, Q02, Q05, Q04, Q08, Q10, Q11,
                   P02, P03, A01, A02, A03, A04, A05, AL01, Q13]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_checks_align_explore.py tests/test_engine_run.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/checks.py tests/test_checks_align_explore.py
git commit -m "feat(AL01,Q13): cross-file alignment and rubric exploration-coverage checks"
```

---

### Task 10: Grounded prose infrastructure — roles & contexts for `P04` and `A06`

**Files:**
- Modify: `autoqc/agent/checks.py` (grounded prose roles + contexts)
- Test: `tests/test_grounded_prose.py`

**Interfaces:**
- Produces:
  - `grounded_prompt_role() -> Role`, `grounded_answer_role() -> Role` (both `tools=factual_tools()`).
  - `grounded_prompt_context(bundle) -> str`, `grounded_answer_context(bundle) -> str` — instruct the agent to use `run_bash` against `/testbed` and submit one finding (`criterion_id="prompt"` / `"answer"`, `check_id` `P04` / `A06`).
- Consumes: `factual_tools` (existing).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grounded_prose.py
import types
from autoqc.agent.checks import (grounded_prompt_role, grounded_answer_role,
                                 grounded_prompt_context, grounded_answer_context)

def _b():
    return types.SimpleNamespace(prompt="Why does the aiohttp pool reopen sockets?",
                                 answer="I traced conn.close() in the cancel handler ...",
                                 repository="aio-libs/aiohttp", base_commit="a" * 40)

def test_grounded_roles_have_run_bash():
    names = {t.name for t in grounded_prompt_role().tools}
    assert "run_bash" in names and "submit_findings" in names

def test_grounded_prompt_context_mentions_p04_and_testbed():
    c = grounded_prompt_context(_b())
    assert "P04" in c and "/testbed" in c and 'criterion_id="prompt"' in c

def test_grounded_answer_context_mentions_a06():
    c = grounded_answer_context(_b())
    assert "A06" in c and 'criterion_id="answer"' in c and "trajectory" in c.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_grounded_prose.py -v`
Expected: FAIL — grounded roles/contexts undefined.

- [ ] **Step 3: Implement grounded roles and contexts**

In `autoqc/agent/checks.py` (near `factual_role`/`factual_context`):

```python
_GROUNDED_PROMPT_SYS = (
    "You verify, against the real repository checked out at /testbed (base_commit, "
    "network-isolated), whether a SWE task PROMPT genuinely requires exploring the codebase. "
    "Use run_bash (cat/grep/rg/find/ls/git) to judge whether a correct answer could be given "
    "from general knowledge alone, or truly needs repo-specific investigation. Finish with "
    "submit_findings: exactly one finding; evidence must include a path:line citation.")

_GROUNDED_ANSWER_SYS = (
    "You verify, against the real repository at /testbed (base_commit, network-isolated), "
    "whether a task's reference ANSWER is trajectory-like: it must demonstrate genuine codebase "
    "exploration and must NOT be a bare direct answer NOR a raw full-trajectory dump. Use "
    "run_bash to confirm the answer's cited paths/mechanisms are real and that answering "
    "required exploration. Finish with submit_findings: one finding; evidence must cite path:line.")

def grounded_prompt_role() -> Role:
    return Role(name="grounded_prompt", system_prompt=_GROUNDED_PROMPT_SYS, tools=factual_tools())

def grounded_answer_role() -> Role:
    return Role(name="grounded_answer", system_prompt=_GROUNDED_ANSWER_SYS, tools=factual_tools())

def _grounded_head(bundle) -> str:
    return (f"Repository: {getattr(bundle, 'repository', '') or ''} at base_commit "
            f"{getattr(bundle, 'base_commit', '') or ''} (checked out at /testbed).\n\n")

def grounded_prompt_context(bundle) -> str:
    return (_grounded_head(bundle) +
            f"Task prompt (tests/prompt.txt):\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            "Check P04 — Requires codebase exploration. VIOLATES (passed=false) if a correct "
            "answer could be produced without exploring THIS repo (general knowledge / direct "
            "answer). passed=true if answering demands repo-specific investigation. Submit EXACTLY "
            "ONE finding: check_id=P04, criterion_id=\"prompt\"; evidence must cite path:line.")

def grounded_answer_context(bundle) -> str:
    return (_grounded_head(bundle) +
            f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"Reference answer (solution/answer.txt):\n{getattr(bundle, 'answer', '') or ''}\n\n"
            "Check A06 — Trajectory-like exploration. VIOLATES (passed=false) if the answer is a "
            "bare direct answer with no exploration, OR a raw full-trajectory dump, OR its cited "
            "exploration does not hold up against the repo. passed=true if it shows genuine, real "
            "codebase exploration. Submit EXACTLY ONE finding: check_id=A06, "
            "criterion_id=\"answer\"; evidence must cite path:line.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_grounded_prose.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/checks.py tests/test_grounded_prose.py
git commit -m "feat(P04,A06): grounded prose roles and repo-verification contexts"
```

---

### Task 11: Grounded stage — share one container across `Q06` + `P04` + `A06`

**Files:**
- Modify: `autoqc/agent/engine.py` (generalize the factual runner; add `run_grounded_stage`; call it from `run_semantic`)
- Test: `tests/test_grounded_stage.py`, `tests/test_factual.py` (verify Q06 unchanged)

**Interfaces:**
- Produces:
  - `run_grounded_prose(check_id, name, severity, role, context_text, client, ctx) -> CheckResult` — two independent grounded passes over one unit; both must agree, else `needs_human`.
  - `run_grounded_stage(bundle, client, votes_log=None, limits=None, docker=docker_available) -> list[CheckResult]` — builds ONE `ContainerSession`, runs `run_factual` (Q06) + `P04` + `A06` against it, returns three results; on no-Dockerfile / no-Docker / setup error returns all three as `needs_human` (fail-safe), reusing the `_q06_needs_human` pattern generalized to each id.
- Consumes: `ContainerSession`, `factual_tools`, grounded roles/contexts (Task 10), `AgentContext`.
- Changes: `run_semantic` calls `run_grounded_stage` (returning a list) instead of appending only `run_factual_stage`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grounded_stage.py
import types
from autoqc.model import Stage, Severity, Verdict
from autoqc.agent.engine import run_grounded_stage

class _FakeSession:
    def __init__(self, bundle, **kw): pass
    def ensure_image(self): return "img"
    def start(self): return "c"
    def exec(self, cmd, **kw): return "aiohttp/client_reqrep.py:631: conn.close()"
    def stop(self): pass

def _bundle():
    return types.SimpleNamespace(
        root=".", rubrics=[{"id": "a" * 32, "title": "1.1: States X",
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}],
        prompt="Why does the pool reopen sockets?", answer="I traced conn.close() ...",
        instruction="i", repository="aio-libs/aiohttp", base_commit="a" * 40,
        files_present={"environment/Dockerfile": True})

def test_grounded_stage_returns_q06_p04_a06(monkeypatch):
    from autoqc.agent import engine
    monkeypatch.setattr(engine, "ContainerSession", _FakeSession)
    from autoqc.llm import FakeLLMClient
    def responder(m, t):
        # every grounded agent submits a passing finding for its own unit
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "Q06", "criterion_id": "a" * 32, "passed": True, "evidence": ["client_reqrep.py:631"]},
            {"check_id": "P04", "criterion_id": "prompt", "passed": True, "evidence": ["client_reqrep.py:631"]},
            {"check_id": "A06", "criterion_id": "answer", "passed": True, "evidence": ["client_reqrep.py:631"]}]}}]}
    results = run_grounded_stage(_bundle(), FakeLLMClient(responder), docker=lambda: True)
    ids = {r.id for r in results}
    assert ids == {"Q06", "P04", "A06"}
    assert all(r.stage is Stage.FACTUAL for r in results)

def test_grounded_stage_failsafe_without_docker():
    results = run_grounded_stage(_bundle(), None, docker=lambda: False)
    assert {r.id for r in results} == {"Q06", "P04", "A06"}
    assert all(r.needs_human for r in results)  # never hard-reject on infra absence
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_grounded_stage.py -v`
Expected: FAIL — `run_grounded_stage` undefined.

- [ ] **Step 3: Implement the grounded stage**

In `autoqc/agent/engine.py`, add imports:

```python
from autoqc.agent.checks import (grounded_prompt_role, grounded_answer_role,
                                 grounded_prompt_context, grounded_answer_context)
from autoqc.model import Severity
```

Add a generic two-pass grounded prose runner and the stage. Reuse `_files` (already in engine) for same-file agreement:

```python
def _grounded_needs_human(cid, name, severity, reason) -> CheckResult:
    return CheckResult(id=cid, name=name, stage=Stage.FACTUAL, severity=severity,
                       passed=False, needs_human=True, detail=f"{cid} not run: {reason}")

def run_grounded_prose(cid, name, severity, role, context_text, client, ctx,
                       unit, votes_log=None) -> CheckResult:
    rounds = []
    for _ in range(2):
        res = run_agent(role, context_text, client, ctx, max_turns=FACTUAL_MAX_TURNS)
        if votes_log is not None:
            votes_log.append({"check": cid, "role": "grounded", "ok": res.ok, "findings": res.findings})
        own = _own(res.findings, cid, {unit}) if res.ok else []
        v = next((f for f in own if f.get("criterion_id") == unit), None)
        rounds.append((bool(v.get("passed")) if v else None,
                       _files(v.get("evidence")) if v else set(),
                       list(v.get("evidence") or []) if v else []))
    (p1, f1, e1), (p2, f2, e2) = rounds
    evidence = (e1 + e2)[:20]
    if p1 is None or p2 is None or p1 != p2:
        return CheckResult(id=cid, name=name, stage=Stage.FACTUAL, severity=severity,
                           passed=False, needs_human=True, evidence=evidence,
                           detail="grounded rounds disagreed or did not submit")
    if p1:
        return CheckResult(id=cid, name=name, stage=Stage.FACTUAL, severity=severity,
                           passed=True, evidence=evidence)
    same = bool(f1 & f2)
    return CheckResult(id=cid, name=name, stage=Stage.FACTUAL, severity=severity,
                       passed=False, needs_human=not same, evidence=evidence,
                       detail="grounded rounds agree the document violates the check")

def run_grounded_stage(bundle, client, votes_log=None, limits=None,
                       docker=docker_available) -> list[CheckResult]:
    specs = [("Q06", Q06.name, Severity.REJECT),
             ("P04", "Requires codebase exploration", Severity.REJECT),
             ("A06", "Trajectory-like exploration", Severity.REJECT)]
    def _all_needs_human(reason):
        return [_grounded_needs_human(cid, name, sev, reason) for cid, name, sev in specs]

    if not getattr(bundle, "files_present", {}).get("environment/Dockerfile", True):
        return _all_needs_human("no environment/Dockerfile in bundle")
    if not docker():
        return _all_needs_human("Docker is not available")
    try:
        session = ContainerSession(bundle, limits=limits)
        session.ensure_image()
        session.start()
    except ContainerError as e:
        return _all_needs_human(str(e))
    except Exception as e:
        return _all_needs_human(f"container setup error: {e}")
    try:
        ctx = AgentContext(bundle_dir=bundle.root, container=session)
        out = [run_factual(bundle, client, ctx, votes_log=votes_log)]
        out.append(run_grounded_prose(
            "P04", "Requires codebase exploration", Severity.REJECT,
            grounded_prompt_role(), grounded_prompt_context(bundle), client, ctx,
            "prompt", votes_log=votes_log))
        out.append(run_grounded_prose(
            "A06", "Trajectory-like exploration", Severity.REJECT,
            grounded_answer_role(), grounded_answer_context(bundle), client, ctx,
            "answer", votes_log=votes_log))
        return out
    except Exception as e:
        return _all_needs_human(f"grounded pass error: {e}")
    finally:
        session.stop()
```

Update `run_semantic` (replace the `if factual:` block):

```python
def run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None, factual=True):
    results = run_checks_parallel(checks, bundle, client, ctx, k, votes_log=votes_log)
    results += [fn(bundle) for fn in DETERMINISTIC_CHECKS]
    if factual:
        results += run_grounded_stage(bundle, client, votes_log=votes_log)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_grounded_stage.py tests/test_factual.py tests/test_engine_run.py -v`
Expected: PASS (Q06 behavior preserved; grounded stage returns three results).

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/engine.py tests/test_grounded_stage.py
git commit -m "feat(engine): shared-container grounded stage (Q06 + P04 + A06)"
```

---

### Task 12: Calibrate the Phase 1.5 set against the Batch-1 negatives and the clean sample

**Files:**
- Create: `scripts/calibrate_phase15.py`
- Test: `tests/test_calibrate_phase15.py`

**Interfaces:**
- Consumes: `autoqc.cli.run`, the 12 Batch-1 negative bundles, the clean `qc-sample-pack` bundles.
- Produces: `summarize_phase15(records) -> dict` with `recall` (share of known-reject negatives that got a Phase-1.5 reject) and `reject_false_fires` (Phase-1.5 rejects on the clean set — target 0), plus a per-check fire table. The script prints this for operator inspection.

> This task delivers a reusable, *testable* summarizer (unit-tested offline) plus an operator entry point. Running it against live bundles + gateway is a manual calibration step, not part of the test suite.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate_phase15.py
from scripts.calibrate_phase15 import summarize_phase15

PHASE15_IDS = {"P01","P02","P03","P04","A01","A02","A03","A04","A05","A06","AL01","H01","Q13"}

def _rec(results):
    return {"results": results}

def test_recall_counts_phase15_rejects_on_negatives():
    # one negative bundle where P01 rejected -> counts as recalled
    neg = [_rec([{"id": "P01", "severity": "reject", "passed": False, "needs_human": False}])]
    clean = []
    s = summarize_phase15(negatives=neg, cleans=clean)
    assert s["recall"] == 1.0

def test_reject_false_fires_counts_clean_rejects():
    clean = [_rec([{"id": "A04", "severity": "reject", "passed": False, "needs_human": False}])]
    s = summarize_phase15(negatives=[], cleans=clean)
    assert s["reject_false_fires"] == 1

def test_warn_and_disputed_are_not_false_fires():
    clean = [_rec([
        {"id": "A01", "severity": "warn", "passed": False, "needs_human": False},
        {"id": "P04", "severity": "reject", "passed": False, "needs_human": True}])]  # disputed -> human
    s = summarize_phase15(negatives=[], cleans=clean)
    assert s["reject_false_fires"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_calibrate_phase15.py -v`
Expected: FAIL — module `scripts.calibrate_phase15` does not exist.

- [ ] **Step 3: Implement the summarizer + operator entry point**

```python
# scripts/calibrate_phase15.py
"""Score the Phase 1.5 checks: recall on Batch-1 negatives, reject-false-fires on the
clean qc-sample-pack. Offline summarizer is unit-tested; live runs are manual."""
from __future__ import annotations
import sys
from pathlib import Path

PHASE15_IDS = {"P01","P02","P03","P04","A01","A02","A03","A04","A05","A06","AL01","H01","Q13"}

def _hard_reject_ids(record) -> set[str]:
    out = set()
    for r in record.get("results", []):
        if (r.get("id") in PHASE15_IDS and r.get("severity") == "reject"
                and not r.get("passed") and not r.get("needs_human")):
            out.add(r["id"])
    return out

def summarize_phase15(negatives: list[dict], cleans: list[dict]) -> dict:
    recalled = sum(1 for rec in negatives if _hard_reject_ids(rec))
    recall = (recalled / len(negatives)) if negatives else 0.0
    false_fires = sum(len(_hard_reject_ids(rec)) for rec in cleans)
    fires: dict[str, int] = {}
    for rec in negatives + cleans:
        for cid in _hard_reject_ids(rec):
            fires[cid] = fires.get(cid, 0) + 1
    return {"recall": recall, "recalled": recalled, "n_negatives": len(negatives),
            "reject_false_fires": false_fires, "n_cleans": len(cleans), "per_check_fires": fires}

def _run_dir(bundle_dirs: list[str]) -> list[dict]:
    import json
    from autoqc.cli import run
    from autoqc.model import Verdict
    out = []
    for i, d in enumerate(bundle_dirs):
        od = Path("phase15_out") / f"b{i}"
        run(Path(d), od)  # writes review_record.json
        out.append(json.loads((od / "review_record.json").read_text()))
    return out

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: calibrate_phase15.py <neg_dir>... -- <clean_dir>...", file=sys.stderr)
        return 64
    cut = argv.index("--")
    negs, cleans = _run_dir(argv[:cut]), _run_dir(argv[cut + 1:])
    s = summarize_phase15(negs, cleans)
    print(f"recall={s['recall']:.2f} ({s['recalled']}/{s['n_negatives']})  "
          f"reject_false_fires={s['reject_false_fires']} over {s['n_cleans']} clean")
    print("per-check fires:", s["per_check_fires"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_calibrate_phase15.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite, then commit**

```bash
python3 -m pytest -q
git add scripts/calibrate_phase15.py tests/test_calibrate_phase15.py
git commit -m "feat(calibrate): Phase 1.5 recall / reject-false-fire summarizer"
```

- [ ] **Step 6: Manual live calibration (operator, not CI)**

Extract the negatives and clean sample, then run against the gateway (Docker up for the grounded stage):

```bash
# negatives: the placeholder/prose reworks; cleans: qc-sample-pack Turing tasks
python3 scripts/calibrate_phase15.py <neg_bundle_1> <neg_bundle_2> ... -- <clean_1> <clean_2> ...
```

Record recall and `reject_false_fires` (target 0 on clean). Note per-check flip behavior on the grounded checks (the known single-run instability caveat).

---

## Self-Review

**1. Spec coverage** — every spec §4 check maps to a task: P01→T3, H01→T4, P02/P03→T7, A01–A05→T8, A04 (reject) in T8, AL01/Q13→T9, P04/A06→T10–T11. Spec §5.1 (deterministic module)→T3–T5; §5.2 (whole-doc modes, prose roles, engine role-select)→T6–T9; §5.3 (shared container)→T11; §5.4 (`bundle.instruction`, verdict/report unchanged)→T1 (+ verdict/report untouched by construction); §6 (offline unit tests + calibration)→every task's tests + T12.

**2. Placeholder scan** — no "TBD/handle edge cases/similar to Task N". Every code and test step carries real content; guidance strings are the actual check definitions.

**3. Type consistency** — `SemanticCheck(id,name,severity,scope,guidance,unit_mode,role_kind)` used identically in T6–T9; `check_p01`/`check_h01`/`TEXT_DETERMINISTIC_CHECKS` names match across T3–T5; `run_grounded_stage`/`run_grounded_prose` signatures match their call sites in T11; `criterion_id` unit ids (`"prompt"`,`"answer"`,`"bundle"`,`"rubric"`) match between contexts (T6/T10) and the `_check_prep`-derived `allowed` sets; `summarize_phase15(negatives, cleans)` matches its test and `main` call (T12).

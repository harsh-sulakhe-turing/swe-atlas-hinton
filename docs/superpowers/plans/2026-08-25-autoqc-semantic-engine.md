# AutoQC Semantic Engine (Milestone 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Stage-2 semantic layer to AutoQC: a mockable LLM client, an ensemble + asymmetric-adversary judging engine, the first semantic check (Q07 negative-semantics), a seed-defect generator, and pipeline wiring — all CI-testable with a fake LLM (no network, no keys).

**Architecture:** `llm.py` defines an `LLMClient` interface with a real `GatewayLLMClient` (OpenAI-compatible, env-configured, lazy-imports `openai`) and a deterministic `FakeLLMClient` for tests. `semantic/base.py` defines the `SemanticCheck` contract (units + proposer/adversary message builders + a result parser). `semantic/engine.py` runs one check: K independent proposers → majority aggregate → one asymmetric adversary round (defend rejects / attack passes) → adjudicate (unanimous-and-unrefuted stands; split or overturned → `needs_human`) → one `CheckResult`. `semantic/checks.py` implements Q07 and the check registry. `seed.py` injects one known defect into a clean rubric to produce labeled test mutants. The CLI runs structural then semantic; semantic is skipped when no client is available, so the existing structural-only tests stay green.

**Tech Stack:** Python 3.11+ target (runs on this machine's 3.10 via existing tomli fallback). `pytest`. The real client uses the `openai` SDK (already a bundle dependency) but it is **lazy-imported inside `judge()`**, so nothing in this plan requires `openai` to be installed — all tests use `FakeLLMClient`.

**Spec:** `docs/superpowers/specs/2026-08-25-autoqc-execution-harness-design.md` (§3 the semantic engine, §5 adjudication) and the rubric spec `2026-08-25-autoqc-quality-rubric-design.md` (Q07 definition).

## Global Constraints

- Run every test with `python3 -m pytest` (bare `pytest` is not on PATH on this machine).
- **All tests use `FakeLLMClient`** — no test may make a network call or require `openai`/keys. The real `GatewayLLMClient.judge()` is exercised only in manual/live runs, never in the suite.
- Reuse the existing `CheckResult`/`Verdict`/`Severity`/`Stage` model from `autoqc/model.py` unchanged. Semantic results use `stage=Stage.SEMANTIC`.
- Each semantic check produces exactly ONE `CheckResult` (id like `"Q07"`), aggregating its per-unit judgments: `passed` = all units passed; `needs_human` = any unit needs human; `detail` names the failing units.
- Q07 severity is `Severity.REJECT`. Q07 is a per-criterion check over the NEGATIVE (`2.x`) criteria only.
- Gateway env var names (mirror the bundle's `evaluate_answer.py`): `EVAL_API_KEY`, `EVAL_BASE_URL`, `EVAL_MODEL` (default `anthropic/claude-opus-4-5-20251101`).
- Backward compatibility: `cli.run(bundle_dir, out_dir)` with no client and no gateway env must behave exactly as Milestone 1 (structural-only) so existing `tests/test_cli_e2e.py` stays green.

---

## File Structure

- `autoqc/llm.py` — `LLMClient`, `FakeLLMClient`, `GatewayLLMClient`.
- `autoqc/semantic/__init__.py` — package marker.
- `autoqc/semantic/base.py` — `SemanticJudgment`, `Unit`, `SemanticCheck`.
- `autoqc/semantic/engine.py` — `run_check(check, bundle, llm, k=3) -> CheckResult`.
- `autoqc/semantic/checks.py` — `NegativeSemanticsCheck` (Q07) + `SEMANTIC_CHECKS` registry.
- `autoqc/seed.py` — defect injectors (`seed_bad_negative`).
- `autoqc/cli.py` — MODIFIED: run structural + semantic, skip semantic when no client.
- `tests/` — one module per new source file.

---

### Task 1: LLM client interface (fake + gateway)

**Files:**
- Create: `autoqc/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `LLMClient` (base, `judge(messages: list[dict]) -> dict` raising NotImplementedError). `FakeLLMClient(responder)` where `responder` is a callable `(messages) -> dict`; `judge` returns `responder(messages)`. `GatewayLLMClient(api_key=None, base_url=None, model=None)` reading the `EVAL_*`/`OPENAI_*` env fallbacks; `.available() -> bool` (true when api_key and base_url are both set); `judge()` lazy-imports `openai`. `default_client() -> LLMClient | None` returns a `GatewayLLMClient` when it is `.available()`, else `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import pytest
from autoqc.llm import LLMClient, FakeLLMClient, GatewayLLMClient, default_client


def test_fake_client_returns_responder_output():
    fake = FakeLLMClient(lambda messages: {"passed": True, "seen": len(messages)})
    out = fake.judge([{"role": "user", "content": "hi"}])
    assert out == {"passed": True, "seen": 1}


def test_base_client_is_abstract():
    with pytest.raises(NotImplementedError):
        LLMClient().judge([])


def test_gateway_available_reflects_env(monkeypatch):
    monkeypatch.delenv("EVAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    assert GatewayLLMClient().available() is False
    assert default_client() is None
    monkeypatch.setenv("EVAL_API_KEY", "k")
    monkeypatch.setenv("EVAL_BASE_URL", "http://gw")
    assert GatewayLLMClient().available() is True
    dc = default_client()
    assert isinstance(dc, GatewayLLMClient) and dc.model  # default model set


def test_gateway_default_model():
    g = GatewayLLMClient(api_key="k", base_url="http://gw")
    assert g.model == "anthropic/claude-opus-4-5-20251101"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.llm'`

- [ ] **Step 3: Write `autoqc/llm.py`**

```python
from __future__ import annotations
import json
import os


class LLMClient:
    """Interface: judge(messages) -> parsed JSON dict."""
    def judge(self, messages: list[dict]) -> dict:
        raise NotImplementedError


class FakeLLMClient(LLMClient):
    """Deterministic client for tests. `responder` maps messages -> dict."""
    def __init__(self, responder):
        self._responder = responder
        self.calls: list[list[dict]] = []

    def judge(self, messages: list[dict]) -> dict:
        self.calls.append(messages)
        return self._responder(messages)


class GatewayLLMClient(LLMClient):
    """OpenAI-compatible client pointed at the token gateway. Lazy-imports openai."""
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.environ.get("EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("EVAL_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        self.model = model or os.environ.get("EVAL_MODEL", "anthropic/claude-opus-4-5-20251101")

    def available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def judge(self, messages: list[dict]) -> dict:
        from openai import OpenAI  # lazy: not needed for tests
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model, messages=messages,
            response_format={"type": "json_object"}, max_tokens=1024,
        )
        return json.loads(resp.choices[0].message.content)


def default_client() -> LLMClient | None:
    g = GatewayLLMClient()
    return g if g.available() else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_llm.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/llm.py tests/test_llm.py
git commit -m "feat(autoqc): LLM client interface (fake + gateway)"
```

---

### Task 2: Semantic check contract

**Files:**
- Create: `autoqc/semantic/__init__.py`
- Create: `autoqc/semantic/base.py`
- Test: `tests/test_semantic_base.py`

**Interfaces:**
- Produces: `SemanticJudgment(passed: bool, evidence: list[str]=[], reason: str="")`. `Unit(key: str, payload: dict)`. `SemanticCheck` with class attrs `id/name` (str), `severity` (Severity, default REJECT) and methods `units(bundle) -> list[Unit]`, `proposer_messages(bundle, unit) -> list[dict]`, `adversary_messages(bundle, unit, agg_passed: bool) -> list[dict]`, `parse(raw: dict) -> SemanticJudgment` — the base raises NotImplementedError for the four methods.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_semantic_base.py
import pytest
from autoqc.model import Severity
from autoqc.semantic.base import SemanticJudgment, Unit, SemanticCheck


def test_judgment_defaults():
    j = SemanticJudgment(passed=True)
    assert j.evidence == [] and j.reason == ""


def test_unit_holds_payload():
    u = Unit(key="abc", payload={"title": "1.1: x"})
    assert u.key == "abc" and u.payload["title"] == "1.1: x"


def test_base_check_methods_abstract():
    c = SemanticCheck()
    assert c.severity is Severity.REJECT
    for call in (lambda: c.units(None),
                 lambda: c.proposer_messages(None, None),
                 lambda: c.adversary_messages(None, None, True),
                 lambda: c.parse({})):
        with pytest.raises(NotImplementedError):
            call()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_semantic_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.semantic'`

- [ ] **Step 3: Write the files**

```python
# autoqc/semantic/__init__.py
```

```python
# autoqc/semantic/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from autoqc.model import Severity


@dataclass
class SemanticJudgment:
    passed: bool
    evidence: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class Unit:
    key: str
    payload: dict


class SemanticCheck:
    id: str = ""
    name: str = ""
    severity: Severity = Severity.REJECT

    def units(self, bundle) -> list[Unit]:
        raise NotImplementedError

    def proposer_messages(self, bundle, unit: Unit) -> list[dict]:
        raise NotImplementedError

    def adversary_messages(self, bundle, unit: Unit, agg_passed: bool) -> list[dict]:
        raise NotImplementedError

    def parse(self, raw: dict) -> SemanticJudgment:
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_semantic_base.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/semantic/__init__.py autoqc/semantic/base.py tests/test_semantic_base.py
git commit -m "feat(autoqc): semantic check contract"
```

---

### Task 3: The ensemble + adversary engine

**Files:**
- Create: `autoqc/semantic/engine.py`
- Test: `tests/test_semantic_engine.py`

**Interfaces:**
- Consumes: `SemanticCheck`, `Unit`, `SemanticJudgment` (Task 2); `LLMClient`/`FakeLLMClient` (Task 1); `CheckResult`, `Stage`, `Severity` (model).
- Produces: `run_check(check: SemanticCheck, bundle, llm: LLMClient, k: int = 3) -> CheckResult`. Per unit: call `llm.judge(check.proposer_messages(...))` k times → `check.parse` each → majority (`votes*2 > k`) = `agg_passed`; `split = votes not in (0, k)`; call `llm.judge(check.adversary_messages(..., agg_passed))` once → `overturn = adv.passed != agg_passed`; unit `needs_human = split or overturn`. The check's `CheckResult`: `passed = all unit agg_passed`, `needs_human = any unit needs_human`, `evidence` = concatenated (capped 20), `detail` names failing unit keys (empty if passed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_semantic_engine.py
from autoqc.model import Severity, Stage
from autoqc.semantic.base import SemanticCheck, SemanticJudgment, Unit
from autoqc.llm import FakeLLMClient
from autoqc.semantic.engine import run_check


class StubCheck(SemanticCheck):
    """One unit; proposer verdict driven by the responder; role tag in messages lets the
    responder distinguish proposer vs adversary calls."""
    id = "QX"
    name = "Stub"
    severity = Severity.REJECT

    def units(self, bundle):
        return [Unit(key="u1", payload={})]

    def proposer_messages(self, bundle, unit):
        return [{"role": "system", "content": "PROPOSER"}]

    def adversary_messages(self, bundle, unit, agg_passed):
        return [{"role": "system", "content": f"ADVERSARY agg={agg_passed}"}]

    def parse(self, raw):
        return SemanticJudgment(passed=bool(raw.get("passed")), evidence=raw.get("evidence", []))


def _responder(proposer_passed, adversary_passed):
    def r(messages):
        tag = messages[0]["content"]
        if tag.startswith("PROPOSER"):
            return {"passed": proposer_passed, "evidence": ["p"]}
        return {"passed": adversary_passed, "evidence": ["a"]}
    return r


def test_unanimous_pass_unrefuted_is_sound():
    llm = FakeLLMClient(_responder(True, True))  # adversary attacks a pass but agrees it passes
    r = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert r.id == "QX" and r.stage is Stage.SEMANTIC
    assert r.passed is True and r.needs_human is False


def test_unanimous_reject_unrefuted_stays_reject():
    llm = FakeLLMClient(_responder(False, False))  # adversary defends but agrees it fails
    r = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert r.passed is False and r.needs_human is False
    assert "u1" in r.detail


def test_adversary_overturn_flags_needs_human():
    # proposers unanimously pass, adversary finds it should fail -> overturn
    llm = FakeLLMClient(_responder(True, False))
    r = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert r.passed is True and r.needs_human is True


def test_split_vote_flags_needs_human():
    # alternate proposer verdicts to force a split; adversary agrees with majority
    state = {"n": 0}
    def r(messages):
        if messages[0]["content"].startswith("PROPOSER"):
            state["n"] += 1
            return {"passed": state["n"] % 2 == 0, "evidence": []}  # T/F/T over 3 calls -> 1 pass? adjust
        return {"passed": False, "evidence": []}
    llm = FakeLLMClient(r)
    res = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert res.needs_human is True


def test_evidence_is_capped():
    llm = FakeLLMClient(lambda m: {"passed": True, "evidence": ["x"] * 50})
    r = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert len(r.evidence) <= 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_semantic_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.semantic.engine'`

- [ ] **Step 3: Write `autoqc/semantic/engine.py`**

```python
from __future__ import annotations
from autoqc.model import CheckResult, Stage
from autoqc.semantic.base import SemanticCheck


def run_check(check: SemanticCheck, bundle, llm, k: int = 3) -> CheckResult:
    unit_passed: list[bool] = []
    unit_needs_human: list[bool] = []
    failing_keys: list[str] = []
    evidence: list[str] = []

    for unit in check.units(bundle):
        judgments = [check.parse(llm.judge(check.proposer_messages(bundle, unit)))
                     for _ in range(k)]
        votes = sum(1 for j in judgments if j.passed)
        agg_passed = votes * 2 > k
        split = votes not in (0, k)

        adv = check.parse(llm.judge(check.adversary_messages(bundle, unit, agg_passed)))
        overturn = adv.passed != agg_passed
        needs_human = split or overturn

        unit_passed.append(agg_passed)
        unit_needs_human.append(needs_human)
        if not agg_passed:
            failing_keys.append(unit.key)
        for j in judgments + [adv]:
            evidence.extend(j.evidence)

    passed = all(unit_passed)  # True when there are no units
    needs_human = any(unit_needs_human)
    detail = "" if passed else "failing units: " + ", ".join(failing_keys)
    return CheckResult(
        id=check.id, name=check.name, stage=Stage.SEMANTIC, severity=check.severity,
        passed=passed, needs_human=needs_human, evidence=evidence[:20], detail=detail,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_semantic_engine.py -v`
Expected: PASS (5 tests)

If `test_split_vote_flags_needs_human` does not split with k=3 (the T/F/T pattern yields votes=1, which is a split since 1 not in (0,3)), it still asserts `needs_human is True` — correct. Keep as written.

- [ ] **Step 5: Commit**

```bash
git add autoqc/semantic/engine.py tests/test_semantic_engine.py
git commit -m "feat(autoqc): ensemble + asymmetric-adversary engine"
```

---

### Task 4: Q07 negative-semantics check

**Files:**
- Create: `autoqc/semantic/checks.py`
- Test: `tests/test_check_q07.py`

**Interfaces:**
- Consumes: `SemanticCheck`, `SemanticJudgment`, `Unit` (Task 2); `Severity`.
- Produces: `NegativeSemanticsCheck` (id `"Q07"`, name `"Negative score-flip semantics"`, severity REJECT) — `units` yields one Unit per NEGATIVE (`2.x`) criterion, keyed by the criterion id, payload `{"title": <title>}`; proposer/adversary message builders; `parse`. Also `SEMANTIC_CHECKS = [NegativeSemanticsCheck()]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_check_q07.py
import types
from autoqc.semantic.checks import NegativeSemanticsCheck, SEMANTIC_CHECKS
from autoqc.semantic.engine import run_check
from autoqc.llm import FakeLLMClient
from autoqc.model import Severity


def _bundle(items):
    return types.SimpleNamespace(rubrics=items)


def _pos(idx, id_):
    return {"id": id_, "title": f"{idx}: States X",
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def _neg(idx, id_, title):
    return {"id": id_, "title": title,
            "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def test_units_are_only_negatives():
    c = NegativeSemanticsCheck()
    b = _bundle([_pos("1.1", "a" * 32),
                 _neg("2.1", "b" * 32, "2.1: Claims that X is false")])
    units = c.units(b)
    assert len(units) == 1 and units[0].key == "b" * 32


def test_registry_contains_q07():
    assert any(c.id == "Q07" for c in SEMANTIC_CHECKS)
    assert NegativeSemanticsCheck().severity is Severity.REJECT


def test_parse_reads_passed_and_evidence():
    j = NegativeSemanticsCheck().parse({"passed": True, "evidence": ["ok"], "reason": "r"})
    assert j.passed is True and j.evidence == ["ok"] and j.reason == "r"


def test_prompts_include_the_title():
    c = NegativeSemanticsCheck()
    u = c.units(_bundle([_neg("2.1", "b" * 32, "2.1: Does not claim bytes fail")]))[0]
    pm = c.proposer_messages(None, u)
    am = c.adversary_messages(None, u, True)
    assert any("Does not claim bytes fail" in m["content"] for m in pm)
    assert any("Does not claim bytes fail" in m["content"] for m in am)
    # adversary stance flips on agg_passed
    am_pass = " ".join(m["content"] for m in c.adversary_messages(None, u, True))
    am_fail = " ".join(m["content"] for m in c.adversary_messages(None, u, False))
    assert am_pass != am_fail


def test_end_to_end_flags_bad_negative_with_fake_llm():
    # A well-phrased negative -> proposers pass; a "Does not claim" one -> proposers fail.
    def responder(messages):
        text = " ".join(m["content"] for m in messages)
        bad = "Does not claim" in text
        # proposer: passed=not bad ; adversary agrees (no overturn)
        return {"passed": (not bad), "evidence": []}
    good = _bundle([_neg("2.1", "b" * 32, "2.1: Claims that bytes bodies fail")])
    bad = _bundle([_neg("2.1", "c" * 32, "2.1: Does not claim that bytes bodies fail")])
    r_good = run_check(NegativeSemanticsCheck(), good, FakeLLMClient(responder), k=3)
    r_bad = run_check(NegativeSemanticsCheck(), bad, FakeLLMClient(responder), k=3)
    assert r_good.passed is True
    assert r_bad.passed is False and "c" * 32 in r_bad.detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_check_q07.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.semantic.checks'`

- [ ] **Step 3: Write `autoqc/semantic/checks.py`**

```python
from __future__ import annotations
from autoqc.model import Severity
from autoqc.semantic.base import SemanticCheck, SemanticJudgment, Unit


def _is_negative(it: dict) -> bool:
    return isinstance(it, dict) and "negative" in str(
        it.get("annotations", {}).get("type", "") if isinstance(it.get("annotations"), dict) else "")


class NegativeSemanticsCheck(SemanticCheck):
    """Q07: a negative criterion must state the FALSE assertion whose presence should fail
    the answer ("Claims that ..."), never "Does not claim...", an omission, or the correct
    behavior."""
    id = "Q07"
    name = "Negative score-flip semantics"
    severity = Severity.REJECT

    def units(self, bundle):
        items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
        return [Unit(key=str(it.get("id", "?")), payload={"title": str(it.get("title", ""))})
                for it in items if _is_negative(it)]

    def proposer_messages(self, bundle, unit):
        system = (
            "You judge whether a NEGATIVE rubric criterion is phrased correctly. A negative "
            "criterion must state a FALSE assertion whose PRESENCE in an answer should fail it "
            "(for example: 'Claims that every retry uses exponential backoff'). It is INCORRECT "
            "if phrased as 'Does not claim...', as a required omission, or as the correct behavior. "
            'Reply ONLY as JSON: {"passed": <bool>, "reason": <str>, "evidence": [<str>]}. '
            "passed=true means it is correctly phrased as a false assertion to catch."
        )
        user = f"Negative criterion title:\n{unit.payload['title']}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def adversary_messages(self, bundle, unit, agg_passed):
        if agg_passed:
            stance = ("A prior review judged this negative criterion CORRECTLY phrased. Adversarially "
                      "look for ANY way it actually reads as an omission, a 'does not claim' form, or the "
                      "correct behavior rather than a false assertion to catch.")
        else:
            stance = ("A prior review judged this negative criterion INCORRECTLY phrased. Adversarially "
                      "check whether it actually IS a valid false assertion whose presence should fail an answer.")
        system = (
            "You are an adversarial reviewer of negative-rubric phrasing. " + stance +
            ' Reply ONLY as JSON: {"passed": <bool>, "reason": <str>, "evidence": [<str>]}, '
            "where passed=true means the criterion IS correctly phrased as a false assertion to catch."
        )
        user = f"Negative criterion title:\n{unit.payload['title']}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def parse(self, raw):
        ev = raw.get("evidence") or []
        if not isinstance(ev, list):
            ev = [str(ev)]
        return SemanticJudgment(
            passed=bool(raw.get("passed")),
            evidence=[str(x) for x in ev],
            reason=str(raw.get("reason", "")),
        )


SEMANTIC_CHECKS = [NegativeSemanticsCheck()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_check_q07.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/semantic/checks.py tests/test_check_q07.py
git commit -m "feat(autoqc): Q07 negative-semantics check"
```

---

### Task 5: Seed-defect generator

**Files:**
- Create: `autoqc/seed.py`
- Test: `tests/test_seed.py`

**Interfaces:**
- Consumes: nothing from autoqc (operates on a rubric item list).
- Produces: `seed_bad_negative(items: list[dict]) -> tuple[list[dict], str | None]` — returns a deep-copied item list in which the first negative criterion's title is rewritten into a "Does not claim..." form (a Q07 defect), plus the id of the mutated criterion (or `None` if there was no negative). Original `items` is not modified.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed.py
import copy
from autoqc.seed import seed_bad_negative
from autoqc.semantic.checks import NegativeSemanticsCheck
from autoqc.semantic.engine import run_check
from autoqc.llm import FakeLLMClient
import types


def _neg(id_, title):
    return {"id": id_, "title": title,
            "annotations": {"type": "negative hli verifier", "importance": "must have"}}


def _pos(id_):
    return {"id": id_, "title": "1.1: States X",
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def test_seed_rewrites_first_negative_and_reports_id():
    items = [_pos("a" * 32), _neg("b" * 32, "2.1: Claims that bytes bodies fail")]
    original = copy.deepcopy(items)
    mutated, changed_id = seed_bad_negative(items)
    assert changed_id == "b" * 32
    assert items == original  # input not mutated
    neg = [it for it in mutated if "negative" in it["annotations"]["type"]][0]
    assert "Does not claim" in neg["title"]
    assert "Claims that" not in neg["title"]


def test_seed_returns_none_when_no_negative():
    items = [_pos("a" * 32)]
    mutated, changed_id = seed_bad_negative(items)
    assert changed_id is None


def test_seeded_mutant_trips_q07_via_engine():
    items = [_neg("b" * 32, "2.1: Claims that bytes bodies fail")]
    mutated, _ = seed_bad_negative(items)

    def responder(messages):
        text = " ".join(m["content"] for m in messages)
        return {"passed": ("Does not claim" not in text), "evidence": []}

    r = run_check(NegativeSemanticsCheck(), types.SimpleNamespace(rubrics=mutated),
                  FakeLLMClient(responder), k=3)
    assert r.passed is False  # the seeded defect is caught
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.seed'`

- [ ] **Step 3: Write `autoqc/seed.py`**

```python
from __future__ import annotations
import copy


def seed_bad_negative(items: list[dict]) -> tuple[list[dict], str | None]:
    """Inject a Q07 defect: rewrite the first negative criterion's title into a
    'Does not claim...' form. Returns (mutated_items, mutated_id). Does not mutate input."""
    mutated = copy.deepcopy(items)
    for it in mutated:
        if not isinstance(it, dict):
            continue
        ann = it.get("annotations")
        typ = ann.get("type", "") if isinstance(ann, dict) else ""
        if "negative" in str(typ):
            title = str(it.get("title", ""))
            if "Claims that" in title:
                it["title"] = title.replace("Claims that", "Does not claim that", 1)
            else:
                it["title"] = title + " (does not claim this)"
            return mutated, str(it.get("id")) if it.get("id") is not None else None
    return mutated, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_seed.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/seed.py tests/test_seed.py
git commit -m "feat(autoqc): seed-defect generator (Q07 bad-negative)"
```

---

### Task 6: Pipeline wiring (structural + semantic)

**Files:**
- Modify: `autoqc/cli.py`
- Test: `tests/test_pipeline_semantic.py`

**Interfaces:**
- Consumes: `run_structural`, `compute_verdict`, `to_record`, `to_markdown` (existing); `run_check`, `SEMANTIC_CHECKS` (Tasks 3–4); `default_client`, `LLMClient` (Task 1).
- Produces: MODIFIED `run(bundle_dir, out_dir, llm=None, k=3) -> Verdict`. Always runs structural. Resolves a client = `llm or default_client()`; if the client is not None, appends `run_check(c, bundle, client, k)` for every `c in SEMANTIC_CHECKS`; if it is None, semantic is skipped (structural-only, Milestone-1 behavior preserved). `main` unchanged in signature.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_semantic.py
import json
from pathlib import Path
from autoqc.cli import run
from autoqc.llm import FakeLLMClient
from autoqc.model import Verdict


def _good_bundle(root: Path, neg_title: str):
    (root / "tests").mkdir(parents=True)
    (root / "tests/rubrics.json").write_text(json.dumps([
        {"id": "a" * 32, "title": "1.1: States X",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}},
        {"id": "b" * 32, "title": neg_title,
         "annotations": {"type": "negative hli verifier", "importance": "must have"}},
    ]))
    (root / "tests/prompt.txt").write_text("q")
    (root / "solution").mkdir()
    (root / "solution/answer.txt").write_text("a")
    (root / "environment").mkdir()
    (root / "environment/Dockerfile").write_text("FROM x")
    (root / "task.toml").write_text(
        '[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))


def _responder(messages):
    text = " ".join(m["content"] for m in messages)
    return {"passed": ("Does not claim" not in text), "evidence": []}


def test_semantic_runs_when_client_given_good_negative(tmp_path):
    b = tmp_path / "task"; b.mkdir()
    _good_bundle(b, "2.1: Claims that X fails")
    out = tmp_path / "out"
    v = run(b, out, llm=FakeLLMClient(_responder))
    assert v is Verdict.SOUND
    rec = json.loads((out / "review_record.json").read_text())
    assert any(r["id"] == "Q07" for r in rec["results"])  # semantic ran


def test_semantic_flags_bad_negative(tmp_path):
    b = tmp_path / "task"; b.mkdir()
    _good_bundle(b, "2.1: Does not claim that X fails")
    out = tmp_path / "out"
    v = run(b, out, llm=FakeLLMClient(_responder))
    assert v is Verdict.NOT_SOUND  # Q07 is reject-severity
    assert "Q07" in (out / "report.md").read_text()


def test_structural_only_when_no_client(tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    b = tmp_path / "task"; b.mkdir()
    _good_bundle(b, "2.1: Does not claim that X fails")  # would trip Q07 IF semantic ran
    out = tmp_path / "out"
    v = run(b, out)  # no client, no env -> structural only
    assert v is Verdict.SOUND
    rec = json.loads((out / "review_record.json").read_text())
    assert not any(r["id"] == "Q07" for r in rec["results"])  # semantic skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_pipeline_semantic.py -v`
Expected: FAIL (run() has no `llm` parameter yet → TypeError)

- [ ] **Step 3: Modify `autoqc/cli.py`**

Add imports at the top with the existing ones:

```python
from autoqc.llm import default_client
from autoqc.semantic.engine import run_check
from autoqc.semantic.checks import SEMANTIC_CHECKS
```

Replace the existing `run` function with:

```python
def run(bundle_dir, out_dir, llm=None, k: int = 3) -> Verdict:
    bundle = load_bundle(Path(bundle_dir))
    results = run_structural(bundle)

    client = llm if llm is not None else default_client()
    if client is not None:
        for check in SEMANTIC_CHECKS:
            results.append(run_check(check, bundle, client, k=k))

    verdict = compute_verdict(results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_record.json").write_text(
        json.dumps(to_record(bundle, results, verdict), indent=2))
    (out / "report.md").write_text(to_markdown(bundle, results, verdict))
    return verdict
```

`main` is unchanged (it calls `run(bundle_dir, out_dir)`; when a gateway is configured in the environment, semantic runs automatically; otherwise structural-only).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pipeline_semantic.py tests/test_cli_e2e.py -v`
Expected: PASS — the three new tests AND the existing e2e tests (which pass no client and set no gateway env, so they stay structural-only).

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: PASS (all Milestone 1 + Milestone 2 tests).

- [ ] **Step 6: Commit**

```bash
git add autoqc/cli.py tests/test_pipeline_semantic.py
git commit -m "feat(autoqc): wire semantic stage into the pipeline"
```

---

## Self-Review

**Spec coverage (semantic-engine slice):**
- Mockable LLM client + gateway → Task 1. ✓
- SemanticCheck contract (units, proposer/adversary, parse) → Task 2. ✓
- Ensemble (K proposers, majority) + asymmetric adversary (1 round) + adjudication (split/overturn → needs_human) → Task 3. ✓
- Q07 as first check, per-criterion over negatives → Task 4. ✓
- Seed-defect generator (labeled mutant) → Task 5. ✓
- Pipeline wiring, backward-compatible structural-only → Task 6. ✓
- Deferred to next plans (correctly absent): Q03 + remaining semantic checks; Q06 in-container factual; live-gateway calibration run; ensemble-size/threshold tuning.

**Placeholder scan:** none — every step has real code and real tests.

**Type consistency:** `SemanticJudgment(passed, evidence, reason)` and `Unit(key, payload)` defined in Task 2, used unchanged in Tasks 3–5. `run_check(check, bundle, llm, k)` defined in Task 3, consumed in Tasks 4–6. `FakeLLMClient(responder)` / `default_client()` from Task 1 used in Tasks 3–6. `NegativeSemanticsCheck` / `SEMANTIC_CHECKS` from Task 4 used in Tasks 5–6. `run(bundle_dir, out_dir, llm, k)` extended in Task 6 preserves the Milestone-1 two-arg call. Consistent.

---

## Notes for the next plan (Milestone 2b — not in scope here)

Adds the remaining semantic checks (Q03 wildcard/interchangeability first, then Q01/Q02/Q04/Q05/Q08–Q12), each with proposer/adversary prompts and seed mutants; a live-gateway smoke run over the internal 10 + a public sample; and the ensemble-size (K) and split-threshold tuning against the seeded corpus. It also resolves the deferred Milestone-1 minor: unify the record schema (retire/rewire the unused `ReviewRecord`).

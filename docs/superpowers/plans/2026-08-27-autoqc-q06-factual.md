# Q06 Factual-Soundness Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Q06, the container-backed check that judges whether each rubric criterion is factually well-founded against the repo at `base_commit`.

**Architecture:** A new container subsystem builds the task's faithful image and runs one long-lived hardened container (mini-swe-agent's `docker exec` model). A factual-role agent uses a guarded `run_bash` tool inside that container to verify each criterion; two independent rounds are adjudicated (agree → stands, disagree → needs_human, both-reject must cite the same file). Q06 wires into the existing semantic engine behind a Docker-availability gate.

**Tech Stack:** Python 3.10 (default `python3`), pytest, Docker CLI via `subprocess`. Reuses `AgentRunner` (`autoqc/agent/runner.py`), the `submit_findings` contract (`autoqc/agent/tools.py`), and `FakeLLMClient` (`autoqc/llm.py`).

**Spec:** `docs/superpowers/specs/2026-08-27-autoqc-q06-factual-design.md`

## Global Constraints

- **Test runner:** always `python3 -m pytest` (only default `python3` 3.10.18 has pytest). `pyproject` sets `pythonpath=["."]`.
- **Whole suite must stay offline & deterministic.** No test may invoke Docker or the gateway. The container subsystem takes an **injected `runner`**; tests pass a fake. Agent tests use `FakeLLMClient`.
- **Never crash / never silent-pass (I1):** every Docker failure, build failure, agent error, timeout, or missing-evidence finding degrades to a `needs_human` result. A genuine confirmed defect is `passed=False, needs_human=False` (→ `not_sound`); anything uncertain is `needs_human=True` (→ `needs_human_review`).
- **Container hardening (spec §2.1), exact flags asserted by tests:** `docker run -d --network=none --cap-drop=ALL --security-opt no-new-privileges --read-only --tmpfs /scratch:rw,size=<n> --memory <n> --cpus <n> --pids-limit <n> -w /testbed <image> sleep <ttl>`. `--network=none` is the hard boundary.
- **Offline build caches:** at `docker exec`, set `GOCACHE=/scratch/go-build` and `TMPDIR=/scratch/tmp` (writable tmpfs, since root fs is read-only) but **never override `GOMODCACHE`/`GOPATH`/`HOME`** — the module cache is baked into the image at default locations and `--network=none` blocks re-download.
- **Resource-cap numbers and cache prune policy are calibration-tunable** (spec §7). Defaults here: memory `2g`, cpus `2`, pids `256`, scratch `512m`, per-exec timeout `120s`, output cap `20000` chars, build timeout `1800s`, container ttl `2h`.
- **Q06 is NOT added to `SEMANTIC_CHECKS`** (that list drives the proposer/adversary text loop). It runs as its own factual stage.
- **Never print or commit secrets** (`.env` holds `EVAL_API_KEY`).

---

## File Structure

- **Create `autoqc/agent/container.py`** — Docker mechanics: `RunResult`, `_subprocess_runner`, `docker_available`, `image_tag`, `Limits`, `ContainerError`, `ContainerSession` (ensure_image / start / exec / stop). Injected `runner` for offline tests. No policy/agent logic.
- **Modify `autoqc/agent/tools.py`** — add `container` field to `AgentContext`; add `guard_command`, the `run_bash` `Tool`, and `factual_tools()`.
- **Modify `autoqc/agent/checks.py`** — add `Q06` `SemanticCheck`, `_FACTUAL_SYS`, `factual_role()`, `factual_context()`.
- **Modify `autoqc/agent/engine.py`** — add `adjudicate_factual`, `run_factual`, `run_factual_stage`; call the stage from `run_semantic` behind a `factual` gate.
- **Modify `autoqc/cli.py`** — thread a `factual` flag (default on) through `run`.
- **Modify `autoqc/seed.py`** — add `seed_factual` (calibration recall seed).
- **Create tests:** `tests/test_container.py`, and extend `tests/test_agent_tools.py`, `tests/test_agent_checks.py`, `tests/test_engine_core.py` (or new `tests/test_factual.py`), `tests/test_seed.py`.

---

## Task 1: Container subsystem

**Files:**
- Create: `autoqc/agent/container.py`
- Test: `tests/test_container.py`

**Interfaces:**
- Consumes: `Bundle` (`autoqc/bundle.py`) — uses `bundle.root` (a `Path`) and the file `environment/Dockerfile`.
- Produces:
  - `RunResult(returncode: int, output: str)`
  - `docker_available(runner=_subprocess_runner) -> bool`
  - `image_tag(bundle) -> str`
  - `Limits` dataclass (fields per Global Constraints)
  - `ContainerError(Exception)`
  - `ContainerSession(bundle, runner=_subprocess_runner, name=None, limits=None)` with `.tag: str`, `.name: str`, `.container_id: str|None`, methods `ensure_image() -> str`, `start() -> str`, `exec(cmd, cwd="/testbed", timeout=None) -> str`, `stop() -> None`.
  - The injected `runner` is a callable `(argv: list[str], timeout: float|None) -> RunResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_container.py
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.agent.container import (
    RunResult, docker_available, image_tag, Limits, ContainerError, ContainerSession)


def _bundle(tmp_path: Path):
    (tmp_path / "environment").mkdir(parents=True)
    (tmp_path / "environment/Dockerfile").write_text("FROM busybox\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/rubrics.json").write_text("[]")
    (tmp_path / "task.toml").write_text('schema_version="1.1"\n')
    return load_bundle(tmp_path)


class FakeRunner:
    """Records argv; returns queued RunResults (or a default success)."""
    def __init__(self, results=None):
        self.calls = []
        self._results = list(results or [])
    def __call__(self, argv, timeout=None):
        self.calls.append((argv, timeout))
        return self._results.pop(0) if self._results else RunResult(0, "")


def test_image_tag_is_deterministic_and_hashed(tmp_path):
    b = _bundle(tmp_path)
    tag = image_tag(b)
    assert tag.startswith("autoqc-q06/")
    assert ":" in tag
    assert image_tag(b) == tag  # stable
    (tmp_path / "environment/Dockerfile").write_text("FROM busybox\nRUN true\n")
    assert image_tag(load_bundle(tmp_path)) != tag  # hash changes with Dockerfile


def test_docker_available_reflects_returncode():
    assert docker_available(FakeRunner([RunResult(0, "ok")])) is True
    assert docker_available(FakeRunner([RunResult(1, "cannot connect")])) is False


def test_ensure_image_skips_build_on_cache_hit(tmp_path):
    r = FakeRunner([RunResult(0, "exists")])  # docker image inspect -> hit
    s = ContainerSession(_bundle(tmp_path), runner=r)
    assert s.ensure_image() == s.tag
    assert r.calls[0][0][:3] == ["docker", "image", "inspect"]
    assert not any(c[0][:2] == ["docker", "build"] for c in r.calls)


def test_ensure_image_builds_on_miss(tmp_path):
    r = FakeRunner([RunResult(1, "no such image"), RunResult(0, "built")])
    s = ContainerSession(_bundle(tmp_path), runner=r)
    assert s.ensure_image() == s.tag
    build = [c for c in r.calls if c[0][:2] == ["docker", "build"]][0][0]
    assert "-t" in build and s.tag in build and "-f" in build


def test_ensure_image_raises_on_build_failure(tmp_path):
    r = FakeRunner([RunResult(1, "miss"), RunResult(1, "build error: boom")])
    s = ContainerSession(_bundle(tmp_path), runner=r)
    try:
        s.ensure_image()
        assert False, "expected ContainerError"
    except ContainerError as e:
        assert "boom" in str(e)


def test_start_uses_hardening_flags(tmp_path):
    r = FakeRunner([RunResult(0, "cid")])
    s = ContainerSession(_bundle(tmp_path), runner=r, name="c1")
    assert s.start() == "c1"
    argv = r.calls[0][0]
    for flag in ["--network=none", "--cap-drop=ALL", "--read-only",
                 "--security-opt", "no-new-privileges", "--pids-limit"]:
        assert flag in argv, f"missing {flag}"
    assert "-w" in argv and "/testbed" in argv
    tmpfs = argv[argv.index("--tmpfs") + 1]
    assert tmpfs.startswith("/scratch:")


def test_start_raises_on_failure(tmp_path):
    s = ContainerSession(_bundle(tmp_path), runner=FakeRunner([RunResult(1, "denied")]), name="c1")
    try:
        s.start(); assert False
    except ContainerError as e:
        assert "denied" in str(e)


def test_exec_sets_offline_caches_and_caps_output(tmp_path):
    long = "x" * 50000
    r = FakeRunner([RunResult(0, long)])
    s = ContainerSession(_bundle(tmp_path), runner=r, name="c1", limits=Limits(output_cap=100))
    s.container_id = "c1"
    out = s.exec("go build ./...")
    argv = r.calls[0][0]
    assert argv[:3] == ["docker", "exec", "-w"]
    assert "GOCACHE=/scratch/go-build" in argv and "TMPDIR=/scratch/tmp" in argv
    assert not any("GOMODCACHE" in a for a in argv)  # baked module cache untouched
    assert argv[-3:] == ["bash", "-lc", "go build ./..."]
    assert len(out) < 50000 and "truncated" in out


def test_exec_timeout_surfaces_as_output(tmp_path):
    r = FakeRunner([RunResult(124, "error: command timed out after 5s")])
    s = ContainerSession(_bundle(tmp_path), runner=r, name="c1", limits=Limits(exec_timeout=5))
    s.container_id = "c1"
    assert "timed out" in s.exec("sleep 999")


def test_stop_is_best_effort(tmp_path):
    r = FakeRunner()
    s = ContainerSession(_bundle(tmp_path), runner=r, name="c1")
    s.container_id = "c1"
    s.stop()  # must not raise
    assert any("c1" in " ".join(c[0]) for c in r.calls)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_container.py -v`
Expected: FAIL — `ModuleNotFoundError: autoqc.agent.container`.

- [ ] **Step 3: Implement the container subsystem**

```python
# autoqc/agent/container.py
from __future__ import annotations
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    returncode: int
    output: str


def _subprocess_runner(argv, timeout=None) -> RunResult:
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        return RunResult(p.returncode, (p.stdout or "") + (p.stderr or ""))
    except subprocess.TimeoutExpired:
        return RunResult(124, f"error: command timed out after {timeout}s")
    except (OSError, ValueError) as e:
        return RunResult(127, f"error: could not run {argv[:2]}: {e}")


def docker_available(runner=_subprocess_runner) -> bool:
    return runner(["docker", "version"], timeout=20).returncode == 0


def image_tag(bundle) -> str:
    df = (Path(bundle.root) / "environment/Dockerfile").read_bytes()
    h = hashlib.sha256(df).hexdigest()[:12]
    task = Path(bundle.root).name or "task"
    safe = re.sub(r"[^a-z0-9._-]", "-", task.lower()) or "task"
    return f"autoqc-q06/{safe}:{h}"


class ContainerError(Exception):
    pass


@dataclass
class Limits:
    memory: str = "2g"
    cpus: str = "2"
    pids: int = 256
    scratch_size: str = "512m"
    exec_timeout: float = 120.0
    output_cap: int = 20000
    build_timeout: float = 1800.0
    ttl: str = "2h"


class ContainerSession:
    def __init__(self, bundle, runner=_subprocess_runner, name=None, limits=None):
        self.bundle = bundle
        self.runner = runner
        self.limits = limits or Limits()
        self.tag = image_tag(bundle)
        self.name = name or ("autoqc_q06_" + self.tag.split(":")[-1])
        self.container_id = None

    def ensure_image(self) -> str:
        if self.runner(["docker", "image", "inspect", self.tag], timeout=30).returncode == 0:
            return self.tag
        env = Path(self.bundle.root) / "environment"
        r = self.runner(["docker", "build", "-f", str(env / "Dockerfile"),
                         "-t", self.tag, str(env)], timeout=self.limits.build_timeout)
        if r.returncode != 0:
            raise ContainerError(f"image build failed: {r.output[-800:]}")
        return self.tag

    def start(self) -> str:
        lim = self.limits
        argv = ["docker", "run", "-d", "--name", self.name,
                "--network=none", "--cap-drop=ALL",
                "--security-opt", "no-new-privileges",
                "--read-only", "--tmpfs", f"/scratch:rw,size={lim.scratch_size}",
                "--memory", lim.memory, "--cpus", lim.cpus,
                "--pids-limit", str(lim.pids), "-w", "/testbed",
                self.tag, "sleep", lim.ttl]
        r = self.runner(argv, timeout=60)
        if r.returncode != 0:
            raise ContainerError(f"container start failed: {r.output[-800:]}")
        self.container_id = self.name
        return self.name

    def exec(self, cmd, cwd="/testbed", timeout=None) -> str:
        argv = ["docker", "exec", "-w", cwd,
                "-e", "GOCACHE=/scratch/go-build", "-e", "TMPDIR=/scratch/tmp",
                self.name, "bash", "-lc", cmd]
        out = self.runner(argv, timeout=timeout or self.limits.exec_timeout).output or ""
        cap = self.limits.output_cap
        if len(out) > cap:
            out = out[:cap] + f"\n...[truncated {len(out) - cap} chars]"
        return out

    def stop(self) -> None:
        self.runner(["bash", "-lc",
                     f"(timeout 60 docker stop {self.name} || docker rm -f {self.name}) "
                     f">/dev/null 2>&1 &"], timeout=10)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_container.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/container.py tests/test_container.py
git commit -m "feat(q06): container subsystem (hardened long-lived docker exec)"
```

---

## Task 2: `run_bash` tool + guard + container on AgentContext

**Files:**
- Modify: `autoqc/agent/tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `ContainerSession.exec(cmd) -> str` (Task 1).
- Produces:
  - `AgentContext(bundle_dir, container=None)` — new optional `container` field.
  - `guard_command(cmd: str) -> str | None` — returns a rejection reason, or `None` if allowed.
  - `run_bash` `Tool` (name `"run_bash"`, params `{cmd: string}`), whose `run(args, ctx)` guards then calls `ctx.container.exec(cmd)`.
  - `factual_tools() -> list[Tool]` = `[read_bundle_file, list_dir, run_bash, SUBMIT_FINDINGS]`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_agent_tools.py
from autoqc.agent.tools import guard_command, run_bash, factual_tools, AgentContext


class _FakeContainer:
    def __init__(self): self.ran = []
    def exec(self, cmd, cwd="/testbed", timeout=None):
        self.ran.append(cmd); return f"OUT:{cmd}"


def test_guard_blocks_network_installs_and_root_writes():
    for bad in ["curl http://x", "wget x", "sudo rm -rf /", "apt-get install foo",
                "pip install bar", "go get x", "echo hi > /etc/passwd", "rm -rf /"]:
        assert guard_command(bad) is not None, bad


def test_guard_allows_inspection_and_scratch_writes():
    for ok in ["cat go.mod", "grep -rn Retry ./pkg", "git log -1", "go build ./...",
               "go test ./pkg/foo/...", "echo hi > /scratch/x"]:
        assert guard_command(ok) is None, ok


def test_run_bash_execs_in_container():
    c = _FakeContainer()
    out = run_bash.run({"cmd": "cat go.mod"}, AgentContext(bundle_dir=".", container=c))
    assert out == "OUT:cat go.mod" and c.ran == ["cat go.mod"]


def test_run_bash_blocks_denied_command_before_exec():
    c = _FakeContainer()
    out = run_bash.run({"cmd": "curl http://evil"}, AgentContext(bundle_dir=".", container=c))
    assert "blocked" in out and c.ran == []


def test_run_bash_without_container_is_error_not_crash():
    out = run_bash.run({"cmd": "cat x"}, AgentContext(bundle_dir="."))
    assert out.startswith("error:")


def test_factual_tools_includes_run_bash_and_submit():
    names = {t.name for t in factual_tools()}
    assert {"run_bash", "submit_findings", "read_bundle_file"} <= names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_tools.py -k "guard or run_bash or factual_tools" -v`
Expected: FAIL — `ImportError` for `guard_command` / `run_bash` / `factual_tools`.

- [ ] **Step 3: Implement in `autoqc/agent/tools.py`**

Add `container` to the dataclass:

```python
@dataclass
class AgentContext:
    bundle_dir: Path
    container: object | None = None
```

Add the guard, tool, and factual tool list (after `SUBMIT_FINDINGS`):

```python
import re

_DENY = [
    r"\bcurl\b", r"\bwget\b", r"\bsudo\b", r"\bnc\b",
    r"\bapt(?:-get)?\s+install\b", r"\bpip3?\s+install\b",
    r"\bgo\s+get\b", r"\bnpm\s+(?:install|i)\b",
    r"\brm\s+-rf\s+/(?!scratch)", r">>?\s*/(?!scratch)",
]


def guard_command(cmd: str) -> str | None:
    """Denylist (defense-in-depth; the network-less container is the real control).
    Returns a rejection reason, or None if allowed."""
    text = (cmd or "").strip()
    if not text:
        return "error: empty command"
    for pat in _DENY:
        if re.search(pat, text):
            return ("error: command blocked by Q06 safety guard "
                    f"(pattern {pat!r}); no network/installs, write only under /scratch")
    return None


def _run_bash(args: dict, ctx: AgentContext) -> str:
    cmd = str(args.get("cmd", ""))
    reason = guard_command(cmd)
    if reason:
        return reason
    container = getattr(ctx, "container", None)
    if container is None:
        return "error: no container available for run_bash"
    try:
        return container.exec(cmd)
    except Exception as e:  # never crash the agent loop
        return f"error: run_bash failed: {e}"


run_bash = Tool(
    name="run_bash",
    description=("Run a shell command in the task's container. The repo is checked out at "
                 "/testbed at base_commit. No network; writes only under /scratch."),
    parameters={"type": "object", "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"]},
    run=_run_bash)


def factual_tools() -> list[Tool]:
    return [read_bundle_file, list_dir, run_bash, SUBMIT_FINDINGS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_tools.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/tools.py tests/test_agent_tools.py
git commit -m "feat(q06): run_bash tool + denylist guard + container on AgentContext"
```

---

## Task 3: Factual role, Q06 check, and context

**Files:**
- Modify: `autoqc/agent/checks.py`
- Test: `tests/test_agent_checks.py`

**Interfaces:**
- Consumes: `SemanticCheck` and `_all_criteria` (both already in `checks.py`); `Role` (`runner.py`); `factual_tools()` (Task 2); `Severity.REJECT`.
- Produces:
  - `Q06: SemanticCheck` — `id="Q06"`, `name="Factual soundness"`, `severity=REJECT`, `scope=_all_criteria`, `unit_mode="criterion"`.
  - `factual_role() -> Role` (tools = `factual_tools()`).
  - `factual_context(bundle, criteria) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_agent_checks.py
from autoqc.agent.checks import Q06, factual_role, factual_context
from autoqc.model import Severity


def test_q06_is_reject_percriterion():
    assert Q06.id == "Q06" and Q06.severity is Severity.REJECT
    assert Q06.unit_mode == "criterion"


def test_factual_role_has_run_bash():
    names = {t.name for t in factual_role().tools}
    assert "run_bash" in names and "submit_findings" in names


def test_factual_context_mentions_base_commit_and_criteria():
    class B:
        prompt = "Explain the retry logic."
        base_commit = "eea1d62f0438f75075d9feb2c022a86083e618b2"
        repository = "cosi-project/runtime"
    crit = [{"id": "1.1", "title": "States the default retry count is 3"}]
    ctx = factual_context(B(), crit)
    assert "eea1d62f" in ctx and "1.1" in ctx and "/testbed" in ctx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_checks.py -k "q06 or factual" -v`
Expected: FAIL — `ImportError` for `Q06` / `factual_role` / `factual_context`.

- [ ] **Step 3: Implement in `autoqc/agent/checks.py`**

Add the import at the top (next to the existing `default_tools` import):

```python
from autoqc.agent.tools import default_tools, factual_tools
```

Add the check, role, and context (after the `Q11`/`SEMANTIC_CHECKS` block):

```python
Q06 = SemanticCheck(
    id="Q06", name="Factual soundness", severity=Severity.REJECT, scope=_all_criteria,
    guidance=("A criterion VIOLATES this if the repo at base_commit does NOT support what it "
              "grades on. A POSITIVE criterion must assert a fact that is TRUE in the repo; a "
              "NEGATIVE criterion asserts a FALSE claim, so it is sound only if that claim is "
              "actually FALSE in the repo. passed=true means the code backs the criterion. "
              "Judge only criteria that make a repo-checkable claim; mark passed=true for "
              "criteria that make no code claim (subjective/phrasing is out of scope)."))

_FACTUAL_SYS = (
    "You verify a grading rubric against the actual repository, which is checked out at "
    "/testbed at base_commit inside a network-isolated container. For each criterion, decide "
    "whether the code supports what it grades on: a POSITIVE criterion is sound iff its fact is "
    "TRUE in the repo; a NEGATIVE criterion states a FALSE assertion and is sound iff that "
    "assertion is actually FALSE in the repo. Use run_bash to read source (cat/grep/rg/find/ls/"
    "git log|show) and, only when a claim needs it, to build/test (writes go under /scratch; no "
    "network). If a criterion makes no repo-checkable claim, mark it passed=true. Finish by "
    "calling submit_findings with exactly one finding per criterion; every finding's evidence "
    "must include at least one path:line citation from the repo.")


def factual_role() -> Role:
    return Role(name="factual", system_prompt=_FACTUAL_SYS, tools=factual_tools())


def factual_context(bundle, criteria) -> str:
    return (f"Repository: {getattr(bundle, 'repository', '') or ''} at base_commit "
            f"{getattr(bundle, 'base_commit', '') or ''} (checked out at /testbed).\n\n"
            f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"Check Q06 — Factual soundness.\n{Q06.guidance}\n\n"
            f"Verify each of these criteria against the repo (submit one finding per criterion, "
            f"check_id=Q06):\n{_criteria_block(criteria)}\n\n"
            "passed=true if the repo supports the criterion, passed=false if it contradicts or "
            "lacks it. Evidence must cite path:line.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_checks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/checks.py tests/test_agent_checks.py
git commit -m "feat(q06): factual role, Q06 check, and repo-aware context"
```

---

## Task 4: Two-round adjudication and `run_factual`

**Files:**
- Modify: `autoqc/agent/engine.py`
- Test: `tests/test_factual.py` (create)

**Interfaces:**
- Consumes: `run_agent` (`runner.py`); `factual_role`, `factual_context`, `Q06` (Task 3); `_own` and `validate_findings` (already in `engine.py`/`tools.py`); `CheckResult`, `Stage`, `Severity`; `AgentContext`.
- Produces:
  - `adjudicate_factual(r1_findings, r2_findings, allowed_ids) -> dict[str, dict]` where each value is `{"passed": bool, "needs_human": bool}`.
  - `run_factual(bundle, client, ctx, votes_log=None) -> CheckResult` — runs two factual rounds over `Q06.scope(bundle.rubrics)`, adjudicates, returns one `CheckResult(id="Q06", stage=Stage.FACTUAL)`.
- **Adjudication rule (spec §2.4):** per criterion, using each round's `(passed, files)` where `files` are the path prefixes parsed from evidence — both agree pass → pass; disagree on `passed` → needs_human; both reject → same file overlap → reject, else needs_human; a round missing a verdict for the criterion → needs_human.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_factual.py
from pathlib import Path
from autoqc.agent.engine import adjudicate_factual, run_factual
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient
from autoqc.model import Stage, Severity


def _f(cid, passed, ev):
    return {"check_id": "Q06", "criterion_id": cid, "passed": passed, "evidence": ev}


def test_adjudicate_agree_pass():
    a = adjudicate_factual([_f("1", True, ["a.go:1"])], [_f("1", True, ["a.go:9"])], {"1"})
    assert a["1"] == {"passed": True, "needs_human": False}


def test_adjudicate_agree_reject_same_file_is_hard_reject():
    a = adjudicate_factual([_f("1", False, ["a.go:1"])], [_f("1", False, ["a.go:40"])], {"1"})
    assert a["1"] == {"passed": False, "needs_human": False}


def test_adjudicate_reject_different_files_needs_human():
    a = adjudicate_factual([_f("1", False, ["a.go:1"])], [_f("1", False, ["b.go:1"])], {"1"})
    assert a["1"]["needs_human"] is True


def test_adjudicate_disagreement_needs_human():
    a = adjudicate_factual([_f("1", True, ["a.go:1"])], [_f("1", False, ["a.go:1"])], {"1"})
    assert a["1"]["needs_human"] is True


def test_adjudicate_missing_round_needs_human():
    a = adjudicate_factual([_f("1", True, ["a.go:1"])], [], {"1"})
    assert a["1"]["needs_human"] is True


def _bundle_with(criteria):
    class B:
        prompt = "p"; base_commit = "abc"; repository = "r"
        rubrics = criteria
    return B()


def _responder_all(passed, ev):
    def r(messages, tools):
        # single-turn submit for every criterion mentioned in the user context
        import re
        user = next(m["content"] for m in messages if m["role"] == "user")
        ids = re.findall(r"criterion_id=(\S+)", user)
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {
            "findings": [_f(i, passed, ev) for i in ids]}}]}
    return r


def test_run_factual_confirmed_reject(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    res = run_factual(b, FakeLLMClient(_responder_all(False, ["x.go:3"])), ctx)
    assert res.id == "Q06" and res.stage is Stage.FACTUAL and res.severity is Severity.REJECT
    assert res.passed is False and res.needs_human is False  # -> not_sound


def test_run_factual_all_pass(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    res = run_factual(b, FakeLLMClient(_responder_all(True, ["x.go:3"])), ctx)
    assert res.passed is True and res.needs_human is False


def test_run_factual_no_criteria_passes(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    res = run_factual(_bundle_with([]), FakeLLMClient(_responder_all(True, ["x:1"])), ctx)
    assert res.passed is True and res.needs_human is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_factual.py -v`
Expected: FAIL — `ImportError` for `adjudicate_factual` / `run_factual`.

- [ ] **Step 3: Implement in `autoqc/agent/engine.py`**

Add imports (extend the existing `from autoqc.agent.checks import ...`) and the model imports:

```python
from autoqc.model import CheckResult, Stage, Severity
from autoqc.agent.checks import (SEMANTIC_CHECKS, proposer_role, adversary_role,
                                 proposer_context, adversary_context,
                                 Q06, factual_role, factual_context)
```

Add near the other helpers:

```python
def _files(evidence) -> set[str]:
    """Path prefixes cited in evidence: the token before ':' (or the first
    slash-bearing whitespace token). Used to require same-file agreement."""
    out = set()
    for e in evidence or []:
        s = str(e).strip()
        if not s:
            continue
        head = s.split()[0]
        out.add(head.split(":", 1)[0])
    return out


def _round_map(findings, allowed_ids):
    """First (passed, files) per criterion in a round's own Q06 findings."""
    m = {}
    for f in _own(findings, "Q06", allowed_ids):
        cid = f.get("criterion_id")
        if cid in allowed_ids and cid not in m:
            m[cid] = (bool(f.get("passed")), _files(f.get("evidence")))
    return m


def adjudicate_factual(r1_findings, r2_findings, allowed_ids) -> dict:
    m1 = _round_map(r1_findings, allowed_ids)
    m2 = _round_map(r2_findings, allowed_ids)
    out = {}
    for cid in allowed_ids:
        v1, v2 = m1.get(cid), m2.get(cid)
        if v1 is None or v2 is None:
            out[cid] = {"passed": False, "needs_human": True}
            continue
        p1, f1 = v1
        p2, f2 = v2
        if p1 != p2:
            out[cid] = {"passed": False, "needs_human": True}
        elif p1:  # both pass
            out[cid] = {"passed": True, "needs_human": False}
        else:      # both reject: require same-file overlap
            same = bool(f1 & f2)
            out[cid] = {"passed": False, "needs_human": not same}
    return out


def run_factual(bundle, client, ctx, votes_log=None) -> CheckResult:
    items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
    criteria = Q06.scope(items)
    if not criteria:
        return CheckResult(id="Q06", name=Q06.name, stage=Stage.FACTUAL,
                           severity=Severity.REJECT, passed=True)
    allowed = {c["id"] for c in criteria}
    p_ctx = factual_context(bundle, criteria)
    rounds = []
    for _ in range(2):
        res = run_agent(factual_role(), p_ctx, client, ctx)
        if votes_log is not None:
            votes_log.append({"check": "Q06", "role": "factual",
                              "ok": res.ok, "findings": res.findings})
        rounds.append(res.findings if res.ok else [])

    adj = adjudicate_factual(rounds[0], rounds[1], allowed)
    passed = all(v["passed"] for v in adj.values()) if adj else True
    needs_human = any(v["needs_human"] for v in adj.values())
    problems = [cid for cid, v in adj.items() if (not v["passed"]) or v["needs_human"]]
    evidence = []
    for fs in rounds:
        for f in _own(fs, "Q06", allowed):
            evidence.extend(f.get("evidence") or [])
    detail = "" if passed and not needs_human else "criteria needing attention: " + ", ".join(problems)
    return CheckResult(id="Q06", name=Q06.name, stage=Stage.FACTUAL, severity=Severity.REJECT,
                       passed=passed, needs_human=needs_human, evidence=evidence[:20], detail=detail)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_factual.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/engine.py tests/test_factual.py
git commit -m "feat(q06): two-round same-file adjudication + run_factual"
```

---

## Task 5: Factual stage wiring + Docker gate

**Files:**
- Modify: `autoqc/agent/engine.py`, `autoqc/cli.py`
- Test: `tests/test_factual.py` (extend)

**Interfaces:**
- Consumes: `ContainerSession`, `docker_available`, `ContainerError` (Task 1); `run_factual` (Task 4); `AgentContext`.
- Produces:
  - `run_factual_stage(bundle, client, votes_log=None, limits=None, docker=docker_available) -> CheckResult` — the full lifecycle: Docker gate → ensure image → start → `run_factual` (ctx carries the container) → stop. Any `ContainerError`/exception/unavailable-Docker → one `needs_human` `CheckResult`.
  - `run_semantic(..., factual=True)` — appends `run_factual_stage(...)` when `factual` is true.
  - `cli.run(bundle_dir, out_dir, llm=None, k=3, factual=True)` — threads `factual` into `run_semantic`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_factual.py
from autoqc.agent.engine import run_factual_stage, run_semantic


def test_stage_degrades_to_needs_human_when_docker_down(tmp_path):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment/Dockerfile").write_text("FROM busybox\n")
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    b.root = tmp_path
    res = run_factual_stage(b, FakeLLMClient(_responder_all(True, ["x:1"])),
                            docker=lambda runner=None: False)
    assert res.id == "Q06" and res.needs_human is True and res.passed is False
    assert "docker" in res.detail.lower()


def test_run_semantic_skips_factual_when_disabled(tmp_path):
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    b.rubrics = [{"id": "1.1", "title": "t", "annotations": {"type": "positive"}}]
    ctx = AgentContext(bundle_dir=tmp_path)
    results = run_semantic(b, FakeLLMClient(_responder_all(True, ["x:1"])), ctx,
                           checks=[], k=1, factual=False)
    assert not any(r.id == "Q06" for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_factual.py -k "stage or skips_factual" -v`
Expected: FAIL — `ImportError: run_factual_stage`, and `run_semantic()` has no `factual` kwarg.

- [ ] **Step 3: Implement**

In `autoqc/agent/engine.py` add the container imports and the stage; update `run_semantic`:

```python
from autoqc.agent.container import ContainerSession, docker_available, ContainerError
from autoqc.agent.tools import validate_findings, AgentContext


def _q06_needs_human(reason: str) -> CheckResult:
    return CheckResult(id="Q06", name=Q06.name, stage=Stage.FACTUAL, severity=Severity.REJECT,
                       passed=False, needs_human=True, detail=f"Q06 not run: {reason}")


def run_factual_stage(bundle, client, votes_log=None, limits=None,
                      docker=docker_available) -> CheckResult:
    if not getattr(bundle, "files_present", {}).get("environment/Dockerfile", True):
        return _q06_needs_human("no environment/Dockerfile in bundle")
    if not docker():
        return _q06_needs_human("Docker is not available")
    session = ContainerSession(bundle, limits=limits)
    try:
        session.ensure_image()
        session.start()
    except ContainerError as e:
        return _q06_needs_human(str(e))
    except Exception as e:  # defensive: never crash the run
        return _q06_needs_human(f"container setup error: {e}")
    try:
        ctx = AgentContext(bundle_dir=bundle.root, container=session)
        return run_factual(bundle, client, ctx, votes_log=votes_log)
    except Exception as e:
        return _q06_needs_human(f"factual pass error: {e}")
    finally:
        session.stop()
```

Update `run_semantic` signature and body:

```python
def run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None, factual=True):
    results = [run_check(c, bundle, client, ctx, k=k, votes_log=votes_log) for c in checks]
    results += [fn(bundle) for fn in DETERMINISTIC_CHECKS]
    if factual:
        results.append(run_factual_stage(bundle, client, votes_log=votes_log))
    return results
```

In `autoqc/cli.py`, thread the flag:

```python
def run(bundle_dir, out_dir, llm=None, k: int = 3, factual: bool = True) -> Verdict:
    ...
    if client is not None:
        ctx = AgentContext(bundle_dir=Path(bundle_dir))
        results.extend(run_semantic(bundle, client, ctx, k=k, factual=factual))
    ...
```

- [ ] **Step 4: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS — all prior 122 tests plus the new ones. (Existing `test_cli_semantic.py` / `test_engine_run.py` use `FakeLLMClient`; `factual` defaults on but degrades to `needs_human` when Docker is absent in CI. If any existing CLI/e2e test asserts an exact verdict or result count, pass `factual=False` in that test call so its expectation is unchanged — note it in the commit.)

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/engine.py autoqc/cli.py tests/test_factual.py
git commit -m "feat(q06): factual stage lifecycle + docker gate wired into run_semantic"
```

---

## Task 6: Factual-defect seed (calibration recall)

**Files:**
- Modify: `autoqc/seed.py`
- Test: `tests/test_seed.py`

**Interfaces:**
- Consumes: nothing new (mirrors existing `seed_bad_negative` / `seed_wildcard` shape).
- Produces: `seed_factual(items: list[dict]) -> tuple[list[dict], str | None]` — injects a claim about a symbol guaranteed absent from any repo into the first positive criterion's title, so Q06 must reject it. Does not mutate input.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_seed.py
from autoqc.seed import seed_factual


def test_seed_factual_injects_nonexistent_symbol_into_positive():
    items = [{"id": "1.1", "title": "States the retry count is 3",
              "annotations": {"type": "positive"}},
             {"id": "2.1", "title": "Claims X", "annotations": {"type": "negative"}}]
    mutated, mid = seed_factual(items)
    assert mid == "1.1"
    assert "nonexistent_autoqc_symbol" in mutated[0]["title"]
    assert items[0]["title"] == "States the retry count is 3"  # input untouched


def test_seed_factual_no_positive_returns_none():
    mutated, mid = seed_factual([{"id": "2.1", "title": "n", "annotations": {"type": "negative"}}])
    assert mid is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_seed.py -k factual -v`
Expected: FAIL — `ImportError: seed_factual`.

- [ ] **Step 3: Implement in `autoqc/seed.py`**

```python
def seed_factual(items: list[dict]) -> tuple[list[dict], str | None]:
    """Inject a Q06 defect: append a claim about a symbol guaranteed absent from
    any repo to the first positive criterion's title, so the code cannot support
    it. Returns (mutated_items, mutated_id). Does not mutate input."""
    mutated = copy.deepcopy(items)
    for it in mutated:
        if not isinstance(it, dict):
            continue
        ann = it.get("annotations")
        typ = ann.get("type", "") if isinstance(ann, dict) else ""
        if "positive" in str(typ):
            title = str(it.get("title", "")).rstrip(".")
            it["title"] = title + ", implemented by the function `nonexistent_autoqc_symbol_xyz`"
            return mutated, str(it.get("id")) if it.get("id") is not None else None
    return mutated, None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_seed.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqc/seed.py tests/test_seed.py
git commit -m "feat(q06): factual-defect seed for recall calibration"
```

---

## Post-plan: live calibration (not in the offline suite)

After Task 6, with Docker up and all 10 internal Dockerfiles building, run a live smoke + calibration (spec §6): one clean bundle through `python3 -m autoqc <bundle> autoqc_out` to confirm Q06 builds/starts/verifies end-to-end; then a `seed_factual` recall check (mutated criterion → Q06 reject; unmutated → pass) and a clean-set false-fire sweep via `scripts/calibrate_clean.py`. Watch run-to-run nondeterminism (per-candidate flip-rate). Tuning the resource caps, cache prune policy, and concurrency is the step-2 (parallelism) work — out of scope here.

---

## Self-Review

**Spec coverage:**
- §1 / §1.1 (what Q06 judges, positive/negative semantics, evidence) → Task 3 guidance + system prompt; verdict mapping → Task 4/5 (`passed`/`needs_human` → `compute_verdict`).
- §2.1 container subsystem (long-lived exec, hardening flags, deterministic tag + cache-skip, offline caches, teardown) → Task 1.
- §2.2 `run_bash` + guard, factual-role-only → Task 2.
- §2.3 factual role + Q06 check + context, not in `SEMANTIC_CHECKS` → Task 3.
- §2.4 two-round same-file adjudication → Task 4.
- §2.5 engine/CLI integration + Docker gate + degrade → Task 5.
- §4 error handling (every failure → needs_human) → Tasks 1 (ContainerError), 2 (tool errors), 4/5 (round/stage degrade).
- §5 testing (offline, fake runner + FakeLLMClient) → every task.
- §6 calibration seed → Task 6; live calibration → post-plan note.
- §7/§8 deferred (cap numbers, cache policy, concurrency, container-reuse) → Global Constraints defaults + post-plan note.

**Placeholder scan:** no TBD/TODO; every code step is complete and runnable; resource numbers are concrete defaults, not placeholders.

**Type consistency:** `runner(argv, timeout) -> RunResult` used the same way in Task 1 impl and tests; `ContainerSession.exec` signature matches `_run_bash`'s call (`container.exec(cmd)`) in Task 2; `AgentContext(bundle_dir, container=None)` consistent across Tasks 2/4/5; `adjudicate_factual(r1, r2, allowed_ids)` and `run_factual(bundle, client, ctx, votes_log)` consistent between Task 4 def and Task 5 use; `run_factual_stage(bundle, client, votes_log, limits, docker)` consistent between Task 5 def and its tests; `CheckResult(id="Q06", stage=Stage.FACTUAL, severity=Severity.REJECT)` uniform.

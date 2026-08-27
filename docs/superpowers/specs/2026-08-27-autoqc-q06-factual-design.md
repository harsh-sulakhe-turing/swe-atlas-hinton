# AutoQC Q06 — Factual Soundness Check (design)

Date: 2026-08-27
Status: Draft for review
Builds on: `2026-08-25-autoqc-agent-architecture.md` (§3.1 run_bash safety, §5.4 factual pass),
`2026-08-25-autoqc-execution-harness-design.md`, `2026-08-25-autoqc-quality-rubric-design.md`

## 1. What Q06 is

Q06 is the last unbuilt semantic check. Like every AutoQC check, it judges whether the
trainer-authored **task rubric** is soundly constructed — it does **not** grade any model
answer, and it does **not** free-form fact-check the repository.

Q06 asks one question per criterion: **does the code at `base_commit` actually support what
this criterion grades on?** A criterion that grades on a fact the repo contradicts (or that
does not exist in the repo) is unsound — it would credit a wrong answer or fail a correct
one — so Q06 rejects it.

Severity: **reject** (same class as Q01–Q05, Q07). Stage: `Stage.FACTUAL` (the enum value
already exists in `autoqc/model.py`).

### 1.1 Verdict semantics per criterion

Q06 runs only over criteria that make a **repo-checkable claim** about the code (a symbol,
file, path, signature, value, call relationship, or observable behavior). Subjective or
non-code criteria are out of scope — Q02 (binary-judgeability) already owns those.

- **Positive criterion** — `passed=true` iff the fact it grades on is **true in the repo**.
  It VIOLATES Q06 if it asserts or presupposes something **false or absent** at `base_commit`.
- **Negative criterion** — by construction states a *false* assertion whose presence in an
  answer should fail it (see Q07). It is sound iff that assertion is **actually false in the
  repo**. If the "wrong" claim is in fact **true** in the repo, the negative is broken (it
  would penalize a correct answer) → VIOLATES Q06.
- **Evidence**: one or more `path:line` citations in the repo backing the verdict (the
  `submit_findings` contract already reserves `path:line` evidence for Q06).

The one-line version: *Q06 tests the rubric against reality, not answers against the rubric.*

## 2. Architecture

Q06 reuses the existing `AgentRunner` primitive (`autoqc/agent/runner.py`) and the
`submit_findings` structured contract (`autoqc/agent/tools.py`) unchanged. Three new pieces:

1. a **container subsystem** that builds the task's faithful image and runs a long-lived
   hardened container;
2. a **`run_bash` tool** bound to that container, enabled only for the factual role;
3. the **`factual` role + Q06 check registration + two-round adjudication**, wired into the
   engine behind the existing client-present gate.

The agent loop stays **outside** the container and calls the gateway (native tool-calling,
protocol A — unchanged). Only `run_bash` crosses into the container. The container runs with
`--network=none`, so the loop's gateway calls are unaffected and the container has no external
reach.

### 2.1 Container subsystem (`autoqc/agent/container.py`, new)

Execution mechanism follows mini-swe-agent's docker environment: **one long-lived container
per task, `docker exec` per command** — filesystem/scratch state persists across the agent's
turns (needed for multi-step runtime checks: `cd`, set up, run test). We layer the spec §3.1
hardening onto mini-swe-agent's `docker run -d` startup (its base image does none of this).

**Build** — from the bundle's `environment/Dockerfile` (a required bundle file). The image is
tagged deterministically:

```
autoqc-q06/<task-id>:<short-hash-of-Dockerfile-bytes>
```

Before building, `docker image inspect <tag>`; on a hit, skip the build. The Dockerfile hash
in the tag means a changed Dockerfile rebuilds automatically while identical ones hit cache.
The image is **always** reused across the two adjudication rounds of one task, and reused
across a calibration sweep. A bounded-cache guard (prune / max-images / build-then-remove-per-
task mode) keeps a 134-task sweep from exhausting disk; the persist-vs-prune default is a
step-2 (parallelism) tuning knob sized against real timings, not fixed here (see §7).

**Startup** (once per task):

```
docker run -d --name <c> \
  --network=none --cap-drop=ALL --user <nonroot> \
  --read-only --tmpfs /scratch:rw,size=<cap> \
  --memory=<cap> --cpus=<cap> --pids-limit=<cap> \
  <image> sleep <ttl>
```

`--network=none` is the hard boundary: the repo and deps are baked in at `base_commit` during
build, so nothing needs network at run time; this kills exfiltration and external reach. Plus
non-root, all caps dropped, read-only root fs with a single writable `/scratch` tmpfs, no host
mounts, no docker socket, and cpu/memory/pids caps.

**Per command:** `docker exec -w <cwd> <c> bash -lc "<cmd>"`, wrapped in
`subprocess.run(..., timeout=<per-cmd-timeout>)` with stdout+stderr merged, utf-8 decode with
replacement, and an output-size cap (truncate + note). A per-command timeout and output cap
apply to every call.

**Teardown:** background `(timeout 60 docker stop <c> || docker rm -f <c>)`, run after Q06
finishes for the task. Every container is disposable.

The subsystem exposes a small object — build/ensure image, start container, `exec(cmd)`,
stop — so the engine can treat "the task's container" as a handle and the `run_bash` tool can
close over it.

### 2.2 `run_bash` tool (`autoqc/agent/tools.py`)

A new `Tool` named `run_bash`, present only in the factual role's tool list (text roles keep
`default_tools()` unchanged — no `run_bash`, so text checks cannot execute anything). Its
`run` closes over the container handle and applies the §3.1 guard before executing.

The guard is defense-in-depth, **not** the primary control (§3.1: the container is the primary
control; a command allowlist permissive enough to run the project's own build/test must include
an interpreter, so it gives false confidence). Two flavors of guidance/guard, both inside the
one tool:

- **Q06a (source reads, the majority of claims):** inspection-only usage — `cat`, `grep`/`rg`,
  `find`, `ls`, `head`/`tail`, `git show`, `git log`. No writes, no execution.
- **Q06b (runtime, the minority):** the full sandboxed shell for claims that genuinely need
  running the build/test/repro, guarded by a denylist for obviously out-of-scope patterns
  (`curl`/`wget`, `sudo`, package installs, writes outside `/scratch`, `rm -rf /`). The
  network-isolated ephemeral container is what makes running arbitrary project code safe.

The split is expressed as **role guidance + a single guard**, not two separately-scheduled
passes: the agent reads source for most claims and runs code only when a claim requires it,
exactly as one shell in mini-swe-agent.

### 2.3 Factual role and check (`autoqc/agent/checks.py`)

- `Q06 = SemanticCheck(id="Q06", name="Factual soundness", severity=REJECT,
  scope=<repo-checkable criteria>, unit_mode="criterion")`. The scope selector picks criteria
  that make a repo-checkable claim; the agent is instructed to return `passed=true` (out of
  scope / not a code claim) rather than guess when a criterion is not repo-checkable, so no
  criterion is silently dropped.
- `factual_role()` — a `Role` whose system prompt states the §1.1 semantics (positive =
  true-in-repo, negative = false-in-repo, evidence = `path:line`) and whose tools are
  `[read_bundle_file, list_dir, run_bash, SUBMIT_FINDINGS]`.
- `factual_context(bundle, criteria)` — includes the task prompt, the repository + `base_commit`
  (from `bundle.base_commit`), the criteria to judge, and the instruction to verify each
  against the checked-out repo at `/testbed`.

`Q06` is **not** added to `SEMANTIC_CHECKS` (that list drives the proposer/adversary text
loop). Q06 runs as its own pass with its own adjudication (§2.4).

### 2.4 Two-round adjudication (`autoqc/agent/engine.py`)

Q06 does not use the K-proposer + 1-adversary text ensemble (too many container-agent runs for
the expensive check). Instead, **two independent factual rounds**: two `run_agent(factual_role,
...)` runs, each with fresh agent state (independent reasoning + a clean message context),
each judging every in-scope criterion. Both rounds run against the same task container by
default (§8) — the independence that matters is the agent's reasoning, not fresh disk.

Per-criterion adjudication:

- both rounds agree `passed` → verdict stands;
- disagree on `passed` → `needs_human`;
- both reject → require they implicate the **same file/symbol** (not the identical line — line
  drift is normal; exact-line agreement would spuriously escalate). Same file + same verdict
  confirms the reject; different files → `needs_human`.

Rolls up to one `CheckResult(id="Q06", stage=Stage.FACTUAL, severity=REJECT, passed=all,
needs_human=any)`, feeding the existing C1-fixed `compute_verdict`. Every round and its findings
are appended to the per-vote log (auditability + T03 repeatability), same as the text passes.

### 2.5 Engine + CLI integration

`run_semantic` gains a Q06 pass after the text checks and deterministic checks, gated so it runs
only when Docker is usable and the bundle has a buildable `environment/Dockerfile`. The pass:
ensures the image, starts the container, runs the two factual rounds, adjudicates, tears the
container down, appends the `CheckResult`. A `run_bash`-disabled path is not needed — if the
container cannot be built or started, the whole Q06 pass degrades to a single `needs_human`
`CheckResult` (never a crash, never a silent pass), consistent with the I1 robustness rule.
`cli.run` needs no signature change; Q06 rides inside `run_semantic`.

## 3. Data flow

```
bundle ──> ensure image (build or cache hit) ──> start hardened container
      ──> factual round 1 (agent: read source / run build; submit_findings)
      ──> factual round 2 (independent)
      ──> adjudicate per criterion (agree / same-file / needs_human)
      ──> CheckResult(Q06) ──> compute_verdict
      ──> teardown container
```

## 4. Error handling (I1, never crash / never silent pass)

Each degrades the Q06 pass (or the affected item) to `needs_human`, never a crash and never a
silent `sound`; structural and text results already computed are never discarded:

- Docker unavailable / daemon down → Q06 → single `needs_human` CheckResult, with detail.
- Image build fails or exceeds build timeout → `needs_human`.
- Container start fails → `needs_human`.
- A `run_bash` command times out or is denied by the guard → the tool returns an error string
  to the agent (the agent may adapt), exactly like today's tool-error path.
- An agent round errors / times out / submits malformed findings → that round is empty; if only
  one round is usable, Q06 → `needs_human` (no cross-check possible).
- A finding without `path:line` evidence → rejected by `validate_findings` → treated as
  `needs_human` for that criterion.

## 5. Testing (all offline, FakeLLMClient + a fake container)

Unit / integration, no Docker and no gateway in the suite:

- **Container subsystem:** the docker calls go through a thin runner injected into the
  subsystem; tests use a fake runner that records the argv and returns canned output. Assert the
  hardened `docker run` flags are present (`--network=none`, `--cap-drop=ALL`, non-root,
  read-only + tmpfs, memory/cpus/pids), the deterministic tag + hash, the `docker image inspect`
  cache-hit skip, `docker exec -w`, per-command timeout wiring, and teardown.
- **`run_bash` guard:** denylist blocks `curl`/`sudo`/package-install/writes-outside-scratch;
  inspection commands pass; guard runs before exec.
- **Factual role + adjudication:** `FakeLLMClient` scripts a `run_bash` turn → tool result →
  `submit_findings`. Cases: both rounds agree pass; both agree reject same file → reject; agree
  reject different files → needs_human; disagree → needs_human; one round errors → needs_human;
  missing-evidence finding → needs_human.
- **Engine gate:** Docker-unavailable → single needs_human CheckResult; results from other
  stages preserved.

## 6. Calibration (after build, needs Docker + all 10 Dockerfiles building)

A new **factual-defect seed** (`autoqc/seed.py`): mutate a criterion into a false claim about
the repo (e.g. change a cited symbol/value/path/behavior to one the code contradicts) and
confirm Q06 rejects it; confirm the unmutated criterion passes (false-fire control). Run over
the internal 10 (all Dockerfiles must build) plus a clean-set false-fire sweep. Reuses the
`scripts/calibrate_clean.py` harness (bundle-parallel). Watch the run-to-run nondeterminism
already seen in text checks; a per-candidate flip-rate check applies here too.

## 7. Deferred / step-2 (parallelism)

- **Cache persist-vs-prune default** across a sweep (disk vs rebuild cost) — sized against real
  container-build timings in the parallelism step, not fixed here.
- **Narrow-pool concurrency** for Q06 container builds (disk/CPU-bound) vs the wide pool for
  text checks — the two-tier design from the parallelism decision; Q06's real timings size it.
- **Exact resource caps** (memory/cpus/pids/tmpfs size, per-command + overall Q06 wall-clock)
  — start from task.toml's declared `environment` values (e.g. cpus 16, memory 16384) and the
  `build_timeout_sec`, tune during calibration.

## 8. Open questions

- Whether the two rounds should run against **one** container (reused) or **two** fresh
  containers. Default: one container reused across both rounds (cheaper; the independence that
  matters is the agent's reasoning + fresh context, not fresh disk). Revisit if a runtime check
  in round 1 leaves `/scratch` state that could bias round 2 — tmpfs is wiped only on container
  restart, so if that proves to matter, restart between rounds.
- Scope selector precision: whether to pre-filter repo-checkable criteria in Python or let the
  agent mark non-code criteria `passed=true`. Default: let the agent mark them (no criterion
  silently dropped); measure over-scoping in the smoke run.
```

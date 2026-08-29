# AutoQC — Quality Rubric for SWE Atlas Codebase Q&A Tasks

AutoQC judges whether a trainer-authored **task rubric** for a SWE Atlas Codebase Q&A
task is *soundly constructed*. It ingests a task bundle and returns a verdict —
`sound`, `needs_human_review`, or `not_sound` — together with specific, actionable
rework instructions.

## The two rubrics (what we are and are not checking)

Every SWE Atlas task carries **two** rubrics, and they are easy to confuse:

1. **The task rubric** — authored by the trainer. A list of `positive` / `negative`
   criteria that a model's answer is graded against (all-or-nothing under the
   downstream grader).
2. **The quality rubric** — *this project*. It does not grade any model answer. It
   checks whether **rubric #1 is well-formed**: atomic, binary-judgeable, free of
   escape hatches, factually correct against the repo, and so on.

AutoQC is the executable implementation of **rubric #2**. It is a QC gate on rubric
authoring, not a grader. How a task rubric is *scored* against answers is a separate,
downstream concern that AutoQC deliberately stays out of.

**Phase 1.5** extends rubric #2 beyond `tests/rubrics.json` itself to the other
non-rubric task files it depends on: the prompt (`tests/prompt.txt`), the rendered
instruction (`instruction.md`), and the reference answer (`solution/answer.txt`).
It adds 13 checks in four new namespaces — **P** (prompt), **A** (answer), **AL**
(cross-file alignment), **H** (hygiene) — plus **Q13**, which extends the existing
Q-family to verify the rubric itself grades demonstrated codebase exploration. See
`docs/superpowers/specs/2026-08-28-autoqc-phase1.5-prompt-answer-design.md` for the
full design and calibration basis.

## Design: two stages × two severities

Checks are organized on two independent axes:

- **Stage** — *how/when* a check runs. Stage 1 is deterministic Python (no LLM,
  fail-fast). Stage 2 is semantic (LLM judgment); a subset of Stage 2 checks
  (Q06, and Phase 1.5's P04/A06) are **grounded** — they need a live repo
  checkout, not just the bundle text.
- **Severity** — *what a failure means*. `reject` failures make a rubric `not_sound`
  (must be reworked). `warn` failures route to `needs_human_review` (a low-signal
  smell, not a hard defect).

These axes are orthogonal: a semantic check can be either severity.

### Stage 1 — structural / deterministic (`autoqc/structural.py`)

Pure Python, runs in milliseconds, no model calls.

| ID  | Checks | Severity |
|-----|--------|----------|
| S01 | `rubrics.json` is a JSON array | reject |
| S02 | Each item has the required shape (`id`, `title`, `annotations`) | reject |
| S03 | `id` matches the id regex and is unique | reject |
| S04 | Title numbering/format is consistent (`1.x` positives ↔ `2.x` negatives) | reject |
| S05 | At least one positive criterion exists | reject |
| S06 | Bundle is complete (all required files present) | reject |
| S07 | `base_commit` present and well-formed (40-char SHA) | reject |
| S08 | Title numbering has no gaps; type vocabulary is known | warn |

### Stage 2 — semantic (agent ensemble)

Each semantic check is executed by a tool-using LLM agent (see *Execution* below).

| ID  | Checks | Severity | Status |
|-----|--------|----------|--------|
| Q01 | **Atomicity** — a criterion grades one independently-gradable fact | reject | ✅ built |
| Q02 | **Binary judgeability** — pass/fail is unambiguous, no undefined completeness claims | reject | ✅ built |
| Q03 | **No wildcard / escape hatch** — no `*`, "or similar", non-interchangeable e.g. lists | reject | ✅ built |
| Q04 | **Prompt coverage** — every obligation in the prompt is graded | reject | ✅ built |
| Q05 | **No unrequested scope** — criteria don't grade facts the prompt never asked for | reject | ✅ built |
| Q06 | **Factual correctness** — each criterion's claim is true at `base_commit` | reject | ✅ built |
| Q07 | **Negative score-flip semantics** — a negative states the FALSE assertion that should fail | reject | ✅ built |
| Q08 | **Discriminating negatives** — at least one negative adds grading power beyond the positives | warn | ✅ built |
| Q09 | **Negative present** — the rubric has ≥1 negative criterion | warn | ✅ built (deterministic) |
| Q10 | **Empirical result graded** — for run-the-code tasks, an observed result is graded | warn | ✅ built |
| Q11 | **Not lookup-dominated** — the rubric isn't mostly trivial fact-lookup | warn | ✅ built |
| Q12 | **Economical** — not oversized (>18 criteria → human) | warn | ✅ built (size only) |

> Q09 and Q12 are pure-Python deterministic checks (`autoqc/agent/deterministic.py`);
> everything else in Stage 2 is agent-driven.

### Stage 1.5 — prompt/answer/hygiene (`autoqc/text_deterministic.py`)

Two more pure-Python checks, run alongside `structural.py` in the no-LLM phase.

| ID  | Checks | Severity |
|-----|--------|----------|
| P01 | `instruction.md` is rendered from the real prompt, not the authoring placeholder | reject |
| H01 | No committed build/cache artifacts (`__pycache__`, `*.pyc`) | warn |

### Stage 2.5 — prompt & answer quality (semantic + grounded)

Phase 1.5 adds 13 checks judging the prompt, the answer, their alignment with
`instruction.md`, and one more rubric check (Q13). Most run as whole-document
semantic-text checks through the same proposer/adversary ensemble as Q01–Q11; two
(`P04`, `A06`) need the repo checkout and run **grounded**, sharing Q06's
container session.

| ID   | Axis | Checks | Severity | Tier |
|------|------|--------|----------|------|
| P01  | Prompt | `instruction.md` rendered, not the authoring placeholder | reject | deterministic |
| P02  | Prompt | Single coherent goal — not multiple independent tasks bundled | reject | semantic-text |
| P03  | Prompt | Natural conversational request — not a rigid checklist; doesn't telegraph the result | reject | semantic-text |
| P04  | Prompt | Requires codebase exploration — not answerable from general knowledge alone | reject | grounded |
| A01  | Answer | Opens investigation-first, not conclusion-first | warn | semantic-text |
| A02  | Answer | Continuous interleaved narrative, not section headers | warn | semantic-text |
| A03  | Answer | Sustained first-person "I" voice | warn | semantic-text |
| A04  | Answer | Every empirical claim shows command **and** output inline | reject | semantic-text |
| A05  | Answer | Bash-only investigation method, not a script/file as the deliverable | warn | semantic-text |
| A06  | Answer | Trajectory-like: genuine exploration, not a bare answer or a raw trajectory dump | reject | grounded |
| AL01 | Align | `instruction.md` / `prompt.txt` / `answer.txt` describe the same task | warn | semantic-text |
| H01  | Hygiene | No committed build/cache artifacts | warn | deterministic |
| Q13  | Rubric | Rubric has ≥1 must-have criterion verifying demonstrated exploration | reject | semantic-text |

> P04 and A06 run inside the same `ContainerSession` as Q06 (one container build
> per task, three grounded checks against it) via `run_grounded_stage`.

### Verdict logic (`autoqc/verdict.py`)

1. Any **undisputed** `reject` failure → **`not_sound`** + rework instructions.
2. A **disputed** `reject` (the ensemble split, or the adversary overturned it) →
   **`needs_human_review`** — never a silent hard reject on a contested call.
3. Otherwise, any `warn` failure or any escalation → **`needs_human_review`**.
4. Otherwise → **`sound`**.

Difficulty / pass@k is a separate downstream gate and is out of scope here.

## Execution

**Stage 1** is straight Python over the parsed bundle.

**Stage 2 (semantic)** runs each check as a **tool-using agent** over an
OpenAI-compatible gateway (OpenRouter; judge model `z-ai/glm-5.2`). The agent is a
*judge*, not a code-solver: it emits a structured finding contract
(`submit_findings`) bound to a `check_id` and a criterion. Each check runs an
**ensemble** for robustness:

- **K = 3 proposer passes** — independent judgments of the criteria.
- **1 asymmetric adversary pass** — defends rejects / attacks passes to surface
  disagreement.
- **Adjudication** — a split proposer vote or an adversary overturn escalates the
  check to `needs_human_review` rather than committing to a verdict.

Every pass is isolated; a gateway/parse/timeout error fails **safe** (routes to a
human, never crashes and never hard-rejects on infrastructure failure).

**Q06 (factual correctness)**, plus Phase 1.5's **P04** and **A06**, are the checks
that need more than the bundle text: they verify claims against the actual
repository at `base_commit`. They run inside the task's own faithful Docker
container (built from `environment/Dockerfile`, network-disabled, non-root,
capability-dropped, ephemeral), sharing a single container build via
`run_grounded_stage` rather than paying the build/clone cost three times. Q06
verifies each rubric criterion's factual claim; P04 verifies the prompt genuinely
requires exploring the repo; A06 verifies the reference answer's claimed
exploration holds up and is neither a bare direct answer nor a raw trajectory dump.

## Bundle format

A task bundle is the standard Harbor layout:

```
task.toml                      environment/Dockerfile
instruction.md                 tests/{prompt.txt, rubrics.json, system_prompt.txt,
canary.txt                            user_prompt_template.txt, evaluate_answer.py, test.sh}
                               solution/{answer.txt, solve.sh}
```

AutoQC reads `tests/rubrics.json` (the task rubric under test), the prompt, and — for
Q06 — the pinned repo. `rubrics.json` is an array of criteria, each with an `id`, a
`title` (the criterion text, numbered `1.x` for positives / `2.x` for negatives), and
`annotations` (`type`, `importance`). Phase 1.5's checks also read `instruction.md`
and `solution/answer.txt`, and P04/A06 use the pinned repo the same way Q06 does.

## Usage

```bash
# Structural-only (no API key needed):
python3 -m autoqc <bundle_dir> [out_dir]

# Full run (structural + semantic) needs gateway credentials in a gitignored .env:
#   EVAL_API_KEY=<openrouter key>
#   EVAL_BASE_URL=https://openrouter.ai/api/v1
#   EVAL_MODEL=z-ai/glm-5.2
python3 -m autoqc <bundle_dir> autoqc_out
```

Outputs two files in `out_dir`: `review_record.json` (structured verdict + per-check
results) and `report.md` (a human-readable, reject-first rework report). Exit codes:
`0` sound, `1` needs_human_review, `2` not_sound.

The gateway client uses only the Python standard library (`urllib`) — no third-party
SDK. `.env` holds the API key and is never committed.

## Calibration

`autoqc/calibrate.py` builds labeled corpora (clean + seeded-defect variants) from
real bundles and scores AutoQC on recall / false-reject-rate / detection-accuracy.
Operator scripts:

- `scripts/calibrate.py <bundle_dir>…` — seeded recall/false-reject on Q07/Q03.
- `scripts/calibrate_clean.py <bundle_dir>…` — runs the full check set on clean "good
  set" bundles and separates **reject false-fires** (the danger, want 0) from
  expected **warn fires** (e.g. redundant negatives), with live per-check progress.

Early live results (`z-ai/glm-5.2`): seeded Q07/Q03 recall 1.0 and false-reject 0.0
on two bundles; broader clean-set runs show 0 hard reject false-fires with the
disputed→human safety net absorbing borderline calls. A known caveat: single-run
verdicts are not fully stable (a borderline check can flip between *escalate* and
*reject* across runs), so a per-candidate **flip-rate** pass is the next calibration
step.

## Status & roadmap

- ✅ **Stage 1 structural (S01–S08)** — built, reviewed, validated on 134 real bundles.
- ✅ **Stage 2 text checks (Q01–Q05, Q07–Q12)** — built, reviewed, live-calibrated.
- ✅ **Q06 factual correctness** — built. The container-build harness, `run_bash`
  tool, and Q06 agent role are complete and live-validated.
- ✅ **Phase 1.5 — prompt/answer/alignment/hygiene (P01–P04, A01–A06, AL01, H01,
  Q13)** — built. Deterministic P01/H01 run in Stage 1.5; the semantic-text checks
  run in the existing ensemble; P04/A06 run grounded, sharing Q06's container
  session. See `docs/superpowers/specs/2026-08-28-autoqc-phase1.5-prompt-answer-design.md`.
- 226 tests passing (all offline via `FakeLLMClient`).
- ⏭️ **Performance (step 2)** — the text-check agents currently spend extra turns
  calling read tools even though the bundle is preloaded, and run serially per task
  (~10 min/task). Planned: drop read tools for text checks (force single-turn
  submit), cap reasoning tokens, use a cheaper model for mechanical checks, and add
  check-level + pass-level parallelism. Target: ~10 min → ~1–2 min of model work per
  task.

## Repository layout

```
autoqc/
  structural.py          Stage 1 checks (S01–S08)
  text_deterministic.py  Stage 1.5 checks (P01, H01), no LLM
  verdict.py             verdict composition
  report.py              JSON record + markdown rework report
  cli.py                 entrypoint (structural, + semantic when a client is present)
  bundle.py              tolerant bundle loader (incl. instruction.md)
  llm.py                 native tool-calling client (Fake for tests, Gateway via urllib)
  agent/
    runner.py            run_agent — the tool-calling loop
    tools.py             Tool, AgentContext, read_bundle_file, submit_findings contract
    engine.py            run_check (ensemble + adversary + adjudication), run_semantic,
                          run_grounded_stage (Q06/P04/A06 sharing one container session)
    checks.py            SemanticCheck registry (Q01–Q05, Q07, Q08, Q10, Q11, plus
                          Phase 1.5's P02, P03, A01–A05, AL01, Q13); Q06/P04/A06 roles
    deterministic.py     Q09, Q12 (no LLM)
  seed.py, calibrate.py  seeded-defect corpora + scoring
scripts/                 operator scripts (smoke, calibration)
docs/superpowers/        design specs and implementation plans
tests/                   226 unit tests (all offline via FakeLLMClient)
```

Design details live in `docs/superpowers/specs/` — the quality-rubric design, the
execution-harness design, and the agent-architecture design (including the `run_bash`
safety model for Q06).

# AutoQC Phase 1.5 — Prompt & Answer Quality Rubric — Design

Date: 2026-08-28
Status: Draft for review
Owner: Harsh Sulakhe · Successor: Vitor Cirilo Araujo Santos

## 1. Purpose

Phase 1 (the shipped `S`/`Q` engine) judges whether a task's `tests/rubrics.json`
is soundly constructed. It deliberately left **prompt-only and answer-only
quality out of scope** (see the Phase 1 spec, §2). Reviewer feedback on the first
Batch-1 pass shows that is now the dominant source of rework: annotators are
making major mistakes on `tests/prompt.txt`, `instruction.md`, and
`solution/answer.txt` — not on the rubric, which the reviewers largely accept.

**Phase 1.5** adds a second quality rubric covering the **prompt** and **answer**
axes, plus the cross-file **alignment** and **hygiene** concerns that surfaced
alongside them. It reuses the Phase 1 machinery wholesale — the two-stage
(deterministic / semantic-ensemble) engine, the disputed→human verdict logic, the
parallel runner, and the Q06 container harness — and adds new check families
rather than a new mechanism.

## 2. Scope

**In scope** (text-checkable and repo-grounded quality of the non-rubric task
files):
- **Prompt** (`tests/prompt.txt`): single-goal, natural/conversational, and
  *requires codebase exploration* (no direct answer possible).
- **Instruction** (`instruction.md`): rendered from the real prompt, not the
  authoring placeholder.
- **Answer** (`solution/answer.txt`): investigation-first, continuous first-person
  narrative, evidence shown inline, bash-only method, and **trajectory-like** —
  genuine codebase exploration, neither a bare direct answer nor a raw full
  trajectory dump.
- **Alignment**: `instruction.md` ↔ `prompt.txt` ↔ `answer.txt` describe the same
  task.
- **Hygiene**: no committed build/cache artifacts.
- **Rubric exploration coverage** (extends the Phase 1 `Q` family): the rubric has
  a must-have criterion verifying the model actually explored the codebase.

**Out of scope** — the infrastructure-evidence rework reasons, which no text or
single-repo check can establish. These stay with the human reviewer / Harbor
pipeline:
- Harbor runs, model pass@k evidence, Docker-validation trustworthiness, the
  `evaluate_answer.py` harness-invariant, export/import round-trips, local
  checkout preparation under `taskapp/repos`.

## 3. Calibration basis

The standard is **derived from the reviewer corpus + a gold reference**, since no
written prompt/answer style guideline exists in the repo:

- **Negatives** — the 12 Batch-1 reworks (`conversations.json`, all `status:
  rework`, reviewer scores 2.5–3.0) and the `negetive_downloads` manifest, whose
  per-task feedback prose names the exact defects.
- **Directive** — Chirag Rade's Slack threads: the original prompt/answer defect
  list, and the later clarification that answer naturalness means *trajectory-like
  with genuine codebase exploration*, that the prompt must *require* exploration,
  and that the rubric must carry a must-have exploration criterion.
- **Gold reference** — the clean `qc-sample-pack` Turing `answer.txt`: opens with
  an investigation acknowledgment, sustained first-person "I", interleaves bash +
  output as continuous narrative, no section headers, bash-only. This is the
  positive exemplar every answer-axis check is calibrated against.

Provenance is recorded so the codified standard can later be reviewed by
Chirag/Anshul; it is not gated on that review for this build.

## 4. The rubric (13 checks)

New namespaces: **`P`** (prompt), **`A`** (answer), **`AL`** (alignment), **`H`**
(hygiene). **`Q13`** extends the existing Phase 1 `Q` (rubric) family. Severity
follows the Phase 1 philosophy: `reject` only for high-confidence, near-objective
defects; everything subjective/stylistic is `warn` → `needs_human_review`, with the
ensemble's disputed→human net still absorbing borderline semantic calls.

| ID | Axis | Check | Tier | Severity |
|----|------|-------|------|----------|
| **P01** | Prompt | `instruction.md` is rendered from the real prompt, not the authoring placeholder | Deterministic | reject |
| **P02** | Prompt | Single coherent goal — not multiple independent tasks bundled | Semantic (text) | reject |
| **P03** | Prompt | Natural conversational request; not a rigid enumerated checklist; does not telegraph the measured result | Semantic (text) | reject |
| **P04** | Prompt | Requires codebase exploration — not answerable as a direct/general-knowledge reply | Semantic (grounded) | reject |
| **A01** | Answer | Opens investigation-first — no conclusion-first "short answer" lede (a summary *after* the investigation ack is fine) | Semantic (text) | warn |
| **A02** | Answer | Continuous interleaved narrative — not broken into bold/numbered section headers | Semantic (text) | warn |
| **A03** | Answer | Sustained first-person "I" voice | Semantic (text) | warn |
| **A04** | Answer | Every empirical claim shows the actual command **and** its output inline | Semantic (text) | reject |
| **A05** | Answer | Bash-only investigation method — not "I created a script/file" as the deliverable | Semantic (text) | warn |
| **A06** | Answer | Trajectory-like: genuine codebase exploration — rejected if a bare direct answer **or** a raw full-trajectory dump | Semantic (grounded) | reject |
| **AL01** | Align | `instruction.md` ↔ `prompt.txt` ↔ `answer.txt` describe the same task | Semantic (text) | warn |
| **H01** | Hygiene | No committed build/cache artifacts (`tests/__pycache__`, etc.) | Deterministic | warn |
| **Q13** | Rubric | Rubric has ≥1 must-have criterion verifying the model explored the codebase | Semantic (text) | reject |

Reject set: **P01, P02, P03, P04, A04, A06, Q13** (7). Warn set: **A01, A02, A03,
A05, AL01, H01** (6). Stylistic answer prose (A01/A02/A03/A05) stays `warn` by
decision; the exploration checks (P04/A06/Q13) are `reject` per Chirag's "not
acceptable" framing.

## 5. Architecture & integration

Three tiers, mapping onto machinery that already exists.

### 5.1 Deterministic tier — `P01`, `H01`
New module `autoqc/text_deterministic.py`, run in the Stage-1 deterministic phase
alongside `structural.py`, emitting `CheckResult(stage=STRUCTURAL)`.
- **P01**: fire (reject) if `instruction.md` is absent or contains any known
  authoring-template marker. Markers are a small constant list taken verbatim from
  the placeholder bundles, e.g. `Describe the developer's realistic, multi-part
  question here`, the empty `<question>…</question>` scaffold, and the
  `I've uploaded a code repository in the directory /app` boilerplate. Pure string
  match — no heuristic judgment of "looks like a placeholder".
- **H01**: fire (warn) if the bundle tree contains committed cache artifacts
  (`__pycache__`, `*.pyc`).

### 5.2 Semantic text tier — `P02, P03, A01–A05, AL01, Q13`
Added to `autoqc/agent/checks.py` as `SemanticCheck`s, run by the existing
proposer(K=3)+adversary ensemble via the parallel engine. Two small extensions:
- **Whole-document unit modes.** New `unit_mode` values `"prompt"`, `"answer"`,
  `"bundle"` follow the existing single-synthetic-unit path already used by
  `unit_mode="rubric"`. `engine._check_prep` is generalized so `allowed =
  {check.unit_mode}` for any non-`criterion` mode (currently hardcodes `"rubric"`);
  each scope returns a one-element synthetic unit, gated on the relevant file being
  present (absence is already an S06 completeness reject, so these pass vacuously
  rather than double-reporting).
- **Prose review roles.** New `prose_proposer_role` / `prose_adversary_role` with
  system prompts oriented to reviewing a *document* (not rubric criteria), and new
  branches in `proposer_context`/`adversary_context` that inject `bundle.prompt`,
  `bundle.answer`, and `bundle.instruction`.

`Q13` reuses the existing rubric roles and `unit_mode="rubric"`; it judges whether
the criterion set includes a must-have exploration/empirical criterion (adjacent
to, but distinct from, the warn-level `Q10`).

### 5.3 Grounded tier — `P04`, `A06` (shared with `Q06`)
`P04` and `A06` need the repo checkout. They **share one `ContainerSession`** with
`Q06`: `run_factual_stage` is refactored into a **grounded stage** that builds the
task image and starts the network-isolated container **once**, then runs `Q06`
(rubric-vs-repo), `P04` (is the prompt answerable without the repo?), and `A06`
(does the answer's claimed exploration hold up against the repo, and is it
exploration rather than a direct answer / raw trajectory?) against that single
session, using the existing `factual_tools` (`run_bash`) role pattern. This avoids
paying the container build/clone cost per check.

### 5.4 Unchanged
`bundle.py` gains one field: `instruction: str | None` (reads `instruction.md` at
bundle root). `verdict.py` and `report.py` are untouched — new `CheckResult`s flow
through the existing reject-first composition and rework report automatically.

## 6. Testing & calibration

- **Offline unit tests** (`FakeLLMClient`, no network — matches the existing 122):
  a pass fixture and a fail fixture per semantic check; placeholder/cache fixtures
  for `P01`/`H01`. The clean `qc-sample-pack` answer is the pass fixture for the
  answer axis.
- **Calibration**, mirroring the Q06 approach: run the full Phase 1.5 set against
  the 12 Batch-1 negatives (measure **recall** on the reject-worthy defects the
  reviewers named) and against the clean `qc-sample-pack` (**reject false-fire =
  0** target; expected `warn` fires are tolerated). Report per-check recall and
  false-fire, and a flip-rate pass on the grounded checks (the known single-run
  instability caveat).

## 7. Rollout

Additive and behind the same entrypoint. Deterministic `P01`/`H01` need no
credentials; the semantic and grounded tiers run when a gateway client (and, for
the grounded tier, Docker) is present, exactly as Q01–Q06 do today. No change to
verdict semantics or exit codes.

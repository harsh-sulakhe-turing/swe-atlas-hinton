# AutoQC Quality Rubric — Design

Date: 2026-08-25
Status: Draft for review
Owner: Harsh Sulakhe · Successor: Vitor Cirilo Araujo Santos

## 1. Purpose

Define the **quality rubric** ("rubric #2"): the set of checks that decide
whether a trainer-authored SWE-Atlas Codebase Q&A task's `tests/rubrics.json`
(the "task rubric", rubric #1) is *soundly constructed*. AutoQC runs this rubric
on a task bundle before human review, returning a verdict plus specific rework
instructions.

This document specifies **the rubric itself** — the criteria and how each is
judged. It deliberately does **not** design how the checks execute (the
verification agent, container orchestration, hosting, output plumbing). That is a
separate, later phase (see §7).

## 2. Scope

**In scope.** Soundness of the task rubric, judged in the context of its prompt,
its pinned repository at `base_commit`, and its reference answer:
- structure/schema of `rubrics.json` and the surrounding Harbor bundle;
- each criterion being atomic, binary-judgeable, specific, correctly typed;
- factual correctness of each criterion at the pinned commit;
- the criterion set completely and non-redundantly covering the prompt;
- quality signals (discriminating negatives, graded runtime result).

**Out of scope.**
- The answer-grading mechanism (LLM judge, score-flipping, all-or-nothing
  reward). We check whether the rubric is sound, not how it is graded.
- **Difficulty calibration** (pass@k against Nemotron / Kimi). Decoupled by
  decision; it runs as a separate downstream gate *after* AutoQC passes.
- Prompt-only and environment-only quality except where a rubric criterion
  depends on it (e.g. factual correctness needs the repo; coverage needs the
  prompt).

## 3. Background and calibration basis

Findings from the first manual QC pass (10 internal candidate tasks + the
124-task public Scale AI `data/qa` set — see the manual-QC memory):

1. **Task-rubric structure.** `rubrics.json` is a flat JSON array. Each item is
   `{ id (32 hex), title ("N.x: <claim>"), annotations{ type, importance } }`.
   `1.x` = positive (a fact a correct answer must contain), `2.x` = negative (a
   false claim the answer must not make). The source/evidence pointer lives
   inside the title prose; there is no separate evidence field. Grading is
   **all-or-nothing**: one defective criterion breaks an otherwise-correct
   answer's score, which is why rubric soundness matters.

2. **The good-set bar is not uniform.** All 10 internal samples carry negatives
   and a separately-graded empirical result — but those negatives are mostly
   redundant inverses of positives (§8.1), so their mere presence is not evidence
   of quality. In the public good-set, **72/124 (58%) have zero negatives**, mean
   ~10.5 criteria, and some use wildcards (`/read/*`, "or similar") and lookup
   bundles. The public set is treated as high-quality by assumption. Therefore
   checks that the good-set itself routinely violates **cannot be hard rejects** —
   they are warnings.

   The 10 internal samples have now been **Q06-verified** (cloned at commit,
   every source-level claim checked: 149/149 sound — see §8.0). They are
   factually excellent; their only defects are low-severity Q08/Q12 warnings, so
   they resolve to `needs_human_review`. They still lack client acceptance and
   their runtime-measured claims (Q06b) were not executed.

3. **The handed-down LLM-generated spec is a baseline, not law.** Chirag's
   `TASK_QUALITY_GUIDELINE.md` makes mandatory negatives (R01/R08) a hard gate
   and would reject the majority of the good-set. It even flags its own cited
   example as defective. We reuse its vocabulary but recalibrate severities
   against the data.

## 4. Model: two stages × two severities

Every check has a **stage** (how/when it runs) and a **severity** (what its
failure means). The two axes are independent.

- **Stage 1 — structural / deterministic.** Pure code, no model, near-zero cost.
  Runs first as a fail-fast pre-filter so no tokens are spent on a malformed
  bundle.
- **Stage 2 — semantic.** Requires reading the prompt, rubric, answer, and
  (for factual checks) the pinned repository. Model/agent work.

- **`reject`** — the rubric is objectively broken; the task cannot pass AutoQC.
  Returned with a rework instruction.
- **`warn`** — a quality signal the good-set itself sometimes violates. Does not
  auto-reject; routes the task to human review with the warning attached.

Rationale for keeping severity separate from stage: a Stage-2 semantic failure
can be either fatal (a factually wrong criterion) or a warning (no negative
present). Stage alone cannot express that.

## 5. The quality rubric

Each check below is stated as: **ID · name · stage · severity · assertion ·
how to judge · example**. Examples cite real tasks from the calibration set
(`bNN` = internal sample; `task-6905…` = public good-set).

### 5.1 Stage 1 — structural (deterministic)

| ID | Name | Severity | Assertion |
|----|------|----------|-----------|
| S01 | Parses as JSON array | reject | `tests/rubrics.json` is valid JSON and a non-empty array. |
| S02 | Item shape | reject | Every item has `id`, `title`, `annotations.type`, `annotations.importance`. |
| S03 | ID format & uniqueness | reject | Every `id` matches `^[0-9a-f]{32}$` and is unique within the file. |
| S04 | Type ↔ number consistency | reject | Title is prefixed `N.x:`; `1.x` ⇒ `type` contains `positive`, `2.x` ⇒ `type` contains `negative`. |
| S05 | Has a positive | reject | At least one positive (`1.x`) criterion exists. |
| S06 | Bundle completeness | reject | Required files present: `tests/prompt.txt`, `tests/rubrics.json`, `solution/answer.txt`, `task.toml`, `environment/Dockerfile`. `task.toml` declares `repository` and a 40-char `base_commit`. |
| S07 | Type vocabulary | warn | `annotations.type` ∈ {`positive hli verifier`, `negative hli verifier`}; `importance` = `must have` (dataset norm — flag deviations, don't reject). |
| S08 | Sequential numbering | warn | `1.x` and `2.x` indices are sequential with no gaps/dupes (cosmetic; flag only). |

Stage 1 is necessary but low-value on its own: it catches only trivially broken
bundles. Its purpose is cost control (fail before Stage 2), not quality.

### 5.2 Stage 2 — semantic

Reject-severity checks (rubric is broken):

- **Q01 · Atomicity · reject.** No single criterion bundles two or more
  independently gradable facts. A cause and its direct effect may stay together
  only when partial satisfaction would be meaningless.
  *How to judge:* could the criterion be true of one fact and false of another
  and still be marked "met"? If yes, it is non-atomic.
  *Fail example (seed):* "States the port, readiness message, and two CSS
  variables" — four independent facts. *Borderline (passes Q01):* b03 `1.9`
  bundles `maxBufferLength 30`, `maxBufferSize 60MB`, and one-of-three others —
  this does **not** violate Q01 because the facts are tightly coupled and partial
  satisfaction is still meaningful; note it informationally but do not reject.
  The Q01 boundary is "independently gradable AND independently meaningful".

- **Q02 · Binary judgeability · reject.** Each criterion is decidable met/not-met
  from the answer text alone, with no subjective quality words ("thorough",
  "clearly", "adequately") and no undefined completeness claim.
  *Fail example:* "Explains the caching well."

- **Q03 · No wildcard / escape hatch · reject.** No criterion can be satisfied by
  vague or open-ended wording: bare `*`, "or similar", or an `e.g.`/"such as"
  list whose items are **not** genuinely interchangeable ways to satisfy the same
  requirement.
  *How to judge:* if a wrong or evasive answer can pass by matching the wildcard,
  it fails. An `e.g.` list is acceptable only when *any* listed item fully
  satisfies the requirement (record the accepted alternatives).
  *Fail example (public):* `task-6905…ba998` `1.8`/`1.9` "…or similar" — an open
  escape hatch; and `1.4` "the endpoint pattern, e.g. `/read/*`" — the wildcard
  passes without naming the endpoints that populate the stream. *Pass example:*
  b01 `1.9` "…such as connection-reuse counts, pooled-connection counts, or
  should_close values" — each is an interchangeable form of the same observed
  evidence.
  **CRITICAL GUARDRAIL (data-backed):** `e.g.`/`such as` appears in **118 of 124**
  good-set tasks and is the normal, legitimate way to list interchangeable
  acceptable answers. Q03 must **NOT** fire on the presence of `e.g.`/`such as`
  alone — doing so would reject ~95% of the good set. Q03 fires only on (a) true
  open escape hatches — `or similar`, `or other`, `and so on`, `among others`,
  bare trailing `*`; or (b) a list a reviewer judges **non-interchangeable** (its
  items are not each a full answer, e.g. `/read/*` standing in for "the specific
  endpoints"). The bare-regex signal is only a candidate; interchangeability is a
  semantic judgment.

- **Q04 · Prompt coverage · reject.** Every explicit obligation in `prompt.txt`
  maps to at least one positive criterion. The central runtime-only fact has its
  own positive, not buried in a broad one.
  *How to judge:* enumerate prompt obligations `O01…`; each must have ≥1 positive
  that fully grades it. An uncovered obligation is a reject.

- **Q05 · No unrequested scope · reject.** Every positive is either requested by
  the prompt or an indispensable causal link needed to answer it. The rubric does
  not grade facts the annotator merely discovered.

- **Q06 · Factual correctness at `base_commit` · reject.** Every positive is true
  and every negative is false at the pinned commit. Paths, symbols, types,
  defaults, exact strings, call order, and runtime results are checked against
  source or repeated execution — not copied trust from the reference answer.
  *How to judge:* requires the pinned repo/environment; deferred execution
  mechanics in §7. This is the highest-value check: an all-or-nothing grader
  turns one false criterion into a correct answer scoring 0.
  *Fail example:* a criterion naming a symbol that does not exist at the commit.
  **Two sub-modes (empirically established, §8):**
  - **Q06a — source-level facts** (paths, symbols, constants, defaults, exact
    strings, call structure). Verifiable by clone-at-commit + read; no build. This
    covers the large majority of criteria and was run on all 10 internal samples.
  - **Q06b — runtime-measured claims** (a measured ratio, a bundle-size delta, an
    observed exit code). Cannot be settled by reading source. For these, Q06
    verifies the **underlying mechanism exists** in source and marks the measured
    value RUNTIME-ONLY; asserting the value needs the full execution phase (§7).
    Do not reject a runtime-only criterion for lack of a static proof.

- **Q07 · Negative score-flip semantics · reject.** Every negative is phrased as
  the **false assertion whose presence should fail** the answer ("Claims that
  …"), never as "Does not claim…", a required omission, or the correct behavior.
  *Fail example (seed):* "Does not claim that bytes bodies fail." *Pass example:*
  b01 `2.2` "Claims that plain bytes bodies are affected by the same failure…".

Warn-severity checks (quality signals; route to human):

- **Q08 · Discriminating negatives · warn (low severity).** A negative adds
  grading power only when **no positive already requires the correct version of
  the same fact**. Because the grader is all-or-nothing, if some positive forces
  the answer to state the correct fact, the paired negative can never
  independently change the outcome — it is redundant. Redundant negatives are
  *low-signal, not wrong*: flag, never reject.
  *How to judge:* for each negative, look for a positive that requires the answer
  to assert the correct version of that fact. If one exists ⇒ redundant (warn).
  If none exists ⇒ discriminating (good).
  *Redundant example:* b05 `2.1` "Claims exponentially increasing backoff" is
  fully covered by positive `1.3` ("delay does NOT grow"), so it adds nothing.
  This is the dominant pattern in the internal samples (see §8.1). Even b01 `2.2`
  ("bytes bodies affected") is redundant, because positive `1.10` already
  requires stating bytes reuse normally.
  *Discriminating example:* b02 `2.2` "each controller registers its own watch" —
  no positive forces the shared-aggregated-watch statement, so it independently
  catches a plausible wrong belief.

- **Q09 · Negative present · warn.** At least one negative exists. (Warn, not
  reject: 58% of the good-set has none.)

- **Q10 · Empirical result graded · warn.** When the prompt requires running the
  software, at least one positive grades the *specific* observed result
  (comparison, value, state transition, artifact, or error) — not merely that the
  answer "ran" or "reported evidence".
  *Good example:* b02 `1.14` "~200 writes producing a single-digit reconcile
  count". *Weak example (seed):* "Reports empirical evidence." alone.

- **Q11 · Not lookup-dominated · warn.** The criterion set is not mostly trivia
  lookups (ports, constants, file lists) with no causal/synthesis content.

- **Q12 · Non-redundant & economical · warn.** No two criteria grade the same
  fact under different wording; set size is proportionate (the public good-set
  spans 5–26, mean ~10.5). More than 18 criteria always routes to human review.

## 6. Verdict

Applied after all checks run:

1. Any **reject** check fails ⇒ `not_sound`. Return the failing check IDs with
   per-criterion rework instructions.
2. Else any **warn** fires ⇒ `needs_human_review`, with the warnings attached.
3. Else ⇒ `sound`.

`needs_human_review` is a routing state, not a softened pass. Difficulty
(pass@k) is a separate gate that runs only after a `sound` verdict.

## 7. Deferred: execution (later phase, not designed here)

Per the current decision, we specify the rubric first and design execution
afterward. Open items, recorded so they are not lost:

- **How each Stage-2 check is executed** — one agent per criterion, a single
  agent over the whole rubric, or a hybrid; prompt design; structured
  per-check output.
- **Factual verification (Q06) environment** — expected to run inside the task's
  own `environment/Dockerfile`, reusing the `_evidence/validate_task.sh` pattern
  (same repo/commit/toolchain as the solver). Likely a custom tool-using agent.
- **Hosting** — local-first for this sprint, then server-hosted (GCP/Daytona) so
  the script cannot be tampered with; model access via the existing token
  gateway; no API keys distributed to annotators.
- **Output/review-record contract** — emit a structured record (reuse the shape
  of `task_quality_spec.json`) plus human-readable rework text.
- **Cost** — must fit the sub-$200/task pipeline budget; Q06 is the dominant
  cost driver.

## 8. Calibration & acceptance for the rubric

This section is now grounded in an executed verification pass, not assumption.

### 8.0 Empirical Q06 result on the 10 internal samples

Each of the 10 internal tasks was checked by cloning its repo at the exact
`base_commit` and verifying every criterion's source-level claim (Q06a). Result:

**149/149 criteria are factually sound at the pinned commit — zero
contradictions.** Every source-checkable positive was confirmed at the cited
lines (including the subtle traps: the `INTERSTITALS` misspelling, the
`unreachable!("stack underflow")` branch, the constant-vs-exponential limiter,
the `arg0/arg1` reflection path); every negative is genuinely false at the
commit. Two cosmetic notes only, neither touching a criterion (b02 says field
"Type" where source is `Typ`; a b09 *answer* prose line misplaces one helper
function). Runtime-measured claims (Q06b: coalescing ratio, bundle-size deltas)
were not executed — no build toolchain in the review sandbox — but their source
mechanisms are all present.

Consequences:
- The 10 are **factually excellent and structurally sound**: they pass all
  reject-severity checks (Q01–Q07) by inspection. The caution to verify rather
  than assume was right; the verification says they are clean.
- Their only defects are **low-severity warnings**: redundant negatives (Q08,
  pervasive) and a few empirical overlaps (Q12). So under this rubric they resolve
  to **`needs_human_review`**, not `not_sound` — the correct outcome for
  high-quality-but-unreviewed tasks.
- Therefore the 10 yield **Q08/Q12 warn** calibration examples and **boundary
  guards** (§8.1/§8.2), but **no reject examples** — reject cases must be seeded.

### 8.1 Negative-calibration set (mined from the 10 + public + seeded)

Warn examples (real, from data) — AutoQC must fire the tagged warn:

| Source | Defect | Check · severity |
|---|---|---|
| b05/b06/b07/b08/b09/b0a `2.1–2.3` | all negatives are inverses of positives (all-or-nothing ⇒ zero added power) | Q08 · warn |
| b01 `2.2`,`2.3`; b02 `2.4`; b03 `2.3`; b04 `2.3` | inverse-of-positive negatives | Q08 · warn |
| b02 `1.13`/`1.14`; b03 `1.8`/`1.13` | two criteria grade the same observed value | Q12 · warn |
| public `task-…ba998` | zero negatives, lookup-dominated, no runtime grading | Q09 + Q11 + Q10 · warn |
| public `task-…baa2a` | 25 criteria (>18), zero negatives, numbering gap (no `1.10`) | Q12 (→human) + Q09 + S08 · warn |
| 72/124 public tasks | zero negatives | Q09 · warn |

Reject examples — the 10 contain none, so these are **seeded** into copies of a
known-clean task; each must trip exactly its check:

| Seed | Check · severity |
|---|---|
| rename a symbol in a positive to one absent at the commit | Q06a · reject |
| merge two unrelated positives into one criterion | Q01 · reject |
| add a criterion `"…endpoints, or similar"` (open hatch) | Q03 · reject |
| delete the positive covering one prompt obligation | Q04 · reject |
| rewrite a negative as `"Does not claim …"` | Q07 · reject |

### 8.2 Good set and boundary guards

- **Good set** = the 10 internal samples (now Q06-verified) as the *exemplary*
  anchor, plus a curated set of public tasks that carry ≥1 negative and no escape
  hatch (43 of 124 qualify; e.g. `a9aa` — real HTTP/SMTP/log runtime evidence + 2
  discriminating negatives — `aa0c`, `aa17`) as the *adequate* anchor. AutoQC must
  return `sound` or `needs_human_review` (never `not_sound`) on these.
- **Boundary guards — AutoQC must NOT reject/flag:**
  - any criterion using `e.g.`/`such as` with interchangeable items (118/124 good
    tasks do this) — Q03 must pass;
  - coherent clusters answering one prompt sub-question (b02 `1.15`, b03 `1.9`) —
    Q01 informational only;
  - genuinely discriminating negatives (b02 `2.1`/`2.2`, b03 `2.1`/`2.2`, a9aa
    `2.1`/`2.2`) — Q08 must not flag.

### 8.3 Readiness bar

The rubric is ready when: every §8.1 warn example fires its warn; every seeded
reject trips its reject; no check fires on any §8.2 good-set criterion or boundary
guard. We still lack client negative feedback; until it arrives this set is the
proxy, and severities may be re-tuned when real rejections land.

### 8.1 Negative-calibration set (mined from the 10 internal samples)

Text-visible defects confirmed by inspection (Q06 factual checks excluded — those
require running the repos, deferred to §7). Each is a seed test case: AutoQC must
fire the tagged check at the tagged severity.

| Task · criteria | Defect | Check · severity |
|---|---|---|
| b05 `2.1/2.2/2.3` | all three negatives are inverses of positives `1.3/1.2/1.7` | Q08 · warn |
| b06 `2.1/2.2/2.3` | inverses of `1.5/1.8/1.10` | Q08 · warn |
| b07 `2.1/2.2/2.3` | inverses of `1.4/1.3/1.2` | Q08 · warn |
| b08 `2.1/2.2/2.3` | inverses of `1.3/1.7/1.9` | Q08 · warn |
| b09 `2.1/2.2/2.3` | inverses of `1.9/1.1/1.9` | Q08 · warn |
| b0a `2.1/2.2/2.3` | inverses of `1.3/1.7/1.9` | Q08 · warn |
| b01 `2.2`, `2.3` | inverses of `1.10`, `1.6` | Q08 · warn |
| b02 `2.4` | inverse of `1.15` | Q08 · warn |
| b03 `2.3` | inverse of `1.14` | Q08 · warn |
| b04 `2.3` | inverse of `1.7/1.8` | Q08 · warn |
| b02 `1.13` vs `1.14` | both grade observed wakeup counts | Q12 · warn |
| b03 `1.8` vs `1.13` | both grade measured bundle sizes | Q12 · warn |

**Boundary guards (AutoQC must NOT reject these — they test over-firing):**

| Task · criterion | Why it looks like a defect but is not | Guarded check |
|---|---|---|
| b02 `1.15` | bundles shared+shared / excl+excl / shared+excl, but they form one coherent answer to a single prompt sub-question ("what happens on conflicting outputs") | Q01 — informational smell, not reject |
| b03 `1.9` | bundles `maxBufferLength 30`, `maxBufferSize 60MB`, one-of-three — one coherent "buffering defaults" answer | Q01 — informational smell, not reject |
| b04 `1.10` | "names weak algorithms such as <list>" — the items are interchangeable and a judge reads it as "at least one" | Q02/Q03 — must pass |
| b02 `2.1` (mutex), `2.2` (own-watch); b03 `2.1` (runtime flags), `2.2` (webpack/HMR) | genuinely discriminating negatives — no positive forces the fact | Q08 — must NOT flag |

**No text-visible hard rejects were found in the 10.** By inspection they yield
Q08/Q12 warnings and atomicity smells only. The one unresolved reject risk is
**Q06 factual correctness**, which text inspection cannot settle — it requires
running the pinned repos (deferred, §7). This is precisely why the 10 must not be
assumed to pass.

## 9. Open questions

- Do any Stage-2 warns deserve promotion to reject once we see trainee output
  (as opposed to the single-author internal samples)? The 10 are one author's
  clean template; real trainee rubrics will fail differently, and the
  reject/warn line may need to move once we have that distribution.
- Q06b (runtime-measured claims) was not executed in this pass — the source
  mechanisms were confirmed but the measured values (ratios, sizes, exit codes)
  were not reproduced. Confirming that Q06b is affordable within the per-task
  budget is part of the deferred execution phase (§7).
- Verifying negatives' falsity (Q06 on `2.x`) proved as cheap as verifying
  positives' truth in this pass, so both are kept in scope.
- Should "coherent-cluster" atomicity (b02 `1.15`, b03 `1.9`) ever be promoted
  from informational to a Q01 warn, or does that create noise on good tasks?

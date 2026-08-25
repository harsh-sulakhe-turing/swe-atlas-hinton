# AutoQC Execution Harness — Design

Date: 2026-08-25
Status: Draft for review
Owner: Harsh Sulakhe · Successor: Vitor Cirilo Araujo Santos
Companion: `2026-08-25-autoqc-quality-rubric-design.md` (defines S01–S08, Q01–Q12)

## 1. Purpose and scope

Implement the quality rubric (rubric #2) as an executable pipeline that ingests a
Harbor task bundle and returns a soundness verdict plus specific rework feedback,
before human review.

**In scope:** the check-execution engine (deterministic + semantic + factual),
the ensemble/adversary machinery, the verdict, both report forms, and the test
harness that validates AutoQC itself.

**Out of scope (deferred):** hosting/deployment specifics, the token-gateway
wiring, and difficulty (pass@k) calibration — a separate downstream gate that
runs only after a `sound` verdict.

## 2. Pipeline architecture

```
bundle (prompt, rubric, answer, task.toml, Dockerfile)
  │
  ├─ Stage 1  DETERMINISTIC  (S01–S08)  ── Python, no model, fail-fast
  │     └─ any reject → stop, emit not_sound
  │
  ├─ Stage 2a SEMANTIC       (Q01–Q05, Q07–Q12)  ── batched agent pass
  │     per-criterion: Q01, Q07, Q08 (each criterion)
  │     per-rubric:    Q04, Q09, Q11, Q12 (whole set)
  │     each check → K-vote ensemble + 1 asymmetric adversary round
  │
  ├─ Stage 2b FACTUAL        (Q06)  ── inside the built container
  │     per-criterion, independent; Q06a source + Q06b run; 2 rounds
  │
  ├─ ADJUDICATE  → per-check verdict + evidence, all votes logged
  └─ VERDICT (§6 of rubric spec) → not_sound / needs_human_review / sound
         └─ outputs: structured record + markdown rework report
```

Build the container **once per task** and run all of that task's Q06 checks
inside it (§4).

## 3. The semantic judgment engine (Stage 2a)

Every semantic check runs the same shape, at its natural unit:

1. **Proposers (ensemble):** K independent agents evaluate the check, each
   returning `{verdict, confidence, evidence}`. Evidence is mandatory and
   concrete — quoted prompt spans, rubric IDs, and (for Q06) `file:line`. A
   verdict without evidence is discarded.
2. **Aggregate:** majority verdict; unanimous → high confidence, split → flag.
3. **Adversary (1 round, asymmetric by verdict direction):**
   - aggregated verdict = **reject** → adversary must *defend* the criterion
     (guards the expensive false-reject / annotator-trust error);
   - aggregated verdict = **pass** on a check → adversary must *attack* it
     (guards the false-accept / quality error).
4. **Adjudicate:** adversary concurs → final. Adversary overturns *with concrete
   evidence*, or the ensemble was split → **`needs_human_review`** with the
   dissent surfaced. Never a third round — disagreement escalates to a human.

**Units.** Per-criterion checks (Q01 atomicity, Q07 negative semantics, Q08
discriminating negatives) evaluate each criterion; per-rubric checks (Q04
coverage, Q09 negative-present, Q11 lookup-share, Q12 redundancy) evaluate the
whole set. Batch the cheap per-criterion checks into one structured pass per
proposer to bound agent count, while keeping Q06 per-criterion-independent.

**Evidence + logging.** Every proposer/adversary vote, with its evidence, is
persisted to the review record. This gives auditability and the repeatability the
rubric guideline (T03) expects from nondeterministic judges.

## 4. Factual verification in the faithful container (Stage 2b)

Q06 runs against ground truth — the repository at `base_commit` — because it is
author-independent and therefore breaks the golden↔rubric circularity.

- Build `environment/Dockerfile` once per task; run all Q06 checks inside it, so
  the review sees the same repo, commit, and toolchain as the solver.
- **Q06a (source-level):** verify paths, symbols, constants, defaults, exact
  strings, call structure by reading source. (Proven feasible: the calibration
  pass confirmed 149/149 internal criteria this way, via clone; the container is
  the faithful version.)
- **Q06b (runtime-measured):** actually run the programs to confirm measured
  claims (ratios, sizes, exit codes). The container makes these verifiable rather
  than merely mechanism-confirmed — closing the one gap the read-only pass left.
- **Adversary = independent re-derivation** (not debate): a second agent re-checks
  each claim against source/runtime and must agree on the `file:line` evidence;
  2 rounds; unresolved disagreement → human.
- The golden answer is **inadmissible** as Q06 evidence.

## 5. Adjudication and verdict

Per-check verdicts aggregate via the rubric spec §6:
1. any reject-severity check fails → `not_sound`;
2. else any warn fires → `needs_human_review`;
3. else → `sound`.

`needs_human_review` also absorbs every ensemble split and adversary overturn.

## 6. Golden answer — role

Context only. It informs Q04/Q05/Q10 (what a complete answer covers) but is never
an arbiter of soundness, and never Q06 evidence. A "does the golden satisfy the
rubric" check is deliberately **excluded**: golden and rubric are co-authored to
match, so the check is tautological when they agree and unactionable when they
disagree (we may be editing the rubric either way).

## 7. Outputs

Both, from one run:
- **Structured record** — reuses the shape of `task_quality_spec.json`: per-check
  results, evidence references, votes, and the verdict. Consumed by the pipeline.
- **Markdown rework report** — for the annotator: reject-first, each failing check
  → offending criterion ID → evidence (quoted span / `file:line`) → a concrete
  "change X → Y" instruction; warnings listed separately as "review these". This
  is the "specific, reworkable feedback" the project requires.

## 8. Testing strategy (how we validate AutoQC itself)

AutoQC is a judge; we score it against cases whose correct verdict is known.

- **Stage 1 — deterministic unit tests.** Table-driven: malformed JSON, missing
  file, bad ID regex, `1.x` typed negative → assert the exact S-check fires.
- **Backbone A — seeded-defect mutation corpus (scalable).** Inject exactly one
  known defect into a known-clean task; assert AutoQC fires exactly that check and
  no other. We inject, so labels are free. Substrate = the 10 internal **and** the
  124 public tasks (we trust our injected label, not Nvidia's). ~134 tasks ×
  ~10 defect types ≈ 1,300+ labeled bad cases. Measures per-check recall +
  misattribution.
- **Backbone B — human-labeled oracle.** For the subtle judgments (atomicity,
  discriminating negatives) that seeds can't cover, Harsh + Vitor label real
  criteria sound/defective. A starter set already exists from the manual QC
  (redundant negatives across b05–b0a, atomicity smells, 149 Q06-verified
  criteria); extend incrementally.
- **Negative controls.** Good set (10 verified internal + curated public) must
  never be `not_sound` → false-reject rate. Boundary guards (`e.g.` lists,
  coherent clusters, discriminating negatives) must not fire → over-firing.
- **Metrics:** false-reject rate; per-check recall + misattribution; verdict
  accuracy vs the human oracle; **stability** (N-run verdict flip-rate, plus the
  ablation that ensemble+adversary actually reduces it vs single-shot).
- **Hygiene:** tiny N → leave-one-out over the 10, not a fixed split; hold seed
  *types* out per task. Regression CI re-runs the full labeled suite on every
  prompt/K/round change.
- **Honest limits:** no client-labeled negatives yet, so passing the suite ≠
  client acceptance. The 10 are one author/one template → prompts can overfit;
  mitigate by seeding across the diverse public 124 and by treating the first
  trainee-authored batches (authoring began Aug 25) as the real generalization
  test, expecting to re-tune. The corpus grows as production QC runs.

## 9. Deployment (deferred)

Local-first for this sprint; then server-hosted (GCP/Daytona) so the script can't
be tampered with; model access via the existing token gateway; no API keys to
annotators. Algorithm is host-agnostic; this is a packaging concern for later.

## 10. Feasibility risks

- **Multi-language container builds** are the expensive, fragile part — the 10
  span Python/Go/Node/.NET/Java/Rust/PHP, each a full image build (guideline
  allots ≤1800s each). Build breakage is the main engineering cost; hence
  build-once-per-task and run all Q06 inside.
- **Cost:** all-semantic ensemble + adversary + per-task container is expensive by
  design (safety over cost, per decision). K and rounds are the tuning knobs once
  false-reject/accept rates are known.
- **Nondeterminism:** mitigated by ensembles, stability tests, and logged
  evidence; never fully eliminated.

## 11. Build order (milestones)

1. **Skeleton + Stage 1** — bundle loader, deterministic S01–S08, verdict plumbing,
   structured-record + markdown report scaffolding.
2. **Seed harness + 2 semantic checks end-to-end** — the mutation corpus
   generator, plus (say) Q03 and Q07 with the full ensemble+adversary shape and
   their tests. Proves the pattern before scaling.
3. **Remaining Stage 2a semantic checks** — Q01, Q02, Q04, Q05, Q08–Q12.
4. **Stage 2b container + Q06** — build-once, Q06a source, Q06b runtime,
   independent re-derivation.
5. **Reports + calibration run** — full good-set/boundary/seed suite, metrics,
   leave-one-out, regression CI.

Each milestone is independently testable and leaves a runnable system.

## 12. Open questions

- K (ensemble size) and the confidence thresholds for "split → human" — set
  empirically in milestone 5.
- Which model tier per stage (deterministic none; cheap for batched semantic;
  strongest for Q06 and adversary) — decide against the gateway's menu.
- Whether any warn graduates to reject once trainee output is seen (carried from
  the rubric spec).

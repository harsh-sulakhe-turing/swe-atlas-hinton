# AutoQC Performance — Parallel Semantic Engine + Single-Turn Text Checks (design)

Date: 2026-08-27
Status: Draft for review
Builds on: `2026-08-25-autoqc-agent-architecture.md`, the merged Q06 work
(`2026-08-27-autoqc-q06-factual-design.md`)

## 1. Goal

Cut single-task latency of the semantic engine from ~10 min toward ~1–2 min, and
reduce run-to-run nondeterminism, without changing verdicts. Two levers only
(other optimizations explicitly deferred, §7):

1. **Parallelism** — run the text-check agent passes concurrently over one bounded
   thread pool instead of 36 sequential passes.
2. **Single-turn text checks** — give the text roles only `submit_findings` and a
   low turn cap, so each pass is one gateway call instead of ~3 (the agent no
   longer burns turns re-reading a bundle that is already inlined).

The two multiply: fewer calls per pass × passes running concurrently.

## 2. Current cost (single-task path, `python3 -m autoqc`)

The engine is fully serial. `run_semantic` runs 9 text checks one after another;
each `run_check` runs K=3 proposers then 1 adversary sequentially → **36 sequential
agent passes**. Each pass allows up to 12 turns (`run_agent(max_turns=12)`), and
`proposer_role`/`adversary_role` carry `read_bundle_file` + `list_dir` even though
the whole bundle is already inlined into their context — so passes routinely spend
turns "reading" files they already have (~12 gateway calls measured for a 4-pass
check that needs ~4). The Q06 factual stage (container build + 2 serial rounds) is
separate and unchanged by this work.

`scripts/calibrate_clean.py` already parallelizes at the **bundle** level for
sweeps; this design parallelizes the **single-task** engine a trainer runs.

## 3. Lever A — parallel scheduler (flat bounded pool)

### 3.1 One pool, one knob

A single `concurrent.futures.ThreadPoolExecutor` with width `W` read from
`AUTOQC_ENGINE_WORKERS` (default 8). The GatewayLLMClient is stdlib `urllib`,
stateless per call, and already proven thread-safe (calibrate shares one client
across threads) — so this is threads, no async rewrite. Total concurrent gateway
calls are capped at `W` (important: the gateway showed mild contention past ~6-way;
one knob bounds pressure).

### 3.2 Decompose `run_check` into leaves

`run_check` is split into reusable pieces so the scheduler can treat a single agent
pass as the unit of concurrency:

- `proposer_pass(check, bundle, client, ctx, p_ctx) -> PassResult` — one
  `run_agent(proposer_role(), …)`; returns `{ok, findings, log_entry}`.
- `finalize_check(check, finding_sets, adversary_result) -> CheckResult` — the
  existing aggregate → adjudicate → roll-up logic, unchanged in behavior.

`run_check` stays as a **thin serial wrapper** over these leaves (K proposer_pass
calls, then the adversary, then finalize), so `scripts/calibrate_clean.py` and the
existing tests keep working unchanged.

### 3.3 The scheduler

`run_semantic` (when parallel) drives dependencies without ever blocking a pool
worker on another pool task (no nested-pool deadlock):

```
pool = ThreadPoolExecutor(max_workers=W)
# submit every proposer pass for every check up front
prop_futs = { pool.submit(proposer_pass, check, …): (check, i)
              for check in text_checks for i in range(k) }
# gather per check; when a check has all k, submit its adversary
# when the adversary future returns, finalize_check -> CheckResult
```

- **Per-check barrier:** a check's adversary is submitted only after its k proposer
  passes complete (the adversary needs the aggregated verdict). Different checks'
  proposers and adversaries interleave freely up to `W`.
- **Determinism:** majority-vote aggregation is order-independent; the final results
  list is reassembled in check order. The parallel run yields the **same
  CheckResults** as the serial run.
- **No shared mutable state / no locks:** each pass returns its own findings and its
  own `votes_log` entry; the scheduler assembles the `votes_log` single-threaded in
  a deterministic order after all passes complete.
- **Error isolation:** every `future.result()` is wrapped in `try/except` (mirroring
  `calibrate_clean`); a pass that raises degrades to empty findings (`ok=False`) →
  that check may go `needs_human`, but one bad pass never sinks the run. `run_agent`
  already turns gateway/parse errors into `ok=False`.
- The **deterministic checks** (Q09/Q12) and the **Q06 factual stage** run after the
  text pool, unchanged. Q06 stays serial (its parallelism was explicitly deferred).

## 4. Lever B — single-turn text checks

- New `text_tools() -> [submit_findings]` in `autoqc/agent/tools.py`. `proposer_role`
  and `adversary_role` switch from `default_tools()` to `text_tools()` — they lose
  `read_bundle_file` and `list_dir`. `factual_role` is unchanged (keeps
  `factual_tools()` with `run_bash`).
- Text passes call `run_agent(..., max_turns=TEXT_MAX_TURNS)` with
  `TEXT_MAX_TURNS = 3` — a backstop; with only `submit_findings` available the agent
  is expected to submit on turn 1. The runner already nudges once on a no-tool-call
  turn and degrades to `needs_human` if the cap is hit without a submit.
- **Safety of removing read tools:** everything a text check needs is already inlined
  by `proposer_context`/`adversary_context` — the full task prompt, the criteria (for
  per-criterion checks), and the full rubric (for rubric-mode checks). The golden
  answer is context-only per the rubric design and is not needed by any text check.
  So no text check depends on a read tool.
- Effect: ~12 → ~4 gateway calls per check, and less flip-rate nondeterminism (the
  agent no longer branches on whether to read).

## 5. Behavior-change note (deliberate)

Lever B changes live judgment behavior (text agents can no longer pull `answer.txt`
or re-read files mid-check). It is safe from the inlined context, but it means the
**next live calibration validates this faster config**, not the pre-parallel one.
This is the correct order given calibration is being deferred to after the perf work.
Offline tests are unaffected (they drive FakeLLMClient deterministically).

## 6. Testing (all offline, FakeLLMClient)

- **Keystone equivalence:** the parallel `run_semantic` produces the **same
  CheckResults** as a serial run over the identical fake responder — proves
  concurrency changes only speed, not verdicts. Run it a few times to confirm the
  result is stable regardless of completion order.
- Aggregation order-independence (shuffle proposer completion order → same verdict).
- `votes_log` completeness and deterministic ordering under the parallel scheduler.
- `proposer_role().tools` and `adversary_role().tools` == `[submit_findings]` (no
  read tools); `factual_role().tools` still includes `run_bash`.
- A responder that never submits is bounded by `TEXT_MAX_TURNS` → the pass is
  `ok=False` → the check degrades to `needs_human` (never hangs, never crashes).
- Per-future error isolation: a pass that raises leaves other checks' results intact.
- Existing `run_check` serial wrapper tests still pass unchanged.

## 7. Non-goals (deferred)

- **Model/token tiering** (cheaper model for mechanical checks, lower
  `EVAL_MAX_TOKENS`).
- **Q06 round parallelism** (the 2 factual rounds concurrently) and a **container
  build pool** for sweeps.
- **Overlapping the Q06 container build with the text pool** — a real latency win,
  but adds cross-resource error handling; a candidate for a later round.
- The width default and any per-check tuning are calibration-time concerns; this
  design ships one env-tunable knob with a sane default.

## 8. Files touched

- `autoqc/agent/engine.py` — split `run_check` into `proposer_pass` +
  `finalize_check`; add the parallel scheduler path to `run_semantic`; keep
  `run_check` as a serial wrapper.
- `autoqc/agent/tools.py` — add `text_tools()`.
- `autoqc/agent/checks.py` — `proposer_role`/`adversary_role` use `text_tools()`.
- `autoqc/agent/runner.py` — no change (max_turns is already a call arg); the low
  cap is passed by the text passes.
- Tests: a new `tests/test_engine_parallel.py` for the scheduler + equivalence;
  extend `tests/test_agent_checks.py` / `tests/test_agent_tools.py` for the tool
  changes.

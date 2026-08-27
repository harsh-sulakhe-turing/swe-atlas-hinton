"""Clean-bundle false-reject calibration across the FULL 11-check set.

Runs the shipped semantic engine (all 11 checks + verdict) live against each
internal "good set" bundle and reports, per bundle, which checks FIRED
(passed=False) — split into:

  * REJECT false-fires  -> the real danger: a clean rubric hard-flagged NOT_SOUND
  * WARN fires          -> expected/acceptable (e.g. Q08 redundant negatives are
                           pervasive in the internal set per manual QC; these are
                           TRUE fires, not false-rejects)

The internal bundles are clean ONLY w.r.t. reject checks; they legitimately
carry Q08/Q12 warns. So the headline number is REJECT_FALSE_FIRES == 0.

Bundles run concurrently over a bounded thread pool (the engine is unchanged
and fully synchronous; the calls are I/O-bound on the gateway, so threads win).
Tune width with AUTOQC_CALIB_WORKERS (default 6). Rather than call run_semantic
(which hides its per-check loop), we drive each check ourselves via run_check so
every check emits a live start/done line with timing — granular progress across
all bundles. The final table is reassembled in bundle order at the end.

Usage: python3 scripts/calibrate_clean.py <bundle_dir> [<bundle_dir> ...]
"""
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, ".")
for _line in open(".env"):
    _line = _line.strip()
    if "=" in _line and not _line.startswith("#"):
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k, _v)

from autoqc.bundle import load_bundle
from autoqc.agent.engine import run_check
from autoqc.agent.checks import SEMANTIC_CHECKS
from autoqc.agent.deterministic import DETERMINISTIC_CHECKS
from autoqc.agent.tools import AgentContext
from autoqc.verdict import compute_verdict
from autoqc.model import Severity
from autoqc.llm import GatewayLLMClient

bundles = sys.argv[1:]
if not bundles:
    print("usage: python3 scripts/calibrate_clean.py <bundle_dir> [...]", file=sys.stderr)
    raise SystemExit(64)

client = GatewayLLMClient()  # stdlib urllib, stateless per call -> shareable across threads
assert client.available(), ".env missing EVAL_API_KEY / EVAL_BASE_URL"
workers = min(int(os.environ.get("AUTOQC_CALIB_WORKERS", "6")), len(bundles))

_CHECKS_PER_BUNDLE = len(SEMANTIC_CHECKS) + len(DETERMINISTIC_CHECKS)
_total_checks = len(bundles) * _CHECKS_PER_BUNDLE
_print_lock = threading.Lock()
_done_count = 0  # completed checks across all bundles (guarded by _print_lock)

print(f"model={client.model}  bundles={len(bundles)}  k=3  workers={workers}  "
      f"checks/bundle={_CHECKS_PER_BUNDLE}  total_checks={_total_checks}\n", flush=True)


def _emit(msg):
    with _print_lock:
        print(msg, flush=True)


def _outcome(r):
    if r.needs_human:
        return "ESCALATED"
    if r.passed:
        return "pass"
    return "REJECT-FIRE" if r.severity == Severity.REJECT else "warn-fire"


def score_bundle(b):
    """Drive every check for one bundle, emitting a start/done line per check.
    Own bundle / own AgentContext / own results — no shared mutable state, so
    many run concurrently against the shared stateless client."""
    global _done_count
    bdir = Path(b)
    name = bdir.name[-3:]
    bundle = load_bundle(bdir)
    ctx = AgentContext(bundle_dir=bdir)
    results = []

    for c in SEMANTIC_CHECKS:
        _emit(f"  [{name}] {c.id:4} start   ({c.name})")
        t0 = time.time()
        r = run_check(c, bundle, client, ctx, k=3)
        results.append(r)
        with _print_lock:
            _done_count += 1
            n = _done_count
        _emit(f"  [{name}] {r.id:4} DONE  {_outcome(r):11} {time.time()-t0:5.1f}s"
              f"   [{n}/{_total_checks} checks]")

    for fn in DETERMINISTIC_CHECKS:  # Q09/Q12: no gateway call, instant
        r = fn(bundle)
        results.append(r)
        with _print_lock:
            _done_count += 1
            n = _done_count
        _emit(f"  [{name}] {r.id:4} DONE  {_outcome(r):11}   0.0s (det)"
              f"   [{n}/{_total_checks} checks]")

    verdict = compute_verdict(results)
    rejects = [r.id for r in results if not r.passed and r.severity == Severity.REJECT and not r.needs_human]
    warns = [r.id for r in results if not r.passed and r.severity == Severity.WARN]
    humans = [r.id for r in results if r.needs_human]

    fire_lines = []
    for r in results:
        if r.passed:
            continue
        if r.severity == Severity.REJECT and not r.needs_human:
            fire_lines.append(f"    !! {r.id} REJECT-fired on clean {name}: {r.detail} | {r.evidence[:3]}")
        elif r.needs_human:
            fire_lines.append(f"    ~~ {r.id} escalated (disputed) on clean {name}: {r.detail} | {r.evidence[:3]}")

    _emit(f"[BUNDLE DONE] {name} -> {verdict.value}  "
          f"R={','.join(rejects) or '-'} W={','.join(warns) or '-'} H={','.join(humans) or '-'}")
    return {"name": name, "verdict": verdict, "rejects": rejects,
            "warns": warns, "humans": humans, "fire_lines": fire_lines}


scored = [None] * len(bundles)
with ThreadPoolExecutor(max_workers=workers) as pool:
    fut_to_idx = {pool.submit(score_bundle, b): i for i, b in enumerate(bundles)}
    for fut in as_completed(fut_to_idx):
        i = fut_to_idx[fut]
        try:
            scored[i] = fut.result()
        except Exception as exc:  # never let one bundle sink the whole sweep
            scored[i] = {"name": Path(bundles[i]).name[-3:], "verdict": None, "rejects": [],
                         "warns": [], "humans": [], "fire_lines": [f"    ERROR: {exc!r}"]}

print(flush=True)
reject_ff = warn_fires = human_esc = 0
hdr = f"{'bundle':10} {'verdict':22} {'REJECT-fires':28} {'WARN-fires':22} {'needs_human':18}"
print(hdr)
print("-" * len(hdr))
for s in scored:
    reject_ff += len(s["rejects"])
    warn_fires += len(s["warns"])
    human_esc += len(s["humans"])
    vlabel = s["verdict"].value if s["verdict"] is not None else "ERROR"
    print(f"{s['name']:10} {vlabel:22} {(','.join(s['rejects']) or '-'):28} "
          f"{(','.join(s['warns']) or '-'):22} {(','.join(s['humans']) or '-'):18}")
    for line in s["fire_lines"]:
        print(line)

print("\n" + "=" * 60)
print(f"REJECT false-fires (KEY, want 0): {reject_ff}")
print(f"WARN fires (expected, e.g. Q08 redundant negatives): {warn_fires}")
print(f"needs_human escalations (disputed reject / warn->human): {human_esc}")
print("A clean 'good set' should show REJECT false-fires = 0; WARN fires are")
print("expected and correct on the internal bundles (redundant negatives etc.).")

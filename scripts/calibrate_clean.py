"""Clean-bundle false-reject calibration across the FULL 11-check set.

Runs the shipped semantic engine (run_semantic -> all 11 checks + verdict) live
against each internal "good set" bundle and reports, per bundle, which checks
FIRED (passed=False) — split into:

  * REJECT false-fires  -> the real danger: a clean rubric hard-flagged NOT_SOUND
  * WARN fires          -> expected/acceptable (e.g. Q08 redundant negatives are
                           pervasive in the internal set per manual QC; these are
                           TRUE fires, not false-rejects)

The internal bundles are clean ONLY w.r.t. reject checks; they legitimately
carry Q08/Q12 warns. So the headline number is REJECT_FALSE_FIRES == 0.

Bundles run concurrently over a bounded thread pool (the engine is unchanged
and fully synchronous; the calls are I/O-bound on the gateway, so threads win
here). Tune width with AUTOQC_CALIB_WORKERS (default 6). Per-bundle output is
buffered and printed in submission order after all bundles finish, so the
table never interleaves.

Usage: python3 scripts/calibrate_clean.py <bundle_dir> [<bundle_dir> ...]
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
for _line in open(".env"):
    _line = _line.strip()
    if "=" in _line and not _line.startswith("#"):
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k, _v)

from autoqc.bundle import load_bundle
from autoqc.agent.engine import run_semantic
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
print(f"model={client.model}  bundles={len(bundles)}  k=3  workers={workers}  "
      f"(all 11 semantic checks)\n")


def score_bundle(b):
    """Run the full check set on one bundle. Pure per-bundle work: own bundle,
    own AgentContext, own results list — no shared mutable state, so it is safe
    to run many of these concurrently against the shared stateless client."""
    bdir = Path(b)
    name = bdir.name[-3:]
    bundle = load_bundle(bdir)
    ctx = AgentContext(bundle_dir=bdir)
    results = run_semantic(bundle, client, ctx, k=3)
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

    return {"name": name, "verdict": verdict, "rejects": rejects,
            "warns": warns, "humans": humans, "fire_lines": fire_lines}


# executor.map preserves input order, so the printed table stays in bundle order.
with ThreadPoolExecutor(max_workers=workers) as pool:
    scored = list(pool.map(score_bundle, bundles))

reject_ff = warn_fires = human_esc = 0
hdr = f"{'bundle':10} {'verdict':22} {'REJECT-fires':28} {'WARN-fires':22} {'needs_human':18}"
print(hdr)
print("-" * len(hdr))
for s in scored:
    reject_ff += len(s["rejects"])
    warn_fires += len(s["warns"])
    human_esc += len(s["humans"])
    print(f"{s['name']:10} {s['verdict'].value:22} {(','.join(s['rejects']) or '-'):28} "
          f"{(','.join(s['warns']) or '-'):22} {(','.join(s['humans']) or '-'):18}")
    for line in s["fire_lines"]:
        print(line)

print("\n" + "=" * 60)
print(f"REJECT false-fires (KEY, want 0): {reject_ff}")
print(f"WARN fires (expected, e.g. Q08 redundant negatives): {warn_fires}")
print(f"needs_human escalations (disputed reject / warn->human): {human_esc}")
print("A clean 'good set' should show REJECT false-fires = 0; WARN fires are")
print("expected and correct on the internal bundles (redundant negatives etc.).")

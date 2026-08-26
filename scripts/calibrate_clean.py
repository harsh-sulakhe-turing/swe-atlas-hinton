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

Usage: python3 scripts/calibrate_clean.py <bundle_dir> [<bundle_dir> ...]
"""
import os
import sys
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

client = GatewayLLMClient()
assert client.available(), ".env missing EVAL_API_KEY / EVAL_BASE_URL"
print(f"model={client.model}  bundles={len(bundles)}  k=3  (all 11 semantic checks)\n")

reject_ff = 0   # KEY danger metric: reject checks firing on a clean bundle
warn_fires = 0  # expected / acceptable
human_esc = 0

hdr = f"{'bundle':10} {'verdict':22} {'REJECT-fires':28} {'WARN-fires':22} {'needs_human':18}"
print(hdr)
print("-" * len(hdr))

for b in bundles:
    bdir = Path(b)
    name = bdir.name[-3:]
    bundle = load_bundle(bdir)
    ctx = AgentContext(bundle_dir=bdir)
    results = run_semantic(bundle, client, ctx, k=3)
    verdict = compute_verdict(results)

    rejects = [r.id for r in results if not r.passed and r.severity == Severity.REJECT and not r.needs_human]
    warns = [r.id for r in results if not r.passed and r.severity == Severity.WARN]
    humans = [r.id for r in results if r.needs_human]

    reject_ff += len(rejects)
    warn_fires += len(warns)
    human_esc += len(humans)

    print(f"{name:10} {verdict.value:22} {(','.join(rejects) or '-'):28} "
          f"{(','.join(warns) or '-'):22} {(','.join(humans) or '-'):18}")

    # surface evidence for any reject false-fire so it can be triaged immediately
    for r in results:
        if not r.passed and r.severity == Severity.REJECT and not r.needs_human:
            print(f"    !! {r.id} REJECT-fired on clean {name}: {r.detail} | {r.evidence[:2]}")

print("\n" + "=" * 60)
print(f"REJECT false-fires (KEY, want 0): {reject_ff}")
print(f"WARN fires (expected, e.g. Q08 redundant negatives): {warn_fires}")
print(f"needs_human escalations (disputed reject / warn->human): {human_esc}")
print("A clean 'good set' should show REJECT false-fires = 0; WARN fires are")
print("expected and correct on the internal bundles (redundant negatives etc.).")

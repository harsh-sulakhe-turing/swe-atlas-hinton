"""Live calibration: build labeled corpora from base bundles (clean + seeded
Q07/Q03 variants), run AutoQC against the real gateway, print per-case scores +
summary metrics (recall, false_reject_rate, detection_accuracy). Operator-run.

Usage: python3 scripts/calibrate.py <base_bundle_dir> [<base_bundle_dir> ...]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
for line in open(".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)

from autoqc.calibrate import build_corpus, run_corpus, summarize
from autoqc.llm import GatewayLLMClient

bases = sys.argv[1:]
if not bases:
    print("usage: python3 scripts/calibrate.py <base_bundle_dir> [...]", file=sys.stderr)
    raise SystemExit(64)

client = GatewayLLMClient()
assert client.available(), ".env missing EVAL_API_KEY / EVAL_BASE_URL"
work = Path(os.environ.get("AUTOQC_CALIB_DIR",
           "/private/tmp/claude-502/-Users-harsh-sulakhe-git-swe-atlas-hinton/"
           "c42695a5-efac-4259-a578-04a4e2da781a/scratchpad/autoqc_calib"))

cases = []
for i, b in enumerate(bases):
    cases += build_corpus(Path(b), work / f"bundle{i}")
print(f"model={client.model}  bases={len(bases)}  cases={len(cases)}")

scored = run_corpus(cases, client, work / "out", k=3)

print(f"\n{'case':10} {'verdict':20} {'Q07':>7} {'Q03':>7}")
for s in scored:
    q07 = "OK" if s["checks"]["Q07"]["correct"] else "WRONG"
    q03 = "OK" if s["checks"]["Q03"]["correct"] else "WRONG"
    print(f"{s['name']:10} {s.get('verdict', '?'):20} {q07:>7} {q03:>7}")

# I2 guard: recall/false-reject are only trustworthy if the clean cases pass both
# checks (Q03 scores over the whole rubric, so a dirty clean case makes seeded
# attribution ambiguous).
dirty_clean = [s for s in scored if s["name"] == "clean"
               and any(c["got_fail"] for c in s["checks"].values())]
if dirty_clean:
    print("\n*** WARNING: a clean case failed a check -> seeded recall/false-reject "
          "numbers are UNRELIABLE (attribution ambiguous). Investigate before trusting. ***")

m = summarize(scored)
print(f"\nrecall={m['recall']}  false_reject_rate={m['false_reject_rate']}  "
      f"detection_accuracy={m['detection_accuracy']}  n_cases={m['n_cases']}")
print("(recall = seeded defects caught; false_reject_rate = clean checks wrongly "
      "failed; detection_accuracy = reject-check flipped as expected. Real per-case "
      "verdict is the 'verdict' column above.)")

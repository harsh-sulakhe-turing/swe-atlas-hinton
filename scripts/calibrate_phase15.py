"""Score the Phase 1.5 checks: recall on Batch-1 negatives, reject-false-fires on the
clean qc-sample-pack. Offline summarizer is unit-tested; live runs are manual."""
from __future__ import annotations
import sys
from pathlib import Path

PHASE15_IDS = {"P01","P02","P03","P04","A01","A02","A03","A04","A05","A06","AL01","H01","Q13"}

def _hard_reject_ids(record) -> set[str]:
    out = set()
    for r in record.get("results", []):
        if (r.get("id") in PHASE15_IDS and r.get("severity") == "reject"
                and not r.get("passed") and not r.get("needs_human")):
            out.add(r["id"])
    return out

def summarize_phase15(negatives: list[dict], cleans: list[dict]) -> dict:
    recalled = sum(1 for rec in negatives if _hard_reject_ids(rec))
    recall = (recalled / len(negatives)) if negatives else 0.0
    false_fires = sum(len(_hard_reject_ids(rec)) for rec in cleans)
    fires: dict[str, int] = {}
    for rec in negatives + cleans:
        for cid in _hard_reject_ids(rec):
            fires[cid] = fires.get(cid, 0) + 1
    return {"recall": recall, "recalled": recalled, "n_negatives": len(negatives),
            "reject_false_fires": false_fires, "n_cleans": len(cleans), "per_check_fires": fires}

def _run_dir(bundle_dirs: list[str]) -> list[dict]:
    import json
    from autoqc.cli import run
    from autoqc.model import Verdict
    out = []
    for i, d in enumerate(bundle_dirs):
        od = Path("phase15_out") / f"b{i}"
        run(Path(d), od)  # writes review_record.json
        out.append(json.loads((od / "review_record.json").read_text()))
    return out

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: calibrate_phase15.py <neg_dir>... -- <clean_dir>...", file=sys.stderr)
        return 64
    cut = argv.index("--")
    negs, cleans = _run_dir(argv[:cut]), _run_dir(argv[cut + 1:])
    s = summarize_phase15(negs, cleans)
    print(f"recall={s['recall']:.2f} ({s['recalled']}/{s['n_negatives']})  "
          f"reject_false_fires={s['reject_false_fires']} over {s['n_cleans']} clean")
    print("per-check fires:", s["per_check_fires"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

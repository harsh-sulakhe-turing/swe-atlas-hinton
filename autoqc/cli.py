from __future__ import annotations
import json
import sys
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.structural import run_structural
from autoqc.verdict import compute_verdict
from autoqc.report import to_record, to_markdown
from autoqc.model import Verdict

_EXIT = {Verdict.SOUND: 0, Verdict.NEEDS_HUMAN_REVIEW: 1, Verdict.NOT_SOUND: 2}


def run(bundle_dir, out_dir) -> Verdict:
    bundle = load_bundle(Path(bundle_dir))
    results = run_structural(bundle)
    verdict = compute_verdict(results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_record.json").write_text(
        json.dumps(to_record(bundle, results, verdict), indent=2))
    (out / "report.md").write_text(to_markdown(bundle, results, verdict))
    return verdict


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 1:
        print("usage: autoqc <bundle_dir> [out_dir]", file=sys.stderr)
        return 64
    bundle_dir = argv[0]
    out_dir = argv[1] if len(argv) > 1 else "autoqc_out"
    verdict = run(bundle_dir, out_dir)
    print(f"verdict: {verdict.value}  (reports in {out_dir})")
    return _EXIT[verdict]


if __name__ == "__main__":
    raise SystemExit(main())

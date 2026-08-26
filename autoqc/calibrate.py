from __future__ import annotations
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from autoqc.seed import seed_bad_negative, seed_wildcard
from autoqc.cli import run


@dataclass
class Case:
    name: str
    bundle_dir: Path
    expected_flags: dict = field(default_factory=dict)  # check_id -> set(criterion_ids)

    @property
    def expected_not_sound(self) -> bool:
        return bool(self.expected_flags)


def _copy_bundle(base: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(base, dest)
    return dest


def _seed_into(bundle_dir: Path, seed_fn):
    rf = bundle_dir / "tests" / "rubrics.json"
    items = json.loads(rf.read_text())
    mutated, changed = seed_fn(items)
    rf.write_text(json.dumps(mutated, indent=2))
    return changed


def build_corpus(base_bundle_dir, work_dir) -> list[Case]:
    base = Path(base_bundle_dir)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    clean = _copy_bundle(base, work / "clean")
    q07 = _copy_bundle(base, work / "q07_bad")
    q07_id = _seed_into(q07, seed_bad_negative)
    q03 = _copy_bundle(base, work / "q03_bad")
    q03_id = _seed_into(q03, seed_wildcard)

    if q07_id is None or q03_id is None:
        raise ValueError(
            "build_corpus needs a base bundle with at least one negative AND one "
            f"positive criterion to seed (q07_id={q07_id}, q03_id={q03_id})")

    return [
        Case("clean", clean, {}),
        Case("q07_bad", q07, {"Q07": {q07_id}}),
        Case("q03_bad", q03, {"Q03": {q03_id}}),
    ]


CHECKS = ("Q07", "Q03")


def score_case(case, results) -> dict:
    by = {r.id: r for r in results}
    checks = {}
    any_expected_fail = bool(case.expected_flags)
    any_got_fail = False
    for cid in CHECKS:
        expected_fail = cid in case.expected_flags
        got_fail = (cid in by) and (by[cid].passed is False)
        any_got_fail = any_got_fail or got_fail
        checks[cid] = {"expected_fail": expected_fail, "got_fail": got_fail,
                       "correct": expected_fail == got_fail}
    return {"name": case.name, "checks": checks,
            "verdict_correct": any_got_fail == any_expected_fail}


def summarize(scored) -> dict:
    exp_total = exp_caught = clean_total = clean_failed = 0
    for s in scored:
        for cid, c in s["checks"].items():
            if c["expected_fail"]:
                exp_total += 1
                exp_caught += 1 if c["got_fail"] else 0
            else:
                clean_total += 1
                clean_failed += 1 if c["got_fail"] else 0
    vc = sum(1 for s in scored if s["verdict_correct"])
    return {
        "recall": (exp_caught / exp_total) if exp_total else None,
        "false_reject_rate": (clean_failed / clean_total) if clean_total else None,
        "detection_accuracy": (vc / len(scored)) if scored else None,
        "n_cases": len(scored),
    }


def run_corpus(cases, client, out_root, k=3) -> list[dict]:
    from autoqc.model import CheckResult, Stage, Severity
    out_root = Path(out_root)
    scored = []
    for case in cases:
        verdict = run(case.bundle_dir, out_root / case.name, llm=client, k=k)
        rec = json.loads((out_root / case.name / "review_record.json").read_text())
        results = [CheckResult(id=r["id"], name=r["name"],
                               stage=Stage(r["stage"]), severity=Severity(r["severity"]),
                               passed=r["passed"], needs_human=r["needs_human"])
                   for r in rec["results"] if r["stage"] == "semantic"]
        s = score_case(case, results)
        s["verdict"] = verdict.value
        scored.append(s)
    return scored

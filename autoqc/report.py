from __future__ import annotations
from autoqc.bundle import Bundle
from autoqc.model import CheckResult, Verdict


def _result_dict(r: CheckResult) -> dict:
    return {
        "id": r.id, "name": r.name,
        "stage": r.stage.value, "severity": r.severity.value,
        "passed": r.passed, "needs_human": r.needs_human,
        "evidence": list(r.evidence), "detail": r.detail,
    }


def to_record(bundle: Bundle, results: list[CheckResult],
              verdict: Verdict, guideline_version: str = "1.0.0") -> dict:
    return {
        "guideline_version": guideline_version,
        "task": {
            "name": bundle.root.name,
            "repository": bundle.repository,
            "base_commit": bundle.base_commit,
        },
        "verdict": verdict.value,
        "results": [_result_dict(r) for r in results],
    }

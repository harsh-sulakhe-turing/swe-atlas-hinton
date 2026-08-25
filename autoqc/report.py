from __future__ import annotations
from autoqc.bundle import Bundle
from autoqc.model import CheckResult, Severity, Verdict


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


def to_markdown(bundle: Bundle, results: list[CheckResult], verdict: Verdict) -> str:
    lines = [f"# AutoQC report — {bundle.root.name}",
             "", f"**Verdict:** `{verdict.value}`", ""]
    rejects = [r for r in results if r.severity is Severity.REJECT and not r.passed]
    warns = [r for r in results if r.severity is Severity.WARN and not r.passed]

    if rejects:
        lines.append("## Must fix (reject)")
        for r in rejects:
            lines.append(f"- **{r.id} {r.name}** — {r.detail}")
            for e in r.evidence:
                lines.append(f"  - evidence: {e}")
        lines.append("")
    if warns:
        lines.append("## Review these (warn)")
        for r in warns:
            lines.append(f"- **{r.id} {r.name}** — {r.detail}")
        lines.append("")
    if not rejects and not warns:
        lines.append("No issues found.")
    return "\n".join(lines)

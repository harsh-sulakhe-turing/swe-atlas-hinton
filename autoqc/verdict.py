from __future__ import annotations
from autoqc.model import CheckResult, Severity, Verdict


def compute_verdict(results: list[CheckResult]) -> Verdict:
    # An *undisputed* reject-severity failure is a hard rejection. A *disputed*
    # one (ensemble split or adversary overturn set needs_human) escalates to a
    # human instead — otherwise the adversary's defend side would be dead code.
    if any(r.severity is Severity.REJECT and not r.passed and not r.needs_human
           for r in results):
        return Verdict.NOT_SOUND
    warn_fail = any(r.severity is Severity.WARN and not r.passed for r in results)
    needs_human = any(r.needs_human for r in results)
    if warn_fail or needs_human:
        return Verdict.NEEDS_HUMAN_REVIEW
    return Verdict.SOUND

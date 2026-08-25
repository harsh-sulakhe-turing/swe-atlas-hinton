from autoqc.model import CheckResult, Stage, Severity, Verdict
from autoqc.verdict import compute_verdict


def _r(sev, passed, needs_human=False):
    return CheckResult(id="x", name="n", stage=Stage.STRUCTURAL,
                       severity=sev, passed=passed, needs_human=needs_human)


def test_reject_failure_is_not_sound():
    results = [_r(Severity.REJECT, False), _r(Severity.WARN, True)]
    assert compute_verdict(results) is Verdict.NOT_SOUND


def test_warn_failure_is_needs_human():
    results = [_r(Severity.REJECT, True), _r(Severity.WARN, False)]
    assert compute_verdict(results) is Verdict.NEEDS_HUMAN_REVIEW


def test_needs_human_flag_routes_to_human():
    results = [_r(Severity.REJECT, True, needs_human=True)]
    assert compute_verdict(results) is Verdict.NEEDS_HUMAN_REVIEW


def test_all_pass_is_sound():
    results = [_r(Severity.REJECT, True), _r(Severity.WARN, True)]
    assert compute_verdict(results) is Verdict.SOUND


def test_reject_beats_warn():
    results = [_r(Severity.REJECT, False), _r(Severity.WARN, False)]
    assert compute_verdict(results) is Verdict.NOT_SOUND


def test_disputed_reject_routes_to_human():
    # A reject-severity check that failed but is flagged for human review
    # (ensemble split or adversary overturn) must escalate, not auto-reject.
    results = [_r(Severity.REJECT, False, needs_human=True)]
    assert compute_verdict(results) is Verdict.NEEDS_HUMAN_REVIEW

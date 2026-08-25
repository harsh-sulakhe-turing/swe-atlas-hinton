from autoqc.model import Severity, Stage, Verdict, CheckResult, ReviewRecord


def test_checkresult_defaults():
    r = CheckResult(id="S01", name="Parses as JSON array",
                    stage=Stage.STRUCTURAL, severity=Severity.REJECT, passed=True)
    assert r.evidence == []
    assert r.needs_human is False
    assert r.detail == ""


def test_enum_values_are_report_strings():
    assert Severity.REJECT.value == "reject"
    assert Verdict.NEEDS_HUMAN_REVIEW.value == "needs_human_review"
    assert Stage.FACTUAL.value == "factual"


def test_review_record_holds_results():
    r = CheckResult(id="S05", name="Has a positive", stage=Stage.STRUCTURAL,
                    severity=Severity.REJECT, passed=False, detail="no positive criterion")
    rec = ReviewRecord(task_name="t", guideline_version="1.0.0",
                       results=[r], verdict=Verdict.NOT_SOUND)
    assert rec.results[0].passed is False
    assert rec.verdict is Verdict.NOT_SOUND

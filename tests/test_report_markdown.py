# tests/test_report_markdown.py
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.model import CheckResult, Stage, Severity, Verdict
from autoqc.report import to_markdown


def _min_bundle(tmp_path: Path):
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/rubrics.json").write_text("[]")
    (tmp_path / "task.toml").write_text('[metadata]\nrepository="o/r"\n')
    return load_bundle(tmp_path)


def test_markdown_lists_rejects_first_then_warns(tmp_path):
    b = _min_bundle(tmp_path)
    results = [
        CheckResult(id="S07", name="Type vocabulary", stage=Stage.STRUCTURAL,
                    severity=Severity.WARN, passed=False, detail="odd type"),
        CheckResult(id="S01", name="Parses as JSON array", stage=Stage.STRUCTURAL,
                    severity=Severity.REJECT, passed=False, detail="bad json",
                    evidence=["line 1"]),
        CheckResult(id="S05", name="Has a positive", stage=Stage.STRUCTURAL,
                    severity=Severity.REJECT, passed=True),
    ]
    md = to_markdown(b, results, Verdict.NOT_SOUND)
    assert "not_sound" in md
    # reject appears before warn
    assert md.index("S01") < md.index("S07")
    assert "bad json" in md and "line 1" in md
    # passing check not shown
    assert "S05" not in md


def test_markdown_sound_has_no_issues_section(tmp_path):
    b = _min_bundle(tmp_path)
    results = [CheckResult(id="S01", name="Parses as JSON array",
                           stage=Stage.STRUCTURAL, severity=Severity.REJECT, passed=True)]
    md = to_markdown(b, results, Verdict.SOUND)
    assert "sound" in md
    assert "Must fix" not in md

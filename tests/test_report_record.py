import json
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.model import CheckResult, Stage, Severity, Verdict
from autoqc.report import to_record


def _min_bundle(tmp_path: Path):
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/rubrics.json").write_text("[]")
    (tmp_path / "task.toml").write_text(
        '[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))
    return load_bundle(tmp_path)


def test_record_is_json_serializable_and_shaped(tmp_path):
    b = _min_bundle(tmp_path)
    results = [CheckResult(id="S01", name="Parses as JSON array",
                           stage=Stage.STRUCTURAL, severity=Severity.REJECT,
                           passed=False, detail="bad json", evidence=["x"])]
    rec = to_record(b, results, Verdict.NOT_SOUND)
    s = json.dumps(rec)  # must not raise
    back = json.loads(s)
    assert back["verdict"] == "not_sound"
    assert back["task"]["repository"] == "o/r"
    assert back["results"][0]["severity"] == "reject"
    assert back["results"][0]["passed"] is False
    assert back["results"][0]["detail"] == "bad json"

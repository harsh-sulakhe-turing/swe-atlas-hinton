import json
from pathlib import Path
from autoqc.cli import run
from autoqc.model import Verdict


def _good(tmp_path: Path):
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/rubrics.json").write_text(json.dumps([
        {"id": "a" * 32, "title": "1.1: States X",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}},
        {"id": "b" * 32, "title": "2.1: Claims X is false",
         "annotations": {"type": "negative hli verifier", "importance": "must have"}},
    ]))
    (tmp_path / "tests/prompt.txt").write_text("q")
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution/answer.txt").write_text("a")
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment/Dockerfile").write_text("FROM x")
    (tmp_path / "task.toml").write_text(
        '[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))


def test_good_bundle_is_sound_and_writes_reports(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    _good(bundle)
    out = tmp_path / "out"
    verdict = run(bundle, out, factual=False)
    assert verdict is Verdict.SOUND
    rec = json.loads((out / "review_record.json").read_text())
    assert rec["verdict"] == "sound"
    assert (out / "report.md").read_text().startswith("# AutoQC report")


def test_unquoted_toml_date_base_commit_does_not_raise(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    _good(bundle)
    (bundle / "task.toml").write_text(
        '[metadata]\nrepository="o/r"\nbase_commit = 2021-01-01\n')
    out = tmp_path / "out"
    verdict = run(bundle, out, factual=False)
    assert verdict is Verdict.NOT_SOUND
    rec = json.loads((out / "review_record.json").read_text())
    assert rec["task"]["base_commit"] is None


def test_malformed_bundle_is_not_sound(tmp_path):
    bundle = tmp_path / "task"
    bundle.mkdir()
    (bundle / "tests").mkdir()
    (bundle / "tests/rubrics.json").write_text("{ not json ]")
    out = tmp_path / "out"
    verdict = run(bundle, out, factual=False)
    assert verdict is Verdict.NOT_SOUND
    assert "Must fix" in (out / "report.md").read_text()

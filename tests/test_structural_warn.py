import json
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.structural import run_structural
from autoqc.model import Severity


def _bundle(tmp_path: Path, rubrics):
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/rubrics.json").write_text(json.dumps(rubrics))
    (tmp_path / "tests/prompt.txt").write_text("q")
    (tmp_path / "solution").mkdir(exist_ok=True)
    (tmp_path / "solution/answer.txt").write_text("a")
    (tmp_path / "environment").mkdir(exist_ok=True)
    (tmp_path / "environment/Dockerfile").write_text("FROM x")
    (tmp_path / "task.toml").write_text(
        '[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))
    return load_bundle(tmp_path)


def _by_id(results):
    return {r.id: r for r in results}


def _item(idx, id_, typ):
    return {"id": id_, "title": f"{idx}: text",
            "annotations": {"type": typ, "importance": "must have"}}


def test_s07_bad_vocab_warns(tmp_path):
    bad = _item("1.1", "a" * 32, "positive verifier")  # wrong type string
    r = _by_id(run_structural(_bundle(tmp_path, [bad])))["S07"]
    assert r.passed is False and r.severity is Severity.WARN


def test_s08_numbering_gap_warns(tmp_path):
    items = [_item("1.1", "a" * 32, "positive hli verifier"),
             _item("1.3", "b" * 32, "positive hli verifier")]  # gap: no 1.2
    r = _by_id(run_structural(_bundle(tmp_path, items)))["S08"]
    assert r.passed is False and r.severity is Severity.WARN


def test_s07_s08_pass_on_clean(tmp_path):
    items = [_item("1.1", "a" * 32, "positive hli verifier"),
             _item("1.2", "b" * 32, "positive hli verifier"),
             _item("2.1", "c" * 32, "negative hli verifier")]
    res = _by_id(run_structural(_bundle(tmp_path, items)))
    assert res["S07"].passed is True and res["S08"].passed is True
    assert len(run_structural(_bundle(tmp_path, items))) == 8

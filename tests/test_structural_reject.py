# tests/test_structural_reject.py
import json
from pathlib import Path
from autoqc.bundle import load_bundle
from autoqc.structural import run_structural
from autoqc.model import Severity


def _bundle(tmp_path: Path, rubrics, with_files=True):
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/rubrics.json").write_text(
        rubrics if isinstance(rubrics, str) else json.dumps(rubrics))
    if with_files:
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


def _positive(idx="1.1"):
    return {"id": "a" * 32, "title": f"{idx}: States X",
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}


def test_s01_invalid_json_fails(tmp_path):
    b = _bundle(tmp_path, "{ not json ]")
    r = _by_id(run_structural(b))["S01"]
    assert r.passed is False and r.severity is Severity.REJECT


def test_s01_non_array_fails(tmp_path):
    b = _bundle(tmp_path, {"not": "an array"})
    assert _by_id(run_structural(b))["S01"].passed is False


def test_s03_bad_id_fails(tmp_path):
    bad = _positive()
    bad["id"] = "XYZ"
    assert _by_id(run_structural(_bundle(tmp_path, [bad])))["S03"].passed is False


def test_s03_duplicate_id_fails(tmp_path):
    a, c = _positive("1.1"), _positive("1.2")  # same id "a"*32
    assert _by_id(run_structural(_bundle(tmp_path, [a, c])))["S03"].passed is False


def test_s04_type_number_mismatch_fails(tmp_path):
    item = _positive("2.1")  # negative number, positive type
    assert _by_id(run_structural(_bundle(tmp_path, [item])))["S04"].passed is False


def test_s05_no_positive_fails(tmp_path):
    neg = {"id": "b" * 32, "title": "2.1: Claims wrongly",
           "annotations": {"type": "negative hli verifier", "importance": "must have"}}
    assert _by_id(run_structural(_bundle(tmp_path, [neg])))["S05"].passed is False


def test_s06_missing_dockerfile_fails(tmp_path):
    b = _bundle(tmp_path, [_positive()], with_files=True)
    (tmp_path / "environment/Dockerfile").unlink()
    b = load_bundle(tmp_path)
    assert _by_id(run_structural(b))["S06"].passed is False


def test_all_pass_on_good_bundle(tmp_path):
    good = [_positive("1.1"),
            {"id": "b" * 32, "title": "2.1: Claims X is false",
             "annotations": {"type": "negative hli verifier", "importance": "must have"}}]
    results = _by_id(run_structural(_bundle(tmp_path, good)))
    for sid in ["S01", "S02", "S03", "S04", "S05", "S06"]:
        assert results[sid].passed is True, sid

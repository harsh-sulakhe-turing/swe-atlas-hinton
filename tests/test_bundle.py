import json
from pathlib import Path
from autoqc.bundle import load_bundle


def _write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _good_bundle(root: Path):
    _write(root, "tests/prompt.txt", "How does X work?")
    _write(root, "tests/rubrics.json", json.dumps([
        {"id": "a" * 32, "title": "1.1: States X",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}}
    ]))
    _write(root, "solution/answer.txt", "X works via Y.")
    _write(root, "environment/Dockerfile", "FROM python:3.11")
    _write(root, "task.toml",
           '[metadata]\nrepository = "org/repo"\nbase_commit = "%s"\n' % ("a" * 40))


def test_loads_good_bundle(tmp_path):
    _good_bundle(tmp_path)
    b = load_bundle(tmp_path)
    assert b.rubrics_error is None
    assert isinstance(b.rubrics, list)
    assert b.repository == "org/repo"
    assert b.base_commit == "a" * 40
    assert b.prompt.startswith("How does")
    assert all(b.files_present.values())


def test_invalid_json_is_recorded_not_raised(tmp_path):
    _good_bundle(tmp_path)
    (tmp_path / "tests/rubrics.json").write_text("{ not json ]")
    b = load_bundle(tmp_path)
    assert b.rubrics is None
    assert b.rubrics_error is not None


def test_missing_files_flagged(tmp_path):
    # only prompt exists
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/prompt.txt").write_text("q")
    b = load_bundle(tmp_path)
    assert b.files_present["tests/prompt.txt"] is True
    assert b.files_present["environment/Dockerfile"] is False
    assert b.task_toml is None


def test_malformed_toml_metadata_not_dict(tmp_path):
    """Test that non-dict metadata in task.toml does not raise AttributeError."""
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/prompt.txt").write_text("q")
    # Write task.toml where metadata is a scalar, not a table
    (tmp_path / "task.toml").write_text('metadata = 5\n')
    b = load_bundle(tmp_path)
    # Should not raise, and repository/base_commit should be None
    assert b.repository is None
    assert b.base_commit is None
    # task_toml should still be parsed successfully
    assert isinstance(b.task_toml, dict)


def test_non_utf8_rubrics_not_raised(tmp_path):
    """Test that non-UTF-8 bytes in rubrics.json do not raise UnicodeDecodeError."""
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/prompt.txt").write_text("q")
    # Write non-UTF-8 bytes to rubrics.json
    (tmp_path / "tests/rubrics.json").write_bytes(b"\xff\xfe\x00")
    b = load_bundle(tmp_path)
    # Should not raise
    assert b.rubrics is None
    assert b.rubrics_error is not None

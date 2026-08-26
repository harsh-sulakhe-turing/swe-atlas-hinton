from pathlib import Path
from autoqc.agent.tools import (Tool, AgentContext, read_bundle_file, list_dir,
                                SUBMIT_FINDINGS, default_tools, validate_findings, CHECK_IDS)


def _bundle(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/rubrics.json").write_text("[]")
    (tmp_path / "tests/prompt.txt").write_text("the question")
    return AgentContext(bundle_dir=tmp_path)


def test_read_bundle_file_reads_whitelisted(tmp_path):
    ctx = _bundle(tmp_path)
    out = read_bundle_file.run({"path": "tests/prompt.txt"}, ctx)
    assert "the question" in out


def test_read_bundle_file_rejects_non_whitelisted(tmp_path):
    ctx = _bundle(tmp_path)
    (tmp_path / "secret.txt").write_text("nope")
    out = read_bundle_file.run({"path": "secret.txt"}, ctx)
    assert "not readable" in out.lower() or "not allowed" in out.lower()


def test_read_bundle_file_missing_is_error_not_raise(tmp_path):
    ctx = _bundle(tmp_path)
    out = read_bundle_file.run({"path": "solution/answer.txt"}, ctx)  # whitelisted but absent
    assert "error" in out.lower() or "not found" in out.lower()


def test_list_dir(tmp_path):
    ctx = _bundle(tmp_path)
    out = list_dir.run({"path": "tests"}, ctx)
    assert "rubrics.json" in out


def test_tool_schema_shape():
    s = read_bundle_file.schema()
    assert s["type"] == "function"
    assert s["function"]["name"] == "read_bundle_file"
    assert "path" in s["function"]["parameters"]["properties"]


def test_submit_findings_is_terminal():
    assert SUBMIT_FINDINGS.terminal is True
    assert SUBMIT_FINDINGS in default_tools()


def test_validate_findings_accepts_good():
    good = [{"check_id": "Q07", "criterion_id": "b" * 32, "passed": False,
             "evidence": ["phrased as 'Does not claim'"], "reason": "r"}]
    valid, problems = validate_findings(good, allowed_criterion_ids={"b" * 32})
    assert len(valid) == 1 and problems == []


def test_validate_findings_flags_bad():
    bad = [
        {"check_id": "Q99", "criterion_id": "b" * 32, "passed": False, "evidence": ["x"]},   # bad check
        {"check_id": "Q07", "criterion_id": "zzz", "passed": False, "evidence": ["x"]},       # unknown criterion
        {"check_id": "Q07", "criterion_id": "rubric", "passed": "no", "evidence": ["x"]},    # passed not bool
        {"check_id": "Q07", "criterion_id": "rubric", "passed": True, "evidence": []},        # empty evidence
    ]
    valid, problems = validate_findings(bad, allowed_criterion_ids={"b" * 32})
    assert valid == [] and len(problems) == 4


def test_read_bundle_file_non_utf8_does_not_raise(tmp_path):
    ctx = _bundle(tmp_path)
    (tmp_path / "task.toml").write_bytes(b"\xff\xfe\x00bad")   # task.toml is whitelisted
    out = read_bundle_file.run({"path": "task.toml"}, ctx)     # must NOT raise
    assert "error" in out.lower()

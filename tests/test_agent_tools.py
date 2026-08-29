from pathlib import Path
from autoqc.agent.tools import (Tool, AgentContext, read_bundle_file, list_dir,
                                SUBMIT_FINDINGS, default_tools, validate_findings, CHECK_IDS,
                                guard_command, run_bash, factual_tools, text_tools)


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


def test_list_dir_rejects_outside_bundle(tmp_path):
    ctx = _bundle(tmp_path)
    out = list_dir.run({"path": "../.."}, ctx)
    assert "outside the bundle" in out.lower()


def test_validate_findings_rejects_empty_or_nonstring_evidence():
    bad = [
        {"check_id":"Q07","criterion_id":"rubric","passed":True,"evidence":[""]},
        {"check_id":"Q07","criterion_id":"rubric","passed":True,"evidence":[123]},
    ]
    valid, problems = validate_findings(bad, allowed_criterion_ids=set())
    assert valid == [] and len(problems) == 2


class _FakeContainer:
    def __init__(self): self.ran = []
    def exec(self, cmd, cwd="/testbed", timeout=None):
        self.ran.append(cmd); return f"OUT:{cmd}"


def test_guard_blocks_network_installs_and_root_writes():
    for bad in ["curl http://x", "wget x", "sudo rm -rf /", "apt-get install foo",
                "pip install bar", "go get x", "echo hi > /etc/passwd", "rm -rf /"]:
        assert guard_command(bad) is not None, bad


def test_guard_allows_inspection_and_scratch_writes():
    for ok in ["cat go.mod", "grep -rn Retry ./pkg", "git log -1", "go build ./...",
               "go test ./pkg/foo/...", "echo hi > /scratch/x"]:
        assert guard_command(ok) is None, ok


def test_run_bash_execs_in_container():
    c = _FakeContainer()
    out = run_bash.run({"cmd": "cat go.mod"}, AgentContext(bundle_dir=".", container=c))
    assert out == "OUT:cat go.mod" and c.ran == ["cat go.mod"]


def test_run_bash_blocks_denied_command_before_exec():
    c = _FakeContainer()
    out = run_bash.run({"cmd": "curl http://evil"}, AgentContext(bundle_dir=".", container=c))
    assert "blocked" in out and c.ran == []


def test_run_bash_without_container_is_error_not_crash():
    out = run_bash.run({"cmd": "cat x"}, AgentContext(bundle_dir="."))
    assert out.startswith("error:")


def test_factual_tools_includes_run_bash_and_submit():
    names = {t.name for t in factual_tools()}
    assert {"run_bash", "submit_findings", "read_bundle_file"} <= names


def test_run_bash_wraps_container_exception():
    class Boom:
        def exec(self, cmd, cwd="/testbed", timeout=None): raise RuntimeError("kaboom")
    out = run_bash.run({"cmd": "cat go.mod"}, AgentContext(bundle_dir=".", container=Boom()))
    assert out.startswith("error:") and "kaboom" in out


def test_text_tools_is_submit_only():
    names = [t.name for t in text_tools()]
    assert names == ["submit_findings"]


# Phase 1.5 check ID and read allow-list tests
from autoqc.agent.tools import ALLOWED_READ

def test_check_ids_include_phase15():
    for cid in ("Q13", "P01", "P04", "A01", "A06", "AL01", "H01"):
        assert cid in CHECK_IDS

def test_validate_findings_accepts_answer_unit():
    valid, problems = validate_findings(
        [{"check_id": "A01", "criterion_id": "answer", "passed": True, "evidence": ["ok"]}],
        {"answer"})
    assert len(valid) == 1 and problems == []

def test_instruction_is_readable():
    assert "instruction.md" in ALLOWED_READ

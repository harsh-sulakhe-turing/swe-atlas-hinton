from pathlib import Path
from autoqc.agent import engine as engine_mod
from autoqc.agent.engine import adjudicate_factual, run_factual, run_factual_stage, run_semantic
from autoqc.agent.container import ContainerError
from autoqc.agent.tools import AgentContext
from autoqc.llm import FakeLLMClient
from autoqc.model import Stage, Severity


def _f(cid, passed, ev):
    return {"check_id": "Q06", "criterion_id": cid, "passed": passed, "evidence": ev}


def test_adjudicate_agree_pass():
    a = adjudicate_factual([_f("1", True, ["a.go:1"])], [_f("1", True, ["a.go:9"])], {"1"})
    assert a["1"] == {"passed": True, "needs_human": False}


def test_adjudicate_agree_reject_same_file_is_hard_reject():
    a = adjudicate_factual([_f("1", False, ["a.go:1"])], [_f("1", False, ["a.go:40"])], {"1"})
    assert a["1"] == {"passed": False, "needs_human": False}


def test_adjudicate_reject_different_files_needs_human():
    a = adjudicate_factual([_f("1", False, ["a.go:1"])], [_f("1", False, ["b.go:1"])], {"1"})
    assert a["1"]["needs_human"] is True


def test_adjudicate_disagreement_needs_human():
    a = adjudicate_factual([_f("1", True, ["a.go:1"])], [_f("1", False, ["a.go:1"])], {"1"})
    assert a["1"]["needs_human"] is True


def test_adjudicate_missing_round_needs_human():
    a = adjudicate_factual([_f("1", True, ["a.go:1"])], [], {"1"})
    assert a["1"]["needs_human"] is True


def test_adjudicate_agree_reject_prose_evidence_same_file_is_hard_reject():
    a = adjudicate_factual(
        [_f("1", False, ["The bug is in pkg/foo.go:42 per the constant"])],
        [_f("1", False, ["see pkg/foo.go:9 also"])],
        {"1"},
    )
    assert a["1"] == {"passed": False, "needs_human": False}


def test_adjudicate_agree_reject_prose_evidence_different_files_needs_human():
    a = adjudicate_factual(
        [_f("1", False, ["The bug is in pkg/foo.go:42 per the constant"])],
        [_f("1", False, ["see pkg/bar.go:9 also"])],
        {"1"},
    )
    assert a["1"]["needs_human"] is True


def _bundle_with(criteria):
    class B:
        prompt = "p"; base_commit = "abc"; repository = "r"
        rubrics = criteria
    return B()


def _responder_all(passed, ev):
    def r(messages, tools):
        # single-turn submit for every criterion mentioned in the user context
        import re
        user = next(m["content"] for m in messages if m["role"] == "user")
        ids = re.findall(r"criterion_id=(\S+)", user)
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {
            "findings": [_f(i, passed, ev) for i in ids]}}]}
    return r


def test_run_factual_confirmed_reject(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    res = run_factual(b, FakeLLMClient(_responder_all(False, ["x.go:3"])), ctx)
    assert res.id == "Q06" and res.stage is Stage.FACTUAL and res.severity is Severity.REJECT
    assert res.passed is False and res.needs_human is False  # -> not_sound


def test_run_factual_all_pass(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    res = run_factual(b, FakeLLMClient(_responder_all(True, ["x.go:3"])), ctx)
    assert res.passed is True and res.needs_human is False


def test_run_factual_no_criteria_passes(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    res = run_factual(_bundle_with([]), FakeLLMClient(_responder_all(True, ["x:1"])), ctx)
    assert res.passed is True and res.needs_human is False


def test_run_factual_votes_log_records_both_rounds(tmp_path):
    ctx = AgentContext(bundle_dir=tmp_path, container=None)
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    log = []
    run_factual(b, FakeLLMClient(_responder_all(True, ["x.go:3"])), ctx, votes_log=log)
    assert len([e for e in log if e["role"] == "factual"]) == 2


def test_stage_degrades_to_needs_human_when_docker_down(tmp_path):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment/Dockerfile").write_text("FROM busybox\n")
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    b.root = tmp_path
    res = run_factual_stage(b, FakeLLMClient(_responder_all(True, ["x:1"])),
                            docker=lambda runner=None: False)
    assert res.id == "Q06" and res.needs_human is True and res.passed is False
    assert "docker" in res.detail.lower()


def test_run_semantic_skips_factual_when_disabled(tmp_path):
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    b.rubrics = [{"id": "1.1", "title": "t", "annotations": {"type": "positive"}}]
    ctx = AgentContext(bundle_dir=tmp_path)
    results = run_semantic(b, FakeLLMClient(_responder_all(True, ["x:1"])), ctx,
                           checks=[], k=1, factual=False)
    assert not any(r.id == "Q06" for r in results)


def test_stage_needs_human_when_no_dockerfile_in_bundle(tmp_path):
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    b.root = tmp_path
    b.files_present = {"environment/Dockerfile": False}
    # docker=True proves the stage short-circuits on the missing-Dockerfile check
    # rather than ever reaching the Docker gate.
    res = run_factual_stage(b, FakeLLMClient(_responder_all(True, ["x:1"])),
                            docker=lambda runner=None: True)
    assert res.id == "Q06" and res.needs_human is True and res.passed is False
    assert "dockerfile" in res.detail.lower()


class _FakeSessionStartFails:
    """No real Docker: ensure_image succeeds, start() raises ContainerError."""

    def __init__(self, bundle, limits=None):
        pass

    def ensure_image(self):
        return "tag"

    def start(self):
        raise ContainerError("start failed: boom")

    def stop(self):
        pass


def test_stage_never_crashes_when_container_start_raises(tmp_path, monkeypatch):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment/Dockerfile").write_text("FROM busybox\n")
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    b.root = tmp_path
    monkeypatch.setattr(engine_mod, "ContainerSession", _FakeSessionStartFails)
    res = run_factual_stage(b, FakeLLMClient(_responder_all(True, ["x:1"])),
                            docker=lambda runner=None: True)
    assert res.id == "Q06" and res.needs_human is True and res.passed is False


class _FakeSessionOK:
    """No real Docker: build/start succeed instantly; tracks whether stop() ran."""

    instances: list["_FakeSessionOK"] = []

    def __init__(self, bundle, limits=None):
        self.stopped = False
        _FakeSessionOK.instances.append(self)

    def ensure_image(self):
        return "tag"

    def start(self):
        return "name"

    def stop(self):
        self.stopped = True


def test_stage_never_crashes_when_factual_pass_raises_and_stops_session(tmp_path, monkeypatch):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment/Dockerfile").write_text("FROM busybox\n")
    b = _bundle_with([{"id": "1.1", "title": "t"}])
    b.root = tmp_path
    _FakeSessionOK.instances.clear()
    monkeypatch.setattr(engine_mod, "ContainerSession", _FakeSessionOK)

    def boom(*args, **kwargs):
        raise RuntimeError("factual pass exploded")

    monkeypatch.setattr(engine_mod, "run_factual", boom)
    res = run_factual_stage(b, FakeLLMClient(_responder_all(True, ["x:1"])),
                            docker=lambda runner=None: True)
    assert res.id == "Q06" and res.needs_human is True and res.passed is False
    assert len(_FakeSessionOK.instances) == 1
    assert _FakeSessionOK.instances[0].stopped is True

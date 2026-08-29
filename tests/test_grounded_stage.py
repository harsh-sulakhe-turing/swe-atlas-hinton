import types
from pathlib import Path

from autoqc.model import Stage, Severity
from autoqc.agent.engine import run_grounded_stage, run_grounded_prose
from autoqc.agent.container import ContainerError
from autoqc.agent.tools import AgentContext, factual_tools
from autoqc.agent.runner import Role
from autoqc.llm import FakeLLMClient


class _FakeSession:
    instances: list["_FakeSession"] = []

    def __init__(self, bundle, **kw):
        self.stopped = False
        _FakeSession.instances.append(self)

    def ensure_image(self): return "img"
    def start(self): return "c"
    def exec(self, cmd, **kw): return "aiohttp/client_reqrep.py:631: conn.close()"
    def stop(self): self.stopped = True


def _bundle():
    return types.SimpleNamespace(
        root=".", rubrics=[{"id": "a" * 32, "title": "1.1: States X",
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}],
        prompt="Why does the pool reopen sockets?", answer="I traced conn.close() ...",
        instruction="i", repository="aio-libs/aiohttp", base_commit="a" * 40,
        files_present={"environment/Dockerfile": True})


def test_grounded_stage_returns_q06_p04_a06(monkeypatch):
    from autoqc.agent import engine
    _FakeSession.instances.clear()
    monkeypatch.setattr(engine, "ContainerSession", _FakeSession)
    def responder(m, t):
        # every grounded agent submits a passing finding for its own unit
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "Q06", "criterion_id": "a" * 32, "passed": True, "evidence": ["client_reqrep.py:631"]},
            {"check_id": "P04", "criterion_id": "prompt", "passed": True, "evidence": ["client_reqrep.py:631"]},
            {"check_id": "A06", "criterion_id": "answer", "passed": True, "evidence": ["client_reqrep.py:631"]}]}}]}
    results = run_grounded_stage(_bundle(), FakeLLMClient(responder), docker=lambda: True)
    ids = {r.id for r in results}
    assert ids == {"Q06", "P04", "A06"}
    assert all(r.stage is Stage.FACTUAL for r in results)
    # ONE shared container across all three checks, not one per check.
    assert len(_FakeSession.instances) == 1


def test_grounded_stage_failsafe_without_docker():
    results = run_grounded_stage(_bundle(), None, docker=lambda: False)
    assert {r.id for r in results} == {"Q06", "P04", "A06"}
    assert all(r.needs_human for r in results)  # never hard-reject on infra absence


# --- run_grounded_prose: the 2-round adjudicator -----------------------------

def _ctx(tmp_path: Path) -> AgentContext:
    return AgentContext(bundle_dir=tmp_path, container=None)


def _role() -> Role:
    return Role(name="grounded_test", system_prompt="s", tools=factual_tools())


def _submit(unit, passed, evidence):
    return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
        {"check_id": "P04", "criterion_id": unit, "passed": passed, "evidence": evidence}]}}]}


def _by_call_count(outcomes):
    """responder(messages, tools) that returns outcomes[i] on the i-th client.chat
    call (0-indexed), clamping to the last entry once exhausted."""
    calls = {"n": 0}
    def responder(m, t):
        i = calls["n"]
        calls["n"] += 1
        return outcomes[min(i, len(outcomes) - 1)]
    return responder


def test_grounded_prose_rounds_disagree_needs_human(tmp_path):
    responder = _by_call_count([
        _submit("prompt", True, ["a.go:1"]),
        _submit("prompt", False, ["a.go:1"]),
    ])
    res = run_grounded_prose("P04", "name", Severity.REJECT, _role(), "ctx",
                             FakeLLMClient(responder), _ctx(tmp_path), "prompt")
    assert res.id == "P04" and res.stage is Stage.FACTUAL
    assert res.passed is False and res.needs_human is True


def test_grounded_prose_round_fails_to_submit_needs_human(monkeypatch, tmp_path):
    from autoqc.agent import engine
    monkeypatch.setattr(engine, "FACTUAL_MAX_TURNS", 2)
    # round 1 submits cleanly; round 2 never calls submit_findings, even on the
    # final forced-submit turn, so run_agent returns ok=False for that round.
    responder = _by_call_count([
        _submit("prompt", True, ["a.go:1"]),
        {"tool_calls": []},
    ])
    res = run_grounded_prose("P04", "name", Severity.REJECT, _role(), "ctx",
                             FakeLLMClient(responder), _ctx(tmp_path), "prompt")
    assert res.passed is False and res.needs_human is True


def test_grounded_prose_both_reject_different_files_needs_human(tmp_path):
    responder = _by_call_count([
        _submit("prompt", False, ["a.go:1"]),
        _submit("prompt", False, ["b.go:9"]),
    ])
    res = run_grounded_prose("P04", "name", Severity.REJECT, _role(), "ctx",
                             FakeLLMClient(responder), _ctx(tmp_path), "prompt")
    assert res.passed is False and res.needs_human is True


def test_grounded_prose_both_reject_same_file_is_hard_reject(tmp_path):
    responder = _by_call_count([
        _submit("prompt", False, ["a.go:1"]),
        _submit("prompt", False, ["a.go:40"]),
    ])
    res = run_grounded_prose("P04", "name", Severity.REJECT, _role(), "ctx",
                             FakeLLMClient(responder), _ctx(tmp_path), "prompt")
    assert res.passed is False and res.needs_human is False


# --- run_grounded_stage: fail-safe branches ----------------------------------

def _responder_all_pass():
    def responder(m, t):
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": "Q06", "criterion_id": "a" * 32, "passed": True, "evidence": ["x.go:1"]},
            {"check_id": "P04", "criterion_id": "prompt", "passed": True, "evidence": ["x.go:1"]},
            {"check_id": "A06", "criterion_id": "answer", "passed": True, "evidence": ["x.go:1"]}]}}]}
    return responder


def _assert_all_needs_human_never_hard_reject(results):
    assert {r.id for r in results} == {"Q06", "P04", "A06"}
    for r in results:
        assert r.needs_human is True
        assert not (r.passed is False and r.needs_human is False)  # never a silent hard reject


def test_grounded_stage_needs_human_when_no_dockerfile_in_bundle():
    b = _bundle()
    b.files_present = {"environment/Dockerfile": False}
    # docker=True proves the stage short-circuits on the missing-Dockerfile check
    # rather than ever reaching the Docker gate.
    results = run_grounded_stage(b, FakeLLMClient(_responder_all_pass()), docker=lambda: True)
    _assert_all_needs_human_never_hard_reject(results)
    assert all("dockerfile" in r.detail.lower() for r in results)


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


def test_grounded_stage_needs_human_when_container_error_on_setup(monkeypatch):
    from autoqc.agent import engine
    monkeypatch.setattr(engine, "ContainerSession", _FakeSessionStartFails)
    results = run_grounded_stage(_bundle(), FakeLLMClient(_responder_all_pass()),
                                 docker=lambda: True)
    _assert_all_needs_human_never_hard_reject(results)


def test_grounded_stage_needs_human_and_stops_session_when_pass_raises(monkeypatch):
    from autoqc.agent import engine
    _FakeSession.instances.clear()
    monkeypatch.setattr(engine, "ContainerSession", _FakeSession)

    def boom(*args, **kwargs):
        raise RuntimeError("factual pass exploded")

    monkeypatch.setattr(engine, "run_factual", boom)
    results = run_grounded_stage(_bundle(), FakeLLMClient(_responder_all_pass()),
                                 docker=lambda: True)
    _assert_all_needs_human_never_hard_reject(results)
    assert len(_FakeSession.instances) == 1
    assert _FakeSession.instances[0].stopped is True

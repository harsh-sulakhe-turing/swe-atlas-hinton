import types
from autoqc.model import Stage, Severity, Verdict
from autoqc.agent.engine import run_grounded_stage

class _FakeSession:
    def __init__(self, bundle, **kw): pass
    def ensure_image(self): return "img"
    def start(self): return "c"
    def exec(self, cmd, **kw): return "aiohttp/client_reqrep.py:631: conn.close()"
    def stop(self): pass

def _bundle():
    return types.SimpleNamespace(
        root=".", rubrics=[{"id": "a" * 32, "title": "1.1: States X",
            "annotations": {"type": "positive hli verifier", "importance": "must have"}}],
        prompt="Why does the pool reopen sockets?", answer="I traced conn.close() ...",
        instruction="i", repository="aio-libs/aiohttp", base_commit="a" * 40,
        files_present={"environment/Dockerfile": True})

def test_grounded_stage_returns_q06_p04_a06(monkeypatch):
    from autoqc.agent import engine
    monkeypatch.setattr(engine, "ContainerSession", _FakeSession)
    from autoqc.llm import FakeLLMClient
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

def test_grounded_stage_failsafe_without_docker():
    results = run_grounded_stage(_bundle(), None, docker=lambda: False)
    assert {r.id for r in results} == {"Q06", "P04", "A06"}
    assert all(r.needs_human for r in results)  # never hard-reject on infra absence

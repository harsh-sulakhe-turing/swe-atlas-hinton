# tests/test_agent_runner.py
from pathlib import Path
from autoqc.agent.runner import Role, AgentResult, run_agent
from autoqc.agent.tools import default_tools, AgentContext
from autoqc.llm import FakeLLMClient


def _ctx(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/rubrics.json").write_text('[{"id":"b"}]')
    return AgentContext(bundle_dir=tmp_path)


def _role():
    return Role(name="proposer", system_prompt="judge.", tools=default_tools())


def test_tool_then_submit(tmp_path):
    # turn 1: read a file; turn 2 (after tool result present): submit
    def responder(messages, tools):
        if any(m["role"] == "tool" for m in messages):
            return {"tool_calls": [{"id": "s", "name": "submit_findings",
                     "args": {"findings": [{"check_id": "Q07", "criterion_id": "b",
                              "passed": True, "evidence": ["ok"]}]}}]}
        return {"tool_calls": [{"id": "r", "name": "read_bundle_file",
                 "args": {"path": "tests/rubrics.json"}}]}
    res = run_agent(_role(), "context", FakeLLMClient(responder), _ctx(tmp_path))
    assert res.ok is True
    assert res.findings[0]["check_id"] == "Q07"


def test_timeout_when_never_submits(tmp_path):
    def responder(messages, tools):
        return {"tool_calls": [{"id": "r", "name": "list_dir", "args": {"path": "tests"}}]}
    res = run_agent(_role(), "context", FakeLLMClient(responder), _ctx(tmp_path), max_turns=3)
    assert res.ok is False and "turn" in res.reason.lower()


def test_chat_error_degrades(tmp_path):
    class Boom(FakeLLMClient):
        def chat(self, messages, tools=None):
            raise RuntimeError("gateway 500")
    res = run_agent(_role(), "context", Boom(lambda m, t: {}), _ctx(tmp_path))
    assert res.ok is False and "gateway 500" in res.reason


def test_unknown_tool_is_reported_not_fatal(tmp_path):
    calls = {"n": 0}
    def responder(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool_calls": [{"id": "x", "name": "no_such_tool", "args": {}}]}
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": []}}]}
    res = run_agent(_role(), "context", FakeLLMClient(responder), _ctx(tmp_path))
    assert res.ok is True  # loop survived the unknown tool and continued to submit


def test_no_tool_calls_gets_nudged_then_submits(tmp_path):
    calls = {"n": 0}
    def responder(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": "I think it's fine."}  # no tool call
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": []}}]}
    res = run_agent(_role(), "context", FakeLLMClient(responder), _ctx(tmp_path))
    assert res.ok is True and calls["n"] == 2


def test_forced_final_submit_salvages_on_budget_exhaustion(tmp_path):
    # Agent explores every turn and never submits during the loop. On budget
    # exhaustion run_agent must make a submit-ONLY call; the agent submits then,
    # so partial findings are salvaged (ok=True) instead of lost.
    def responder(messages, tools):
        names = [t["function"]["name"] for t in (tools or [])]
        if names == ["submit_findings"]:                       # the forced final turn
            return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {
                "findings": [{"check_id": "Q06", "criterion_id": "b",
                              "passed": True, "evidence": ["ok"]}]}}]}
        return {"tool_calls": [{"id": "r", "name": "list_dir", "args": {"path": "tests"}}]}
    res = run_agent(_role(), "context", FakeLLMClient(responder), _ctx(tmp_path), max_turns=3)
    assert res.ok is True and res.findings[0]["criterion_id"] == "b"


def test_forced_final_submit_still_fails_if_agent_refuses(tmp_path):
    # If the agent still won't submit even on the submit-only final turn, the
    # result is the unchanged ok=False fallback.
    def responder(messages, tools):
        names = [t["function"]["name"] for t in (tools or [])]
        if names == ["submit_findings"]:
            return {"text": "still not submitting"}            # no tool call
        return {"tool_calls": [{"id": "r", "name": "list_dir", "args": {"path": "tests"}}]}
    res = run_agent(_role(), "context", FakeLLMClient(responder), _ctx(tmp_path), max_turns=2)
    assert res.ok is False and "max_turns" in res.reason


def test_raising_tool_degrades_not_crash(tmp_path):
    from autoqc.agent.tools import Tool, default_tools
    def boom(args, ctx): raise RuntimeError("kaboom")
    bad = Tool(name="bad", description="", parameters={"type":"object","properties":{}}, run=boom)
    role = Role(name="proposer", system_prompt="x", tools=default_tools() + [bad])
    calls = {"n": 0}
    def responder(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool_calls": [{"id": "b", "name": "bad", "args": {}}]}
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": []}}]}
    res = run_agent(role, "ctx", FakeLLMClient(responder), _ctx(tmp_path))
    assert res.ok is True  # survived the raising tool

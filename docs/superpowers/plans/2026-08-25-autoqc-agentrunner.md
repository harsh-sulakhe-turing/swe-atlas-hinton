# AutoQC AgentRunner Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tool-using agent primitive for AutoQC's semantic layer — a native tool-calling client method, a tool registry with the structured-finding contract, and the `run_agent` loop — all CI-testable with a `FakeLLMClient` (no network, no keys).

**Architecture:** `llm.py` gains a `chat(messages, tools)` method returning a normalized `ChatResponse` (tool calls + text); the real `GatewayLLMClient` implements it over the OpenAI-compatible gateway with stdlib `urllib` (native tool-calling — confirmed working against OpenRouter/GLM), the `FakeLLMClient` implements it from a scripted responder. `agent/tools.py` defines a `Tool` (schema + executor), the built-in tools (`read_bundle_file`, `list_dir`), the terminal `submit_findings` schema (the contract), and a `validate_findings` helper. `agent/runner.py` runs the loop: call the model with the role's tools → dispatch each tool call → when the model calls `submit_findings`, return its findings; a timeout or error returns `ok=False` so the caller can degrade to `needs_human`.

**Tech Stack:** Python 3.11+ target (runs on this machine's 3.10). `pytest`. **No third-party runtime deps** — the gateway call uses stdlib `urllib` (as validated by the probe). Every test uses `FakeLLMClient`.

**Spec:** `docs/superpowers/specs/2026-08-25-autoqc-agent-architecture.md` (§3 the primitive, §3.1 run_bash safety, §4 the contract).

## Global Constraints

- Run every test with `python3 -m pytest`. **All tests use `FakeLLMClient`** — no test may hit the network or require `openai`/keys. The live `GatewayLLMClient.chat()` path (`urllib` POST) is exercised only in manual/live runs; its response *parsing* is unit-tested with a canned dict.
- Native tool-calling is **confirmed** (protocol A) — do not build a text-action fallback.
- Reuse the existing `CheckResult`/`Severity`/`Stage` model unchanged.
- The `submit_findings` tool schema is the structured contract: each finding is `{check_id (Q01–Q12), criterion_id, passed (bool), evidence (non-empty list[str]), reason}`.
- Gateway config from env: `EVAL_API_KEY`, `EVAL_BASE_URL`, `EVAL_MODEL` (already used by `GatewayLLMClient`). Dev endpoint is OpenRouter; judge model `z-ai/glm-5.2`.
- Robustness: any model error, timeout, or malformed submission makes the agent return `ok=False` — never raise out of `run_agent`.
- `if git commit is blocked by a permission classifier, leave files staged and report DONE_WITH_CONCERNS` — the controller commits.

---

## File Structure

- `autoqc/llm.py` — MODIFIED: add `ToolCall`, `ChatResponse`, `chat()` on both clients, `_parse_openai_response`; remove the now-unused `judge()`.
- `autoqc/agent/__init__.py` — package marker.
- `autoqc/agent/tools.py` — `Tool`, `AgentContext`, `read_bundle_file`, `list_dir`, `SUBMIT_FINDINGS`, `default_tools`, `validate_findings`.
- `autoqc/agent/runner.py` — `Role`, `AgentResult`, `run_agent`.
- `tests/` — one module per source file.

---

### Task 1: `chat()` — native tool-calling on the client

**Files:**
- Modify: `autoqc/llm.py`
- Modify: `tests/test_llm.py`
- Test: `tests/test_llm_chat.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ToolCall(id: str, name: str, args: dict)`; `ChatResponse(tool_calls: list[ToolCall], text: str | None = None)`. `LLMClient.chat(messages, tools=None) -> ChatResponse` (base raises NotImplementedError). `FakeLLMClient.chat` calls `responder(messages, tools)` returning a dict `{"tool_calls": [{"id"?, "name", "args"}], "text"?}` normalized via `_to_response`. `GatewayLLMClient.chat` POSTs (stdlib urllib) to `{base}/chat/completions` with `tools=[schema,...]` and parses via `_parse_openai_response(resp_dict) -> ChatResponse`. `judge()` is removed. `default_client()` and `.available()` unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_chat.py
import pytest
from autoqc.llm import (LLMClient, FakeLLMClient, ToolCall, ChatResponse,
                        _parse_openai_response)


def test_base_chat_abstract():
    with pytest.raises(NotImplementedError):
        LLMClient().chat([])


def test_fake_returns_tool_call():
    fake = FakeLLMClient(lambda messages, tools: {
        "tool_calls": [{"id": "c1", "name": "read_bundle_file", "args": {"path": "x"}}]})
    r = fake.chat([{"role": "user", "content": "hi"}], tools=[])
    assert isinstance(r, ChatResponse)
    assert len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert (tc.id, tc.name, tc.args) == ("c1", "read_bundle_file", {"path": "x"})
    assert r.text is None


def test_fake_returns_text_and_synthesizes_ids():
    fake = FakeLLMClient(lambda m, t: {"text": "done", "tool_calls": [{"name": "submit_findings", "args": {}}]})
    r = fake.chat([], tools=[])
    assert r.text == "done"
    assert r.tool_calls[0].id  # an id was synthesized when omitted


def test_parse_openai_response_reads_tool_calls():
    # shape returned by the OpenRouter/GLM probe
    resp = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "call_f443", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}}]}
    r = _parse_openai_response(resp)
    assert r.tool_calls[0].name == "get_weather"
    assert r.tool_calls[0].args == {"city": "Paris"}
    assert r.tool_calls[0].id == "call_f443"


def test_parse_openai_response_plain_text():
    resp = {"choices": [{"message": {"content": "hello", "tool_calls": None}}]}
    r = _parse_openai_response(resp)
    assert r.tool_calls == [] and r.text == "hello"


def test_parse_tolerates_bad_arguments_json():
    resp = {"choices": [{"message": {"tool_calls": [
        {"id": "c", "function": {"name": "f", "arguments": "{not json"}}]}}]}
    r = _parse_openai_response(resp)
    assert r.tool_calls[0].args == {}  # unparseable args degrade to {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_llm_chat.py -v`
Expected: FAIL with `ImportError: cannot import name 'ToolCall'`

- [ ] **Step 3: Rewrite `autoqc/llm.py`**

```python
from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ChatResponse:
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str | None = None


def _to_response(raw: dict) -> ChatResponse:
    tcs = []
    for i, t in enumerate(raw.get("tool_calls") or []):
        tcs.append(ToolCall(id=t.get("id") or f"c{i}", name=t["name"], args=t.get("args") or {}))
    return ChatResponse(tool_calls=tcs, text=raw.get("text"))


def _parse_openai_response(resp: dict) -> ChatResponse:
    msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
    tcs = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", {}) or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                args = {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        tcs.append(ToolCall(id=tc.get("id") or f"c{i}", name=fn.get("name", ""), args=args))
    return ChatResponse(tool_calls=tcs, text=msg.get("content"))


class LLMClient:
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        raise NotImplementedError


class FakeLLMClient(LLMClient):
    """Deterministic client for tests. `responder(messages, tools) -> dict`."""
    def __init__(self, responder):
        self._responder = responder
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None) -> ChatResponse:
        self.calls.append(messages)
        return _to_response(self._responder(messages, tools))


class GatewayLLMClient(LLMClient):
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.environ.get("EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("EVAL_BASE_URL")
                         or os.environ.get("OPENAI_API_BASE") or "").rstrip("/")
        self.model = model or os.environ.get("EVAL_MODEL", "z-ai/glm-5.2")

    def available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat(self, messages, tools=None) -> ChatResponse:
        payload = {"model": self.model, "messages": messages, "max_tokens": 1024}
        if tools:
            payload["tools"] = tools
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://turing.com", "X-Title": "AutoQC"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return _parse_openai_response(json.load(r))


def default_client() -> LLMClient | None:
    g = GatewayLLMClient()
    return g if g.available() else None
```

- [ ] **Step 4: Update `tests/test_llm.py`** — replace the two `judge`-based tests; keep the env/available/default_client tests.

Replace the whole file with:

```python
# tests/test_llm.py
import pytest
from autoqc.llm import LLMClient, GatewayLLMClient, default_client


def test_base_client_is_abstract():
    with pytest.raises(NotImplementedError):
        LLMClient().chat([])


def test_gateway_available_reflects_env(monkeypatch):
    monkeypatch.delenv("EVAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    assert GatewayLLMClient().available() is False
    assert default_client() is None
    monkeypatch.setenv("EVAL_API_KEY", "k")
    monkeypatch.setenv("EVAL_BASE_URL", "http://gw")
    assert GatewayLLMClient().available() is True
    assert isinstance(default_client(), GatewayLLMClient)


def test_gateway_default_model():
    assert GatewayLLMClient(api_key="k", base_url="http://gw").model == "z-ai/glm-5.2"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_llm_chat.py tests/test_llm.py -v`
Expected: PASS (all)

- [ ] **Step 6: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: PASS (M1 + these).

- [ ] **Step 7: Commit**

```bash
git add autoqc/llm.py tests/test_llm.py tests/test_llm_chat.py
git commit -m "feat(autoqc): native tool-calling chat() on the LLM client"
```

---

### Task 2: tools, context, and the finding contract

**Files:**
- Create: `autoqc/agent/__init__.py`
- Create: `autoqc/agent/tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `Bundle` (only conceptually — the tool takes a bundle root path via `AgentContext`).
- Produces:
  - `AgentContext(bundle_dir: Path)` — carries what tools need.
  - `Tool(name: str, description: str, parameters: dict, run, terminal: bool = False)` with `.schema() -> dict` (OpenAI tool schema).
  - `read_bundle_file` / `list_dir` Tool instances; `SUBMIT_FINDINGS` Tool (terminal, `run=None`); `default_tools() -> list[Tool]` (the three above).
  - `CHECK_IDS = {"Q01",...,"Q12"}` (Q06 included for later).
  - `validate_findings(findings, allowed_criterion_ids: set[str]) -> tuple[list[dict], list[str]]` — returns `(valid, problems)`; a finding is valid iff `check_id` in CHECK_IDS, `criterion_id` in allowed set or == `"rubric"`, `passed` is bool, `evidence` a non-empty list.
  - `ALLOWED_READ = {"tests/prompt.txt","tests/rubrics.json","solution/answer.txt","task.toml"}` — `read_bundle_file` only reads these (whitelist), returns an error string otherwise.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_tools.py
import json
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.agent'`

- [ ] **Step 3: Write the files**

```python
# autoqc/agent/__init__.py
```

```python
# autoqc/agent/tools.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CHECK_IDS = {f"Q{n:02d}" for n in range(1, 13)}
ALLOWED_READ = {"tests/prompt.txt", "tests/rubrics.json", "solution/answer.txt", "task.toml"}


@dataclass
class AgentContext:
    bundle_dir: Path


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable | None = None
    terminal: bool = False

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


def _read_bundle_file(args: dict, ctx: AgentContext) -> str:
    path = str(args.get("path", ""))
    if path not in ALLOWED_READ:
        return f"error: '{path}' is not allowed (readable: {sorted(ALLOWED_READ)})"
    p = Path(ctx.bundle_dir) / path
    try:
        return p.read_text()
    except FileNotFoundError:
        return f"error: '{path}' not found in bundle"
    except OSError as e:
        return f"error: could not read '{path}': {e}"


def _list_dir(args: dict, ctx: AgentContext) -> str:
    rel = str(args.get("path", "."))
    p = Path(ctx.bundle_dir) / rel
    try:
        return "\n".join(sorted(x.name for x in p.iterdir()))
    except OSError as e:
        return f"error: could not list '{rel}': {e}"


read_bundle_file = Tool(
    name="read_bundle_file",
    description="Read one file from the task bundle (prompt, rubric, answer, or task.toml).",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    run=_read_bundle_file)

list_dir = Tool(
    name="list_dir",
    description="List the entries of a directory inside the task bundle.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    run=_list_dir)

SUBMIT_FINDINGS = Tool(
    name="submit_findings",
    description="Submit the final structured verdicts and finish. Call this exactly once.",
    parameters={"type": "object", "properties": {"findings": {"type": "array", "items": {
        "type": "object",
        "required": ["check_id", "criterion_id", "passed", "evidence"],
        "properties": {
            "check_id": {"type": "string", "enum": sorted(CHECK_IDS)},
            "criterion_id": {"type": "string"},
            "passed": {"type": "boolean"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"}}}}}, "required": ["findings"]},
    run=None, terminal=True)


def default_tools() -> list[Tool]:
    return [read_bundle_file, list_dir, SUBMIT_FINDINGS]


def validate_findings(findings, allowed_criterion_ids: set[str]):
    """Return (valid, problems). A finding is valid iff check_id is a known check,
    criterion_id is a known rubric id (or 'rubric'), passed is bool, evidence non-empty."""
    valid, problems = [], []
    for i, f in enumerate(findings if isinstance(findings, list) else []):
        if not isinstance(f, dict):
            problems.append(f"finding {i}: not an object"); continue
        if f.get("check_id") not in CHECK_IDS:
            problems.append(f"finding {i}: bad check_id {f.get('check_id')!r}"); continue
        cid = f.get("criterion_id")
        if cid != "rubric" and cid not in allowed_criterion_ids:
            problems.append(f"finding {i}: unknown criterion_id {cid!r}"); continue
        if not isinstance(f.get("passed"), bool):
            problems.append(f"finding {i}: passed not a bool"); continue
        ev = f.get("evidence")
        if not (isinstance(ev, list) and len(ev) > 0):
            problems.append(f"finding {i}: empty/invalid evidence"); continue
        valid.append(f)
    return valid, problems
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_tools.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add autoqc/agent/__init__.py autoqc/agent/tools.py tests/test_agent_tools.py
git commit -m "feat(autoqc): agent tools + submit_findings contract + validation"
```

---

### Task 3: the `run_agent` loop

**Files:**
- Create: `autoqc/agent/runner.py`
- Test: `tests/test_agent_runner.py`

**Interfaces:**
- Consumes: `Tool`, `AgentContext`, `default_tools`, `SUBMIT_FINDINGS` (Task 2); `LLMClient`/`FakeLLMClient`, `ChatResponse` (Task 1).
- Produces:
  - `Role(name: str, system_prompt: str, tools: list[Tool])`.
  - `AgentResult(findings: list[dict], ok: bool, reason: str = "")`.
  - `run_agent(role, context_text, client, ctx, max_turns=12) -> AgentResult` — loop: `client.chat(messages, tools=[t.schema()...])`; append the assistant message; for each tool call, if it is `submit_findings` return `AgentResult(findings=..., ok=True)`, else execute the tool and append a `{"role":"tool",...}` message; if the model returns no tool calls, nudge once to call `submit_findings`; a `client.chat` exception → `AgentResult(ok=False, reason=...)`; hitting `max_turns` → `ok=False`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoqc.agent.runner'`

- [ ] **Step 3: Write `autoqc/agent/runner.py`**

```python
from __future__ import annotations
import json
from dataclasses import dataclass, field
from autoqc.agent.tools import Tool


@dataclass
class Role:
    name: str
    system_prompt: str
    tools: list[Tool]


@dataclass
class AgentResult:
    findings: list[dict] = field(default_factory=list)
    ok: bool = False
    reason: str = ""


def _assistant_msg(resp) -> dict:
    return {"role": "assistant", "content": resp.text or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                           for tc in resp.tool_calls]}


def run_agent(role: Role, context_text: str, client, ctx, max_turns: int = 12) -> AgentResult:
    by_name = {t.name: t for t in role.tools}
    schemas = [t.schema() for t in role.tools]
    messages = [{"role": "system", "content": role.system_prompt},
                {"role": "user", "content": context_text}]
    for _ in range(max_turns):
        try:
            resp = client.chat(messages, tools=schemas)
        except Exception as e:  # gateway/network/parse error
            return AgentResult(ok=False, reason=f"chat error: {e}")

        if not resp.tool_calls:
            messages.append({"role": "assistant", "content": resp.text or ""})
            messages.append({"role": "user", "content": "Call submit_findings to finish."})
            continue

        messages.append(_assistant_msg(resp))
        for call in resp.tool_calls:
            if call.name == "submit_findings":
                findings = call.args.get("findings", []) if isinstance(call.args, dict) else []
                return AgentResult(findings=findings, ok=True)
            tool = by_name.get(call.name)
            out = tool.run(call.args, ctx) if (tool and tool.run) else f"error: unknown tool {call.name!r}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
    return AgentResult(ok=False, reason=f"exceeded max_turns ({max_turns}) without submit_findings")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_runner.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add autoqc/agent/runner.py tests/test_agent_runner.py
git commit -m "feat(autoqc): run_agent loop (tool dispatch, submit-findings terminal, degrade to ok=False)"
```

---

## Self-Review

**Spec coverage (agent-core slice):**
- `chat()` native tool-calling on the client + response parsing → Task 1. ✓
- Tools (`read_bundle_file`, `list_dir`) with a read whitelist → Task 2. ✓
- `submit_findings` structured contract + `validate_findings` → Task 2. ✓
- `run_agent` loop with tool dispatch, terminal submit, timeout/error → `ok=False` (never raise) → Task 3. ✓
- Deferred to the next plans (correctly absent): the engine (K proposer + adversary passes + adjudication), the checks (Q07/Q03 role prompts), pipeline wiring, the live smoke run, and Q06's `run_bash`/container role.

**Placeholder scan:** none — every step has real code and real tests.

**Type consistency:** `ToolCall(id,name,args)` / `ChatResponse(tool_calls,text)` defined in Task 1, consumed by `run_agent` in Task 3 (`resp.tool_calls`, `tc.id/name/args`, `resp.text`). `Tool(name,description,parameters,run,terminal)` + `AgentContext(bundle_dir)` defined in Task 2, consumed in Task 3 (`t.schema()`, `t.name`, `tool.run(args, ctx)`). `FakeLLMClient` responder is `(messages, tools) -> dict` consistently across Tasks 1 and 3. `validate_findings` (Task 2) is not called by `run_agent` — it is applied by the engine in the next plan (the runner returns raw findings; validation needs the rubric's criterion ids).

---

## Notes for the next plan (engine + Q07 + wiring — not in scope here)

Adds `autoqc/agent/engine.py` (`run_check`: K proposer passes + 1 asymmetric adversary pass + adjudication reusing the split/overturn→needs_human logic, calling `validate_findings` against the rubric ids, rolling up to one `CheckResult`), the Q07 (and Q03) role prompts + a `SEMANTIC_CHECKS` registry, pipeline wiring in `cli.run` (structural + agent semantic, structural-only when no client), and the first **live smoke run** against `z-ai/glm-5.2`. Q06's `factual` role + `run_bash` + container is the plan after.

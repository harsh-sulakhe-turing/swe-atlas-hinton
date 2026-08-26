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
        # Generous default: a submit_findings call over a whole rubric emits many
        # findings, and reasoning models spend output tokens on reasoning too. 1024
        # truncated multi-finding submissions in the first live smoke. Env-tunable.
        max_tokens = int(os.environ.get("EVAL_MAX_TOKENS", "8192"))
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
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

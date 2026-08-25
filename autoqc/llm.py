from __future__ import annotations
import json
import os


class LLMClient:
    """Interface: judge(messages) -> parsed JSON dict."""
    def judge(self, messages: list[dict]) -> dict:
        raise NotImplementedError


class FakeLLMClient(LLMClient):
    """Deterministic client for tests. `responder` maps messages -> dict."""
    def __init__(self, responder):
        self._responder = responder
        self.calls: list[list[dict]] = []

    def judge(self, messages: list[dict]) -> dict:
        self.calls.append(messages)
        return self._responder(messages)


class GatewayLLMClient(LLMClient):
    """OpenAI-compatible client pointed at the token gateway. Lazy-imports openai."""
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.environ.get("EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("EVAL_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        self.model = model or os.environ.get("EVAL_MODEL", "anthropic/claude-opus-4-5-20251101")

    def available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def judge(self, messages: list[dict]) -> dict:
        from openai import OpenAI  # lazy: not needed for tests
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model, messages=messages,
            response_format={"type": "json_object"}, max_tokens=1024,
        )
        return json.loads(resp.choices[0].message.content)


def default_client() -> LLMClient | None:
    g = GatewayLLMClient()
    return g if g.available() else None

import pytest
from autoqc.llm import LLMClient, FakeLLMClient, GatewayLLMClient, default_client


def test_fake_client_returns_responder_output():
    fake = FakeLLMClient(lambda messages: {"passed": True, "seen": len(messages)})
    out = fake.judge([{"role": "user", "content": "hi"}])
    assert out == {"passed": True, "seen": 1}


def test_base_client_is_abstract():
    with pytest.raises(NotImplementedError):
        LLMClient().judge([])


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
    dc = default_client()
    assert isinstance(dc, GatewayLLMClient) and dc.model  # default model set


def test_gateway_default_model():
    g = GatewayLLMClient(api_key="k", base_url="http://gw")
    assert g.model == "anthropic/claude-opus-4-5-20251101"

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

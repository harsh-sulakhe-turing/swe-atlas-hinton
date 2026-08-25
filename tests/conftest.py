import pytest


@pytest.fixture(autouse=True)
def _clear_gateway_env(monkeypatch):
    for var in ("EVAL_API_KEY", "EVAL_BASE_URL", "EVAL_MODEL",
                "OPENAI_API_KEY", "OPENAI_API_BASE"):
        monkeypatch.delenv(var, raising=False)

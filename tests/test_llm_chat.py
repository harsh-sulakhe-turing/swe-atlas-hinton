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

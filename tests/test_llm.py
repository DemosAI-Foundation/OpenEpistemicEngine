import json

from oee.config import Settings
from oee.llm import OpenAICompatProvider, _finish_reason
from oee.prompts import PASS2_SCHEMA

BODY = json.dumps({"choices": [{"message": {"content": '{"a": 1}'}}],
                   "model": "m",
                   "usage": {"prompt_tokens": 5, "completion_tokens": 7}})


def _settings(**kw):
    s = Settings()
    s.provider = "openai"
    s.base_url = "http://127.0.0.1:1"
    s.model = "m"
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _capture(seen, replies=None):
    def fake_post(self, payload):
        seen.append(json.loads(json.dumps(payload)))
        if replies:
            return replies[len(seen) - 1]
        return 200, BODY
    return fake_post


def test_the_default_structured_mode_is_tool_calling(monkeypatch):
    seen = []
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen))
    r = OpenAICompatProvider(_settings()).chat("s", "u", schema=PASS2_SCHEMA,
                                               kind="pass2")
    assert seen[0]["tools"][0]["function"]["name"] == "epistemic_output"
    assert r.meta["response_format"] == "tool"
    assert r.meta["response_format_fallbacks"] == []


def test_json_object_mode(monkeypatch):
    seen = []
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen))
    p = OpenAICompatProvider(_settings(structured_outputs="object"))
    r = p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert "tools" not in seen[0]
    assert r.meta["response_format"] == "json_object"
    assert r.meta["response_format_fallbacks"] == []


def test_json_schema_mode_sends_the_full_schema(monkeypatch):
    seen = []
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen))
    p = OpenAICompatProvider(_settings(structured_outputs="schema"))
    p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    fmt = seen[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "epistemic_schema"
    assert fmt["json_schema"]["schema"]["properties"]["chat_reply"]


def test_structured_outputs_can_be_switched_off(monkeypatch):
    seen = []
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen))
    OpenAICompatProvider(_settings(structured_outputs="off")).chat(
        "s", "u", schema=PASS2_SCHEMA, kind="pass2")
    assert "response_format" not in seen[0]


def test_unstructured_calls_send_no_response_format(monkeypatch):
    seen = []
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen))
    OpenAICompatProvider(_settings()).chat("s", "u", kind="generic")
    assert "response_format" not in seen[0]


def test_a_server_that_rejects_json_object_is_retried_unconstrained(monkeypatch):
    seen = []
    replies = [(400, '{"error": "response_format is not supported by this server"}'),
               (200, BODY)]
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen, replies))
    p = OpenAICompatProvider(_settings(structured_outputs="object"))
    r = p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    assert len(seen) == 2
    assert "response_format" not in seen[1]
    assert r.meta["response_format_fallbacks"] == [
        "response_format rejected; retried unconstrained"]


def test_the_downgrade_chain_runs_tool_then_json_object_then_nothing(monkeypatch):
    seen = []
    replies = [(400, '{"error": "tools not supported"}'),
               (400, '{"error": "response_format not supported"}'),
               (200, BODY)]
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen, replies))
    p = OpenAICompatProvider(_settings(structured_outputs="tool"))
    r = p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    assert "tools" in seen[0]
    assert seen[1]["response_format"] == {"type": "json_object"}
    assert "response_format" not in seen[2]
    assert r.text == '{"a": 1}'
    assert "tool calling rejected" in r.meta["response_format_fallbacks"][0]
    assert r.meta["response_format_fallbacks"][1] == (
        "response_format rejected; retried unconstrained")
    assert "response_format" not in seen[2]


def test_json_schema_mode_downgrades_to_json_object_first(monkeypatch):
    seen = []
    replies = [(400, '{"error": "json_schema: strict mode unsupported"}'),
               (200, BODY)]
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen, replies))
    p = OpenAICompatProvider(_settings(structured_outputs="schema"))
    r = p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    assert seen[1]["response_format"] == {"type": "json_object"}
    assert r.meta["response_format_fallbacks"] == [
        "json_schema rejected; retried as json_object"]


TOOL_BODY = json.dumps({
    "choices": [{"message": {"content": None, "tool_calls": [{
        "id": "c1", "type": "function",
        "function": {"name": "epistemic_output",
                     "arguments": '{"chat_reply": "hi [R1-C1]", "basis": ["reference"]}'}}]}}],
    "model": "m", "usage": {"prompt_tokens": 9, "completion_tokens": 11}})


def test_tool_mode_sends_the_schema_as_a_forced_function(monkeypatch):
    seen = []
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen, [(200, TOOL_BODY)]))
    p = OpenAICompatProvider(_settings(structured_outputs="tool"))
    r = p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    tool = seen[0]["tools"][0]
    assert tool["function"]["name"] == "epistemic_output"
    assert tool["function"]["strict"] is True
    assert tool["function"]["parameters"]["properties"]["chat_reply"]
    assert seen[0]["tool_choice"] == {"type": "function",
                                      "function": {"name": "epistemic_output"}}
    assert "response_format" not in seen[0]
    assert json.loads(r.text)["basis"] == ["reference"]
    assert r.meta["response_format"] == "tool"


def test_a_server_that_rejects_tools_falls_back_to_json_object(monkeypatch):
    seen = []
    replies = [(400, '{"error": "tools are not supported by this model"}'), (200, BODY)]
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen, replies))
    p = OpenAICompatProvider(_settings(structured_outputs="tool"))
    r = p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    assert "tools" not in seen[1]
    assert seen[1]["response_format"] == {"type": "json_object"}
    assert "tool calling rejected" in r.meta["response_format_fallbacks"][0]
    assert "<response_contract>" in seen[1]["messages"][1]["content"]
    assert p._tool_rejected is True


def test_an_empty_tool_call_is_retried_with_a_weaker_form(monkeypatch):
    seen = []
    empty = json.dumps({"choices": [{"message": {"content": None, "tool_calls": []}}],
                        "model": "m", "usage": {}})
    replies = [(200, empty), (200, BODY)]
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen, replies))
    p = OpenAICompatProvider(_settings(structured_outputs="tool"))
    r = p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    assert r.text == '{"a": 1}'
    assert "tool calling rejected" in r.meta["response_format_fallbacks"][0]
    assert "<response_contract>" in seen[1]["messages"][1]["content"]
    assert p._tool_rejected is True


def test_plain_content_is_accepted_from_a_tool_capable_server(monkeypatch):
    seen = []
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen, [(200, BODY)]))
    p = OpenAICompatProvider(_settings(structured_outputs="tool"))
    r = p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    assert r.text == '{"a": 1}'


def test_tool_rejection_is_memoised_for_later_calls(monkeypatch):
    seen = []
    replies = [(400, '{"error": "tools not supported"}'), (200, BODY), (200, BODY)]
    monkeypatch.setattr(OpenAICompatProvider, "_post", _capture(seen, replies))
    p = OpenAICompatProvider(_settings(structured_outputs="tool"))
    first = p.chat("s", "u", schema=PASS2_SCHEMA, kind="pass2")
    second = p.chat("s2", "u2", schema=PASS2_SCHEMA, kind="pass2")
    assert "tools" in seen[0]
    assert "tools" not in seen[1] and "tools" not in seen[2]
    assert seen[2]["response_format"] == {"type": "json_object"}
    assert "<response_contract>" in seen[2]["messages"][1]["content"]
    assert first.meta["response_format_fallbacks"]
    assert second.meta["response_format_fallbacks"] == []


def test_finish_reason_reads_length_and_tolerates_missing():
    assert _finish_reason({"choices": [{"finish_reason": "length"}]}) == "length"
    assert _finish_reason({"choices": [{"finish_reason": "stop"}]}) == "stop"
    assert _finish_reason({"choices": []}) == ""
    assert _finish_reason({}) == ""

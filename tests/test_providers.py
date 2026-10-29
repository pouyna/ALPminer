from types import SimpleNamespace

import pytest

from alpminer.providers import (AnthropicBackend, GeminiBackend, LLMError,
                                _jsonschema_to_gemini, get_backend)
from alpminer.schema import EXTRACTION_TOOL, TRIAGE_TOOL


# ---- schema conversion ----------------------------------------------------------

def test_jsonschema_to_gemini_converts_nullable_lists():
    schema = {"type": "object", "properties": {
        "material": {"type": "string"},
        "confidence": {"type": ["number", "null"]},
        "tags": {"type": ["array", "null"],
                 "items": {"type": "string"}},
    }, "required": ["material"]}
    out = _jsonschema_to_gemini(schema)
    assert out["properties"]["material"] == {"type": "string"}
    assert out["properties"]["confidence"] == {"type": "number",
                                                "nullable": True}
    assert out["properties"]["tags"]["type"] == "array"
    assert out["properties"]["tags"]["nullable"] is True
    assert out["properties"]["tags"]["items"] == {"type": "string"}


def test_jsonschema_to_gemini_handles_the_real_extraction_tool():
    # every property of the actual recipe schema must convert without error
    # and end up with a single string type, never a list
    converted = _jsonschema_to_gemini(
        EXTRACTION_TOOL["input_schema"]["properties"]["records"]["items"])
    for name, prop in converted["properties"].items():
        assert isinstance(prop["type"], str), f"{name} still has list type"


# ---- backend selection ----------------------------------------------------------

def test_get_backend_anthropic_missing_key_raises(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg.provider = "anthropic"
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        get_backend(cfg)


def test_get_backend_gemini_missing_key_raises(cfg, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg.provider = "gemini"
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        get_backend(cfg)


def test_get_backend_anthropic_with_key_constructs(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    cfg.provider = "anthropic"
    backend = get_backend(cfg)
    assert isinstance(backend, AnthropicBackend)


def test_get_backend_gemini_with_key_constructs(cfg, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake")
    cfg.provider = "gemini"
    backend = get_backend(cfg)
    assert isinstance(backend, GeminiBackend)


def test_get_backend_openai_uses_openai_dot_com(cfg, monkeypatch):
    from alpminer.providers import OpenAIBackend
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fake")
    cfg.provider = "openai"
    backend = get_backend(cfg)
    assert isinstance(backend, OpenAIBackend)
    assert backend.base_url == "https://api.openai.com/v1"


def test_get_backend_openai_missing_key_raises(cfg, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg.provider = "openai"
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        get_backend(cfg)


# ---- Gemini call_tool response handling (mocked client) -------------------------

class FakeGeminiClient:
    """Stands in for genai.Client(); records the config it was called with
    and returns a scripted response."""

    def __init__(self, response=None, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc
        self.last_config = None
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.last_config = config
        if self._raise_exc:
            raise self._raise_exc
        return self._response


def _backend_with_fake_client(monkeypatch, fake_client):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake")
    from google import genai
    monkeypatch.setattr(genai, "Client", lambda api_key: fake_client)
    return GeminiBackend(api_key="AIza-fake")


def test_gemini_call_tool_returns_function_args(monkeypatch):
    call = SimpleNamespace(name="triage_result",
                           args={"reports_own_ald_experiment": True})
    resp = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")],
                           function_calls=[call])
    backend = _backend_with_fake_client(monkeypatch, FakeGeminiClient(resp))
    out = backend.call_tool("gemini-2.5-flash-lite", "sys", "text",
                            TRIAGE_TOOL, max_tokens=300)
    assert out == {"reports_own_ald_experiment": True}


def test_gemini_call_tool_truncated_output_raises(monkeypatch):
    resp = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        function_calls=[])
    backend = _backend_with_fake_client(monkeypatch, FakeGeminiClient(resp))
    with pytest.raises(LLMError, match="truncated"):
        backend.call_tool("gemini-2.5-flash", "sys", "text",
                          EXTRACTION_TOOL, max_tokens=50)


def test_gemini_call_tool_no_function_call_raises(monkeypatch):
    resp = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")],
                           function_calls=None)
    backend = _backend_with_fake_client(monkeypatch, FakeGeminiClient(resp))
    with pytest.raises(LLMError, match="no function call"):
        backend.call_tool("gemini-2.5-flash", "sys", "text",
                          EXTRACTION_TOOL, max_tokens=300)


def test_gemini_daily_quota_aborts_without_retry(monkeypatch):
    """A free-tier requests-per-DAY 429 cannot be waited out inside a run:
    it must raise QuotaExhausted on the FIRST attempt, not retry five times
    (regression: 41 s retry sleeps against an exhausted daily quota)."""
    from google.genai import errors

    from alpminer.providers import QuotaExhausted

    daily_429 = errors.APIError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "Quota exceeded for metric: generate_content_free_tier_"
                   "requests, quotaId: GenerateRequestsPerDayPerProjectPer"
                   "Model-FreeTier"}})

    calls = []

    class CountingClient(FakeGeminiClient):
        def _generate(self, model, contents, config):
            calls.append(model)
            raise daily_429

    monkeypatch.setattr("time.sleep", lambda s: None)
    backend = _backend_with_fake_client(monkeypatch, CountingClient())
    with pytest.raises(QuotaExhausted, match="daily"):
        backend.call_tool("gemini-2.5-flash", "sys", "text", TRIAGE_TOOL,
                          max_tokens=300)
    assert len(calls) == 1                      # no futile retries

    # an ordinary 429 (per-minute, no PerDay) is still retried
    minute_429 = errors.APIError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "rate limited, retry shortly"}})
    calls.clear()

    class MinuteClient(FakeGeminiClient):
        def _generate(self, model, contents, config):
            calls.append(model)
            raise minute_429

    backend = _backend_with_fake_client(monkeypatch, MinuteClient())
    from alpminer.utils import RetryError
    with pytest.raises(RetryError):
        backend.call_tool("gemini-2.5-flash", "sys", "text", TRIAGE_TOOL,
                          max_tokens=300)
    assert len(calls) == 5                      # full retry budget used


def test_gemini_call_tool_forces_the_right_tool_choice(monkeypatch):
    call = SimpleNamespace(name="record_findings", args={"records": []})
    resp = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")],
                           function_calls=[call])
    fake_client = FakeGeminiClient(resp)
    backend = _backend_with_fake_client(monkeypatch, fake_client)
    backend.call_tool("gemini-2.5-flash", "sys", "text", EXTRACTION_TOOL,
                      max_tokens=8000)
    cfg = fake_client.last_config
    assert cfg.tool_config.function_calling_config.mode.value == "ANY"
    assert cfg.tool_config.function_calling_config.allowed_function_names == [
        "record_findings"]
    assert cfg.tools[0].function_declarations[0].name == "record_findings"


def test_gemini_call_tool_retries_on_429_then_succeeds(monkeypatch):
    from google.genai import errors as genai_errors

    call = SimpleNamespace(name="triage_result",
                           args={"reports_own_ald_experiment": False})
    good_resp = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason="STOP")],
        function_calls=[call])
    err = genai_errors.APIError(429, {"error": {"message": "rate limited"}})

    attempts = {"n": 0}

    class FlakyClient(FakeGeminiClient):
        def _generate(self, model, contents, config):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise err
            return good_resp

    monkeypatch.setattr("time.sleep", lambda s: None)
    backend = _backend_with_fake_client(monkeypatch, FlakyClient())
    out = backend.call_tool("gemini-2.5-flash-lite", "sys", "text",
                            TRIAGE_TOOL, max_tokens=300)
    assert out == {"reports_own_ald_experiment": False}
    assert attempts["n"] == 2


def test_gemini_call_tool_gives_up_on_400(monkeypatch):
    from google.genai import errors as genai_errors

    err = genai_errors.APIError(400, {"error": {"message": "bad schema"}})
    backend = _backend_with_fake_client(
        monkeypatch, FakeGeminiClient(raise_exc=err))
    with pytest.raises(LLMError, match="400"):
        backend.call_tool("gemini-2.5-flash", "sys", "text", EXTRACTION_TOOL,
                          max_tokens=300)


# ---- OpenAI-compatible backend (v2) ---------------------------------------------

from alpminer.providers import (OpenAICompatibleBackend, write_new_plugin,
                                _load_plugin)


class FakeHTTPResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or json_dumps(payload)

    def json(self):
        return self._payload


def json_dumps(x):
    import json
    return json.dumps(x or {})


def _openai_ok_payload(args):
    import json
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "tool_calls": [{"function": {"name": "record_findings",
                                     "arguments": json.dumps(args)}}]}}],
            "usage": {"prompt_tokens": 1234, "completion_tokens": 56}}


def _openai_backend(monkeypatch, responses, base_url="http://localhost:11434/v1"):
    import requests as rq
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(rq, "post", fake_post)
    b = OpenAICompatibleBackend({"base_url": base_url})
    return b, calls


def test_openai_backend_forces_tool_and_parses_args(monkeypatch):
    args = {"relevant": True, "records": [{"material": "ZnO"}]}
    b, calls = _openai_backend(
        monkeypatch, [FakeHTTPResponse(payload=_openai_ok_payload(args))])
    out = b.call_tool("llama3", "sys", "text", EXTRACTION_TOOL, 800)
    assert out == args
    sent = calls[0]["json"]
    assert sent["tool_choice"]["function"]["name"] == "record_findings"
    assert sent["messages"][0]["role"] == "system"
    assert calls[0]["url"].endswith("/chat/completions")
    # provider-reported token usage was recorded on the backend
    assert b.usage == {"input": 1234, "output": 56, "calls": 1}


def test_openai_backend_retries_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    args = {"relevant": False, "records": []}
    b, calls = _openai_backend(monkeypatch, [
        FakeHTTPResponse(status=503),
        FakeHTTPResponse(payload=_openai_ok_payload(args))])
    assert b.call_tool("m", "s", "t", TRIAGE_TOOL, 300) == args
    assert len(calls) == 2


def test_openai_backend_gives_up_on_400_and_truncation(monkeypatch):
    b, _ = _openai_backend(monkeypatch,
                           [FakeHTTPResponse(status=400, text="bad request")])
    with pytest.raises(LLMError, match="400"):
        b.call_tool("m", "s", "t", TRIAGE_TOOL, 300)
    b, _ = _openai_backend(monkeypatch, [FakeHTTPResponse(payload={
        "choices": [{"finish_reason": "length", "message": {}}]})])
    with pytest.raises(LLMError, match="truncated"):
        b.call_tool("m", "s", "t", TRIAGE_TOOL, 300)


def test_openai_backend_requires_key_for_remote_hosts(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAICompatibleBackend({"base_url": "https://api.openai.com/v1"})
    OpenAICompatibleBackend({"base_url": "http://localhost:11434/v1"})  # ok


def test_get_backend_openai_compatible(cfg, monkeypatch):
    cfg.provider = "openai_compatible"
    cfg.provider_settings = {"base_url": "http://localhost:11434/v1"}
    assert isinstance(get_backend(cfg), OpenAICompatibleBackend)


# ---- plugin providers (v2) -------------------------------------------------------

def test_plugin_scaffold_lists_contract_and_loads(cfg, tmp_path):
    dest = write_new_plugin(tmp_path, "my_llm")
    assert "create_backend" in dest.read_text()
    cfg.base_dir = tmp_path
    cfg.provider = "my_llm"
    backend = get_backend(cfg)  # scaffold loads; call_tool raises by design
    with pytest.raises(LLMError, match="implement"):
        backend.call_tool("m", "s", "t", TRIAGE_TOOL, 100)


def test_working_plugin_end_to_end(cfg, tmp_path):
    plugin_dir = tmp_path / "llm_providers"
    plugin_dir.mkdir()
    (plugin_dir / "echo.py").write_text(
        "from alpminer.providers import Backend\n"
        "class EchoBackend(Backend):\n"
        "    name='echo'\n"
        "    def __init__(self,s): self.s=s\n"
        "    def call_tool(self,model,system,user_text,tool,max_tokens):\n"
        "        return {'relevant': True, 'records': [],\n"
        "                'notes': self.s.get('greeting')}\n"
        "def create_backend(settings): return EchoBackend(settings)\n")
    cfg.base_dir = tmp_path
    cfg.provider = "echo"
    cfg.provider_settings = {"greeting": "hi"}
    out = get_backend(cfg).call_tool("m", "s", "t", TRIAGE_TOOL, 100)
    assert out["notes"] == "hi"


def test_missing_plugin_names_the_scaffold_command(cfg, tmp_path):
    cfg.base_dir = tmp_path
    cfg.provider = "ghost"
    with pytest.raises(LLMError, match="providers new ghost"):
        get_backend(cfg)


def test_broken_plugin_reports_the_bug(cfg, tmp_path):
    plugin_dir = tmp_path / "llm_providers"
    plugin_dir.mkdir()
    (plugin_dir / "boom.py").write_text("raise ValueError('kaput')\n")
    cfg.base_dir = tmp_path
    cfg.provider = "boom"
    with pytest.raises(LLMError, match="kaput"):
        get_backend(cfg)


# ---- Anthropic (mocked client) ----------------------------------------------------

from alpminer.providers import _anthropic_effort, _openai_effort  # noqa: E402


@pytest.mark.parametrize("model,effort,expected", [
    ("claude-sonnet-5-5", "low", "low"),
    ("claude-haiku-5-5", "low", "low"),
    ("claude-opus-5-5", "max", "max"),
    ("claude-sonnet-4-6", "xhigh", "high"),     # 4.6 has max but no xhigh
    ("claude-sonnet-4-6", "max", "max"),
    ("claude-opus-4-5", "xhigh", "high"),       # Opus 4.5 stops at high
    ("claude-haiku-4-5", "low", None),          # no effort parameter: 400
    ("claude-sonnet-4-5-20250929", "low", None),
    ("claude-sonnet-4-20250514", "low", None),
    ("claude-3-5-haiku-20241022", "low", None),
    ("claude-sonnet-5-5", None, None),
])
def test_anthropic_effort_is_sent_only_where_supported(model, effort,
                                                       expected):
    assert _anthropic_effort(model, effort) == expected


class _Stream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        if isinstance(self._message, BaseException):
            raise self._message
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class FakeAnthropicClient:
    """Records every request; answers from a script of messages."""

    def __init__(self, *messages):
        self.script = list(messages)
        self.requests = []
        self.messages = SimpleNamespace(stream=self._stream)
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(stream=self._beta_stream))

    def _stream(self, **kw):
        self.requests.append(("messages", kw))
        return _Stream(self.script.pop(0))

    def _beta_stream(self, **kw):
        self.requests.append(("beta", kw))
        return _Stream(self.script.pop(0))


def _claude_msg(content, stop="tool_use", **extra):
    return SimpleNamespace(content=content, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=1000,
                                                 output_tokens=50), **extra)


def _tool_use(args, name="record_findings"):
    return SimpleNamespace(type="tool_use", name=name, input=args)


def _anthropic_backend(client):
    backend = AnthropicBackend("sk-ant-fake")
    backend._client = client
    return backend


def _bad_request(message):
    import anthropic
    exc = anthropic.BadRequestError.__new__(anthropic.BadRequestError)
    Exception.__init__(exc, message)
    return exc


def test_anthropic_does_not_force_the_tool_and_sends_effort():
    args = {"relevant": True, "records": []}
    client = FakeAnthropicClient(_claude_msg([
        SimpleNamespace(type="thinking", thinking=""), _tool_use(args)]))
    backend = _anthropic_backend(client)
    out = backend.call_tool("claude-sonnet-5-5", "sys", "text",
                            EXTRACTION_TOOL, 16000, effort="low")
    assert out == args
    kind, kw = client.requests[0]
    assert "tool_choice" not in kw              # forced choice 400s on 5.5
    assert kw["tools"] == [EXTRACTION_TOOL]
    assert kw["output_config"] == {"effort": "low"}
    # Sonnet 5.5 goes through the beta endpoint with refusal fallback
    assert kind == "beta" and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert backend.usage == {"input": 1000, "output": 50, "calls": 1}


def test_anthropic_haiku_has_no_fallback_and_old_models_no_effort():
    client = FakeAnthropicClient(
        _claude_msg([_tool_use({"relevant": True}, "triage_result")]),
        _claude_msg([_tool_use({"relevant": True}, "triage_result")]))
    backend = _anthropic_backend(client)
    backend.call_tool("claude-haiku-5-5", "s", "t", TRIAGE_TOOL, 2048,
                      effort="low")
    backend.call_tool("claude-haiku-4-5", "s", "t", TRIAGE_TOOL, 2048,
                      effort="low")
    (k1, kw1), (k2, kw2) = client.requests
    assert k1 == k2 == "messages" and "fallbacks" not in kw1
    assert kw1["output_config"] == {"effort": "low"}
    assert "output_config" not in kw2           # Haiku 4.5 rejects effort


def test_anthropic_asks_again_once_when_no_tool_call():
    text_only = _claude_msg([SimpleNamespace(type="text", text="Sure!")],
                            stop="end_turn")
    args = {"relevant": False}
    client = FakeAnthropicClient(
        text_only, _claude_msg([_tool_use(args, "triage_result")]))
    backend = _anthropic_backend(client)
    assert backend.call_tool("claude-haiku-5-5", "s", "t", TRIAGE_TOOL,
                             2048) == args
    assert len(client.requests) == 2
    backend = _anthropic_backend(FakeAnthropicClient(text_only, text_only))
    with pytest.raises(LLMError, match="no tool_use"):
        backend.call_tool("claude-haiku-5-5", "s", "t", TRIAGE_TOOL, 2048)


def test_anthropic_refusal_and_truncation_raise():
    refusal = _claude_msg([], stop="refusal",
                          stop_details=SimpleNamespace(category="bio"))
    backend = _anthropic_backend(FakeAnthropicClient(refusal))
    with pytest.raises(LLMError, match="declined.*bio"):
        backend.call_tool("claude-haiku-5-5", "s", "t", TRIAGE_TOOL, 2048)
    cut = _claude_msg([], stop="max_tokens")
    backend = _anthropic_backend(FakeAnthropicClient(cut))
    with pytest.raises(LLMError, match="truncated"):
        backend.call_tool("claude-sonnet-5-5", "s", "t", EXTRACTION_TOOL, 10)


def test_anthropic_rejected_fallback_is_dropped_for_the_session():
    args = {"relevant": True, "records": []}
    client = FakeAnthropicClient(
        _bad_request("fallbacks: not available for this organization"),
        _claude_msg([_tool_use(args)]), _claude_msg([_tool_use(args)]))
    backend = _anthropic_backend(client)
    assert backend.call_tool("claude-sonnet-5-5", "s", "t", EXTRACTION_TOOL,
                             16000) == args
    assert [k for k, _ in client.requests] == ["beta", "messages"]
    backend.call_tool("claude-sonnet-5-5", "s", "t", EXTRACTION_TOOL, 16000)
    assert client.requests[-1][0] == "messages"   # remembered


def test_anthropic_other_bad_requests_become_llm_errors():
    backend = _anthropic_backend(
        FakeAnthropicClient(_bad_request("prompt is too long")))
    with pytest.raises(LLMError, match="too long"):
        backend.call_tool("claude-haiku-5-5", "s", "t", TRIAGE_TOOL, 2048)


# ---- Gemini thinking level --------------------------------------------------------

def test_gemini_3_gets_a_thinking_level_and_2_5_does_not(monkeypatch):
    call = SimpleNamespace(name="triage_result", args={"relevant": True})
    resp = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")],
                           function_calls=[call],
                           usage_metadata=SimpleNamespace(
                               prompt_token_count=900,
                               candidates_token_count=40,
                               thoughts_token_count=60))
    fake = FakeGeminiClient(resp)
    backend = _backend_with_fake_client(monkeypatch, fake)
    backend.call_tool("gemini-3.5-flash-lite", "s", "t", TRIAGE_TOOL, 2048,
                      effort="low")
    assert fake.last_config.thinking_config.thinking_level.value == "LOW"
    assert backend.usage["output"] == 100        # thinking billed as output
    backend.call_tool("gemini-3.8-flash", "s", "t", TRIAGE_TOOL, 2048,
                      effort="max")
    assert fake.last_config.thinking_config.thinking_level.value == "HIGH"
    backend.call_tool("gemini-2.5-flash", "s", "t", TRIAGE_TOOL, 2048,
                      effort="low")
    assert fake.last_config.thinking_config is None


# ---- OpenAI Responses API ---------------------------------------------------------

@pytest.mark.parametrize("model,effort,expected", [
    ("gpt-6.1-sol", "low", "low"),
    ("gpt-6-luna", "medium", "medium"),
    ("gpt-6-astra", "max", "max"),
    ("gpt-5", "max", "high"),
    ("o3", "xhigh", "high"),
    ("gpt-4o", "low", None),
    ("gpt-4.1-mini", "low", None),
])
def test_openai_effort_only_for_reasoning_models(model, effort, expected):
    assert _openai_effort(model, effort) == expected


def _responses_payload(args, name="record_findings"):
    import json
    return {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "function_call", "name": name, "call_id": "c1",
         "arguments": json.dumps(args)}],
        "usage": {"input_tokens": 1500, "output_tokens": 70}}


def _responses_backend(monkeypatch, responses):
    import requests as rq

    from alpminer.providers import OpenAIBackend
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        return responses.pop(0)

    monkeypatch.setattr(rq, "post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fake")
    return OpenAIBackend({"api_key_env": "OPENAI_API_KEY"}), calls


def test_openai_uses_the_responses_api(monkeypatch):
    args = {"relevant": True, "records": [{"material": "ZnO"}]}
    b, calls = _responses_backend(
        monkeypatch, [FakeHTTPResponse(payload=_responses_payload(args))])
    assert b.call_tool("gpt-6.1-sol", "sys", "text", EXTRACTION_TOOL, 16000,
                       effort="low") == args
    sent = calls[0]["json"]
    assert calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-openai-fake"
    assert sent["instructions"] == "sys" and sent["input"] == "text"
    assert sent["tools"][0]["name"] == "record_findings"
    assert sent["tools"][0]["parameters"] == EXTRACTION_TOOL["input_schema"]
    assert sent["tool_choice"] == {"type": "function",
                                   "name": "record_findings"}
    assert sent["max_output_tokens"] == 16000 and sent["store"] is False
    assert sent["reasoning"] == {"effort": "low"}
    assert "max_tokens" not in sent and "messages" not in sent
    assert b.usage == {"input": 1500, "output": 70, "calls": 1}


def test_openai_non_reasoning_model_gets_no_reasoning_block(monkeypatch):
    b, calls = _responses_backend(monkeypatch, [FakeHTTPResponse(
        payload=_responses_payload({"relevant": True}, "triage_result"))])
    b.call_tool("gpt-4o", "s", "t", TRIAGE_TOOL, 2048, effort="low")
    assert "reasoning" not in calls[0]["json"]


def test_openai_truncation_and_missing_call(monkeypatch):
    b, _ = _responses_backend(monkeypatch, [FakeHTTPResponse(payload={
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": []})])
    with pytest.raises(LLMError, match="truncated"):
        b.call_tool("gpt-6-luna", "s", "t", TRIAGE_TOOL, 10)
    b, _ = _responses_backend(monkeypatch, [FakeHTTPResponse(payload={
        "status": "completed", "output": [{"type": "message"}]})])
    with pytest.raises(LLMError, match="no function call"):
        b.call_tool("gpt-6-luna", "s", "t", TRIAGE_TOOL, 2048)


def test_anthropic_ignores_the_declined_attempt_before_a_fallback():
    partial = _tool_use({"relevant": True, "records": [
        {"material": "Al2O3", "temperature_c": {"value": 2}}]})
    full = _tool_use({"relevant": True, "records": [
        {"material": "Al2O3"}, {"material": "ZnO"}]})
    client = FakeAnthropicClient(_claude_msg(
        [partial, SimpleNamespace(type="fallback"), full]))
    out = _anthropic_backend(client).call_tool(
        "claude-sonnet-5-5", "s", "t", EXTRACTION_TOOL, 16000)
    assert out == full.input


def test_anthropic_takes_tool_calls_only_from_a_tool_use_stop():
    odd = _claude_msg([_tool_use({"relevant": True}, "triage_result")],
                      stop="end_turn")
    good = _claude_msg([_tool_use({"relevant": False}, "triage_result")])
    client = FakeAnthropicClient(odd, good)
    assert _anthropic_backend(client).call_tool(
        "claude-haiku-5-5", "s", "t", TRIAGE_TOOL, 2048) == {"relevant": False}


def test_anthropic_retries_early_ends_and_mid_stream_failures(monkeypatch):
    import anthropic
    import httpx2
    monkeypatch.setattr("time.sleep", lambda s: None)
    overloaded = anthropic.APIStatusError.__new__(anthropic.APIStatusError)
    Exception.__init__(overloaded, "Overloaded")
    overloaded.status_code, overloaded.type = 200, "overloaded_error"
    args = {"relevant": True}
    client = FakeAnthropicClient(
        _claude_msg([_tool_use(args, "triage_result")], stop=None),
        overloaded, httpx2.RemoteProtocolError("peer closed connection"),
        _claude_msg([_tool_use(args, "triage_result")]))
    backend = _anthropic_backend(client)
    assert backend.call_tool("claude-haiku-5-5", "s", "t", TRIAGE_TOOL,
                             2048) == args
    assert len(client.requests) == 4


def test_anthropic_counts_tokens_of_every_attempt():
    usage = SimpleNamespace(input_tokens=500, output_tokens=40, iterations=[
        SimpleNamespace(input_tokens=1000, output_tokens=300),
        SimpleNamespace(input_tokens=1000, output_tokens=40)])
    msg = SimpleNamespace(content=[_tool_use({"relevant": True})],
                          stop_reason="tool_use", usage=usage)
    backend = _anthropic_backend(FakeAnthropicClient(msg))
    backend.call_tool("claude-sonnet-5-5", "s", "t", EXTRACTION_TOOL, 16000)
    assert backend.usage == {"input": 2000, "output": 340, "calls": 1}


def test_1x_style_subclasses_of_the_claude_backend_keep_working():
    import anthropic

    from alpminer.extract import _call

    class Gateway(AnthropicBackend):          # 1.0 contract, own __init__
        def __init__(self, client):
            self._anthropic = anthropic
            self._client = client

        def call_tool(self, model, system, user_text, tool, max_tokens):
            return super().call_tool(model, system, user_text, tool,
                                     max_tokens)

    args = {"relevant": True, "records": []}
    gw = Gateway(FakeAnthropicClient(_claude_msg([_tool_use(args)])))
    assert _call(gw, "claude-sonnet-5-5", "s", "t", EXTRACTION_TOOL, 100,
                 "low") == args

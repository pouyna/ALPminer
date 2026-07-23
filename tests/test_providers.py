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
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fake")
    cfg.provider = "openai"
    backend = get_backend(cfg)
    assert isinstance(backend, OpenAICompatibleBackend)
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

"""LLM backends. Every provider is called through one interface --
``Backend.call_tool`` -- so the rest of the pipeline (extract.py) never
branches on which provider is configured. Backends record the token usage
each response reports (``Backend.record_usage``) so runs can log real spend.

Adding another provider means adding one class here that implements
``call_tool`` and registering it in ``get_backend``; nothing else changes.
The built-in backends also accept an ``effort`` keyword (low ... max) and map
it to the provider's reasoning control; plugins keep the five-argument
contract and never receive it.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .config import Config
from .utils import log, with_retries


class LLMError(RuntimeError):
    """Raised for any unrecoverable LLM call failure (missing key, bad
    schema response, output truncated, etc.)."""


class QuotaExhausted(LLMError):
    """The provider reports a hard quota that retrying within this run
    cannot fix (e.g. a free-tier requests-per-DAY cap). Extraction aborts
    the pass immediately instead of grinding through futile retries."""


TRUNCATED = ("LLM output was truncated at max_output_tokens; raise "
             "max_output_tokens in alpminer.toml and re-run.")


def _as_int(value) -> int:
    """Best-effort int for token counts (SDK mocks / missing fields -> 0)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class Backend:
    """Common interface implemented by each provider backend."""

    name: str
    supports_effort = False      # built-ins: call_tool(..., effort=...)

    def call_tool(self, model: str, system: str, user_text: str,
                 tool: dict, max_tokens: int) -> dict[str, Any]:
        """Have the model call `tool` once and return its arguments as a
        plain dict. Raises LLMError on any non-retryable failure."""
        raise NotImplementedError

    _usage_lock = threading.Lock()   # workers share one backend instance

    def record_usage(self, input_tokens, output_tokens) -> None:
        """Accumulate real token usage reported by the provider. Cached
        responses never reach a backend, so this reflects actual spend.
        Thread-safe: parallel extraction calls this from worker threads."""
        with Backend._usage_lock:
            u = getattr(self, "usage", None)
            if u is None:
                u = {"input": 0, "output": 0, "calls": 0}
                self.usage = u
            u["input"] += _as_int(input_tokens)
            u["output"] += _as_int(output_tokens)
            u["calls"] += 1


# ---- Anthropic ----------------------------------------------------------------

_CLAUDE_VERSION = re.compile(r"^claude-(?:([a-z]+)-)?(\d+)(?:-(\d+))?")

# Server-side refusal fallback ("default" routing) for the models that take
# it on the Claude API; Claude Haiku has no fallback model.
_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5",
                    "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _anthropic_effort(model: str, effort: str | None) -> str | None:
    """The output_config.effort value for this model, or None when the model
    predates the parameter (it returns a 400 there)."""
    if not effort:
        return None
    m = _CLAUDE_VERSION.match(model or "")
    if not m:
        return effort                      # unknown naming: newer model
    family = m.group(1) or ""
    minor = int(m.group(3)) if m.group(3) and len(m.group(3)) < 3 else 0
    version = (int(m.group(2)), minor)
    if version < (4, 5) or (version == (4, 5) and family != "opus"):
        return None
    if version in ((4, 5), (4, 6)) and effort in ("xhigh", "max"):
        # Opus 4.5 stops at high; the 4.6 models have max but no xhigh
        return "max" if version == (4, 6) and effort == "max" else "high"
    return effort


def _transport_errors() -> tuple:
    """Connection failures raised while a stream is being read (the SDK
    wraps only those raised before the response starts)."""
    found = []
    for mod in ("httpx2", "httpx"):
        try:
            found.append(__import__(mod).TransportError)
        except (ImportError, AttributeError):
            pass
    return tuple(found)


_STREAM_RETRY_TYPES = {"overloaded_error", "api_error", "rate_limit_error"}


def _claude_usage(usage) -> tuple[int, int]:
    """Billed input/output tokens; after a refusal fallback the attempts are
    listed in usage.iterations and the top-level numbers cover only the last
    one."""
    iterations = getattr(usage, "iterations", None) or []
    if iterations:
        return (sum(_as_int(getattr(i, "input_tokens", 0))
                    for i in iterations),
                sum(_as_int(getattr(i, "output_tokens", 0))
                    for i in iterations))
    return (_as_int(getattr(usage, "input_tokens", 0)),
            _as_int(getattr(usage, "output_tokens", 0)))


class AnthropicBackend(Backend):
    name = "anthropic"
    supports_effort = True

    def __init__(self, api_key: str):
        import anthropic
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._no_fallback = False

    def call_tool(self, model, system, user_text, tool, max_tokens,
                  effort=None):
        a = self._anthropic
        params = {"model": model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user_text}],
                  "tools": [tool]}
        level = _anthropic_effort(model, effort)
        if level:
            params["output_config"] = {"effort": level}
        # tool_choice stays "auto": the newest Claude models reject a forced
        # tool call (400), and the profile prompts already name the tool

        def _use_fallback() -> bool:
            return (model in _FALLBACK_MODELS
                    and not getattr(self, "_no_fallback", False))

        transport_errors = _transport_errors()

        def _once():
            try:
                if _use_fallback():
                    with self._client.beta.messages.stream(
                            **params, betas=[_FALLBACK_BETA],
                            fallbacks="default") as stream:
                        message = stream.get_final_message()
                else:
                    with self._client.messages.stream(**params) as stream:
                        message = stream.get_final_message()
            except a.APIStatusError as exc:
                # an error event inside an HTTP-200 stream (e.g. overloaded)
                if (exc.status_code == 200
                        and getattr(exc, "type", None) in _STREAM_RETRY_TYPES):
                    raise _Retry(exc) from exc
                raise
            except transport_errors as exc:      # connection dropped mid-stream
                raise _Retry(exc) from exc
            if message.stop_reason is None:      # stream ended without a stop
                raise _Retry("the response stream ended early")
            return message

        retry_on = (_Retry,) + tuple(c for c in (
            getattr(a, n, None) for n in (
                "RateLimitError", "APIConnectionError", "InternalServerError",
                "OverloadedError", "ServiceUnavailableError",
                "DeadlineExceededError")) if isinstance(c, type))

        def _send():
            try:
                return with_retries(
                    _once, desc=f"Anthropic call ({model})", attempts=5,
                    base_delay=5.0, retry_on=retry_on,
                    give_up_on=(a.BadRequestError, a.AuthenticationError,
                                a.PermissionDeniedError, a.NotFoundError))
            except a.BadRequestError as exc:
                if _use_fallback() and "fallback" in str(exc).lower():
                    log.warning("Anthropic rejected the refusal fallback "
                                "(%s); continuing without it.", exc)
                    self._no_fallback = True
                    return _send()
                raise LLMError(f"Anthropic API error: {exc}") from exc

        for attempt in (1, 2):
            resp = _send()
            self.record_usage(*_claude_usage(getattr(resp, "usage", None)))
            if resp.stop_reason == "refusal":
                details = getattr(resp, "stop_details", None)
                category = getattr(details, "category", None) or "unspecified"
                raise LLMError(f"the model declined this paper (refusal, "
                               f"category {category})")
            if resp.stop_reason == "max_tokens":
                raise LLMError(TRUNCATED)
            if resp.stop_reason == "tool_use":
                # after a refusal fallback, blocks before the last "fallback"
                # block belong to the declined (unfinished) attempt
                blocks = list(resp.content)
                cut = max((i for i, b in enumerate(blocks)
                           if getattr(b, "type", None) == "fallback"),
                          default=-1)
                for block in blocks[cut + 1:]:
                    if (getattr(block, "type", None) == "tool_use"
                            and getattr(block, "name", None) == tool["name"]):
                        return dict(block.input)
            if attempt == 1:
                log.info("%s answered without calling %s; asking once more.",
                         model, tool["name"])
        raise LLMError("model returned no tool_use block")


# ---- Gemini ---------------------------------------------------------------------

def _jsonschema_to_gemini(schema: Any) -> Any:
    """Convert the JSON-Schema dialect used in schema.py (nullable fields
    written as e.g. {"type": ["string", "null"]}) into the OpenAPI-style
    subset Gemini's FunctionDeclaration accepts (a single `type` plus a
    `nullable` flag). Recurses through object properties and array items.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    t = out.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        out["type"] = non_null[0] if non_null else "string"
        if "null" in t:
            out["nullable"] = True
    if "properties" in out:
        out["properties"] = {k: _jsonschema_to_gemini(v)
                             for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _jsonschema_to_gemini(out["items"])
    return out


_GEMINI_LEVELS = {"low": "low", "medium": "medium", "high": "high",
                  "xhigh": "high", "max": "high"}


def _gemini_thinking_level(model: str, effort: str | None) -> str | None:
    """thinking_level for Gemini 3 and later; 2.5 models keep their own
    thinking budget (they predate thinking levels)."""
    if effort and re.match(r"^gemini-(?:[3-9]|\d{2,})", model or ""):
        return _GEMINI_LEVELS.get(effort)
    return None


class GeminiBackend(Backend):
    name = "gemini"
    supports_effort = True

    def __init__(self, api_key: str):
        from google import genai
        from google.genai import errors, types
        self._types = types
        self._errors = errors
        self._client = genai.Client(api_key=api_key)

    def call_tool(self, model, system, user_text, tool, max_tokens,
                  effort=None):
        types_, errors_ = self._types, self._errors
        declaration = types_.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters=_jsonschema_to_gemini(tool["input_schema"]),
        )
        extra = {}
        level = _gemini_thinking_level(model, effort)
        if level:
            extra["thinking_config"] = types_.ThinkingConfig(
                thinking_level=level)
        config = types_.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=[types_.Tool(function_declarations=[declaration])],
            tool_config=types_.ToolConfig(
                function_calling_config=types_.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[tool["name"]],
                )
            ),
            **extra,
        )

        def _once():
            return self._client.models.generate_content(
                model=model, contents=user_text, config=config,
            )

        def _retryable(exc: BaseException) -> bool:
            code = getattr(exc, "code", None)
            return isinstance(exc, errors_.ServerError) or code == 429

        def _daily_quota(exc: BaseException) -> bool:
            # free-tier DAILY caps (e.g. quotaId GenerateRequestsPerDay...)
            # cannot be waited out inside a run; retrying is futile
            return (getattr(exc, "code", None) == 429
                    and "PerDay" in str(exc))

        def _call():
            try:
                return _once()
            except errors_.APIError as exc:
                if _daily_quota(exc):
                    raise QuotaExhausted(
                        f"Gemini daily request quota exhausted for {model} "
                        "(the free tier allows only a small number of "
                        "requests per day per model). Switch the provider "
                        "or model in Settings, or re-run tomorrow; "
                        "completed papers are saved.") from exc
                if _retryable(exc):
                    raise _Retry(exc) from exc
                raise LLMError(f"Gemini API error {exc.code}: "
                               f"{exc.message}") from exc

        resp = with_retries(_call, desc=f"Gemini call ({model})",
                            attempts=5, base_delay=5.0,
                            retry_on=(_Retry,))

        meta = getattr(resp, "usage_metadata", None)
        # thinking tokens are billed as output (and count toward the cap)
        self.record_usage(getattr(meta, "prompt_token_count", 0),
                          _as_int(getattr(meta, "candidates_token_count", 0))
                          + _as_int(getattr(meta, "thoughts_token_count", 0)))
        candidates = getattr(resp, "candidates", None) or []
        if candidates and candidates[0].finish_reason == "MAX_TOKENS":
            raise LLMError(TRUNCATED)
        calls = getattr(resp, "function_calls", None)
        if not calls:
            raise LLMError("model returned no function call")
        return dict(calls[0].args)


class _Retry(RuntimeError):
    """Internal signal: wrap a retryable provider error for with_retries."""


# ---- OpenAI-compatible (Ollama, LM Studio, vLLM, DeepSeek, ...) -----------------

class OpenAICompatibleBackend(Backend):
    """Talks to any /v1/chat/completions endpoint with function calling.

    Configured through [provider_settings] in alpminer.toml:
        base_url    default "http://localhost:11434/v1" (Ollama); point it
                    at any OpenAI-style endpoint (LM Studio, vLLM, a
                    hosted service, ...)
        api_key_env name of the env var holding the key (optional for
                    local servers; default "OPENAI_API_KEY")
    """

    name = "openai_compatible"
    default_base_url = "http://localhost:11434/v1"
    default_timeout_s = 300

    def __init__(self, settings: dict):
        import requests
        self._requests = requests
        self.base_url = (settings.get("base_url")
                         or self.default_base_url).rstrip("/")
        env_name = settings.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.environ.get(env_name, "")
        self.timeout = float(settings.get("timeout_s",
                                          self.default_timeout_s))
        if not self.api_key and "localhost" not in self.base_url \
                and "127.0.0.1" not in self.base_url:
            raise LLMError(
                f"{env_name} is not set (needed for {self.base_url}). "
                "Set it in your environment, or point base_url at a local "
                "server that needs no key."
            )

    def _post(self, path: str, payload: dict, model: str) -> dict:
        rq = self._requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def _call():
            try:
                resp = rq.post(f"{self.base_url}{path}", json=payload,
                               headers=headers, timeout=self.timeout)
            except rq.RequestException as exc:
                raise _Retry(exc) from exc
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _Retry(f"HTTP {resp.status_code}")
            if resp.status_code >= 400:
                raise LLMError(f"provider error {resp.status_code}: "
                               f"{resp.text[:300]}")
            return resp.json()

        return with_retries(_call, desc=f"{self.name} call ({model})",
                            attempts=5, base_delay=5.0, retry_on=(_Retry,))

    def call_tool(self, model, system, user_text, tool, max_tokens):
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user_text}],
            "tools": [{"type": "function",
                       "function": {"name": tool["name"],
                                    "description": tool["description"],
                                    "parameters": tool["input_schema"]}}],
            "tool_choice": {"type": "function",
                            "function": {"name": tool["name"]}},
        }
        data = self._post("/chat/completions", payload, model)
        usage = data.get("usage") or {} if isinstance(data, dict) else {}
        self.record_usage(usage.get("prompt_tokens", 0),
                          usage.get("completion_tokens", 0))
        try:
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMError(TRUNCATED)
            calls = choice["message"].get("tool_calls") or []
            if not calls:
                raise LLMError("model returned no tool call")
            return json.loads(calls[0]["function"]["arguments"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected provider response shape: {exc}") \
                from exc


# ---- OpenAI (Responses API) -------------------------------------------------------

_OPENAI_REASONING = re.compile(r"^(?:o\d|gpt-(?:[5-9]|\d{2,})(?![\d]))")


def _openai_effort(model: str, effort: str | None) -> str | None:
    """reasoning.effort for OpenAI reasoning models (o-series, GPT-5 and
    later), or None for models without it (gpt-4o, gpt-4.1)."""
    if not effort or not _OPENAI_REASONING.match(model or ""):
        return None
    if re.match(r"^(?:o\d|gpt-5(?:-|$))", model) and effort in ("xhigh",
                                                               "max"):
        return "high"                    # first-generation levels stop here
    return effort


class OpenAIBackend(OpenAICompatibleBackend):
    """OpenAI through the Responses API (POST /responses). GPT-6.1 Sol and
    GPT-6 Astra call functions only there, and Chat Completions allows
    function calls on GPT-6 Luna only with reasoning switched off, so
    provider "openai" uses this endpoint; openai_compatible keeps Chat
    Completions for local and third-party servers."""

    name = "openai"
    supports_effort = True
    default_base_url = "https://api.openai.com/v1"
    default_timeout_s = 600

    def call_tool(self, model, system, user_text, tool, max_tokens,
                  effort=None):
        payload = {
            "model": model,
            "instructions": system,
            "input": user_text,
            "tools": [{"type": "function", "name": tool["name"],
                       "description": tool["description"],
                       "parameters": tool["input_schema"],
                       "strict": False}],
            "tool_choice": {"type": "function", "name": tool["name"]},
            "parallel_tool_calls": False,
            "max_output_tokens": max_tokens,
            "store": False,
        }
        level = _openai_effort(model, effort)
        if level:
            payload["reasoning"] = {"effort": level}
        data = self._post("/responses", payload, model)
        if not isinstance(data, dict):
            raise LLMError("unexpected provider response shape")
        usage = data.get("usage") or {}
        self.record_usage(usage.get("input_tokens", 0),
                          usage.get("output_tokens", 0))
        status = data.get("status")
        if status == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            if reason == "max_output_tokens":
                raise LLMError(TRUNCATED)
            raise LLMError(f"response incomplete ({reason})")
        if status == "failed":
            raise LLMError(f"response failed: {data.get('error')}")
        for item in data.get("output") or []:
            if (isinstance(item, dict) and item.get("type") == "function_call"
                    and item.get("name") == tool["name"]):
                try:
                    return json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    raise LLMError(f"function arguments were not valid "
                                   f"JSON: {exc}") from exc
        raise LLMError("model returned no function call")


# ---- project plugins ----------------------------------------------------------

PLUGIN_DIR_NAME = "llm_providers"

PLUGIN_TEMPLATE = '''\
"""Custom alpminer LLM provider: {name}

Set  provider = "{name}"  in alpminer.toml to use it. Values placed under
[provider_settings] in alpminer.toml arrive here as the `settings` dict.

Contract: create_backend(settings) returns an object with one method,

    call_tool(model, system, user_text, tool, max_tokens) -> dict

which must have the model call `tool` (a JSON-schema function spec with
"name", "description", "input_schema") and return the call's arguments as a
plain dict. Raise alpminer.providers.LLMError for unrecoverable failures.
The pipeline handles caching, ret/next-run retry, and validation.
"""

from alpminer.providers import Backend, LLMError


class {cls}(Backend):
    name = "{name}"

    def __init__(self, settings: dict):
        self.settings = settings
        # e.g. read settings.get("base_url"), an env var for the key, etc.

    def call_tool(self, model, system, user_text, tool, max_tokens):
        raise LLMError("implement {name}.call_tool()")


def create_backend(settings: dict) -> Backend:
    return {cls}(settings)
'''


def _load_plugin(base_dir, name: str, settings: dict) -> Backend:
    import importlib.util
    path = Path(base_dir) / PLUGIN_DIR_NAME / f"{name}.py"
    if not path.exists():
        raise LLMError(
            f"unknown provider {name!r}: not a built-in "
            f"(anthropic, openai, gemini, openai_compatible) and no plugin "
            f"at {path}. Scaffold one with `alpminer providers new {name}`."
        )
    spec = importlib.util.spec_from_file_location(
        f"alpminer_plugin_{name}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - report plugin bugs clearly
        raise LLMError(f"provider plugin {path} failed to load: "
                       f"{type(exc).__name__}: {exc}") from exc
    factory = getattr(module, "create_backend", None)
    if not callable(factory):
        raise LLMError(f"provider plugin {path} must define "
                       "create_backend(settings) -> Backend")
    backend = factory(settings)
    if not hasattr(backend, "call_tool"):
        raise LLMError(f"plugin {name!r} returned an object without "
                       "call_tool()")
    return backend


def write_new_plugin(base_dir, name: str):
    if not re.match(r"^[a-z][a-z0-9_]*$", name or ""):
        raise LLMError(f"invalid provider name {name!r} (use snake_case)")
    pdir = Path(base_dir) / PLUGIN_DIR_NAME
    pdir.mkdir(parents=True, exist_ok=True)
    dest = pdir / f"{name}.py"
    if dest.exists():
        raise LLMError(f"{dest} already exists")
    cls = "".join(part.capitalize() for part in name.split("_")) + "Backend"
    dest.write_text(PLUGIN_TEMPLATE.format(name=name, cls=cls),
                    encoding="utf-8")
    return dest


# ---- factory ----------------------------------------------------------------

def get_backend(cfg: Config) -> Backend:
    if cfg.provider == "gemini":
        if not cfg.gemini_api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and set it, e.g.\n"
                "    setx GEMINI_API_KEY \"your-key\"      (Windows)\n"
                "    export GEMINI_API_KEY=your-key      (macOS/Linux)"
            )
        log.info("Using Gemini backend (extraction=%s, triage=%s)",
                 cfg.active_extraction_model, cfg.active_triage_model)
        return GeminiBackend(cfg.gemini_api_key)

    if cfg.provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Export it in your shell, e.g.\n"
                "    export ANTHROPIC_API_KEY=sk-ant-..."
            )
        return AnthropicBackend(cfg.anthropic_api_key)

    if cfg.provider == "openai":
        # OpenAI proper: the Responses API at api.openai.com with the
        # OPENAI_API_KEY env var (both overridable via [provider_settings]).
        settings = dict(cfg.provider_settings)
        settings.setdefault("api_key_env", "OPENAI_API_KEY")
        log.info("Using OpenAI backend (extraction=%s, triage=%s)",
                 cfg.active_extraction_model, cfg.active_triage_model)
        return OpenAIBackend(settings)

    if cfg.provider == "openai_compatible":
        log.info("Using OpenAI-compatible backend (%s)",
                 cfg.provider_settings.get("base_url",
                                           "http://localhost:11434/v1"))
        return OpenAICompatibleBackend(cfg.provider_settings)

    log.info("Using plugin provider %r", cfg.provider)
    return _load_plugin(cfg.base_dir, cfg.provider, cfg.provider_settings)

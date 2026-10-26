"""Single-local-LLM provider layer. One endpoint, one model, used for every pass.

Providers: ollama (/api/chat) and openai-compatible (/v1/chat/completions).
There is no offline stand-in: with no local model configured, configuration
raises and the engine does not run. The test suite registers an in-test double
in PROVIDER_REGISTRY (tests/stub_provider.py) so unit tests never touch a
network endpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import requests

from .config import Settings


@dataclass
class LLMResponse:
    text: str
    model: str
    endpoint: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    kind: str = "generic"
    system: str = ""
    user: str = ""
    meta: dict = field(default_factory=dict)


TOOL_NAME = "epistemic_output"


def _message_text(data: dict, expect_tool: bool) -> str:
    """Tool-calling servers put the JSON in a function-call argument string;
    plain servers put it in message content. Accept both."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    for call in (message.get("tool_calls") or []):
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str) and args.strip():
            return args
        params = fn.get("parameters")
        if isinstance(params, dict) and params:
            return json.dumps(params)
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if expect_tool:
        return ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return reasoning if isinstance(reasoning, str) else ""


def _can_downgrade(payload: dict) -> bool:
    return "tools" in payload or "response_format" in payload


_has_structuring = _can_downgrade


def _schema_hint(schema: dict) -> str:
    props = schema.get("properties") or {}
    req = schema.get("required", [])
    lines = [f"- {k}: {json.dumps(_grammar_safe(props[k]), separators=(',', ':'))}"
             for k in props]
    return ("\n\n<response_contract>\nConstrained decoding is not available on this "
            "server, so the shape is spelled out here. Return exactly ONE JSON object "
            f"with these keys (required: {', '.join(req)}):\n"
            + "\n".join(lines)
            + "\nNo keys outside this list, no prose outside the JSON."
            + "\n</response_contract>")


def _downgrade(payload: dict, notes: list[str]) -> None:
    if "tools" in payload:
        payload.pop("tools")
        payload.pop("tool_choice", None)
        payload["response_format"] = {"type": "json_object"}
        notes.append("tool calling rejected; retried as json_object")
        return
    if payload.get("response_format", {}).get("type") == "json_schema":
        payload["response_format"] = {"type": "json_object"}
        notes.append("json_schema rejected; retried as json_object")
    else:
        payload.pop("response_format", None)
        notes.append("response_format rejected; retried unconstrained")


def _effective_mode(payload: dict) -> str | None:
    if "tools" in payload:
        return "tool"
    return (payload.get("response_format") or {}).get("type")


GRAMMAR_KEYS = ("type", "properties", "items", "required", "enum",
                "description", "additionalProperties")


def _grammar_safe(node):
    """Constrained-decoding grammars (llama.cpp GBNF, vLLM xgrammar) reject
    JSON-Schema keywords they cannot compile. Keep the structural keys and drop
    validation-only ones: the full schema still validates the reply in jsonx."""
    if isinstance(node, dict):
        out = {}
        for key in GRAMMAR_KEYS:
            if key in node:
                if key == "properties":
                    out[key] = {name: _grammar_safe(sub)
                                for name, sub in node[key].items()}
                else:
                    out[key] = _grammar_safe(node[key])
        return out
    if isinstance(node, list):
        return [_grammar_safe(v) for v in node]
    return node


def _is_format_rejection(body: str) -> bool:
    low = body.lower()
    return any(t in low for t in ("response_format", "json_schema", "structured output",
                                  "guided decoding", "grammar", "tool", "function",
                                  "no usable content", "empty"))


class ProviderError(RuntimeError):
    pass


class ProviderBase:
    name = "base"

    def __init__(self, settings: Settings):
        self.settings = settings

    def chat(self, system: str, user: str, schema: dict | None = None,
             kind: str = "generic", temperature: float | None = None,
             max_tokens: int | None = None) -> LLMResponse:
        raise NotImplementedError

    def stream_chat(self, system: str, user: str, schema: dict | None = None,
                    kind: str = "generic", temperature: float | None = None,
                    max_tokens: int | None = None):
        """Yield text deltas. The default degrades to a single delta from
        chat(), so every provider works with the streaming endpoint."""
        yield self.chat(system, user, schema=schema, kind=kind,
                        temperature=temperature, max_tokens=max_tokens).text

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        return None

    def health(self) -> dict:
        return {"provider": self.name, "ok": False, "detail": "not implemented"}


class OllamaProvider(ProviderBase):
    name = "ollama"

    def _url(self, path: str) -> str:
        return self.settings.base_url.rstrip("/") + path

    def chat(self, system, user, schema=None, kind="generic", temperature=None,
             max_tokens=None) -> LLMResponse:
        payload = {
            "model": self.settings.model,
            "stream": False,
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.settings.temperature if temperature is None else temperature,
                "num_predict": self.settings.max_tokens if max_tokens is None else max_tokens,
            },
        }
        if schema is not None:
            payload["format"] = schema
        start = time.perf_counter()
        try:
            resp = requests.post(self._url("/api/chat"), json=payload,
                                 timeout=self.settings.request_timeout)
        except requests.RequestException as exc:
            raise ProviderError(f"ollama request failed: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        if resp.status_code >= 400:
            raise ProviderError(f"ollama HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self.settings.model),
            endpoint=self._url("/api/chat"),
            latency_ms=latency,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            kind=kind, system=system, user=user,
            meta={"done_reason": data.get("done_reason")},
        )

    def stream_chat(self, system, user, schema=None, kind="generic",
                    temperature=None, max_tokens=None):
        payload = {
            "model": self.settings.model,
            "stream": True,
            "keep_alive": "30m",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {
                "temperature": self.settings.temperature if temperature is None
                else temperature,
                "num_predict": self.settings.max_tokens if max_tokens is None
                else max_tokens,
            },
        }
        if schema is not None:
            payload["format"] = schema
        parts: list[str] = []
        try:
            resp = requests.post(self._url("/api/chat"), json=payload, stream=True,
                                 timeout=self.settings.request_timeout)
            if resp.status_code >= 400:
                resp.close()
                yield self.chat(system, user, schema=schema, kind=kind,
                                temperature=temperature, max_tokens=max_tokens).text
                return
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                piece = (obj.get("message") or {}).get("content") or ""
                if piece:
                    parts.append(piece)
                    yield piece
                if obj.get("done"):
                    break
            resp.close()
            if not parts:
                yield self.chat(system, user, schema=schema, kind=kind,
                                temperature=temperature, max_tokens=max_tokens).text
        except (requests.RequestException, OSError):
            yield self.chat(system, user, schema=schema, kind=kind,
                            temperature=temperature, max_tokens=max_tokens).text

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        try:
            resp = requests.post(
                self._url("/api/embed"),
                json={"model": self.settings.embed_model, "input": texts},
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            emb = resp.json().get("embeddings")
            if emb and len(emb) == len(texts):
                return emb
        except requests.RequestException:
            pass
        out = []
        for t in texts:
            try:
                resp = requests.post(
                    self._url("/api/embeddings"),
                    json={"model": self.settings.embed_model, "prompt": t},
                    timeout=self.settings.request_timeout,
                )
                resp.raise_for_status()
                v = resp.json().get("embedding")
                if not v:
                    return None
                out.append(v)
            except requests.RequestException:
                return None
        return out

    def health(self) -> dict:
        try:
            resp = requests.get(self._url("/api/tags"), timeout=5)
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {"provider": self.name, "ok": True, "base_url": self.settings.base_url,
                    "model": self.settings.model, "model_present": self.settings.model in models,
                    "available": models[:20]}
        except requests.RequestException as exc:
            return {"provider": self.name, "ok": False, "base_url": self.settings.base_url,
                    "model": self.settings.model, "detail": str(exc)}


class OpenAICompatProvider(ProviderBase):
    name = "openai"

    BUDGET_CAP = 8000
    _tool_rejected = False

    def _url(self, path: str) -> str:
        return self.settings.base_url.rstrip("/") + path

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.settings.api_key}"}

    def _post(self, payload) -> tuple[int, str]:
        try:
            resp = requests.post(self._url("/v1/chat/completions"), json=payload,
                                 headers=self._headers(),
                                 timeout=self.settings.request_timeout)
        except requests.RequestException as exc:
            raise ProviderError(f"openai-compat request failed: {exc}") from exc
        return resp.status_code, resp.text

    def _post_stream(self, payload):
        return requests.post(self._url("/v1/chat/completions"), json=payload,
                             headers=self._headers(), stream=True,
                             timeout=self.settings.request_timeout)

    def stream_chat(self, system, user, schema=None, kind="generic",
                    temperature=None, max_tokens=None):
        """Token deltas from the SSE channel. Tool-mode arguments arrive as
        function-argument fragments: they are the JSON payload, streamed. Any
        failure on the stream degrades to one whole-text delta from chat()."""
        budget = self.settings.max_tokens if max_tokens is None else max_tokens
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.settings.temperature if temperature is None
            else temperature,
            "max_tokens": budget,
            "stream": True,
        }
        mode = self.settings.structured_outputs
        if schema is not None and mode == "tool" and not self._tool_rejected:
            payload["tools"] = [self._tool_definition(schema)]
            payload["tool_choice"] = {"type": "function",
                                      "function": {"name": TOOL_NAME}}
        elif schema is not None and mode != "off":
            payload["response_format"] = self._response_format(schema)
            if payload["response_format"].get("type") == "json_object":
                payload["messages"][1]["content"] = user + _schema_hint(schema)
        try:
            resp = self._post_stream(payload)
            if resp.status_code >= 400:
                resp.close()
                yield self.chat(system, user, schema=schema, kind=kind,
                                temperature=temperature, max_tokens=max_tokens).text
                return
            parts: list[str] = []
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                blob = raw[5:].strip()
                if blob == "[DONE]":
                    break
                try:
                    obj = json.loads(blob)
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    for tc in delta.get("tool_calls") or []:
                        piece += (tc.get("function") or {}).get("arguments") or ""
                except (ValueError, KeyError, TypeError, IndexError):
                    continue
                if piece:
                    parts.append(piece)
                    yield piece
            resp.close()
            if not parts:
                yield self.chat(system, user, schema=schema, kind=kind,
                                temperature=temperature, max_tokens=max_tokens).text
        except (requests.RequestException, OSError):
            yield self.chat(system, user, schema=schema, kind=kind,
                            temperature=temperature, max_tokens=max_tokens).text

    def chat(self, system, user, schema=None, kind="generic", temperature=None,
             max_tokens=None) -> LLMResponse:
        budget = self.settings.max_tokens if max_tokens is None else max_tokens
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": budget,
            "stream": False,
        }
        fmt_fallbacks: list[str] = []
        mode = self.settings.structured_outputs
        user_payload = user
        if schema is not None and mode == "tool" and not self._tool_rejected:
            payload["tools"] = [self._tool_definition(schema)]
            payload["tool_choice"] = {"type": "function",
                                      "function": {"name": TOOL_NAME}}
        elif schema is not None and mode != "off":
            payload["response_format"] = self._response_format(schema)
            if payload["response_format"].get("type") == "json_object":
                user_payload = user + _schema_hint(schema)
        payload["messages"][1]["content"] = user_payload
        start = time.perf_counter()
        retries = 0
        downgrades = 0
        text = ""
        data: dict = {}
        while True:
            payload["messages"][1]["content"] = user_payload
            status, body = self._post(payload)
            if status == 502 and retries < 2 and "structured_output_failed" in body:
                doubled = min(self.BUDGET_CAP, payload["max_tokens"] * 2)
                if doubled > payload["max_tokens"]:
                    payload["max_tokens"] = doubled
                    retries += 1
                    continue
            usable = status < 400
            if usable:
                try:
                    data = json.loads(body)
                    text = _message_text(data, expect_tool="tools" in payload)
                    usable = bool(text.strip())
                except (ValueError, KeyError, TypeError, IndexError):
                    data, text, usable = {}, "", False
            if (not usable and downgrades < 3
                    and (_has_structuring(payload) or "tools" in payload)
                    and (status >= 400 or _is_format_rejection(body)
                         or not text.strip())):
                downgrades += 1
                if "tools" in payload:
                    payload.pop("tools")
                    payload.pop("tool_choice", None)
                    payload["response_format"] = {"type": "json_object"}
                    self._tool_rejected = True
                    if schema is not None:
                        user_payload = user + _schema_hint(schema)
                    fmt_fallbacks.append(
                        "tool calling rejected by this endpoint; switched to json_object "
                        "with the schema spelled out in the prompt (memoised for later calls)")
                else:
                    _downgrade(payload, fmt_fallbacks)
                    if (payload.get("response_format") or {}).get("type") == "json_object" \
                            and schema is not None:
                        user_payload = user + _schema_hint(schema)
                continue
            break
        latency = (time.perf_counter() - start) * 1000
        if status >= 400:
            raise ProviderError(f"openai HTTP {status}: {body[:400]}")
        if not text.strip():
            raise ProviderError(
                "openai-compat returned no usable content: the structured "
                "response (tool call / response_format) produced empty output "
                + ("after " + "; ".join(fmt_fallbacks) if fmt_fallbacks else ""))
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", self.settings.model),
            endpoint=self._url("/v1/chat/completions"),
            latency_ms=latency,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            kind=kind, system=system, user=user,
            meta={"budget_retries": retries, "max_tokens_used": payload["max_tokens"],
                  "response_format": _effective_mode(payload),
                  "response_format_fallbacks": fmt_fallbacks,
                  "tool_calls_used": 1 if (data.get("choices") and (
                      data["choices"][0].get("message") or {}).get("tool_calls")) else 0},
        )

    def _tool_definition(self, schema: dict) -> dict:
        return {"type": "function",
                "function": {"name": TOOL_NAME,
                             "description": (
                                 "Emit the complete epistemic sidecar for this turn. "
                                 "All fields are required; write prose inside string "
                                 "values, never as bare tokens."),
                             "parameters": _grammar_safe(schema),
                             "strict": True}}

    def _response_format(self, schema: dict) -> dict:
        if self.settings.structured_outputs == "schema":
            return {"type": "json_schema",
                    "json_schema": {"name": "epistemic_schema", "strict": True,
                                    "schema": schema}}
        return {"type": "json_object"}

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        try:
            resp = requests.post(
                self._url("/v1/embeddings"),
                json={"model": self.settings.embed_model, "input": texts},
                headers=self._headers(), timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            rows = sorted(resp.json().get("data", []), key=lambda r: r.get("index", 0))
            emb = [r["embedding"] for r in rows]
            return emb if len(emb) == len(texts) else None
        except requests.RequestException:
            return None

    def health(self) -> dict:
        try:
            resp = requests.get(self._url("/v1/models"), headers=self._headers(), timeout=5)
            resp.raise_for_status()
            ids = [m.get("id") for m in resp.json().get("data", [])]
            return {"provider": self.name, "ok": True, "base_url": self.settings.base_url,
                    "model": self.settings.model, "model_present": self.settings.model in ids,
                    "available": ids[:20]}
        except requests.RequestException as exc:
            return {"provider": self.name, "ok": False, "base_url": self.settings.base_url,
                    "model": self.settings.model, "detail": str(exc)}


PROVIDER_REGISTRY: dict[str, type[ProviderBase]] = {
    "ollama": OllamaProvider,
    "openai": OpenAICompatProvider,
}


def build_provider(settings: Settings) -> ProviderBase:
    cls = PROVIDER_REGISTRY.get(settings.provider)
    if cls is None:
        raise ProviderError(
            f"unknown provider {settings.provider!r}: build_provider requires "
            f"{', '.join(sorted(PROVIDER_REGISTRY))}. No offline stand-in exists."
        )
    return cls(settings)


def describe_provider(provider: ProviderBase) -> dict:
    return {"provider": provider.name, "model": provider.settings.model,
            "base_url": provider.settings.base_url, "single_model_policy": True}

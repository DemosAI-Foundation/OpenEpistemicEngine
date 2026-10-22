"""Single local LLM endpoint (port 8082) used for every role, as isolated stateless calls."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .schemas import FieldSpec, validate_fields

DEFAULT_BASE_URL = "http://127.0.0.1:8082"

MAX_CONTEXT_TOKENS = 256000
MAX_OUTPUT_TOKENS = 45000
MIN_OUTPUT_TOKENS = 800
DEFAULT_OUTPUT_TOKENS = 16384


class LLMUnavailable(RuntimeError):
    pass


class UnparseableOutput(LLMUnavailable):
    """The role answered, but the answer could not be parsed. Carries the raw text so the caller
    can salvage a partial packet instead of failing the case."""

    def __init__(self, role: str, reason: str = "", raw: str = "", truncated: bool = False, attempts: int = 1):
        super().__init__(f"role {role} failed: {reason}")
        self.role = role
        self.reason = reason
        self.raw = raw
        self.truncated = truncated
        self.attempts = attempts


class StreamAbort(Exception):
    """Raised by a stream watcher to hard-abort generation and keep what was streamed."""

    def __init__(self, reason: str = "", markers: list[str] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.markers = markers or []


@dataclass
class CallRecord:
    role: str
    mode: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    attempts: int = 1
    error: str = ""
    streamed: bool = False
    negative_prompt: bool = False
    aborted: bool = False


@dataclass
class RoleResult:
    fields: dict[str, Any]
    record: CallRecord
    raw: str = ""
    aborted: bool = False
    abort: StreamAbort | None = None


def _jsonize_schema(spec: list[FieldSpec]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for f in spec:
        if f.kind == "list":
            props[f.name] = {"type": "array", "items": {"type": "string"}}
        elif f.kind == "steps":
            props[f.name] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer"},
                        "valid": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["step", "valid"],
                    "additionalProperties": False,
                },
            }
        else:
            props[f.name] = {"type": "string"}
    return {
        "type": "object",
        "properties": props,
        "required": [f.name for f in spec if f.required],
        "additionalProperties": False,
    }


@dataclass
class LLMClient:
    base_url: str = DEFAULT_BASE_URL
    model: str = ""
    timeout: int = 300
    max_tokens: int = DEFAULT_OUTPUT_TOKENS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    context_tokens: int = MAX_CONTEXT_TOKENS
    temperature: float = 0.0
    use_schema: bool = True
    retries: int = 2
    negative_scale: float = 3.0
    thinking_reserve: float = 0.25
    log: list[CallRecord] = field(default_factory=list)
    last_abort: StreamAbort | None = None
    _model_resolved: bool = False

    def resolve_model(self) -> str:
        if self._model_resolved and self.model:
            return self.model
        try:
            data = requests.get(f"{self.base_url}/v1/models", timeout=10).json()
            ids = [m.get("id", "") for m in data.get("data", [])]
            self.model = "strata" if "strata" in ids else (ids[0] if ids else "strata")
        except Exception as exc:
            raise LLMUnavailable(f"cannot reach LLM at {self.base_url}: {exc}") from exc
        self._model_resolved = True
        if not self.model:
            raise LLMUnavailable("no model served on port 8082")
        return self.model

    def health(self) -> dict[str, Any]:
        return requests.get(f"{self.base_url}/health", timeout=10).json()

    def _post(self, payload: dict[str, Any], on_delta=None) -> tuple[str, int, int, str]:
        if on_delta is None:
            resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content")
            usage = data.get("usage") or {}
            if content is None:
                raise LLMUnavailable("model produced no content within token budget")
            return (
                str(content),
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
                str(choice.get("finish_reason") or ""),
            )
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        self.last_abort = None
        parts: list[str] = []
        ptok = ctok = 0
        finish = ""
        with requests.post(
            f"{self.base_url}/v1/chat/completions", json=stream_payload, timeout=self.timeout, stream=True
        ) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                usage = data.get("usage") or {}
                ptok = int(usage.get("prompt_tokens") or ptok)
                ctok = int(usage.get("completion_tokens") or ctok)
                choice = (data.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finish = str(choice.get("finish_reason"))
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    parts.append(str(delta))
                    try:
                        on_delta(str(delta))
                    except StreamAbort as abort:
                        self.last_abort = abort
                        break
        content = "".join(parts)
        if not content:
            raise LLMUnavailable("model produced no content within token budget")
        if not ctok:
            ctok = len(content) // 4
        return content, ptok, ctok, finish

    def _fit_budget(self, system: str, user: str, budget: int) -> int:
        room = self.context_tokens - self._prompt_estimate(system, user)
        return max(MIN_OUTPUT_TOKENS, min(budget, room, self.max_output_tokens))

    def _prompt_estimate(self, system: str, user: str) -> int:
        return (len(system) + len(user)) // 4 + 64

    def ask(
        self,
        role: str,
        system: str,
        user: str,
        spec: list[FieldSpec],
        max_tokens: int | None = None,
        on_delta=None,
        use_schema: bool | None = None,
        negative_prompt: str = "",
    ) -> RoleResult:
        model = self.resolve_model()
        estimate = self._prompt_estimate(system, user)
        if estimate > self.context_tokens:
            raise LLMUnavailable(
                f"role {role} prompt is ~{estimate} tokens, over the {self.context_tokens} token context budget"
            )
        budget = self._fit_budget(system, user, max_tokens or self.max_tokens)
        schema_on = self.use_schema if use_schema is None else use_schema
        last_error = ""
        last_raw = ""
        last_truncated = False
        for attempt in range(self.retries + 1):
            for mode in self._modes(schema_on):
                reset = getattr(on_delta, "reset", None)
                if callable(reset):
                    reset()
                payload: dict[str, Any] = {
                    "model": model,
                    "temperature": self.temperature,
                    "max_tokens": budget,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                thinking = self._thinking_budget(budget)
                if thinking:
                    payload["thinking_budget"] = thinking
                if negative_prompt:
                    payload["negative_prompt"] = negative_prompt
                    payload["neg_scale"] = self.negative_scale
                if mode == "schema":
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"{role.lower()}_packet",
                            "strict": True,
                            "schema": _jsonize_schema(spec),
                        },
                    }
                started = time.time()
                self.last_abort = None
                try:
                    raw, ptok, ctok, finish = self._post(payload, on_delta=on_delta)
                except requests.RequestException as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    continue
                except LLMUnavailable as exc:
                    last_error = str(exc)
                    budget = self._fit_budget(system, user, int(budget * 2))
                    continue
                abort = self.last_abort
                self.last_abort = None
                last_raw = raw
                truncated = finish == "length"
                last_truncated = truncated
                fields = self._parse(raw, spec, mode)
                if fields is None and truncated:
                    fields = self._parse(raw, spec, mode, partial=True)
                if fields is not None:
                    cut = truncated or abort is not None
                    if truncated:
                        fields["_partial"] = True
                    rec = CallRecord(
                        role=role,
                        mode=mode,
                        prompt_tokens=ptok,
                        completion_tokens=ctok,
                        seconds=round(time.time() - started, 2),
                        attempts=attempt + 1,
                        streamed=on_delta is not None,
                        negative_prompt=bool(negative_prompt),
                        aborted=cut,
                        error=("output cut at the token budget: partial packet kept" if truncated
                               else (abort.reason if abort is not None else "")),
                    )
                    self.log.append(rec)
                    return RoleResult(fields=fields, record=rec, raw=raw, aborted=cut, abort=abort)
                last_error = f"unparseable {mode} output" + (" cut at the token budget" if truncated else "")
                if truncated:
                    break
            if last_truncated:
                break
            budget = self._fit_budget(system, user, int(budget * 1.5))
        rec = CallRecord(
            role=role,
            mode="failed",
            seconds=0.0,
            attempts=self.retries + 1,
            streamed=on_delta is not None,
            negative_prompt=bool(negative_prompt),
            aborted=last_truncated,
            error=last_error,
        )
        self.log.append(rec)
        raise UnparseableOutput(role, last_error, raw=last_raw, truncated=last_truncated, attempts=self.retries + 1)

    def _thinking_budget(self, budget: int) -> int:
        """Reserve output tokens for the answer so a long internal monologue cannot consume the whole budget."""
        if self.thinking_reserve <= 0:
            return 0
        room = int(budget * (1.0 - self.thinking_reserve))
        return room if room >= 256 else 0

    def _modes(self, use_schema: bool | None = None) -> list[str]:
        schema_on = self.use_schema if use_schema is None else use_schema
        return ["schema", "text"] if schema_on else ["text"]

    def _parse(self, raw: str, spec: list[FieldSpec], mode: str, partial: bool = False) -> dict[str, Any] | None:
        if mode == "schema":
            try:
                data = json.loads(_extract_json(raw))
            except json.JSONDecodeError:
                if not partial:
                    return None
                data = _salvage_truncated_json(raw)
                if data is None:
                    return None
            return validate_fields(data, spec, partial=partial)
        return parse_field_text(raw, spec, partial=partial)


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw.strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return raw


def _salvage_truncated_json(raw: str) -> dict[str, Any] | None:
    """A schema call cut mid-generation is not lost: drop the trailing incomplete member, close the
    open containers, and parse what is there."""
    start = raw.find("{")
    if start == -1:
        return None
    body = raw[start:]
    stack: list[str] = []
    in_str = False
    escape = False
    last_complete = -1
    for i, ch in enumerate(body):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if not stack:
                try:
                    return json.loads(body[: i + 1])
                except json.JSONDecodeError:
                    return None
        elif ch == "," and len(stack) == 1:
            last_complete = i
    if last_complete <= 0:
        return None
    frag = body[:last_complete]
    opens: list[str] = []
    in_str = False
    escape = False
    for ch in frag:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            opens.append(ch)
        elif ch in "}]":
            if opens:
                opens.pop()
    closers = "".join("}" if c == "{" else "]" for c in reversed(opens))
    try:
        data = json.loads(frag + closers)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _header_lookup(spec: list[FieldSpec]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for f in spec:
        variants = {f.name, f.name.replace("_", " "), f.name.replace("_", " ")}
        for variant in variants:
            lookup[" ".join(variant.upper().split())] = f.name
    return lookup


def _clean(value: str) -> str:
    return value.strip().strip("*_").strip().strip(":").strip()


def parse_field_text(raw: str, spec: list[FieldSpec], partial: bool = False) -> dict[str, Any] | None:
    lookup = _header_lookup(spec)
    sections: dict[str, list[str]] = {}
    current = ""
    for line in (raw or "").splitlines():
        hit = re.match(r"^\s*(?:[-*#>\s]*)?([A-Za-z][A-Za-z0-9_ /-]{2,})\s*[:\u2192]\s*(.*)$", line)
        if hit:
            name = lookup.get(" ".join(hit.group(1).upper().split()))
            if name:
                current = name
                sections.setdefault(current, [])
                rest = _clean(hit.group(2))
                if rest:
                    sections[current].append(rest)
                continue
        if current:
            text = _clean(line)
            if text:
                sections[current].append(text)
    out: dict[str, Any] = {}
    for f in spec:
        lines = sections.get(f.name, [])
        if f.kind == "list":
            items: list[str] = []
            for line in lines:
                for part in re.split(r";\s*", line):
                    part = part.strip().strip(",").strip()
                    if part and part.upper() not in {"NONE", "N/A"}:
                        items.append(part)
            out[f.name] = items
        else:
            out[f.name] = " ".join(lines).strip()
    required = [f.name for f in spec if f.required]
    if required and not any(out.get(name) for name in required):
        return None
    filled = validate_fields(out, spec, partial=partial)
    if filled is None:
        return None
    filled["_format"] = "text"
    filled["_partial"] = bool(partial)
    return filled



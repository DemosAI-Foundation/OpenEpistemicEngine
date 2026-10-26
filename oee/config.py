"""Runtime configuration. One and only one local LLM may be configured.

If OEE_PROVIDER is unset the endpoint is auto-detected by probing local
servers (OpenAI-compatible first, then Ollama). Nothing is ever resolved
remotely: only loopback/local addresses on the machine are probed.
"""

from __future__ import annotations

import os

import requests
from dataclasses import dataclass, asdict


PROVIDERS = ("ollama", "openai", "mock")
PROBE_OPENAI = os.getenv("OEE_PROBE_OPENAI", "http://127.0.0.1:8082")
PROBE_OLLAMA = os.getenv("OEE_PROBE_OLLAMA", "http://127.0.0.1:11434")

_PROBE_CACHE: tuple | None = None
_UNSET = ""


def probe_endpoints(openai_url: str = PROBE_OPENAI, ollama_url: str = PROBE_OLLAMA,
                    timeout: float = 2.5, refresh: bool = False) -> tuple | None:
    """(provider, base_url, model, context_window, available_models) or None."""
    global _PROBE_CACHE
    if _PROBE_CACHE is not None and not refresh:
        return _PROBE_CACHE

    for label, base in (("openai", openai_url), ("ollama", ollama_url)):
        if not base:
            continue
        base = base.rstrip("/")
        try:
            if label == "openai":
                data = requests.get(base + "/v1/models", timeout=timeout).json()
                ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
                window = 0
                try:
                    health = requests.get(base + "/health", timeout=timeout).json()
                    window = int(health.get("max_context") or 0)
                except Exception:
                    pass
            else:
                tags = requests.get(base + "/api/tags", timeout=timeout).json()
                ids = [t.get("name") for t in tags.get("models", []) if t.get("name")]
                window = 0
            if ids:
                _PROBE_CACHE = (label, base, ids[0], window, ids)
                return _PROBE_CACHE
        except Exception:
            continue
    _PROBE_CACHE = None
    return None


def autoconfigure(settings: "Settings") -> "Settings":
    if settings.provider in PROVIDERS:
        settings.wiring = f"explicit provider={settings.provider}"
        return settings
    found = probe_endpoints(settings.probe_openai, settings.probe_ollama)
    if found:
        provider, base, model, window, ids = found
        settings.provider = provider
        settings.base_url = base
        if not os.getenv("OEE_MODEL"):
            settings.model = model
        if window and not os.getenv("OEE_CONTEXT_WINDOW"):
            settings.context_window = window
        settings.wiring = (f"auto-detected {provider} at {base}; model {settings.model}; "
                           f"models offered: {', '.join(ids)}")
    else:
        settings.provider = "mock"
        settings.wiring = (f"no local LLM answered on {settings.probe_openai} or "
                           f"{settings.probe_ollama}; using the deterministic mock stand-in")
    return settings


@dataclass
class Settings:
    provider: str = os.getenv("OEE_PROVIDER", _UNSET)
    base_url: str = os.getenv("OEE_BASE_URL", "http://127.0.0.1:11434")
    model: str = os.getenv("OEE_MODEL", "qwen2.5:32b-instruct-q5_K_M")
    api_key: str = os.getenv("OEE_API_KEY", "token-abc123")
    host: str = os.getenv("OEE_HOST", "127.0.0.1")
    port: int = int(os.getenv("OEE_PORT", "8080"))
    probe_openai: str = PROBE_OPENAI
    probe_ollama: str = PROBE_OLLAMA
    wiring: str = ""

    max_tokens: int = int(os.getenv("OEE_MAX_TOKENS", "1600"))
    pass2_max_tokens: int = int(os.getenv("OEE_PASS2_MAX_TOKENS", "4000"))
    temperature: float = float(os.getenv("OEE_TEMPERATURE", "0.2"))
    structured_outputs: str = os.getenv("OEE_STRUCTURED_OUTPUTS", "tool")
    best_of: str = os.getenv("OEE_BEST_OF", "auto")
    critic: str = os.getenv("OEE_CRITIC", "auto")
    semantic_mode: str = os.getenv("OEE_SEMANTIC", "auto")
    request_timeout: float = float(os.getenv("OEE_LLM_TIMEOUT", "300"))

    context_window: int = int(os.getenv("OEE_CONTEXT_WINDOW", "32768"))
    leaf_chunk_tokens: int = int(os.getenv("OEE_LEAF_CHUNK_TOKENS", "300"))
    leaf_chunk_overlap: int = int(os.getenv("OEE_LEAF_CHUNK_OVERLAP", "40"))
    cluster_size: int = int(os.getenv("OEE_CLUSTER_SIZE", "4"))
    leaf_k: int = int(os.getenv("OEE_LEAF_K", "8"))
    section_k: int = int(os.getenv("OEE_SECTION_K", "3"))
    rrf_k: int = int(os.getenv("OEE_RRF_K", "60"))
    embed_provider: str = os.getenv("OEE_EMBED_PROVIDER", "auto")
    embed_model: str = os.getenv("OEE_EMBED_MODEL", "nomic-embed-text")
    max_context_tokens: int = int(os.getenv("OEE_MAX_CONTEXT_TOKENS", "9000"))

    db_path: str = os.getenv("OEE_DB", os.path.join(os.getcwd(), "data", "oee.db"))
    require_single_model: bool = True

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.provider not in PROVIDERS:
            problems.append(f"provider must be one of {PROVIDERS}, got {self.provider!r}")
        if self.provider == "mock":
            problems.append(
                "provider='mock' is a deterministic offline stand-in; point "
                "OEE_BASE_URL/OEE_MODEL at your single local LLM for real use"
            )
        if self.require_single_model and self.embed_provider == "external":
            problems.append("external embedding/RAG services are forbidden by design")
        if self.leaf_chunk_overlap >= self.leaf_chunk_tokens:
            problems.append("leaf_chunk_overlap must be smaller than leaf_chunk_tokens")
        return problems

    def as_public_dict(self) -> dict:
        data = asdict(self)
        data.pop("api_key", None)
        return data


def load_settings(**overrides) -> Settings:
    settings = Settings()
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    autoconfigure(settings)
    settings.validate()
    return settings

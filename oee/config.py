"""Runtime configuration. One and only one local LLM may be configured.

If OEE_PROVIDER is unset the endpoint is auto-detected by probing local
servers (OpenAI-compatible first, then Ollama). Nothing is ever resolved
remotely: only loopback/local addresses on the machine are probed. If nothing
answers, configuration fails: the engine has no offline stand-in and refuses
to run without a real model.
"""

from __future__ import annotations

import os

import requests
from dataclasses import dataclass, asdict


PROVIDERS = ("ollama", "openai")
PROBE_OPENAI = os.getenv("OEE_PROBE_OPENAI", "http://127.0.0.1:8082")
PROBE_OLLAMA = os.getenv("OEE_PROBE_OLLAMA", "http://127.0.0.1:11434")

_PROBE_CACHE: tuple | None = None
_UNSET = ""


class ProviderConfigError(RuntimeError):
    """No single local LLM is configured: the engine refuses to run without one."""


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


def known_providers() -> tuple[str, ...]:
    """Providers the provider layer can build, including test doubles registered
    by the test suite (tests never reach a real endpoint)."""
    from .llm import PROVIDER_REGISTRY

    return tuple(PROVIDER_REGISTRY)


def autoconfigure(settings: "Settings") -> "Settings":
    if settings.provider:
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
        raise ProviderConfigError(
            "no single local LLM is configured: nothing answered on "
            f"{settings.probe_openai} (OpenAI-compatible) or {settings.probe_ollama} "
            "(Ollama). Set OEE_PROVIDER=ollama|openai with OEE_BASE_URL/OEE_MODEL, "
            "or start a local server. The engine has no offline stand-in and will "
            "not run without a model."
        )
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
    # Dense-similarity floor: below this the retrieved chunks are topically
    # unrelated to the query, and the grounding is disclosed as FALLBACK.
    # MiniLM-calibrated: unrelated pairs measure <= 0.13, relevant pairs
    # >= 0.39; tune with OEE_WEAK_GROUNDING_COSINE.
    weak_grounding_cosine: float = float(os.getenv("OEE_WEAK_GROUNDING_COSINE", "0.25"))

    db_path: str = os.getenv("OEE_DB", os.path.join(os.getcwd(), "data", "oee.db"))
    require_single_model: bool = True

    def validate(self) -> list[str]:
        problems: list[str] = []
        known = known_providers()
        if self.provider not in known:
            problems.append(f"provider must be one of {known}, got {self.provider!r}")
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

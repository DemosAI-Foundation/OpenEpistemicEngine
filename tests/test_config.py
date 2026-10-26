import pytest

from oee import config
from oee.config import Settings, autoconfigure, load_settings, probe_endpoints


def test_explicit_provider_is_respected(monkeypatch):
    monkeypatch.delenv("OEE_PROVIDER", raising=False)
    calls = []

    def fake_probe(*a, **k):
        calls.append(1)
        return ("openai", "http://127.0.0.1:8082", "swift", 262144, ["swift"])

    monkeypatch.setattr(config, "probe_endpoints", fake_probe)
    s = Settings()
    s.provider = "openai"
    s.base_url = "http://127.0.0.1:9999"
    autoconfigure(s)
    assert s.provider == "openai"
    assert s.base_url == "http://127.0.0.1:9999"
    assert s.wiring.startswith("explicit")
    assert not calls


def test_autoconfig_wires_the_answered_endpoint(monkeypatch):
    monkeypatch.delenv("OEE_MODEL", raising=False)
    monkeypatch.delenv("OEE_CONTEXT_WINDOW", raising=False)

    def fake_probe(*a, **k):
        return ("openai", "http://127.0.0.1:8082", "swift-1.5-iq3_s", 262144,
                ["swift-1.5-iq3_s", "strata"])

    monkeypatch.setattr(config, "probe_endpoints", fake_probe)
    s = Settings()
    s.provider = ""
    autoconfigure(s)
    assert s.provider == "openai"
    assert s.base_url == "http://127.0.0.1:8082"
    assert s.model == "swift-1.5-iq3_s"
    assert s.context_window == 262144
    assert "auto-detected openai" in s.wiring
    assert not s.validate()


def test_no_endpoint_refuses_to_configure(monkeypatch):
    """There is no offline stand-in: with no local model answering, configuration
    raises instead of silently running a non-model."""
    monkeypatch.setattr(config, "probe_endpoints", lambda *a, **k: None)
    s = Settings()
    s.provider = ""
    with pytest.raises(config.ProviderConfigError) as exc:
        autoconfigure(s)
    assert "no single local LLM is configured" in str(exc.value)
    assert "no offline stand-in" in str(exc.value)
    assert s.provider == ""


def test_probe_targets_are_local(monkeypatch):
    seen = []

    class Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "m-1"}]}

    def fake_get(url, timeout=None):
        seen.append(url)
        return Resp()

    monkeypatch.setattr(config, "requests", type("R", (), {"get": staticmethod(fake_get)}))
    config._PROBE_CACHE = None
    found = probe_endpoints("http://127.0.0.1:8082", "", refresh=True)
    config._PROBE_CACHE = None
    assert found[0] == "openai" and found[2] == "m-1"
    assert any(u.endswith("/v1/models") for u in seen)
    assert all(u.startswith("http://127.0.0.1:8082/") for u in seen)


def test_load_settings_never_leaves_provider_unset(monkeypatch):
    monkeypatch.setattr(config, "probe_endpoints", lambda *a, **k: None)
    with pytest.raises(config.ProviderConfigError):
        load_settings()
    s = load_settings(provider="openai", base_url="http://127.0.0.1:8082")
    assert s.provider in config.known_providers()

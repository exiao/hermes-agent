"""Live model discovery must use the active profile's secret scope."""

import json

from agent import secret_scope as ss
import hermes_cli.models as models


def test_anthropic_catalog_uses_scoped_key(monkeypatch):
    captured = {}

    class _Response:
        def read(self):
            return json.dumps({"data": [{"id": "claude-test"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(request, timeout):
        captured["headers"] = dict(request.headers)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr(
        "agent.anthropic_adapter.resolve_anthropic_token",
        lambda: "sk-ant-oat01-default-profile",
    )
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(
        {
            "ANTHROPIC_TOKEN": "   ",
            "ANTHROPIC_API_KEY": "sk-ant-api03-profile-b",
        }
    )
    try:
        assert models._fetch_anthropic_models() == ["claude-test"]
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert captured["headers"]["X-api-key"] == "sk-ant-api03-profile-b"


def test_anthropic_catalog_does_not_borrow_process_credentials(monkeypatch):
    called = False

    def _urlopen(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not probe with a process credential")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})
    monkeypatch.setattr(
        "agent.anthropic_adapter.resolve_anthropic_token",
        lambda: "sk-ant-oat01-default-profile",
    )
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({})
    try:
        assert models._fetch_anthropic_models() is None
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert called is False


def test_anthropic_catalog_uses_profile_auth_store(monkeypatch):
    captured = {}

    class _Response:
        def read(self):
            return json.dumps({"data": [{"id": "claude-profile-auth"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(request, timeout):
        captured["headers"] = dict(request.headers)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr(
        "hermes_cli.auth._load_auth_store",
        lambda: {
            "credential_pool": {
                "anthropic": [{"access_token": "sk-ant-oat01-profile-store"}]
            }
        },
    )
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({})
    try:
        assert models._fetch_anthropic_models() == ["claude-profile-auth"]
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert captured["headers"]["Authorization"] == "Bearer sk-ant-oat01-profile-store"


def test_anthropic_catalog_skips_unavailable_profile_pool_entries(monkeypatch):
    captured = {}

    class _Response:
        def read(self):
            return json.dumps({"data": [{"id": "claude-healthy"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(request, timeout):
        captured["headers"] = dict(request.headers)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr(
        "hermes_cli.auth._load_auth_store",
        lambda: {
            "credential_pool": {
                "anthropic": [
                    {"access_token": "sk-ant-oat01-dead", "last_status": "dead"},
                    {"access_token": "sk-ant-oat01-healthy"},
                ]
            }
        },
    )
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({})
    try:
        assert models._fetch_anthropic_models() == ["claude-healthy"]
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert captured["headers"]["Authorization"] == "Bearer sk-ant-oat01-healthy"


def test_ollama_catalog_uses_scoped_credentials(monkeypatch):
    captured = {}

    def _fetch(key, base_url, timeout):
        captured.update(key=key, base_url=base_url, timeout=timeout)
        return ["qwen3"]

    monkeypatch.setattr(models, "fetch_api_models", _fetch)
    monkeypatch.setattr(
        "agent.models_dev.list_agentic_models", lambda _provider: []
    )
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(
        {
            "OLLAMA_API_KEY": "ollama-profile-b-key",
            "OLLAMA_BASE_URL": "https://profile-b.example/v1",
        }
    )
    try:
        assert models.fetch_ollama_cloud_models(force_refresh=True) == ["qwen3"]
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert captured == {
        "key": "ollama-profile-b-key",
        "base_url": "https://profile-b.example/v1",
        "timeout": 8.0,
    }


def test_openai_catalog_uses_scoped_base_url_not_environ(monkeypatch):
    """OpenAI discovery must keep the scoped key and scoped base URL together."""
    captured = {}

    def _fetch(api_key, base_url, *args, **kwargs):
        captured.update(api_key=api_key, base_url=base_url)
        return ["profile-only-model"]

    monkeypatch.setattr(models, "fetch_api_models", _fetch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-default-leak")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://default.example/v1")

    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(
        {
            "OPENAI_API_KEY": "sk-profile-b",
            "OPENAI_BASE_URL": "https://profile-b.example/v1",
        }
    )
    try:
        assert models.provider_model_ids("openai-api", force_refresh=True) == [
            "profile-only-model"
        ]
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert captured == {
        "api_key": "sk-profile-b",
        "base_url": "https://profile-b.example/v1",
    }

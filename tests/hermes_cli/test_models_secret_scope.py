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


def test_named_profile_catalog_does_not_borrow_process_credentials(
    monkeypatch, tmp_path
):
    """A NAMED profile stays fail-closed: an empty scope never falls back to the
    process/Claude-file identity, so it can't borrow the gateway's catalog."""
    called = False

    def _urlopen(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not probe with a process credential")

    # Named profile home: <home>/profiles/<name> — parent dir == "profiles".
    named_home = tmp_path / "profiles" / "b"
    named_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(named_home))

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


def test_default_profile_catalog_keeps_claude_file_fallback(monkeypatch, tmp_path):
    """The DEFAULT profile keeps its live catalog under multiplexing.

    Every profile — including the default/process-owner one — runs inside a
    secret scope when multiplexing is on. An Anthropic identity that lives only
    in ~/.claude.json (Claude Code login) is NOT carried in the scope's env
    vars or auth.json pool, so a scoped-only lookup would drop the default
    profile's live catalog to the static list. The default profile owns the
    process, so it falls back to resolve_anthropic_token() (Claude-file
    auto-discovery). Named profiles do not (asserted above)."""
    captured = {}

    class _Response:
        def read(self):
            return json.dumps({"data": [{"id": "claude-default-live"}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(request, timeout):
        captured["headers"] = dict(request.headers)
        return _Response()

    # Default profile home: parent dir != "profiles".
    default_home = tmp_path / ".hermes"
    default_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})
    monkeypatch.setattr(
        "agent.anthropic_adapter.resolve_anthropic_token",
        lambda: "sk-ant-oat01-claude-file-login",
    )
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({})
    try:
        assert models._fetch_anthropic_models() == ["claude-default-live"]
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    # OAuth-shaped token routes via Bearer, proving the Claude-file token flowed.
    assert (
        captured["headers"]["Authorization"] == "Bearer sk-ant-oat01-claude-file-login"
    )


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

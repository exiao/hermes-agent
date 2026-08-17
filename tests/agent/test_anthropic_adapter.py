"""Tests for agent/anthropic_adapter.py — Anthropic Messages API adapter."""

import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.prompt_caching import apply_anthropic_cache_control, make_cache_marker
from agent.anthropic_adapter import (
    _is_azure_anthropic_endpoint,
    _is_oauth_token,
    _refresh_oauth_token,
    _resolve_anthropic_pool_token,
    _to_plain_data,
    _write_claude_code_credentials,
    build_anthropic_client,
    build_anthropic_bedrock_client,
    build_anthropic_kwargs,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    is_claude_code_token_valid,
    normalize_model_name,
    read_claude_code_credentials,
    resolve_anthropic_token,
    run_oauth_setup_token,
)
from agent.transports import get_transport


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


class TestIsOAuthToken:
    def test_setup_token(self):
        assert _is_oauth_token("sk-ant-oat01-abcdef1234567890") is True

    def test_api_key(self):
        assert _is_oauth_token("sk-ant-api03-abcdef1234567890") is False





class TestBuildAnthropicClient:


    def test_api_key_uses_api_key(self):
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant-api03-something")
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["api_key"] == "sk-ant-api03-something"
            assert "auth_token" not in kwargs
            # API key auth should still get common betas
            betas = kwargs["default_headers"]["anthropic-beta"]
            assert "interleaved-thinking-2025-05-14" in betas
            assert "context-1m-2025-08-07" not in betas
            assert "oauth-2025-04-20" not in betas  # OAuth-only beta NOT present
            assert "claude-code-20250219" not in betas  # OAuth-only beta NOT present






    def test_opencode_endpoint_gets_attribution_headers(self):
        """OpenCode identifies clients by request headers, like OpenRouter.

        The OpenAI-wire paths get HTTP-Referer / X-Title / User-Agent from
        profile.default_headers. The Anthropic Messages route builds its
        client here and must merge the same set.
        """
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "sk-opencode-secret",
                base_url="https://opencode.ai/zen/go/v1",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            headers = kwargs["default_headers"]
            assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
            assert headers["X-Title"] == "Hermes Agent"
            assert headers["User-Agent"].startswith("HermesAgent/")
            # Auth branch is unchanged: x-api-key via api_key, betas kept.
            assert kwargs["api_key"] == "sk-opencode-secret"
            assert "anthropic-beta" in headers

    def test_minimax_anthropic_endpoint_uses_bearer_auth_for_regular_api_keys(self):
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "minimax-secret-123",
                base_url="https://api.minimax.io/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["auth_token"] == "minimax-secret-123"
            assert "api_key" not in kwargs
            assert kwargs["default_headers"] == {
                "anthropic-beta": "interleaved-thinking-2025-05-14"
            }


    def test_azure_foundry_anthropic_endpoint_uses_bearer_auth(self):
        """Azure AI Foundry's /anthropic endpoint requires Authorization: Bearer.

        Regression test for #26970: without this, builds set api_key (x-api-key)
        and the endpoint returns HTTP 401. Also verifies that Azure retains the
        1M-context beta even though it now matches `_requires_bearer_auth`.
        """
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "azure-foundry-secret-123",
                base_url="https://my-resource.openai.azure.com/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["auth_token"] == "azure-foundry-secret-123"
            assert "api_key" not in kwargs
            # Azure endpoints still get the api-version query param plumbing.
            assert kwargs.get("default_query") == {"api-version": "2025-04-15"}
            # Azure keeps the 1M-context beta (it's not MiniMax).
            betas = kwargs["default_headers"]["anthropic-beta"]
            assert "context-1m-2025-08-07" in betas

    def test_proxy_api_key_header_attached_for_proxy_host(self, monkeypatch):
        """PROXY_API_KEY is sent as x-proxy-api-key only to the proxy host.

        Regression for the CPE Modal proxy: the main pipeline agents build the
        Anthropic client here (not config.anthropic_client()), so without this
        the env-driven inbound key never reaches a key-gated public proxy and
        every Anthropic agent call 401s into the fallback provider. The header
        is a proxy-admission secret, so it attaches only when the client targets
        the configured proxy host (ANTHROPIC_BASE_URL).
        """
        monkeypatch.setenv("PROXY_API_KEY", "inbound-secret")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com/anthropic")
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "sk-ant-api03-something",
                base_url="https://proxy.example.com/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["default_headers"]["x-proxy-api-key"] == "inbound-secret"
            # Must not clobber the beta header it merges onto.
            assert "anthropic-beta" in kwargs["default_headers"]

    def test_proxy_api_key_header_attached_when_base_url_omitted(self, monkeypatch):
        """Omitted base_url routes to ANTHROPIC_BASE_URL, so the header attaches.

        Regression for the env-only proxy deployment (CPE): callers authenticate
        via ANTHROPIC_TOKEN/ANTHROPIC_BASE_URL and don't pass base_url, in which
        case the Anthropic SDK routes the request to ANTHROPIC_BASE_URL itself.
        The effective target is the proxy, so the admission header must attach or
        every call 401s into the fallback.
        """
        monkeypatch.setenv("PROXY_API_KEY", "inbound-secret")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com/anthropic")
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant-api03-something")
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["default_headers"]["x-proxy-api-key"] == "inbound-secret"

    def test_proxy_api_key_header_absent_for_explicit_native_anthropic(self, monkeypatch):
        """The proxy-admission secret never leaks to an explicit native Anthropic URL.

        When the client explicitly targets api.anthropic.com (not the proxy), the
        header must be omitted so the secret isn't sent to a host that has no
        business receiving it.
        """
        monkeypatch.setenv("PROXY_API_KEY", "inbound-secret")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com/anthropic")
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "sk-ant-api03-something",
                base_url="https://api.anthropic.com",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert "x-proxy-api-key" not in (kwargs.get("default_headers") or {})

    def test_proxy_api_key_header_absent_for_unrelated_third_party(self, monkeypatch):
        """The secret is not sent to an unrelated Anthropic-compatible provider."""
        monkeypatch.setenv("PROXY_API_KEY", "inbound-secret")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com/anthropic")
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "third-party-secret",
                base_url="https://api.minimax.io/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert "x-proxy-api-key" not in (kwargs.get("default_headers") or {})

    def test_proxy_api_key_header_absent_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("PROXY_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com/anthropic")
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "sk-ant-api03-something",
                base_url="https://proxy.example.com/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert "x-proxy-api-key" not in (kwargs.get("default_headers") or {})

    def test_proxy_api_key_header_merges_without_clobbering_betas(self, monkeypatch):
        """Attaching the proxy key preserves the existing anthropic-beta header."""
        monkeypatch.setenv("PROXY_API_KEY", "inbound-secret")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com/anthropic")
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "proxy-routed-key",
                base_url="https://proxy.example.com/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["default_headers"]["x-proxy-api-key"] == "inbound-secret"
            assert "anthropic-beta" in kwargs["default_headers"]
    def test_palantir_foundry_anthropic_endpoint_uses_bearer_auth(self):
        """Palantir Foundry's LLM proxy requires Authorization: Bearer.

        Regression test for PR #36043: Palantir's
        ``<org>.palantirfoundry.com/api/v2/llm/proxy/anthropic`` endpoint
        rejects x-api-key with 401 — the SDK must be built with auth_token.
        """
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client(
                "foundry-secret-123",
                base_url="https://acme.palantirfoundry.com/api/v2/llm/proxy/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["auth_token"] == "foundry-secret-123"
            assert "api_key" not in kwargs


    def test_disables_sdk_retries_for_api_key(self):
        """#26293: the SDK's default max_retries=2 ignores Retry-After and
        double-retries inside hermes's outer loop. We delegate retry entirely
        to the outer loop, so the client must be built with max_retries=0."""
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("sk-ant-api03-something")
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["max_retries"] == 0




class TestReadClaudeCodeCredentials:
    @pytest.fixture(autouse=True)
    def no_keychain(self, monkeypatch):
        monkeypatch.setattr(
            "agent.anthropic_adapter._read_claude_code_credentials_from_keychain",
            lambda: None,
        )

    def test_reads_valid_credentials(self, tmp_path, monkeypatch):
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-token",
                "refreshToken": "sk-ant-oat01-refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        creds = read_claude_code_credentials()
        assert creds is not None
        assert creds["accessToken"] == "sk-ant-oat01-token"
        assert creds["refreshToken"] == "sk-ant-oat01-refresh"
        assert creds["source"] == "claude_code_credentials_file"

    def test_ignores_primary_api_key_for_native_anthropic_resolution(self, tmp_path, monkeypatch):
        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(json.dumps({"primaryApiKey": "sk-ant-api03-primary"}))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        creds = read_claude_code_credentials()
        assert creds is None





class TestIsClaudeCodeTokenValid:
    def test_valid_token(self):
        creds = {"accessToken": "tok", "expiresAt": int(time.time() * 1000) + 3600_000}
        assert is_claude_code_token_valid(creds) is True

    def test_expired_token(self):
        creds = {"accessToken": "tok", "expiresAt": int(time.time() * 1000) - 3600_000}
        assert is_claude_code_token_valid(creds) is False

    def test_no_expiry_but_has_token(self):
        creds = {"accessToken": "tok", "expiresAt": 0}
        assert is_claude_code_token_valid(creds) is True


class TestResolveAnthropicToken:
    def _assert_not_called(*_args, **_kwargs):
        raise AssertionError("should not be called when API key is present")

    def test_prefers_oauth_token_over_api_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-mykey")
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-mytoken")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert resolve_anthropic_token() == "sk-ant-oat01-mytoken"

    def test_does_not_resolve_primary_api_key_as_native_anthropic_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        (tmp_path / ".claude.json").write_text(json.dumps({"primaryApiKey": "sk-ant-api03-primary"}))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        assert resolve_anthropic_token() is None

    def test_falls_back_to_api_key_when_no_oauth_sources_exist(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant...ykey")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert resolve_anthropic_token() == "sk-ant...ykey"

    def test_api_key_wins_over_auto_discovered_claude_code_credentials(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant...ykey")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "cc-auto-token",
                "refreshToken": "refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        assert resolve_anthropic_token() == "sk-ant...ykey"

    def test_suppressed_claude_code_source_is_not_read(self, monkeypatch, tmp_path):
        """A user-suppressed claude_code source must not resolve, even for the
        default profile. `hermes auth remove anthropic` leaves the Claude file
        in place (Claude Code owns it) but records a suppression marker; the
        Claude-file read (source #3) must honor it rather than resurrect the
        removed credential.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        # A valid, non-expired Claude Code credential file exists on disk.
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-claudefile",
                "refreshToken": "r",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        # Without suppression it resolves the Claude file token (control).
        monkeypatch.setattr(
            "hermes_cli.auth.is_source_suppressed",
            lambda provider, source: False,
        )
        assert resolve_anthropic_token() == "sk-ant-oat01-claudefile"

        # With the claude_code source suppressed, it must NOT be read.
        monkeypatch.setattr(
            "hermes_cli.auth.is_source_suppressed",
            lambda provider, source: provider == "anthropic" and source == "claude_code",
        )
        assert resolve_anthropic_token() is None

    def test_suppressed_claude_code_source_is_not_even_read(self, monkeypatch, tmp_path):
        """Suppression must be checked BEFORE read_claude_code_credentials so a
        suppressed source never touches the Claude file / macOS Keychain at all.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "hermes_cli.auth.is_source_suppressed",
            lambda provider, source: provider == "anthropic" and source == "claude_code",
        )
        calls = {"n": 0}

        def _boom():
            calls["n"] += 1
            raise AssertionError("read_claude_code_credentials must not be called when suppressed")

        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", _boom
        )
        assert resolve_anthropic_token() is None
        assert calls["n"] == 0

    def test_returns_none_with_no_creds(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert resolve_anthropic_token() is None

    def test_api_key_path_does_not_read_auto_discovered_credentials(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant...ykey")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            self._assert_not_called,
        )

        assert resolve_anthropic_token() == "sk-ant...ykey"

    def test_falls_back_to_claude_code_credentials(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "cc-auto-token",
                "refreshToken": "refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert resolve_anthropic_token() == "cc-auto-token"

    def test_falls_back_to_anthropic_credential_pool_oauth(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        # Isolate source #5 (credential_pool): ensure source #4 (Claude Code
        # creds, incl. the macOS keychain read which Path.home does not cover)
        # returns nothing, mirroring a Hermes-PKCE-only setup.
        monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)

        pool_entry = SimpleNamespace(
            auth_type="oauth",
            access_token="pool-oauth-token",
        )
        pool = SimpleNamespace(
            _available_entries=lambda **_kwargs: ([pool_entry], []),
        )
        monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)

        assert resolve_anthropic_token() == "pool-oauth-token"

    def test_api_key_wins_over_anthropic_credential_pool_oauth(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant...ykey")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            self._assert_not_called,
        )
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            self._assert_not_called,
        )

        assert resolve_anthropic_token() == "sk-ant...ykey"

    def test_pool_entry_with_null_access_token_does_not_crash(self, monkeypatch, tmp_path):
        """A persisted OAuth entry with access_token=None must not crash the
        resolver (None.strip() would escape the helper's try/excepts and take
        down the whole resolver incl. the ANTHROPIC_API_KEY fallback). It should
        be skipped and the api-key fallback (source #3) should win."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant...ykey")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)

        broken_entry = SimpleNamespace(auth_type="oauth", access_token=None)
        pool = SimpleNamespace(
            _available_entries=lambda **_kwargs: ([broken_entry], []),
        )
        monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)

        # Must fall through to source #3 (ANTHROPIC_API_KEY), not raise.
        assert resolve_anthropic_token() == "sk-ant...ykey"

    def test_pool_api_key_only_entry_is_not_returned_as_token(self, monkeypatch, tmp_path):
        """resolve_anthropic_token() returns an OAuth bearer token; a pool entry
        whose auth_type is api_key (not oauth) must NOT be returned from the pool
        path — those are consumed via the aux client's _pool_runtime_api_key
        lane, a different resolution concern."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)

        api_key_entry = SimpleNamespace(auth_type="api_key", access_token="sk-pool-apikey")
        pool = SimpleNamespace(
            _available_entries=lambda **_kwargs: ([api_key_entry], []),
        )
        monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)

        # No OAuth entry and no other source → None (the api_key entry is ignored here).
        assert resolve_anthropic_token() is None


    def test_pool_resolution_is_read_only(self, monkeypatch, tmp_path):
        """The resolver must enumerate the pool read-only — clear_expired and
        refresh must both be False so a bare resolve never writes auth.json or
        triggers a network refresh from diagnostic call sites (#50108 MED)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)

        captured = {}
        pool_entry = SimpleNamespace(auth_type="oauth", access_token="pool-oauth-token")

        def _available_entries(**kwargs):
            captured.update(kwargs)
            return ([pool_entry], [])

        pool = SimpleNamespace(_available_entries=_available_entries)
        monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pool)

        assert resolve_anthropic_token() == "pool-oauth-token"
        assert captured == {"clear_expired": False, "refresh": False}

    def test_prefers_refreshable_claude_code_credentials_over_static_anthropic_token(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-static-token")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "cc-auto-token",
                "refreshToken": "refresh-token",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        assert resolve_anthropic_token() == "cc-auto-token"


    def test_nonmultiplex_scope_falls_back_to_service_env(
        self, monkeypatch
    ):
        from agent import secret_scope as ss

        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-SERVICE-ENV")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._resolve_anthropic_pool_token", lambda **_: None
        )
        ss.set_multiplex_active(False)
        scope_token = ss.set_secret_scope({})
        try:
            assert resolve_anthropic_token() == "sk-ant-oat01-SERVICE-ENV"
        finally:
            ss.reset_secret_scope(scope_token)

    def test_nonmultiplex_scope_secret_beats_service_env_priority(
        self, monkeypatch
    ):
        from agent import secret_scope as ss

        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-SERVICE-ENV")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: {
                "accessToken": "sk-ant-oat01-GLOBAL-CLAUDE",
                "refreshToken": "refresh-global",
                "expiresAt": 9999999999999,
            },
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid", lambda _: True
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._resolve_anthropic_pool_token",
            lambda **_: "sk-ant-oat01-GLOBAL-POOL",
        )
        ss.set_multiplex_active(False)
        scope_token = ss.set_secret_scope(
            {"ANTHROPIC_API_KEY": "sk-ant-api03-SCOPED-CRON"}
        )
        try:
            assert resolve_anthropic_token() == "sk-ant-api03-SCOPED-CRON"
        finally:
            ss.reset_secret_scope(scope_token)

    def test_nonmultiplex_scoped_oauth_keeps_refreshable_claude_preference(
        self, monkeypatch
    ):
        from agent import secret_scope as ss

        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: {
                "accessToken": "sk-ant-oat01-REFRESHABLE-CLAUDE",
                "refreshToken": "refresh-global",
                "expiresAt": 9999999999999,
            },
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid", lambda _: True
        )
        ss.set_multiplex_active(False)
        scope_token = ss.set_secret_scope(
            {"ANTHROPIC_TOKEN": "sk-ant-oat01-STATIC-SCOPED"}
        )
        try:
            assert resolve_anthropic_token() == "sk-ant-oat01-REFRESHABLE-CLAUDE"
        finally:
            ss.reset_secret_scope(scope_token)

    def test_scoped_secret_wins_over_os_environ_under_multiplexing(self, monkeypatch, tmp_path):
        """Regression for t_601f23d1: under gateway profile multiplexing the
        resolver must read ANTHROPIC_TOKEN from the active _SECRET_SCOPE (the
        requesting profile's .env), NOT the process/default-profile os.environ.

        Mirrors tests/gateway/test_credits_usage_profile_scope.py: with a scope
        installed, profile B's token wins; before the get_secret routing it
        returned the default profile's os.environ value.
        """
        from agent import secret_scope as ss

        # Default-profile value living in the process env.
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-DEFAULT")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )

        # Secondary profile B's isolated secret scope.
        token = ss.set_secret_scope({"ANTHROPIC_TOKEN": "sk-ant-oat01-PROFILE-B"})
        try:
            assert resolve_anthropic_token() == "sk-ant-oat01-PROFILE-B"
        finally:
            ss.reset_secret_scope(token)

        # No scope installed → unchanged os.environ behavior.
        assert resolve_anthropic_token() == "sk-ant-oat01-DEFAULT"

    def test_scoped_token_wins_over_global_refreshable_creds_under_multiplexing(
        self, monkeypatch, tmp_path
    ):
        """Regression for the Codex P1 on PR #99: when the host has refreshable
        Claude Code credentials for the DEFAULT profile, a scoped turn must NOT
        let those global creds override the requesting profile's scoped
        ANTHROPIC_TOKEN.

        read_claude_code_credentials() reads the global ~/.claude / Keychain
        record. Before the scope-drops-creds fix, _prefer_refreshable_claude_code_token
        resolved that refreshable global credential and returned it in place of
        the scoped token — authenticating /usage and Anthropic calls as the
        wrong profile. The earlier regression test missed this because it stubbed
        read_claude_code_credentials to None.
        """
        from agent import secret_scope as ss

        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        # Host default-profile refreshable Claude Code credential (global).
        global_creds = {
            "accessToken": "sk-ant-oat01-GLOBAL-DEFAULT",
            "refreshToken": "refresh-default",
            "expiresAt": 9999999999999,
            "source": "file",
        }
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: global_creds,
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid", lambda c: True
        )
        # Active scope belongs to a NON-default profile (profile B), so the
        # host-global ~/.claude credential is not this profile's and must be
        # suppressed. get_active_profile_name reads HERMES_HOME (overridden per
        # turn by _profile_runtime_scope); stub it to the requesting profile.
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "profileB"
        )

        # Requesting profile B carries its own scoped OAuth token.
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(
            {"ANTHROPIC_TOKEN": "sk-ant-oat01-PROFILE-B"}
        )
        try:
            assert resolve_anthropic_token() == "sk-ant-oat01-PROFILE-B"
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        # No scope → the global refreshable credential resolves as before.
        assert resolve_anthropic_token() == "sk-ant-oat01-GLOBAL-DEFAULT"

    def test_scoped_api_key_wins_over_global_creds_at_source_3_under_multiplexing(
        self, monkeypatch, tmp_path
    ):
        """Regression for the second Codex P1 on PR #99 (source-#3 gap): when a
        scoped profile has NO ANTHROPIC_TOKEN / CLAUDE_CODE_OAUTH_TOKEN — only a
        scoped ANTHROPIC_API_KEY — the global Claude Code credential must NOT be
        returned at source #3.

        Setting creds=None under scope does not suppress source #3 on its own,
        because _resolve_claude_code_token_from_credentials(None) re-reads the
        global ~/.claude file. The scoped turn must skip that fallback entirely
        so the profile's own scoped API key (source #5) resolves instead of the
        default profile's global Claude Code token.
        """
        from agent import secret_scope as ss

        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        # Host default-profile refreshable Claude Code credential (global).
        global_creds = {
            "accessToken": "sk-ant-oat01-GLOBAL-DEFAULT",
            "refreshToken": "refresh-default",
            "expiresAt": 9999999999999,
            "source": "file",
        }
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: global_creds,
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid", lambda c: True
        )
        # Active scope belongs to a NON-default profile.
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "profileB"
        )

        # Requesting profile B has only a scoped API key (no OAuth token).
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(
            {"ANTHROPIC_API_KEY": "sk-ant-api03-PROFILE-B"}
        )
        try:
            assert resolve_anthropic_token() == "sk-ant-api03-PROFILE-B"
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        # No scope → the global Claude Code credential still resolves at source #3.
        assert resolve_anthropic_token() == "sk-ant-oat01-GLOBAL-DEFAULT"

    def test_nondefault_scope_does_not_borrow_global_pool_before_scoped_api_key(
        self, monkeypatch, tmp_path
    ):
        """A named profile's scoped API key must beat the root pool fallback.

        ``read_credential_pool`` deliberately borrows root entries when a profile
        has no local entries. That sharing is useful for ordinary profile workers,
        but unsafe inside a multiplexed request whose secret scope belongs to a
        different profile: the borrowed OAuth token would shadow source #5.
        """
        from agent import secret_scope as ss
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        root = tmp_path / ".hermes"
        profile = root / "profiles" / "profileB"
        profile.mkdir(parents=True)
        (root / "auth.json").write_text(
            json.dumps(
                {
                    "credential_pool": {
                        "anthropic": [
                            {
                                "id": "root-oauth",
                                "label": "root-oauth",
                                "auth_type": "oauth",
                                "priority": 0,
                                "source": "manual",
                                "access_token": "sk-ant-oat01-GLOBAL-POOL",
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        (profile / "auth.json").write_text(
            json.dumps(
                {
                    "credential_pool": {
                        "anthropic": [
                            {
                                "id": "profile-claude",
                                "label": "profile-claude",
                                "auth_type": "oauth",
                                "priority": 0,
                                "source": "claude_code",
                                "access_token": "sk-ant-oat01-STALE-PROFILE",
                                "last_status": "dead",
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        (profile / ".anthropic_oauth.json").write_text(
            json.dumps(
                {
                    "accessToken": "sk-ant-oat01-STALE-PKCE",
                    "refreshToken": "refresh-stale",
                    "expiresAt": 9999999999999,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: {
                "accessToken": "sk-ant-oat01-GLOBAL-CLAUDE",
                "refreshToken": "refresh-global",
                "expiresAt": 9999999999999,
            },
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid", lambda c: True
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "profileB"
        )

        home_token = set_hermes_home_override(str(profile))
        scope_token = ss.set_secret_scope(
            {"ANTHROPIC_API_KEY": "sk-ant-api03-PROFILE-B"}
        )
        try:
            assert resolve_anthropic_token() == "sk-ant-api03-PROFILE-B"
        finally:
            ss.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    def test_nondefault_scope_prefers_local_manual_pool_to_scoped_api_key(
        self, monkeypatch
    ):
        from agent import secret_scope as ss

        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "profileB"
        )
        calls = []

        def _pool_token(*, profile_only=False):
            calls.append(profile_only)
            return "sk-ant-oat01-PROFILE-MANUAL"

        monkeypatch.setattr(
            "agent.anthropic_adapter._resolve_anthropic_pool_token", _pool_token
        )
        ss.set_multiplex_active(True)
        scope_token = ss.set_secret_scope(
            {"ANTHROPIC_API_KEY": "sk-ant-api03-PROFILE-B"}
        )
        try:
            assert resolve_anthropic_token() == "sk-ant-oat01-PROFILE-MANUAL"
        finally:
            ss.reset_secret_scope(scope_token)
            ss.set_multiplex_active(False)
        assert calls == [True]

    def test_default_multiplex_scope_prefers_manual_pool_to_scoped_api_key(
        self, monkeypatch
    ):
        """The DEFAULT profile under multiplexing keeps source-#4 pool precedence.

        Regression for the default-profile gap: when the multiplexed gateway
        handles the default profile whose ``.env`` sets ``ANTHROPIC_API_KEY``,
        ``scope_has_api_key`` is true while ``nondefault_scope`` is false. The
        pool-consult gate must still run ``_resolve_anthropic_pool_token`` so a
        manually-added OAuth entry retains its documented precedence over the
        scoped API key — matching both the no-scope default path and the
        non-default multiplex path. Before the fix the gate short-circuited and
        returned the API key, silently dropping the manual OAuth credential.
        """
        from agent import secret_scope as ss

        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "default"
        )
        calls = []

        def _pool_token(*, profile_only=False):
            calls.append(profile_only)
            return "sk-ant-oat01-DEFAULT-MANUAL"

        monkeypatch.setattr(
            "agent.anthropic_adapter._resolve_anthropic_pool_token", _pool_token
        )
        ss.set_multiplex_active(True)
        scope_token = ss.set_secret_scope(
            {"ANTHROPIC_API_KEY": "sk-ant-api03-DEFAULT"}
        )
        try:
            assert resolve_anthropic_token() == "sk-ant-oat01-DEFAULT-MANUAL"
        finally:
            ss.reset_secret_scope(scope_token)
            ss.set_multiplex_active(False)
        # Default profile owns ~/.hermes, so its own pool is read directly
        # (profile_only=True under the active scope), never the borrowed root.
        assert calls == [True]

    def test_nondefault_scope_preserves_profile_local_hermes_pkce(
        self, monkeypatch, tmp_path
    ):
        from agent import secret_scope as ss
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        root = tmp_path / "hermes"
        profile = root / "profiles" / "profileB"
        profile.mkdir(parents=True)
        (profile / ".anthropic_oauth.json").write_text(
            json.dumps(
                {
                    "accessToken": "sk-ant-oat01-PROFILE-PKCE",
                    "refreshToken": "refresh-profile",
                    "expiresAt": 9999999999999,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: {
                "accessToken": "sk-ant-oat01-GLOBAL-CLAUDE",
                "refreshToken": "refresh-global",
                "expiresAt": 9999999999999,
            },
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid", lambda c: True
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "profileB"
        )

        home_token = set_hermes_home_override(str(profile))
        ss.set_multiplex_active(True)
        scope_token = ss.set_secret_scope({})
        try:
            assert resolve_anthropic_token() == "sk-ant-oat01-PROFILE-PKCE"
        finally:
            ss.reset_secret_scope(scope_token)
            ss.set_multiplex_active(False)
            reset_hermes_home_override(home_token)

    def test_nondefault_scope_honors_suppressed_profile_pkce(
        self, monkeypatch, tmp_path
    ):
        from agent import secret_scope as ss
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        root = tmp_path / "hermes"
        profile = root / "profiles" / "profileB"
        profile.mkdir(parents=True)
        (profile / ".anthropic_oauth.json").write_text(
            json.dumps(
                {
                    "accessToken": "sk-ant-oat01-SUPPRESSED-PKCE",
                    "refreshToken": "refresh-suppressed",
                    "expiresAt": 9999999999999,
                }
            ),
            encoding="utf-8",
        )
        (profile / "auth.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "providers": {},
                    "suppressed_sources": {"anthropic": ["hermes_pkce"]},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "profileB"
        )

        home_token = set_hermes_home_override(str(profile))
        scope_token = ss.set_secret_scope({})
        try:
            assert resolve_anthropic_token() is None
        finally:
            ss.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    def test_default_profile_scope_still_resolves_global_claude_code_creds(
        self, monkeypatch, tmp_path
    ):
        """Regression for the Codex P2 on PR #99: under multiplexing the DEFAULT
        profile's own turns also run inside a secret scope, but the DEFAULT
        profile OWNS the host-global ~/.claude credential. The suppression must
        NOT fire for it — otherwise a default profile relying on Claude Code
        OAuth (rather than .env / pool creds) loses Anthropic auth and /usage.

        The default profile has no scoped ANTHROPIC_TOKEN / API key; its only
        Anthropic credential is the global Claude Code record at source #3, which
        must still resolve while a (default-profile) scope is active.
        """
        from agent import secret_scope as ss

        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        global_creds = {
            "accessToken": "sk-ant-oat01-GLOBAL-DEFAULT",
            "refreshToken": "refresh-default",
            "expiresAt": 9999999999999,
            "source": "file",
        }
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: global_creds,
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid", lambda c: True
        )
        # Active scope belongs to the DEFAULT profile (owner of ~/.claude).
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "default"
        )

        # Default profile's scope carries no Anthropic secret of its own.
        token = ss.set_secret_scope({})
        try:
            assert resolve_anthropic_token() == "sk-ant-oat01-GLOBAL-DEFAULT"
        finally:
            ss.reset_secret_scope(token)

    def test_default_multiplex_empty_scope_falls_back_to_service_env(
        self, monkeypatch, tmp_path
    ):
        """The default profile may get its Anthropic token from the service env.

        ``_profile_runtime_scope`` builds the default profile's secret scope from
        ``$HERMES_HOME/.env`` only. If the gateway process instead inherited
        ANTHROPIC_TOKEN from its service environment, the scope is empty while
        ``os.environ`` still owns the DEFAULT profile's credential. Multiplex
        fail-closed isolation must block that fallback only for non-default
        scopes, not for the default profile's own scope.
        """
        from agent import secret_scope as ss

        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-DEFAULT-SERVICE")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._resolve_anthropic_pool_token", lambda **_: None
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "default"
        )

        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            assert resolve_anthropic_token() == "sk-ant-oat01-DEFAULT-SERVICE"
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

    def test_default_multiplex_blank_scoped_slot_beats_service_env(
        self, monkeypatch, tmp_path
    ):
        """A profile that explicitly CLEARS an Anthropic slot must not have that
        clear overridden by the service env.

        The setup helpers write blank slots (``ANTHROPIC_TOKEN=``) to disable a
        path; load_env_file keeps them as present-but-empty scope keys. A blank
        scoped Anthropic slot means the profile opted OUT, so the default-profile
        os.environ fallback must not resurrect a service-level ANTHROPIC_TOKEN.
        Distinct from the empty-scope ({}) case above, where no slot is declared
        and the service-env fallback is legitimate.
        """
        from agent import secret_scope as ss

        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-SERVICE-SHOULD-NOT-WIN")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._resolve_anthropic_pool_token", lambda **_: None
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "default"
        )

        ss.set_multiplex_active(True)
        # Profile .env explicitly cleared the OAuth token (present but blank).
        token = ss.set_secret_scope({"ANTHROPIC_TOKEN": ""})
        try:
            assert resolve_anthropic_token() is None
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

    def test_nondefault_multiplex_empty_scope_does_not_borrow_service_env(
        self, monkeypatch, tmp_path
    ):
        """The default env fallback is not allowed for a secondary profile."""
        from agent import secret_scope as ss

        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-DEFAULT-SERVICE")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._resolve_anthropic_pool_token", lambda **_: None
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "profileB"
        )

        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            assert resolve_anthropic_token() is None
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

    def test_profile_only_pool_prunes_stale_env_oauth_under_explicit_api_key(
        self, monkeypatch, tmp_path
    ):
        """A non-default profile that switched OAuth -> API key must not let a
        stale auto-seeded env OAuth pool entry survive on the profile_only pool
        path.

        Regression: the profile_only reader dropped only ``hermes_pkce`` when an
        explicit API key was present, so an ``env:ANTHROPIC_TOKEN`` /
        ``env:CLAUDE_CODE_OAUTH_TOKEN`` OAuth singleton left in the profile
        auth.json could still be returned by source #4, reviving the Claude Code
        masquerade the API-key path opted out of. It must prune every
        auto-seeded OAuth entry (env:* + hermes_pkce + claude_code) while
        keeping manual entries.
        """
        from agent import secret_scope as ss
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        root = tmp_path / "hermes"
        profile = root / "profiles" / "profileB"
        profile.mkdir(parents=True)
        profile_auth = {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "env:ANTHROPIC_TOKEN",
                        "source": "env:ANTHROPIC_TOKEN",
                        "auth_type": "oauth",
                        "access_token": "sk-ant-oat01-STALE-ENV-OAUTH",
                        "priority": 0,
                    },
                    {
                        "id": "manual:keep",
                        "source": "manual:keep",
                        "auth_type": "api_key",
                        "access_token": "sk-ant-api-MANUAL-KEPT",
                        "priority": 1,
                    },
                ]
            },
        }
        (profile / "auth.json").write_text(json.dumps(profile_auth), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "profileB"
        )

        home_token = set_hermes_home_override(str(profile))
        ss.set_multiplex_active(True)
        # Explicit API-key path: API key set, no OAuth env tokens.
        scope_token = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-SCOPED"})
        try:
            token = _resolve_anthropic_pool_token(profile_only=True)
            # The stale env OAuth token must never be what resolves here.
            assert token != "sk-ant-oat01-STALE-ENV-OAUTH"
        finally:
            ss.reset_secret_scope(scope_token)
            ss.set_multiplex_active(False)
            reset_hermes_home_override(home_token)

    def test_profile_only_pool_manual_oauth_takes_precedence_over_seeded(
        self, monkeypatch, tmp_path
    ):
        """A manually-added OAuth entry must outrank a seeded singleton on the
        profile_only pool path.

        Regression: the profile_only branch builds CredentialPool directly to
        skip load_pool's global seeders, but that also skipped
        _normalize_pool_priorities, so a manual OAuth entry that `hermes auth
        add` appended with a LARGER priority number sorted BEHIND a seeded
        env:ANTHROPIC_TOKEN singleton and the seeded token resolved first.
        Applying the normalization restores manual-over-seeded precedence.
        """
        from agent import secret_scope as ss
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        root = tmp_path / "hermes"
        profile = root / "profiles" / "profileB"
        profile.mkdir(parents=True)
        profile_auth = {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "env:ANTHROPIC_TOKEN",
                        "source": "env:ANTHROPIC_TOKEN",
                        "auth_type": "oauth",
                        "access_token": "sk-ant-oat01-SEEDED-SINGLETON",
                        "priority": 0,
                    },
                    {
                        "id": "manual:mine",
                        "source": "manual:mine",
                        "auth_type": "oauth",
                        "access_token": "sk-ant-oat01-MANUAL-PREFERRED",
                        "priority": 1,
                    },
                ]
            },
        }
        (profile / "auth.json").write_text(json.dumps(profile_auth), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "profileB"
        )

        home_token = set_hermes_home_override(str(profile))
        ss.set_multiplex_active(True)
        # No API-key clear here: both OAuth entries survive; precedence is what's
        # under test.
        scope_token = ss.set_secret_scope({})
        try:
            token = _resolve_anthropic_pool_token(profile_only=True)
            assert token == "sk-ant-oat01-MANUAL-PREFERRED"
        finally:
            ss.reset_secret_scope(scope_token)
            ss.set_multiplex_active(False)
            reset_hermes_home_override(home_token)


class TestRefreshOauthToken:
    def test_returns_none_without_refresh_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        # Neutralize live Claude Code sources (macOS Keychain + ~/.claude file)
        # so the adopt-already-refreshed branch can't short-circuit with a real
        # credential on a dev/CI machine that happens to have Claude Code creds.
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        creds = {"accessToken": "expired", "refreshToken": "", "expiresAt": 0}
        assert _refresh_oauth_token(creds) is None

    def test_successful_refresh(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )

        creds = {
            "accessToken": "old-token",
            "refreshToken": "refresh-123",
            "expiresAt": int(time.time() * 1000) - 3600_000,
        }

        mock_response = json.dumps({
            "access_token": "new-token-abc",
            "refresh_token": "new-refresh-456",
            "expires_in": 7200,
        }).encode()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=MagicMock(
                read=MagicMock(return_value=mock_response)
            ))
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_ctx

            result = _refresh_oauth_token(creds)

        assert result == "new-token-abc"
        # Verify credentials were written back
        cred_file = tmp_path / ".claude" / ".credentials.json"
        assert cred_file.exists()
        written = json.loads(cred_file.read_text())
        assert written["claudeAiOauth"]["accessToken"] == "new-token-abc"
        assert written["claudeAiOauth"]["refreshToken"] == "new-refresh-456"

    def test_failed_refresh_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
        )
        creds = {
            "accessToken": "old",
            "refreshToken": "refresh-123",
            "expiresAt": 0,
        }

        with patch("urllib.request.urlopen", side_effect=Exception("network error")):
            assert _refresh_oauth_token(creds) is None


class TestWriteClaudeCodeCredentials:
    def test_writes_new_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        _write_claude_code_credentials("tok", "ref", 12345)
        cred_file = tmp_path / ".claude" / ".credentials.json"
        assert cred_file.exists()
        data = json.loads(cred_file.read_text())
        assert data["claudeAiOauth"]["accessToken"] == "tok"
        assert data["claudeAiOauth"]["refreshToken"] == "ref"
        assert data["claudeAiOauth"]["expiresAt"] == 12345

    def test_preserves_existing_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        cred_file = cred_dir / ".credentials.json"
        cred_file.write_text(json.dumps({"otherField": "keep-me"}))
        _write_claude_code_credentials("new-tok", "new-ref", 99999)
        data = json.loads(cred_file.read_text())
        assert data["otherField"] == "keep-me"
        assert data["claudeAiOauth"]["accessToken"] == "new-tok"

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows")
    def test_credentials_file_created_with_0o600(self, tmp_path, monkeypatch):
        """Refreshed Claude Code credentials must land on disk at 0o600.

        Regression for the TOCTOU race where ``write_text`` + ``replace``
        + post-write ``chmod`` left both the temp file and the destination
        briefly readable at the process umask (commonly 0o644). Mirrors
        the fix shipped in #19673 (google_oauth) and #21148 (mcp_oauth).
        """
        import stat as _stat
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        _write_claude_code_credentials("tok", "ref", 12345)

        cred_file = tmp_path / ".claude" / ".credentials.json"
        assert cred_file.exists()
        mode = _stat.S_IMODE(cred_file.stat().st_mode)
        assert mode == 0o600, f"creds file mode {oct(mode)} != 0o600 — TOCTOU race regressed"


class TestResolveWithRefresh:
    def test_auto_refresh_on_expired_creds(self, monkeypatch, tmp_path):
        """When cred file has expired token + refresh token, auto-refresh is attempted."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        # Set up expired creds with a refresh token
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "expired-tok",
                "refreshToken": "valid-refresh",
                "expiresAt": int(time.time() * 1000) - 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        # Mock refresh to succeed
        with patch("agent.anthropic_adapter._refresh_oauth_token", return_value="refreshed-token"):
            result = resolve_anthropic_token()

        assert result == "refreshed-token"

    def test_static_env_oauth_token_does_not_block_refreshable_claude_creds(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-expired-env-token")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "expired-claude-creds-token",
                "refreshToken": "valid-refresh",
                "expiresAt": int(time.time() * 1000) - 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("agent.anthropic_adapter._refresh_oauth_token", return_value="refreshed-token"):
            result = resolve_anthropic_token()

        assert result == "refreshed-token"


class TestRunOauthSetupToken:

    def test_returns_token_from_credential_files(self, monkeypatch, tmp_path):
        """After subprocess completes, reads credentials from Claude Code files."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)

        # Pre-create credential files that will be found after subprocess
        cred_file = tmp_path / ".claude" / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "from-cred-file",
                "refreshToken": "refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            token = run_oauth_setup_token()

        assert token == "from-cred-file"
        # Don't assert exact call count — the contract is "credentials flow
        # through", not "exactly one subprocess call". xdist cross-test
        # pollution (other tests shimming subprocess via plugins) has flaked
        # assert_called_once() in CI.
        assert mock_run.called


    def test_returns_none_when_no_creds_found(self, monkeypatch, tmp_path):
        """Returns None when subprocess completes but no credentials are found."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            token = run_oauth_setup_token()

        assert token is None



# ---------------------------------------------------------------------------
# Model name normalization
# ---------------------------------------------------------------------------


class TestNormalizeModelName:
    def test_strips_anthropic_prefix(self):
        assert normalize_model_name("anthropic/claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"




    def test_preserve_dots_for_alibaba_dashscope(self):
        """Alibaba/DashScope use dots in model names (e.g. qwen3.5-plus). Fixes #1739."""
        assert normalize_model_name("qwen3.5-plus", preserve_dots=True) == "qwen3.5-plus"
        assert normalize_model_name("anthropic/qwen3.5-plus", preserve_dots=True) == "qwen3.5-plus"
        assert normalize_model_name("qwen3.5-flash", preserve_dots=True) == "qwen3.5-flash"


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_converts_openai_to_anthropic_format(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        result = convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search the web"
        assert result[0]["input_schema"]["properties"]["query"]["type"] == "string"

    def test_empty_tools(self):
        assert convert_tools_to_anthropic([]) == []
        assert convert_tools_to_anthropic(None) == []

    def test_strips_nullable_union_from_input_schema(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "Run command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                                "default": None,
                            },
                        },
                        "required": ["command"],
                    },
                },
            }
        ]

        result = convert_tools_to_anthropic(tools)

        assert result[0]["input_schema"]["properties"]["timeout"] == {
            "type": "integer",
            "default": None,
        }
        assert result[0]["input_schema"]["required"] == ["command"]


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestConvertMessages:









    def test_strips_tool_use_when_result_not_immediately_adjacent(self):
        """A tool_use whose result appears LATER but not in the immediately
        following user message must be stripped (adjacency, #52145).

        The old logic matched tool_result ids globally across the whole
        transcript, so it would wrongly KEEP such a tool_use; Anthropic then
        400s because the result does not follow the tool_use turn. The adjacency
        rewrite only honors a result in the next user message.
        """
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_late", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "user", "content": "actually, something else"},
            {"role": "assistant", "content": "sure"},
            {"role": "tool", "tool_call_id": "tc_late", "content": "late result"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        for m in result:
            if m["role"] == "assistant" and isinstance(m["content"], list):
                assert all(b.get("type") != "tool_use" for b in m["content"]), (
                    "non-adjacent tool_use should have been stripped"
                )
        for m in result:
            if m["role"] == "user" and isinstance(m["content"], list):
                assert all(b.get("type") != "tool_result" for b in m["content"]), (
                    "orphaned late tool_result should have been stripped"
                )


    def test_system_with_cache_control(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "user", "content": "Hi"},
        ]
        system, result = convert_messages_to_anthropic(messages)
        # When cache_control is present, system should be a list of blocks
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_static_system_prefix_markers_are_preserved(self):
        messages = apply_anthropic_cache_control(
            [
                {"role": "system", "content": "stable\n\nsession context"},
                {"role": "user", "content": "Hi"},
            ],
            static_system_prefix="stable",
        )

        system, _ = convert_messages_to_anthropic(messages)

        # Use make_cache_marker() so the test tracks CACHE_TTL changes
        # (fork defaults to a 1h TTL) instead of freezing a literal dict.
        assert system == [
            {"type": "text", "text": "stable", "cache_control": make_cache_marker()},
            {
                "type": "text",
                "text": "\n\nsession context",
                "cache_control": make_cache_marker(),
            },
        ]

    def test_assistant_cache_control_blocks_are_preserved(self):
        messages = apply_anthropic_cache_control([
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Hello from assistant"},
        ])

        _, result = convert_messages_to_anthropic(messages)
        assistant_msg = next(m for m in result if m["role"] == "assistant")
        assistant_blocks = assistant_msg["content"]

        assert assistant_blocks[0]["type"] == "text"
        assert assistant_blocks[0]["text"] == "Hello from assistant"
        # W1 (cache-ttl-1h-default): default marker now carries ttl='1h'.
        # Use make_cache_marker() so the test tracks CACHE_TTL changes.
        assert assistant_blocks[0]["cache_control"] == make_cache_marker()

    def test_assistant_tool_use_cache_control_is_preserved(self):
        messages = apply_anthropic_cache_control([
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Run the tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "test_tool", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ], native_anthropic=True)

        _, result = convert_messages_to_anthropic(messages)
        assistant_msg = [m for m in result if m["role"] == "assistant"][0]
        tool_use = assistant_msg["content"][-1]

        assert tool_use["type"] == "tool_use"
        assert tool_use["id"] == "tc_1"
        assert tool_use["cache_control"] == make_cache_marker()

    def test_ordered_replay_keeps_cache_control_from_nonempty_content(self):
        """An assistant turn that interleaves signed thinking with a tool_use
        AND has preamble text carries its cache_control INSIDE ``content``
        (apply_anthropic_cache_control marks the last content block, not the
        top level). The ordered-replay branch rebuilds the message from
        ``anthropic_content_blocks`` alone, so without harvesting that marker
        the breakpoint is dropped -- and it is *burned*, because
        _can_carry_marker already spent a budget slot on this message.

        #56195 covers the blank-content shape; this is the non-empty one, which
        is what a Claude thinking+tools turn normally looks like.
        """
        preamble = "I will read a.py now."
        messages = apply_anthropic_cache_control([
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Read a.py"},
            {
                "role": "assistant",
                "content": preamble,
                "anthropic_content_blocks": [
                    {"type": "thinking", "thinking": "Need a tool.", "signature": "sig_1"},
                    {"type": "text", "text": preamble},
                    {"type": "tool_use", "id": "tc_1", "name": "test_tool", "input": {}},
                ],
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "contents"},
        ])

        _system, converted = convert_messages_to_anthropic(messages)
        assistant = next(m for m in converted if m.get("role") == "assistant")
        marked = [
            b for b in assistant["content"]
            if isinstance(b, dict) and b.get("cache_control")
        ]
        assert marked, (
            "the assistant cache breakpoint was dropped by the ordered-replay "
            "path and the budget slot is burned"
        )
        # The signed thinking block must still lead the replayed message.
        assert assistant["content"][0]["type"] == "thinking"

    def test_ordered_replay_tool_use_cache_control_is_preserved(self):
        messages = apply_anthropic_cache_control([
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Run the tool"},
            {
                "role": "assistant",
                "content": "",
                "anthropic_content_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "Need a tool.",
                        "signature": "sig_1",
                    },
                    {
                        "type": "tool_use",
                        "id": "tc_1",
                        "name": "test_tool",
                        "input": {"query": "raw"},
                    },
                ],
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"query":"redacted"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ], native_anthropic=True)

        _, result = convert_messages_to_anthropic(messages)
        assistant_msg = [m for m in result if m["role"] == "assistant"][0]
        thinking, tool_use = assistant_msg["content"]

        assert thinking["type"] == "thinking"
        assert "cache_control" not in thinking
        assert tool_use["type"] == "tool_use"
        assert tool_use["id"] == "tc_1"
        assert tool_use["input"] == {"query": "redacted"}
        assert tool_use["cache_control"] == make_cache_marker()

    def test_tool_cache_control_is_preserved_on_tool_result_block(self):
        messages = apply_anthropic_cache_control([
            {"role": "system", "content": "System prompt"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "test_tool", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ], native_anthropic=True)

        _, result = convert_messages_to_anthropic(messages)
        user_msg = next(
            m for m in result
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        tool_block = user_msg["content"][0]

        assert tool_block["type"] == "tool_result"
        assert tool_block["tool_use_id"] == "tc_1"
        assert tool_block["content"] == "result"
        # W1 (cache-ttl-1h-default): default marker now carries ttl='1h'.
        assert tool_block["cache_control"] == make_cache_marker()





    def test_empty_user_message_string_gets_placeholder(self):
        """Empty user message strings should get '(empty message)' placeholder.

        Anthropic rejects requests with empty user message content.
        Regression test for #3143 — Discord @mention-only messages.
        """
        messages = [
            {"role": "user", "content": ""},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "(empty message)"




    def test_leading_assistant_after_compaction_gets_user_turn_prepended(self):
        """The adapter backstops compactors that emit a leading assistant summary."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "[Context compaction summary] earlier work…"},
            {"role": "user", "content": "continue"},
        ]

        system, result = convert_messages_to_anthropic(messages)

        assert system == "You are helpful."
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "(empty)"}]
        assert result[1]["role"] == "assistant"
        assert any(
            m["role"] == "assistant" and "Context compaction summary" in str(m["content"])
            for m in result
        )





# ---------------------------------------------------------------------------
# Build kwargs
# ---------------------------------------------------------------------------


class TestBuildAnthropicKwargs:




    def test_reasoning_config_maps_to_manual_thinking_for_pre_4_6_models(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "think hard"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 16000
        assert kwargs["temperature"] == 1
        assert kwargs["max_tokens"] >= 16000 + 4096
        assert "output_config" not in kwargs

    def test_reasoning_config_maps_to_adaptive_thinking_for_4_6_models(self):
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "think hard"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        # Adaptive thinking + display="summarized" keeps reasoning text
        # populated in the response stream (Opus 4.7 default is "omitted").
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "high"}
        assert "budget_tokens" not in kwargs["thinking"]
        assert "temperature" not in kwargs
        assert kwargs["max_tokens"] == 4096






    def test_supports_fast_mode_predicate(self):
        """Fast mode is Opus 4.6 only — Opus 4.7 and others must be excluded.

        For Opus 4.8 the fast variant is a separate model ID
        (anthropic/claude-opus-4.8-fast) routed through the normal model
        field, NOT via the ``speed: "fast"`` request parameter. So
        ``_supports_fast_mode`` (which gates the parameter) must stay
        False for both opus-4-8 and opus-4-8-fast.
        """
        from agent.anthropic_adapter import _supports_fast_mode
        assert _supports_fast_mode("claude-opus-4-6") is True
        assert _supports_fast_mode("anthropic/claude-opus-4-6") is True
        assert _supports_fast_mode("claude-opus-4-7") is False
        assert _supports_fast_mode("claude-opus-4-8") is False
        assert _supports_fast_mode("claude-opus-4-8-fast") is False
        assert _supports_fast_mode("claude-sonnet-4-6") is False
        assert _supports_fast_mode("claude-haiku-4-5") is False
        assert _supports_fast_mode("") is False

    def test_fable_class_models_route_as_adaptive_thinking(self):
        """Invariant: unknown/new Claude models default to the modern (4.7+)
        contract — adaptive thinking, xhigh-capable, sampling-params-forbidden —
        without any per-model code change. Named models (claude-fable-5) and
        hypothetical future ones must all classify modern; only the explicit
        legacy list stays on the manual path.
        """
        from agent.anthropic_adapter import (
            _supports_adaptive_thinking,
            _supports_xhigh_effort,
            _forbids_sampling_params,
            _get_anthropic_max_output,
        )
        # New / unknown Claude models → modern contract by default.
        for m in (
            "claude-fable-5",
            "anthropic/claude-fable-5",
            "claude-saga-2",            # hypothetical future named model
            "anthropic/claude-opus-9",  # hypothetical future numbered model
        ):
            assert _supports_adaptive_thinking(m) is True, m
            assert _supports_xhigh_effort(m) is True, m
            assert _forbids_sampling_params(m) is True, m
        # 1M-context reasoning model → highest output ceiling.
        assert _get_anthropic_max_output("anthropic/claude-fable-5") == 128_000



    def test_non_claude_anthropic_models_use_manual_path(self):
        """Non-Claude Anthropic-Messages models (minimax, qwen3, glm) must not
        be misclassified as adaptive by the default-to-modern rule. Kimi is
        the deliberate exception — see test_kimi_family_uses_adaptive_path."""
        from agent.anthropic_adapter import (
            _supports_adaptive_thinking,
            _supports_xhigh_effort,
            _forbids_sampling_params,
        )
        for m in ("minimax-m2", "qwen3-max", "glm-4.6"):
            assert _supports_adaptive_thinking(m) is False, m
            assert _supports_xhigh_effort(m) is False, m
            assert _forbids_sampling_params(m) is False, m


    def test_bare_k3_coding_plan_slug_is_kimi_family(self):
        """Kimi Coding Plan serves K3 as the bare slug ``k3`` — it must be
        classified as Kimi family (adaptive thinking) even on proxied
        endpoints where only the model name is available. Lookalike
        non-Kimi names must NOT match the exact-slug rule."""
        from agent.anthropic_adapter import (
            _model_name_is_kimi_family,
            _supports_adaptive_thinking,
        )
        for m in ("k3", "K3", "moonshotai/k3", "k3.1-preview", "k3-turbo"):
            assert _model_name_is_kimi_family(m) is True, m
        assert _supports_adaptive_thinking("k3") is True
        # Prefix-lookalikes without a separator must not be swept in.
        for m in ("k30", "k3000-chat", "keras-3"):
            assert _model_name_is_kimi_family(m) is False, m

    def test_fast_mode_omitted_for_unsupported_model(self):
        """fast_mode=True on Opus 4.7 must NOT inject speed=fast (API 400s)."""
        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            fast_mode=True,
        )
        # extra_body either absent or doesn't carry "speed"
        assert "speed" not in kwargs.get("extra_body", {})
        # No fast-mode beta header should be added either
        beta_header = (kwargs.get("extra_headers") or {}).get("anthropic-beta", "")
        assert "fast-mode-2026-02-01" not in beta_header













# ---------------------------------------------------------------------------
# Model output limit lookup
# ---------------------------------------------------------------------------


class TestGetAnthropicMaxOutput:
    def test_opus_4_6(self):
        from agent.anthropic_adapter import _get_anthropic_max_output
        assert _get_anthropic_max_output("claude-opus-4-6") == 128_000









# ---------------------------------------------------------------------------
# _to_plain_data hardening
# ---------------------------------------------------------------------------


class TestToPlainData:




    def test_deep_nesting_is_capped(self):
        deep = "leaf"
        for _ in range(25):
            deep = {"nested": deep}
        result = _to_plain_data(deep)
        assert isinstance(result, dict)

    def test_plain_values_pass_through(self):
        assert _to_plain_data("hello") == "hello"
        assert _to_plain_data(42) == 42
        assert _to_plain_data(None) is None

    def test_object_with_dunder_dict(self):
        obj = SimpleNamespace(type="thinking", thinking="reason", signature="sig")
        result = _to_plain_data(obj)
        assert result == {"type": "thinking", "thinking": "reason", "signature": "sig"}


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


class TestNormalizeResponse:
    def _make_response(self, content_blocks, stop_reason="end_turn"):
        resp = SimpleNamespace()
        resp.content = content_blocks
        resp.stop_reason = stop_reason
        resp.usage = SimpleNamespace(input_tokens=100, output_tokens=50)
        return resp


    def test_tool_use_response(self):
        blocks = [
            SimpleNamespace(type="text", text="Searching..."),
            SimpleNamespace(
                type="tool_use",
                id="tc_1",
                name="search",
                input={"query": "test"},
            ),
        ]
        nr = get_transport("anthropic_messages").normalize_response(
            self._make_response(blocks, "tool_use")
        )
        assert nr.content == "Searching..."
        assert nr.finish_reason == "tool_calls"
        assert len(nr.tool_calls) == 1
        assert nr.tool_calls[0].name == "search"
        assert json.loads(nr.tool_calls[0].arguments) == {"query": "test"}

    def test_thinking_response(self):
        blocks = [
            SimpleNamespace(type="thinking", thinking="Let me reason about this..."),
            SimpleNamespace(type="text", text="The answer is 42."),
        ]
        nr = get_transport("anthropic_messages").normalize_response(self._make_response(blocks))
        assert nr.content == "The answer is 42."
        assert nr.reasoning == "Let me reason about this..."
        assert nr.provider_data["reasoning_details"] == [{"type": "thinking", "thinking": "Let me reason about this..."}]


    def test_stop_reason_mapping(self):
        block = SimpleNamespace(type="text", text="x")
        nr1 = get_transport("anthropic_messages").normalize_response(
            self._make_response([block], "end_turn")
        )
        nr2 = get_transport("anthropic_messages").normalize_response(
            self._make_response([block], "tool_use")
        )
        nr3 = get_transport("anthropic_messages").normalize_response(
            self._make_response([block], "max_tokens")
        )
        assert nr1.finish_reason == "stop"
        assert nr2.finish_reason == "tool_calls"
        assert nr3.finish_reason == "length"




# ---------------------------------------------------------------------------
# Role alternation
# ---------------------------------------------------------------------------


class TestRoleAlternation:
    def test_merges_consecutive_user_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]

    def test_preserves_proper_alternation(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["user", "assistant", "user"]


# ---------------------------------------------------------------------------
# Thinking block signature management
# ---------------------------------------------------------------------------


class TestThinkingBlockSignatureManagement:
    """Tests for the thinking block handling strategy:
    strip from old turns, preserve latest signed, downgrade unsigned."""




    def test_redacted_thinking_with_data_preserved(self):
        """Redacted thinking with 'data' field is kept on last turn."""
        messages = [
            {
                "role": "assistant",
                "content": "Response.",
                "reasoning_details": [
                    {"type": "redacted_thinking", "data": "opaque_signature_data"},
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)
        blocks = next(m for m in result if m["role"] == "assistant")["content"]
        redacted = [b for b in blocks if b.get("type") == "redacted_thinking"]
        assert len(redacted) == 1
        assert redacted[0]["data"] == "opaque_signature_data"

    def test_redacted_thinking_without_data_dropped(self):
        """Redacted thinking without 'data' is dropped — can't be validated."""
        messages = [
            {
                "role": "assistant",
                "content": "Response.",
                "reasoning_details": [
                    {"type": "redacted_thinking"},
                    # No 'data' field
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)
        blocks = result[0]["content"]
        assert not any(b.get("type") == "redacted_thinking" for b in blocks)

    def test_cache_control_stripped_from_thinking_blocks(self):
        """cache_control markers are removed from thinking/redacted_thinking blocks."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "t", "arguments": "{}"}},
                ],
                "reasoning_details": [
                    {
                        "type": "thinking",
                        "thinking": "Reasoning.",
                        "signature": "sig_1",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assistant = next(m for m in result if m["role"] == "assistant")
        for block in assistant["content"]:
            if block.get("type") in {"thinking", "redacted_thinking"}:
                assert "cache_control" not in block



    def test_multi_turn_conversation_preserves_only_last(self):
        """Full multi-turn conversation: only last assistant keeps thinking."""
        messages = [
            {"role": "user", "content": "Question 1"},
            {
                "role": "assistant",
                "content": "Answer 1",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Thought 1", "signature": "sig_1"},
                ],
            },
            {"role": "user", "content": "Question 2"},
            {
                "role": "assistant",
                "content": "Answer 2",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Thought 2", "signature": "sig_2"},
                ],
            },
            {"role": "user", "content": "Question 3"},
            {
                "role": "assistant",
                "content": "Answer 3",
                "reasoning_details": [
                    {"type": "thinking", "thinking": "Thought 3", "signature": "sig_3"},
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)

        assistants = [m for m in result if m["role"] == "assistant"]
        assert len(assistants) == 3

        # First two: no thinking blocks
        for a in assistants[:2]:
            assert not any(
                b.get("type") in {"thinking", "redacted_thinking"}
                for b in a["content"]
                if isinstance(b, dict)
            )

        # Last one: thinking preserved
        last_thinking = [
            b for b in assistants[2]["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(last_thinking) == 1
        assert last_thinking[0]["signature"] == "sig_3"




# ---------------------------------------------------------------------------
# Tool choice
# ---------------------------------------------------------------------------


class TestToolChoice:
    _DUMMY_TOOL = [
        {
            "type": "function",
            "function": {
                "name": "test",
                "description": "x",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    def test_auto_tool_choice(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hi"}],
            tools=self._DUMMY_TOOL,
            max_tokens=4096,
            reasoning_config=None,
            tool_choice="auto",
        )
        assert kwargs["tool_choice"] == {"type": "auto"}


    def test_specific_tool_choice(self):
        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hi"}],
            tools=self._DUMMY_TOOL,
            max_tokens=4096,
            reasoning_config=None,
            tool_choice="search",
        )
        assert kwargs["tool_choice"] == {"type": "tool", "name": "search"}



# ---------------------------------------------------------------------------
# max_tokens resolver — openclaw/openclaw#66664 port
# ---------------------------------------------------------------------------

from agent.anthropic_adapter import (
    _resolve_positive_anthropic_max_tokens,
    _resolve_anthropic_messages_max_tokens,
)


class TestResolvePositiveMaxTokens:
    """Unit tests for the positive-int resolver helper."""


    def test_zero_returns_none(self):
        assert _resolve_positive_anthropic_max_tokens(0) is None





    def test_nan_returns_none(self):
        assert _resolve_positive_anthropic_max_tokens(float("nan")) is None


    def test_bool_true_returns_none(self):
        # True is an int subclass but semantically never a real max_tokens value
        assert _resolve_positive_anthropic_max_tokens(True) is None
        assert _resolve_positive_anthropic_max_tokens(False) is None




class TestResolveMessagesMaxTokens:
    """Integration tests for the full Messages resolver."""

    def test_positive_requested_wins(self):
        assert _resolve_anthropic_messages_max_tokens(
            8192, "claude-opus-4-6"
        ) == 8192





    def test_sub_one_float_falls_back(self):
        # 0.5 floors to 0 -> not positive -> falls back to model ceiling
        result = _resolve_anthropic_messages_max_tokens(0.5, "claude-opus-4-6")
        assert result > 0
        assert result != 0


# ---------------------------------------------------------------------------
# convert_tools_to_anthropic — tool dedup at API boundary
# ---------------------------------------------------------------------------

class TestConvertToolsToAnthropicDedup:
    """convert_tools_to_anthropic must deduplicate tool names.

    Anthropic rejects requests with duplicate tool names.  This guard converts
    a hard failure into a warning log.  See:
    https://github.com/NousResearch/hermes-agent/issues/18478
    """

    def _make_openai_tool(self, name: str) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool {name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }


    def test_duplicate_tool_names_are_deduplicated(self):
        """RED test — must fail until dedup guard is added."""
        tools = [
            self._make_openai_tool("lcm_grep"),
            self._make_openai_tool("lcm_describe"),
            self._make_openai_tool("lcm_grep"),  # duplicate
            self._make_openai_tool("lcm_expand"),
            self._make_openai_tool("lcm_describe"),  # duplicate
        ]
        result = convert_tools_to_anthropic(tools)
        names = [t["name"] for t in result]
        assert len(names) == len(set(names)), (
            f"Duplicate tool names found: {names}"
        )
        assert len(result) == 3  # lcm_grep, lcm_describe, lcm_expand


    def test_none_tools_returns_empty(self):
        assert convert_tools_to_anthropic(None) == []


class TestBlankTextBlockFiltering:
    """Regression tests for blank text block filtering in _convert_assistant_message.

    Bedrock and strict Anthropic-compatible endpoints reject text blocks where
    "text" is empty or whitespace-only with HTTP 400. Both the normal list-
    content path and the ordered-replay fast path must drop such blocks while
    preserving tool_use and other block types, and must relocate (not lose)
    any cache_control marker attached to the dropped block.
    """

    def _convert(self, message):
        from agent.anthropic_adapter import _convert_assistant_message
        return _convert_assistant_message(message)



    def test_normal_path_filters_none_text_block_without_crashing(self):
        """Regression (review of #63228): text=None must not raise
        AttributeError. _convert_content_part_to_anthropic() can preserve
        None from an invalid upstream input text block -- a bare .strip()
        on blk.get("text", "") crashes because .get() only substitutes the
        default when the key is ABSENT, not when it's present with value None."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": None},
                {"type": "tool_use", "id": "call_none", "name": "web_search",
                 "input": {"query": "test"}},
            ],
        }
        result = self._convert(msg)  # must not raise
        blocks = result["content"]
        text_blocks = [b for b in blocks if b.get("type") == "text"]
        tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        assert len(text_blocks) == 0, f"None text block not filtered: {text_blocks}"
        assert len(tool_blocks) == 1



    def test_normal_path_relocates_cache_control_from_dropped_block(self):
        """Regression (review of #63228): prompt_caching.py's _apply_cache_marker
        sets cache_control directly on content[-1] for list content. If that
        last part is blank text, dropping it must relocate the marker to the
        surviving last cacheable block (here: the tool_use), not lose it."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll look that up."},
                {"type": "tool_use", "id": "call_cache", "name": "web_search",
                 "input": {"query": "test"}},
                {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
            ],
        }
        result = self._convert(msg)
        blocks = result["content"]
        assert not any(b.get("type") == "text" and not b.get("text", "").strip() for b in blocks), (
            "Blank text block must be dropped"
        )
        cacheable_with_marker = [b for b in blocks if isinstance(b.get("cache_control"), dict)]
        assert len(cacheable_with_marker) == 1, (
            f"cache_control marker must survive on exactly one surviving block: {blocks}"
        )
        assert cacheable_with_marker[0]["type"] == "tool_use", (
            f"Marker must relocate to the new last cacheable block: {blocks}"
        )


    def test_replay_path_relocates_cache_control_from_dropped_block(self):
        """Same cache_control-relocation guarantee on the ordered-replay path:
        a blank text block carrying cache_control (e.g. a stored, previously
        cache-marked turn where prompt_caching later becomes blank on replay)
        must not silently lose the breakpoint when dropped."""
        from agent.anthropic_adapter import _convert_assistant_message
        msg = {
            "role": "assistant",
            "content": "",
            "anthropic_content_blocks": [
                {"type": "tool_use", "id": "call_5", "name": "web_search",
                 "input": {"query": "test"}},
                {"type": "text", "text": "  ", "cache_control": {"type": "ephemeral"}},
            ],
            "tool_calls": [
                {
                    "id": "call_5",
                    "function": {"name": "web_search",
                                 "arguments": '{"query": "test"}'},
                }
            ],
        }
        result = _convert_assistant_message(msg)
        blocks = result["content"]
        assert not any(b.get("type") == "text" for b in blocks), "Blank replay text must be dropped"
        cacheable_with_marker = [b for b in blocks if isinstance(b.get("cache_control"), dict)]
        assert len(cacheable_with_marker) == 1
        assert cacheable_with_marker[0]["type"] == "tool_use"


class TestAllBlankFallbackAndNonStringText:
    """Regression tests for the two bugs found in independent review of
    #68633 (GPT-5.6-sol-xhigh in Codex, egilewski):

    1. `effective = blocks or content` fell back to the RAW, unfiltered
       `content` when every block was filtered out as blank -- restoring
       exactly the invalid (blank/whitespace) payload the filter exists to
       remove, for any message where blank content is the ONLY content
       (no surviving tool_use/text/thinking block).
    2. The normal-path blank-text check used `(blk.get("text") or "").strip()`,
       which is not type-safe for a truthy NON-string, non-None text value
       (e.g. an int) -- `or` doesn't substitute for a truthy value, so
       `(7 or "").strip()` still raises AttributeError.
    """

    def _convert(self, message):
        from agent.anthropic_adapter import _convert_assistant_message
        return _convert_assistant_message(message)



    def test_sole_cache_marked_blank_block_relocates_marker_to_placeholder(self):
        """A message whose ONLY content is a blank text block that also
        carries cache_control: the marker must not be silently dropped just
        because there's nothing else to relocate it onto -- it must land on
        the (empty) placeholder that replaces the dropped block."""
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}],
        }
        result = self._convert(msg)
        blocks = result["content"]
        assert blocks == [
            {"type": "text", "text": "(empty)", "cache_control": {"type": "ephemeral"}}
        ], f"cache_control must relocate onto the (empty) placeholder: {blocks}"


    def test_non_string_truthy_text_treated_as_invalid_not_crash(self):
        """Regression: text=7 (a truthy int, not None) must not reach
        .strip() and raise AttributeError -- it must be treated the same as
        blank/invalid text and dropped."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": 7},
                {"type": "tool_use", "id": "call_int", "name": "web_search",
                 "input": {"query": "test"}},
            ],
        }
        result = self._convert(msg)  # must not raise
        blocks = result["content"]
        text_blocks = [b for b in blocks if b.get("type") == "text"]
        tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        assert len(text_blocks) == 0, f"Non-string text value must be dropped, not kept: {text_blocks}"
        assert len(tool_blocks) == 1


    def test_dict_valued_text_treated_as_invalid_not_crash(self):
        """Another truthy non-string shape (dict) must also be safely dropped."""
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": {"nested": "garbage"}}],
            "tool_calls": [
                {"id": "call_d", "function": {"name": "web_search",
                                               "arguments": '{"query": "test"}'}},
            ],
        }
        result = self._convert(msg)  # must not raise
        blocks = result["content"]
        assert not any(b.get("type") == "text" for b in blocks)


class TestReplayAllBlankFallback:
    """Regression for the final open review point on #68633 (egilewski):

    ``_relocated_replay_cache_control`` was applied only inside ``if
    replayed:``. For ``anthropic_content_blocks`` containing only a blank
    cache-marked text block, ``replayed`` became empty, the function fell
    through to the main path's ``(empty)`` fallback, and the marker was
    lost. A signed-thinking block plus the blank marked text also returned
    without any relocated marker (thinking is not a cacheable carrier).
    The replay branch now resolves a cacheable ``(empty)`` placeholder when
    no cacheable block survives the blank filter.
    """

    def _convert(self, message):
        from agent.anthropic_adapter import _convert_assistant_message
        return _convert_assistant_message(message)

    def test_sole_blank_marked_replay_block_keeps_marker_on_placeholder(self):
        msg = {
            "role": "assistant",
            "content": "",
            "anthropic_content_blocks": [
                {"type": "text", "text": " ", "cache_control": {"type": "ephemeral"}},
            ],
        }
        result = self._convert(msg)
        assert result["content"] == [
            {"type": "text", "text": "(empty)", "cache_control": {"type": "ephemeral"}}
        ], result["content"]

    def test_thinking_plus_blank_marked_text_keeps_thinking_and_marker(self):
        msg = {
            "role": "assistant",
            "content": "",
            "anthropic_content_blocks": [
                {"type": "thinking", "thinking": "reasoning", "signature": "sig-A"},
                {"type": "text", "text": "  ", "cache_control": {"type": "ephemeral"}},
            ],
        }
        result = self._convert(msg)
        blocks = result["content"]
        assert blocks[0] == {"type": "thinking", "thinking": "reasoning", "signature": "sig-A"}
        marked = [b for b in blocks if isinstance(b.get("cache_control"), dict)]
        assert len(marked) == 1 and marked[0]["type"] == "text"
        assert marked[0]["text"].strip(), "placeholder must be non-whitespace"

    def test_thinking_plus_blank_unmarked_text_gets_schema_valid_placeholder(self):
        """Even without a cache marker, dropping the only text block from a
        thinking-only replay must leave schema-valid content."""
        msg = {
            "role": "assistant",
            "content": "",
            "anthropic_content_blocks": [
                {"type": "thinking", "thinking": "reasoning", "signature": "sig-B"},
                {"type": "text", "text": "\n"},
            ],
        }
        result = self._convert(msg)
        texts = [b for b in result["content"] if b.get("type") == "text"]
        assert texts == [{"type": "text", "text": "(empty)"}]


def _find_blank_text_blocks(messages):
    """Recursively scan a converted Anthropic message list (including
    nested tool_result content) for any text block whose text is empty or
    whitespace-only. Returns a list of (message_index, role, location,
    block_index) tuples for every violation found -- empty means the
    payload is safe to send to Anthropic."""
    violations = []
    for m_idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b_idx, blk in enumerate(content):
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "text" and not (
                isinstance(blk.get("text"), str) and blk["text"].strip()
            ):
                violations.append((m_idx, msg.get("role"), "content", b_idx))
            if blk.get("type") == "tool_result" and isinstance(blk.get("content"), list):
                for ib_idx, iblk in enumerate(blk["content"]):
                    if (
                        isinstance(iblk, dict)
                        and iblk.get("type") == "text"
                        and not (isinstance(iblk.get("text"), str) and iblk["text"].strip())
                    ):
                        violations.append((m_idx, msg.get("role"), "tool_result", ib_idx))
    return violations


class TestFinalPayloadHasNoBlankTextBlocks:
    """End-to-end regression tests on the true final payload boundary:
    ``convert_messages_to_anthropic`` -- the last transform before
    ``build_anthropic_kwargs`` hands ``messages`` to the Anthropic SDK.

    Covers the blank-content shapes enumerated for the "text content
    blocks must contain non-whitespace text" HTTP 400 class, verifying the
    final built payload never contains a blank text block while tool_use,
    tool_result, and image content are preserved.
    """

    def test_user_message_empty_string_content(self):
        messages = [{"role": "user", "content": ""}]
        _, result = convert_messages_to_anthropic(messages)
        assert _find_blank_text_blocks(result) == []
        assert result[0]["content"] == "(empty message)"

    def test_user_message_whitespace_only_string_content(self):
        messages = [{"role": "user", "content": "   "}]
        _, result = convert_messages_to_anthropic(messages)
        assert _find_blank_text_blocks(result) == []
        assert result[0]["content"] == "(empty message)"

    def test_user_message_blank_list_content(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        _, result = convert_messages_to_anthropic(messages)
        assert _find_blank_text_blocks(result) == []
        assert result[0]["content"] == [{"type": "text", "text": "(empty message)"}]

    def test_user_message_mixed_blank_and_valid_text_blocks(self):
        """A blank text block sitting alongside a non-blank one must be
        dropped individually -- not left in place (the all-or-nothing bug)
        and not used as an excuse to nuke the valid sibling block."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "real question"},
                    {"type": "text", "text": "   "},
                ],
            }
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert _find_blank_text_blocks(result) == []
        assert result[0]["content"] == [{"type": "text", "text": "real question"}]

    def test_mixed_blank_text_plus_valid_tool_block_preserved(self):
        """Blank text next to a valid non-text block (tool_result) must
        drop only the blank text and keep the tool block intact."""
        messages = [
            {"role": "user", "content": "call a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "web_search", "arguments": '{"query": "x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert _find_blank_text_blocks(result) == []
        assistant_msg = next(m for m in result if m["role"] == "assistant")
        tool_use_blocks = [b for b in assistant_msg["content"] if b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        tool_result_msg = next(
            m
            for m in result
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        assert tool_result_msg is not None

    def test_assistant_tool_call_message_with_blank_content(self):
        """OpenAI-wire-shaped assistant turn: content is a blank string,
        tool_calls carries the real payload. Must not surface a blank text
        block, and the tool_use block must survive untouched."""
        messages = [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": "   ",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "web_search", "arguments": '{"query": "y"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert _find_blank_text_blocks(result) == []
        assistant_msg = next(m for m in result if m["role"] == "assistant")
        assert assistant_msg["content"] == [
            {"type": "tool_use", "id": "call_2", "name": "web_search", "input": {"query": "y"}}
        ]

    def test_leading_synthesized_user_turn_is_non_blank(self):
        """_ensure_leading_user_turn's synthesized filler must itself be
        non-whitespace -- regression for the literal " " placeholder bug."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "[Context compaction summary] earlier work"},
            {"role": "user", "content": "continue"},
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert _find_blank_text_blocks(result) == []
        assert result[0]["content"] == [{"type": "text", "text": "(empty)"}]

    def test_blank_text_nested_in_tool_result_content_is_dropped(self):
        """A blank text part nested inside a tool_result's own multimodal
        content list (e.g. alongside an image) must be scrubbed without
        losing the image."""
        messages = [
            {"role": "user", "content": "screenshot please"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "function": {"name": "screenshot", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_3",
                "content": [
                    {"type": "text", "text": "   "},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ]
        _, result = convert_messages_to_anthropic(messages)
        assert _find_blank_text_blocks(result) == []
        tool_result_msg = next(
            m
            for m in result
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        tool_result_block = next(
            b for b in tool_result_msg["content"] if b.get("type") == "tool_result"
        )
        image_blocks = [b for b in tool_result_block["content"] if b.get("type") == "image"]
        assert len(image_blocks) == 1

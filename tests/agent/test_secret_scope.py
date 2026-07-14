"""Tests for the profile-scoped credential primitive (Workstream A / Phase 2)."""
import pytest

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset_multiplex():
    """Ensure each test starts and ends with multiplexing off (it's a global)."""
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestMultiplexInactiveBackwardCompat:
    """Default deployment: get_secret transparently reads os.environ."""

    def test_reads_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-test"

    def test_missing_returns_default(self, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        assert ss.get_secret("NOPE_KEY") is None
        assert ss.get_secret("NOPE_KEY", "fallback") == "fallback"

    def test_no_raise_without_scope(self, monkeypatch):
        monkeypatch.delenv("SOME_KEY", raising=False)
        # multiplex off => unscoped read is fine, returns default
        assert ss.get_secret("SOME_KEY") is None


class TestMultiplexActiveFailClosed:
    """Multiplex on: an unscoped secret read raises instead of leaking."""

    def test_unscoped_read_raises(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leaky")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            ss.get_secret("ANTHROPIC_API_KEY")

    def test_scoped_read_uses_scope_not_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-environ")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-from-scope"})
        try:
            assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-from-scope"
        finally:
            ss.reset_secret_scope(token)

    def test_scoped_missing_key_returns_default_not_environ(self, monkeypatch):
        # Even though the value exists in os.environ, a scope is authoritative:
        # an absent scope key must NOT fall through to the (cross-profile) env.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-other-profile")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-mine"})
        try:
            assert ss.get_secret("OPENAI_API_KEY") is None
            assert ss.get_secret("OPENAI_API_KEY", "d") == "d"
        finally:
            ss.reset_secret_scope(token)

    def test_global_env_still_reads_environ_under_multiplex(self, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", "/opt/data")
        ss.set_multiplex_active(True)
        # No scope, multiplex on — but HERMES_HOME is global, so no raise.
        assert ss.get_secret("HERMES_HOME") == "/opt/data"

    def test_kanban_prefix_is_global(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_DB", "/x/kanban.db")
        ss.set_multiplex_active(True)
        assert ss.get_secret("HERMES_KANBAN_DB") == "/x/kanban.db"


class TestScopeIsolation:
    """Two scopes never see each other's secrets."""

    def test_nested_scopes_restore(self):
        ss.set_multiplex_active(True)
        t1 = ss.set_secret_scope({"K": "a"})
        try:
            assert ss.get_secret("K") == "a"
            t2 = ss.set_secret_scope({"K": "b"})
            try:
                assert ss.get_secret("K") == "b"
            finally:
                ss.reset_secret_scope(t2)
            assert ss.get_secret("K") == "a"
        finally:
            ss.reset_secret_scope(t1)


class TestEnvFileParsing:
    """load_env_file parses without mutating os.environ."""

    def test_parses_basic(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "# comment\n"
            "ANTHROPIC_API_KEY=sk-abc\n"
            "export OPENAI_API_KEY=sk-def\n"
            'QUOTED="quoted-value"\n'
            "SINGLE='single'\n"
            "\n"
            "BAD_LINE_NO_EQUALS\n"
        )
        out = ss.load_env_file(env)
        assert out == {
            "ANTHROPIC_API_KEY": "sk-abc",
            "OPENAI_API_KEY": "sk-def",
            "QUOTED": "quoted-value",
            "SINGLE": "single",
        }

    def test_does_not_mutate_environ(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZZZ_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("ZZZ_KEY=secret\n")
        ss.load_env_file(env)
        import os
        assert "ZZZ_KEY" not in os.environ

    def test_missing_file_returns_empty(self, tmp_path):
        assert ss.load_env_file(tmp_path / "nope.env") == {}

    def test_build_profile_secret_scope(self, tmp_path):
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-profile\n")
        assert ss.build_profile_secret_scope(tmp_path) == {
            "ANTHROPIC_API_KEY": "sk-profile"
        }

    def test_default_scope_keeps_shell_onepassword_auth(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            "OPENAI_API_KEY=op://Private/Default/key\n", encoding="utf-8"
        )
        monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops-shell-default")

        def _fake_fetch(**kwargs):
            # None preserves fetch_onepassword_secrets' process-env fallback.
            assert kwargs["token_value"] is None
            assert kwargs["include_process_auth"] is True
            return {"OPENAI_API_KEY": "  sk-default-resolved  "}, []

        monkeypatch.setattr(
            "agent.secret_sources.onepassword.fetch_onepassword_secrets",
            _fake_fetch,
        )

        assert ss.build_profile_secret_scope(tmp_path)["OPENAI_API_KEY"] == (
            "  sk-default-resolved  "
        )

    def test_failed_configured_onepassword_ref_drops_stale_env(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / ".env").write_text(
            "OPENAI_API_KEY=sk-stale-plaintext\n", encoding="utf-8"
        )
        (tmp_path / "config.yaml").write_text(
            "secrets:\n"
            "  onepassword:\n"
            "    enabled: true\n"
            "    env:\n"
            "      OPENAI_API_KEY: op://Private/OpenAI/key\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "agent.secret_sources.onepassword.fetch_onepassword_secrets",
            lambda **kwargs: ({}, ["auth failed"]),
        )

        assert "OPENAI_API_KEY" not in ss.build_profile_secret_scope(tmp_path)

    def test_configured_ref_resolved_by_registry_survives_manual_refetch_failure(
        self, tmp_path, monkeypatch
    ):
        """A configured op:// ref that apply_all already resolved must not be
        dropped when the redundant manual op fetch transiently fails."""
        (tmp_path / ".env").write_text(
            "OPENAI_API_KEY=sk-stale-plaintext\n", encoding="utf-8"
        )
        (tmp_path / "config.yaml").write_text(
            "secrets:\n"
            "  onepassword:\n"
            "    enabled: true\n"
            "    env:\n"
            "      OPENAI_API_KEY: op://Private/OpenAI/key\n",
            encoding="utf-8",
        )

        # apply_all (the secret-source registry) resolves the configured ref
        # into the isolated scope mapping — as the real 1Password source does
        # when the op CLI is available.
        def _fake_apply_all(sources_cfg, home, environ=None):
            if environ is not None:
                environ["OPENAI_API_KEY"] = "sk-resolved"

        monkeypatch.setattr(
            "agent.secret_sources.registry.apply_all", _fake_apply_all
        )
        # The redundant manual op re-fetch fails (different auth/cache
        # fingerprint, transient timeout, etc.).
        monkeypatch.setattr(
            "agent.secret_sources.onepassword.fetch_onepassword_secrets",
            lambda **kwargs: ({}, ["auth failed"]),
        )

        scope = ss.build_profile_secret_scope(tmp_path)
        assert scope.get("OPENAI_API_KEY") == "sk-resolved"

    def test_configured_onepassword_ref_overrides_raw_env_ref(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / ".env").write_text(
            "OPENAI_API_KEY=op://Private/Stale/key\n", encoding="utf-8"
        )
        (tmp_path / "config.yaml").write_text(
            "secrets:\n"
            "  onepassword:\n"
            "    enabled: true\n"
            "    env:\n"
            "      OPENAI_API_KEY: op://Private/Configured/key\n",
            encoding="utf-8",
        )

        def _fake_fetch(**kwargs):
            assert kwargs["references"]["OPENAI_API_KEY"] == (
                "op://Private/Configured/key"
            )
            return {"OPENAI_API_KEY": "sk-configured"}, []

        monkeypatch.setattr(
            "agent.secret_sources.onepassword.fetch_onepassword_secrets",
            _fake_fetch,
        )

        assert ss.build_profile_secret_scope(tmp_path)["OPENAI_API_KEY"] == (
            "sk-configured"
        )

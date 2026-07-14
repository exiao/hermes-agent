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
        # when the op CLI is available — and reports it in its provenance, even
        # when the resolved value matches a plaintext already in .env.
        from agent.secret_sources.registry import ApplyReport, AppliedVar

        def _fake_apply_all(sources_cfg, home, environ=None, scoped=False):
            if environ is not None:
                environ["OPENAI_API_KEY"] = "sk-resolved"
            report = ApplyReport()
            report.provenance["OPENAI_API_KEY"] = AppliedVar(
                name="OPENAI_API_KEY",
                source="onepassword",
                shape="mapped",
                overrode_env=True,
            )
            return report

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

    def test_registry_resolved_ref_survives_manual_refetch_even_if_value_matches_env(
        self, tmp_path, monkeypatch
    ):
        """Provenance — not value inequality — decides registry resolution: a
        configured ref that resolves to the SAME plaintext already in .env must
        still survive a transient manual-refetch failure."""
        (tmp_path / ".env").write_text(
            "OPENAI_API_KEY=sk-same\n", encoding="utf-8"
        )
        (tmp_path / "config.yaml").write_text(
            "secrets:\n"
            "  onepassword:\n"
            "    enabled: true\n"
            "    env:\n"
            "      OPENAI_API_KEY: op://Private/OpenAI/key\n",
            encoding="utf-8",
        )

        from agent.secret_sources.registry import ApplyReport, AppliedVar

        def _fake_apply_all(sources_cfg, home, environ=None, scoped=False):
            # Resolves to the SAME value already in .env — value inequality
            # would miss this, but provenance records the applied var.
            if environ is not None:
                environ["OPENAI_API_KEY"] = "sk-same"
            report = ApplyReport()
            report.provenance["OPENAI_API_KEY"] = AppliedVar(
                name="OPENAI_API_KEY",
                source="onepassword",
                shape="mapped",
                overrode_env=True,
            )
            return report

        monkeypatch.setattr(
            "agent.secret_sources.registry.apply_all", _fake_apply_all
        )
        monkeypatch.setattr(
            "agent.secret_sources.onepassword.fetch_onepassword_secrets",
            lambda **kwargs: ({}, ["auth failed"]),
        )

        scope = ss.build_profile_secret_scope(tmp_path)
        assert scope.get("OPENAI_API_KEY") == "sk-same"

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

    def test_default_scope_keeps_shell_bitwarden_bootstrap(
        self, tmp_path, monkeypatch
    ):
        """Default profile: a shell/systemd-supplied BWS_ACCESS_TOKEN (absent
        from .env) must still reach BitwardenSource so its secrets resolve into
        the scope — mirroring the 1Password process-auth preservation."""
        (tmp_path / ".env").write_text("", encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            "secrets:\n"
            "  bitwarden:\n"
            "    enabled: true\n"
            "    project_id: proj-123\n"
            "    auto_install: false\n",
            encoding="utf-8",
        )
        # Token comes from the shell env, NOT from .env.
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "bws-shell-default")

        import agent.secret_sources.bitwarden as bw

        monkeypatch.setattr(bw, "find_bws", lambda **kw: tmp_path / "bws")

        seen = {}

        def _fake_fetch(*, access_token, **kwargs):
            seen["access_token"] = access_token
            return {"STRIPE_KEY": "sk-from-vault"}, []

        monkeypatch.setattr(bw, "fetch_bitwarden_secrets", _fake_fetch)

        scope = ss.build_profile_secret_scope(tmp_path)
        assert seen["access_token"] == "bws-shell-default"
        assert scope.get("STRIPE_KEY") == "sk-from-vault"

    def test_named_profile_does_not_borrow_shell_bitwarden_bootstrap(
        self, tmp_path, monkeypatch
    ):
        """A named profile must NOT seed its scope from the process/default
        profile's shell BWS_ACCESS_TOKEN — that would let it borrow another
        profile's Bitwarden vault under multiplexing."""
        profile_home = tmp_path / "profiles" / "alpha"
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text("", encoding="utf-8")
        (profile_home / "config.yaml").write_text(
            "secrets:\n"
            "  bitwarden:\n"
            "    enabled: true\n"
            "    project_id: proj-123\n"
            "    auto_install: false\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "bws-shell-default")

        import agent.secret_sources.bitwarden as bw

        monkeypatch.setattr(bw, "find_bws", lambda **kw: profile_home / "bws")

        seen = {}

        def _fake_fetch(*, access_token, **kwargs):
            seen["access_token"] = access_token
            return {"STRIPE_KEY": "sk-from-vault"}, []

        monkeypatch.setattr(bw, "fetch_bitwarden_secrets", _fake_fetch)

        scope = ss.build_profile_secret_scope(profile_home)
        # The named profile supplied no token in its own .env, so the source
        # never authenticated with the default profile's shell token.
        assert "access_token" not in seen
        assert "STRIPE_KEY" not in scope

    def test_default_profile_still_runs_legacy_env_less_source(
        self, tmp_path, monkeypatch
    ):
        """The scoped fail-closed rejection is for NAMED profiles only. A
        default-profile legacy source whose fetch() lacks 'environ' must still
        run (env-less reads the owner's own environment, no cross-profile leak),
        so its default-profile vault secrets are not dropped."""
        (tmp_path / ".env").write_text("", encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            "secrets:\n  legacy:\n    enabled: true\n",
            encoding="utf-8",
        )

        from agent.secret_sources.base import (
            FetchResult,
            SECRET_SOURCE_API_VERSION,
            SecretSource,
        )
        from agent.secret_sources import registry as reg

        ran = {}

        class _LegacySrc(SecretSource):
            # No 'environ' param — a pre-contract / third-party source.
            def fetch(self, cfg, home_path):
                ran["hit"] = True
                res = FetchResult()
                res.secrets = {"LEGACY_KEY": "sk-legacy"}
                return res

            def override_existing(self, cfg):
                return False

            def protected_env_vars(self, cfg):
                return frozenset()

        _LegacySrc.name = "legacy"
        _LegacySrc.label = "Legacy"
        _LegacySrc.shape = "mapped"
        _LegacySrc.scheme = None
        _LegacySrc.api_version = SECRET_SOURCE_API_VERSION
        reg.register_source(_LegacySrc(), replace=True)
        try:
            scope = ss.build_profile_secret_scope(tmp_path)
        finally:
            reg._reset_registry_for_tests()

        assert ran.get("hit") is True
        assert scope.get("LEGACY_KEY") == "sk-legacy"

    def test_named_profile_rejects_legacy_env_less_source(
        self, tmp_path, monkeypatch
    ):
        """A named profile must still fail closed on a legacy env-less source
        (it could otherwise read the gateway/default profile's os.environ)."""
        profile_home = tmp_path / "profiles" / "beta"
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text("", encoding="utf-8")
        (profile_home / "config.yaml").write_text(
            "secrets:\n  legacy:\n    enabled: true\n",
            encoding="utf-8",
        )

        from agent.secret_sources.base import (
            FetchResult,
            SECRET_SOURCE_API_VERSION,
            SecretSource,
        )
        from agent.secret_sources import registry as reg

        ran = {}

        class _LegacySrc(SecretSource):
            def fetch(self, cfg, home_path):
                ran["hit"] = True
                res = FetchResult()
                res.secrets = {"LEGACY_KEY": "sk-legacy"}
                return res

            def override_existing(self, cfg):
                return False

            def protected_env_vars(self, cfg):
                return frozenset()

        _LegacySrc.name = "legacy"
        _LegacySrc.label = "Legacy"
        _LegacySrc.shape = "mapped"
        _LegacySrc.scheme = None
        _LegacySrc.api_version = SECRET_SOURCE_API_VERSION
        reg.register_source(_LegacySrc(), replace=True)
        try:
            scope = ss.build_profile_secret_scope(profile_home)
        finally:
            reg._reset_registry_for_tests()

        assert "hit" not in ran  # failed closed before calling fetch()
        assert "LEGACY_KEY" not in scope

    def test_named_profile_inherits_shared_root_env(self, tmp_path, monkeypatch):
        """A named profile with an empty (or partial) .env still sees shared
        root ~/.hermes/.env secrets in its scope — mirroring load_hermes_dotenv's
        root-then-profile merge. Without this, /model provider detection (which
        reads through get_secret) would miss a shared root OPENAI_API_KEY."""
        # Layout: tmp_path is the root home; tmp_path/profiles/alpha is the
        # named profile. get_default_hermes_root(profile_home) → tmp_path.
        (tmp_path / ".env").write_text(
            "OPENAI_API_KEY=sk-shared-root\nSHARED_TOKEN=root-tok\n",
            encoding="utf-8",
        )
        profile_home = tmp_path / "profiles" / "alpha"
        profile_home.mkdir(parents=True)
        # Profile overrides one value, inherits the rest from root.
        (profile_home / ".env").write_text(
            "SHARED_TOKEN=profile-override\n", encoding="utf-8"
        )

        scope = ss.build_profile_secret_scope(profile_home)
        assert scope.get("OPENAI_API_KEY") == "sk-shared-root"  # inherited
        assert scope.get("SHARED_TOKEN") == "profile-override"  # profile wins

    def test_default_profile_does_not_double_load_root(self, tmp_path):
        """The default/root home resolves its root to itself — no duplicate
        base-layer pass, and its own .env is authoritative."""
        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-root-only\n", encoding="utf-8")
        scope = ss.build_profile_secret_scope(tmp_path)
        assert scope.get("OPENAI_API_KEY") == "sk-root-only"

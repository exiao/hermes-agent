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




class TestScopedSingleProfile:
    """Multiplex OFF with a scope installed: the scope is an overlay, not a
    blindfold. The cron scheduler installs a ``<home>/.env`` scope around every
    job unconditionally, and single-profile deployments legitimately supply
    credentials via the process environment only (systemd ``Environment=``,
    ``pass-cli run`` / ``op run`` wrappers) — those must keep resolving."""

    def test_scope_hit_wins_over_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-environ")
        token = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-from-env-file"})
        try:
            assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-from-env-file"
        finally:
            ss.reset_secret_scope(token)


    def test_scope_miss_absent_everywhere_returns_default(self, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        token = ss.set_secret_scope({})
        try:
            assert ss.get_secret("NOPE_KEY") is None
            assert ss.get_secret("NOPE_KEY", "d") == "d"
        finally:
            ss.reset_secret_scope(token)

    def test_multiplex_on_still_authoritative(self, monkeypatch):
        # The fallthrough is strictly multiplex-off behavior: turning
        # multiplexing on must restore scope-authoritative semantics.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-other-profile")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            assert ss.get_secret("OPENAI_API_KEY") is None
        finally:
            ss.reset_secret_scope(token)


class TestRoutedForeignHomeScope:
    """Multiplex OFF but the bound scope serves a DIFFERENT home (routed profile:
    dashboard/desktop backend, per-profile cron ticker). os.environ is the LAUNCH
    profile's env there, so a scoped miss must fail closed exactly like multiplex —
    the .env-overlay fallthrough is only safe when the scope's home IS ours."""

    def test_scoped_miss_under_foreign_home_returns_default(self, monkeypatch, tmp_path):
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        monkeypatch.setenv("OPENAI_API_KEY", "sk-launch-profile")
        home_token = set_hermes_home_override(str(tmp_path / "other-profile"))
        token = ss.set_secret_scope({})
        try:
            assert ss.serves_routed_profile() is True
            assert ss.get_secret("OPENAI_API_KEY") is None
            assert ss.get_secret("OPENAI_API_KEY", "d") == "d"
        finally:
            ss.reset_secret_scope(token)
            reset_hermes_home_override(home_token)

    def test_scoped_miss_under_own_home_keeps_env_overlay(self, monkeypatch, tmp_path):
        """The deliberate single-profile overlay: a scope bound for the process's
        own home still falls through to os.environ (systemd / op run credentials)."""
        from hermes_constants import get_process_hermes_home, set_hermes_home_override, reset_hermes_home_override

        monkeypatch.setenv("OPENAI_API_KEY", "sk-own-env")
        home_token = set_hermes_home_override(str(get_process_hermes_home()))
        token = ss.set_secret_scope({})
        try:
            assert ss.serves_routed_profile() is False
            assert ss.get_secret("OPENAI_API_KEY") == "sk-own-env"
        finally:
            ss.reset_secret_scope(token)
            reset_hermes_home_override(home_token)

    def test_scope_hit_under_foreign_home_still_wins(self, monkeypatch, tmp_path):
        """A scoped hit is unaffected: only the miss branch changes."""
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        monkeypatch.setenv("OPENAI_API_KEY", "sk-launch-profile")
        home_token = set_hermes_home_override(str(tmp_path / "other-profile"))
        token = ss.set_secret_scope({"OPENAI_API_KEY": "sk-served-profile"})
        try:
            assert ss.get_secret("OPENAI_API_KEY") == "sk-served-profile"
        finally:
            ss.reset_secret_scope(token)
            reset_hermes_home_override(home_token)

    def test_stamped_foreign_scope_miss_fails_closed_without_override(self, monkeypatch, tmp_path):
        """The kanban/MCP shape: a foreign-home scope bound WITHOUT the HERMES_HOME
        override (deliberate — those paths need the dispatcher's policy reads).
        The profile_home stamp makes serves_routed_profile see it anyway."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-launch-profile")
        token = ss.set_secret_scope({}, profile_home=str(tmp_path / "other-profile"))
        try:
            assert ss.serves_routed_profile() is True
            assert ss.get_secret("OPENAI_API_KEY") is None
            assert ss.get_secret("OPENAI_API_KEY", "d") == "d"
        finally:
            ss.reset_secret_scope(token)

    def test_stamped_own_home_scope_keeps_env_overlay(self, monkeypatch):
        """A scope stamped with the process's own home is not routed: env
        fallthrough stays, matching launch_secret_scope's documented precedence."""
        from hermes_constants import get_process_hermes_home

        monkeypatch.setenv("OPENAI_API_KEY", "sk-own-env")
        token = ss.set_secret_scope({}, profile_home=str(get_process_hermes_home()))
        try:
            assert ss.serves_routed_profile() is False
            assert ss.get_secret("OPENAI_API_KEY") == "sk-own-env"
        finally:
            ss.reset_secret_scope(token)

    def test_profile_runtime_scope_binds_foreign_home_e2e(self, monkeypatch, tmp_path):
        """End to end through the real binder: ``_profile_runtime_scope`` (the same
        guard the desktop backend and routed turns use) installs the home override +
        secret scope together. Inside it, a miss must not borrow launch env, while
        the foreign profile's own .env resolves and the adapter-facing reader agrees."""
        from gateway.platforms._shared import get_scoped_secret
        from gateway.run import _profile_runtime_scope

        foreign = tmp_path / "profiles" / "team_b"
        foreign.mkdir(parents=True)
        (foreign / ".env").write_text("TEAM_B_KEY=from-b\n", encoding="utf-8")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-launch-profile")

        with _profile_runtime_scope(foreign, hydrate_secrets=False):
            assert ss.serves_routed_profile() is True
            assert ss.get_secret("OPENAI_API_KEY") is None
            assert ss.get_secret("TEAM_B_KEY") == "from-b"
            # The reader adapters actually call takes the same fail-closed path.
            assert get_scoped_secret("OPENAI_API_KEY") is None

    def test_worker_profile_scope_bind_home_false_e2e(self, monkeypatch, tmp_path):
        """The kanban spawn-env build binds the assignee's secret scope with
        ``bind_home=False`` — no home override, because the passthrough POLICY
        belongs to the dispatcher. The profile_home stamp must still make scoped
        misses fail closed, or the dispatcher's env leaks into B's worker env."""
        from hermes_cli.kanban_db_dispatch import _worker_profile_scope

        foreign = tmp_path / "profiles" / "assignee"
        foreign.mkdir(parents=True)
        (foreign / ".env").write_text("ASSIGNEE_KEY=from-assignee\n", encoding="utf-8")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-dispatcher")

        with _worker_profile_scope(str(foreign), bind_home=False):
            assert ss.serves_routed_profile() is True
            assert ss.get_secret("OPENAI_API_KEY") is None
            assert ss.get_secret("ASSIGNEE_KEY") == "from-assignee"


class TestScopeSetupRecovery:
    """A raise mid-scope-setup must release whatever was already bound — a leaked
    HERMES_HOME override or secret scope silently re-homes every later read in
    the caller's context."""

    def test_profile_runtime_scope_setup_failure_restores_override(self, monkeypatch, tmp_path):
        from gateway.run import _profile_runtime_scope
        from hermes_constants import get_hermes_home_override

        foreign = tmp_path / "profiles" / "b"
        foreign.mkdir(parents=True)

        def boom(home):
            raise RuntimeError("corrupt profile home")

        monkeypatch.setattr("agent.secret_scope.build_profile_secret_scope", boom)
        with pytest.raises(RuntimeError):
            with _profile_runtime_scope(foreign, hydrate_secrets=False):
                pass
        assert get_hermes_home_override() is None
        assert ss.current_secret_scope() is None

    def test_worker_profile_scope_setup_failure_restores_override(self, monkeypatch, tmp_path):
        from hermes_cli.kanban_db_dispatch import _worker_profile_scope
        from hermes_constants import get_hermes_home_override

        foreign = tmp_path / "profiles" / "assignee"
        foreign.mkdir(parents=True)

        def boom(home):
            raise RuntimeError("corrupt profile home")

        monkeypatch.setattr("agent.secret_scope.build_profile_secret_scope", boom)
        with pytest.raises(RuntimeError):
            with _worker_profile_scope(str(foreign), bind_home=True):
                pass
        assert get_hermes_home_override() is None
        assert ss.current_secret_scope() is None

    def test_model_switch_bind_releases_partial_scopes_on_raise(self, monkeypatch, tmp_path):
        """The scopes object never reaches the caller when the bind raises, so the
        bind must release what it already bound (home override + secret scope).
        Driven through ``server`` — the split module's functions run rebound on
        server.py's globals (``bind_module``)."""
        from tui_gateway import server
        from hermes_constants import get_hermes_home_override

        home = tmp_path / "profiles" / "b"
        home.mkdir(parents=True)

        def boom(home, env_overlay=None):
            raise RuntimeError("terminal policy unreadable")

        monkeypatch.setattr("tools.terminal_scope.install_profile_terminal_scope", boom)
        with pytest.raises(RuntimeError):
            server._profile_runtime_scope_tokens(home, hydrate_secrets=False)
        assert get_hermes_home_override() is None
        assert ss.current_secret_scope() is None


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

    def test_load_env_file_unescapes_quoted_values(self, tmp_path):
        """Values written by save_env_value must round-trip byte-exactly.

        Regression: load_env_file stripped only the outer quotes, leaving
        the writer's \\" and \\\\ escapes literal — credentials containing
        '\"' or '\\' worked interactively but were corrupted under scoped
        (cron / multiplex) resolution.
        """
        from hermes_cli.config import _quote_env_value

        original = 'tok"en\\with spaces'
        (tmp_path / ".env").write_text(f"MY_TOKEN={_quote_env_value(original)}\n")
        assert ss.load_env_file(tmp_path / ".env") == {"MY_TOKEN": original}

    def test_load_env_file_single_quotes_and_plain_values(self, tmp_path):
        (tmp_path / ".env").write_text(
            "PLAIN=abc123\nQUOTED='single quoted'\nEMPTY=\n"
        )
        assert ss.load_env_file(tmp_path / ".env") == {
            "PLAIN": "abc123",
            "QUOTED": "single quoted",
            "EMPTY": "",
        }

    def test_inline_comment_stripped_from_unquoted_value(self, tmp_path):
        """`KEY=value # comment` → `value` (python-dotenv semantics)."""
        (tmp_path / ".env").write_text("KEY=value # comment\nTABBED=foo\t#tabbed\n")
        assert ss.load_env_file(tmp_path / ".env") == {
            "KEY": "value",
            "TABBED": "foo",
        }

    def test_hash_without_preceding_whitespace_is_not_a_comment(self, tmp_path):
        """`KEY=foo#bar` stays intact — dotenv only strips `#` after whitespace."""
        (tmp_path / ".env").write_text("KEY=foo#bar\nLEAD=#leading\n")
        assert ss.load_env_file(tmp_path / ".env") == {
            "KEY": "foo#bar",
            "LEAD": "#leading",
        }

    def test_inline_comment_after_quoted_value(self, tmp_path):
        """Quotes strip AND the trailing comment drops; inner `#` survives."""
        (tmp_path / ".env").write_text(
            "DQ=\"has # inside\" # trailing\n"
            "SQ='single # inside' # trailing\n"
        )
        assert ss.load_env_file(tmp_path / ".env") == {
            "DQ": "has # inside",
            "SQ": "single # inside",
        }

    def test_inline_comment_with_escaped_quote_inside_value(self, tmp_path):
        r"""Escape-aware close-quote scan: `\"` must not terminate the value."""
        (tmp_path / ".env").write_text(
            'KEY="a \\" quote # x" # trail\n'
        )
        assert ss.load_env_file(tmp_path / ".env") == {"KEY": 'a " quote # x'}

    def test_round_trip_writer_value_with_trailing_comment(self, tmp_path):
        """A value quoted by the save_env_value writer survives an appended
        inline comment byte-exactly."""
        from hermes_cli.config import _quote_env_value

        original = 'we#ird "tok\\en" # not a comment'
        quoted = _quote_env_value(original)
        (tmp_path / ".env").write_text(f"MY_TOKEN={quoted} # rotated 2026-08\n")
        assert ss.load_env_file(tmp_path / ".env") == {"MY_TOKEN": original}




    def test_strips_utf8_bom_from_first_key(self, tmp_path):
        """Windows editors often save .env as UTF-8 with BOM (EF BB BF).

        Plain utf-8 keeps U+FEFF on the first key name, so get_secret('NAME')
        misses under an installed scope. utf-8-sig strips the leading BOM.
        """
        env = tmp_path / ".env"
        env.write_bytes(
            b"\xef\xbb\xbfANTHROPIC_API_KEY=sk-x\nOPENAI_API_KEY=sk-y\n"
        )
        out = ss.load_env_file(env)
        assert out == {
            "ANTHROPIC_API_KEY": "sk-x",
            "OPENAI_API_KEY": "sk-y",
        }
        assert "\ufeffANTHROPIC_API_KEY" not in out

        scope = ss.build_profile_secret_scope(tmp_path)
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(scope)
        try:
            assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-x"
            assert ss.get_secret("OPENAI_API_KEY") == "sk-y"
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

    def test_build_profile_secret_scope(self, tmp_path):
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-profile\n")
        # tmp_path is a default/root home (parent != "profiles"), so its scope
        # additionally seeds the process-owner's own os.environ (below .env) —
        # assert the .env contract rather than exact dict equality, which would
        # be a change-detector against the ambient shell environment.
        scope = ss.build_profile_secret_scope(tmp_path)
        assert scope["ANTHROPIC_API_KEY"] == "sk-profile"

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

    def test_default_profile_scope_keeps_shell_only_provider_key(
        self, tmp_path, monkeypatch
    ):
        """Codex P2 on #100: under gateway.multiplex_profiles the default profile
        also runs inside a secret scope, and get_secret becomes authoritative
        (no os.environ fallback) once any scope is installed. A default/process-
        owner profile that supplies a provider key via the shell/systemd env
        rather than .env (e.g. `export OPENAI_API_KEY=...` in the gateway's
        shell) must NOT lose it — the default scope seeds the owner's os.environ
        as its lowest layer."""
        # No .env at all: the key lives only in the process/shell environment.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell-default")
        scope = ss.build_profile_secret_scope(tmp_path)
        assert scope.get("OPENAI_API_KEY") == "sk-shell-default"

    def test_default_profile_env_file_overrides_shell(self, tmp_path, monkeypatch):
        """The os.environ seed is the LOWEST layer — a value in .env still wins
        over the same key in the process env."""
        (tmp_path / ".env").write_text(
            "OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8"
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell-default")
        scope = ss.build_profile_secret_scope(tmp_path)
        assert scope.get("OPENAI_API_KEY") == "sk-from-dotenv"

    def test_named_profile_does_not_seed_process_env(self, tmp_path, monkeypatch):
        """A NAMED profile is a hard isolation boundary: it must never borrow a
        provider key from the gateway/default process env, even one absent from
        its own .env."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-gateway-shell")
        profile_home = tmp_path / "profiles" / "alpha"
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text(
            "ANTHROPIC_API_KEY=sk-alpha\n", encoding="utf-8"
        )
        scope = ss.build_profile_secret_scope(profile_home)
        assert scope.get("ANTHROPIC_API_KEY") == "sk-alpha"
        assert "OPENAI_API_KEY" not in scope, (
            "named profile must not seed the default process's shell provider key"
        )

    def test_build_profile_secret_scope_includes_home_external_secrets(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / ".env").write_text("XIAOMI_API_KEY=placeholder\n")
        from hermes_cli import env_loader

        home_key = str(tmp_path.resolve())
        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_VALUES_BY_HOME,
            home_key,
            {"XIAOMI_API_KEY": "sk-from-bitwarden"},
        )

        scope = ss.build_profile_secret_scope(tmp_path)
        assert scope["XIAOMI_API_KEY"] == "sk-from-bitwarden"

    def test_build_profile_secret_scope_ignores_other_home_external_secrets(
        self, tmp_path, monkeypatch
    ):
        profile = tmp_path / "profile"
        other = tmp_path / "other"
        profile.mkdir()
        other.mkdir()
        from hermes_cli import env_loader

        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_VALUES_BY_HOME,
            str(other.resolve()),
            {"XIAOMI_API_KEY": "sk-other-profile"},
        )

        # Upstream asserts the whole scope is empty here. That holds upstream,
        # which never seeds the scope from os.environ. This fork deliberately
        # seeds the DEFAULT/process-owner scope with the process environment
        # (see build_profile_secret_scope: a non-"profiles" parent is not a
        # named profile, so the seed fires and the scope is legitimately
        # non-empty). The invariant under test is unchanged and still enforced:
        # the OTHER home's third-party secret must not leak into this scope.
        scope = ss.build_profile_secret_scope(profile)
        assert "XIAOMI_API_KEY" not in scope


class TestApiServerListenerGlobals:
    """API_SERVER listener settings are deployment config (#69379), not
    profile secrets: the scoped runner reload must keep seeing container env
    (Docker compose ``environment:`` block). API_SERVER_KEY IS a credential
    and stays profile-scoped."""

    LISTENER_VARS = (
        "API_SERVER_ENABLED",
        "API_SERVER_HOST",
        "API_SERVER_PORT",
        "API_SERVER_CORS_ORIGINS",
    )

    def test_listener_vars_read_environ_even_when_scoped_multiplex(self, monkeypatch):
        for name in self.LISTENER_VARS:
            monkeypatch.setenv(name, f"container-{name.lower()}")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"TELEGRAM_BOT_TOKEN": "scoped"})
        try:
            for name in self.LISTENER_VARS:
                assert ss.get_secret(name) == f"container-{name.lower()}"
        finally:
            ss.reset_secret_scope(token)

    def test_api_server_key_stays_profile_scoped(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_KEY", "default-profile-key-0123456789abcdef")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"OTHER": "x"})
        try:
            # A scoped miss must NOT borrow the (potentially cross-profile)
            # environ value: API_SERVER_KEY is a credential.
            assert ss.get_secret("API_SERVER_KEY") is None
        finally:
            ss.reset_secret_scope(token)
        assert not ss._is_global_env("API_SERVER_KEY")


class TestRelayRoutingStampGlobals:
    """GATEWAY_RELAY_* ROUTING stamps are deployment config, not profile
    secrets: config's relay enablement/sweep and gateway.relay's readers
    (relay_url(), registration, self-provision) must resolve the same
    process-env value under any scope, or the gateway enters a split-brain
    state (adapter registered but Platform.RELAY absent from config, or vice
    versa). Auth material (GATEWAY_RELAY_SECRET / _ID / _DELIVERY_KEY and the
    IDP_* credentials) stays profile-scoped with the fail-closed guard —
    mirroring the API_SERVER_KEY line above and the terminal env blocklist
    (tools/environments/local.py)."""

    ROUTING_VARS = (
        "GATEWAY_RELAY_URL",
        "GATEWAY_RELAY_ENDPOINT",
        "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS",
        "GATEWAY_RELAY_PLATFORMS",
        "GATEWAY_RELAY_BOT_IDS",
        "GATEWAY_RELAY_ROUTE_KEYS",
        "GATEWAY_RELAY_INSTANCE_ID",
        "GATEWAY_RELAY_WAKE_URL",
        "GATEWAY_RELAY_DISPLAY_NAME",
    )
    AUTH_VARS = (
        "GATEWAY_RELAY_SECRET",
        "GATEWAY_RELAY_ID",
        "GATEWAY_RELAY_DELIVERY_KEY",
        "GATEWAY_RELAY_IDP_CLIENT_SECRET",
        "GATEWAY_RELAY_IDP_CLIENT_ID",
        "GATEWAY_RELAY_IDP_TOKEN_URL",
    )

    def test_routing_stamps_read_environ_even_when_scoped_multiplex(self, monkeypatch):
        for name in self.ROUTING_VARS:
            monkeypatch.setenv(name, f"deploy-{name.lower()}")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"TELEGRAM_BOT_TOKEN": "scoped"})
        try:
            for name in self.ROUTING_VARS:
                assert ss.get_secret(name) == f"deploy-{name.lower()}", name
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

    def test_relay_auth_material_stays_profile_scoped(self, monkeypatch):
        for name in self.AUTH_VARS:
            monkeypatch.setenv(name, "cross-profile-credential")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"OTHER": "x"})
        try:
            for name in self.AUTH_VARS:
                # A scoped miss must NOT borrow the (potentially
                # cross-profile) environ value: relay auth is a credential.
                assert ss.get_secret(name) is None, name
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)
        for name in self.AUTH_VARS:
            assert not ss._is_global_env(name), name


class TestSecretScopeAcrossExecutorThreads:
    """Multiplexed profile state must reach pool workers (see #95119).

    The context-compression timeout fence runs auxiliary LLM calls in a
    daemon thread pool.  Bundled CPython runtime builds omit
    ``ThreadPoolExecutor``'s context propagation, so the profile secret
    scope was absent in the worker and ``get_secret`` failed closed with
    ``UnscopedSecretError``, silently degrading compression to lossy
    deterministic summaries.  ``DaemonThreadPoolExecutor.submit`` restores
    stdlib context semantics; these tests lock that in.
    """

    def test_scoped_read_works_in_daemon_pool_worker(self, monkeypatch):
        from tools.daemon_pool import DaemonThreadPoolExecutor

        monkeypatch.setenv("SURPLUS_API_KEY", "env-key")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SURPLUS_API_KEY": "scope-key"})
        pool = DaemonThreadPoolExecutor(max_workers=1)
        try:
            # The scope (authoritative under multiplex) must reach the worker.
            seen = pool.submit(ss.get_secret, "SURPLUS_API_KEY").result(timeout=10)
            assert seen == "scope-key"
            # A scoped miss must still not borrow the (cross-profile) env value.
            monkeypatch.setenv("OPENAI_API_KEY", "env-leak")
            assert pool.submit(ss.get_secret, "OPENAI_API_KEY").result(timeout=10) is None
        finally:
            pool.shutdown(wait=True)
            ss.reset_secret_scope(token)


class TestUnscopedSecretErrorSignature:
    def test_named_secret_leads_the_user_sentence(self):
        err = ss.UnscopedSecretError("SURPLUS_API_KEY", "get_secret('SURPLUS_API_KEY') with no scope")
        assert err.secret_name == "SURPLUS_API_KEY" and "SURPLUS_API_KEY" in str(err)
        assert err.developer_detail in getattr(err, "__notes__", [])

    def test_legacy_single_message_positional_is_the_developer_detail(self):
        """Older callers passed the whole sentence positionally; it must not be read as a name."""
        err = ss.UnscopedSecretError("get_secret('X') called with no profile secret scope active.")
        assert err.secret_name == ""
        assert err.developer_detail.startswith("get_secret('X')")
        assert "get_secret" not in str(err)

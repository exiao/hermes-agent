"""End-to-end credential isolation proof for multiplex mode (Workstream A).

These exercise the REAL resolution path (runtime_provider, secret scope, MCP
interpolation) rather than mocking it, proving the property that matters: two
profiles with different keys never see each other's, and an unscoped read in
multiplex mode fails closed instead of leaking.
"""
import pytest

from pathlib import Path

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestRuntimeProviderUsesScope:
    """hermes_cli.runtime_provider._getenv resolves through the secret scope."""

    def test_getenv_reads_scope_under_multiplex(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-global-leak")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-profileA"})
        try:
            assert _getenv("ANTHROPIC_API_KEY") == "sk-profileA"
        finally:
            ss.reset_secret_scope(tok)

    def test_getenv_two_profiles_isolated(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        ss.set_multiplex_active(True)

        tok_a = ss.set_secret_scope({"OPENAI_API_KEY": "sk-A"})
        try:
            assert _getenv("OPENAI_API_KEY") == "sk-A"
        finally:
            ss.reset_secret_scope(tok_a)

        tok_b = ss.set_secret_scope({"OPENAI_API_KEY": "sk-B"})
        try:
            assert _getenv("OPENAI_API_KEY") == "sk-B"
        finally:
            ss.reset_secret_scope(tok_b)

    def test_getenv_fails_closed_unscoped(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-leak")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            _getenv("OPENROUTER_API_KEY")

    def test_getenv_global_var_still_reads_environ(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        monkeypatch.setenv("HERMES_MAX_ITERATIONS", "42")
        ss.set_multiplex_active(True)
        # global var: no scope needed, no raise
        assert _getenv("HERMES_MAX_ITERATIONS") == "42"


class TestMcpInterpolationUsesScope:
    """MCP config ${VAR} interpolation resolves through the secret scope."""

    def test_interpolation_reads_scope(self, monkeypatch):
        from tools.mcp_tool import _interpolate_env_vars
        monkeypatch.setenv("MY_MCP_TOKEN", "global-token")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"MY_MCP_TOKEN": "profile-token"})
        try:
            cfg = {"env": {"TOKEN": "${MY_MCP_TOKEN}"}}
            assert _interpolate_env_vars(cfg) == {"env": {"TOKEN": "profile-token"}}
        finally:
            ss.reset_secret_scope(tok)

    def test_interpolation_unset_keeps_placeholder(self, monkeypatch):
        from tools.mcp_tool import _interpolate_env_vars
        monkeypatch.delenv("UNSET_MCP_VAR", raising=False)
        # multiplex off: unset var keeps literal placeholder (legacy behavior)
        assert _interpolate_env_vars("${UNSET_MCP_VAR}") == "${UNSET_MCP_VAR}"

    def test_interpolation_off_reads_environ(self, monkeypatch):
        from tools.mcp_tool import _interpolate_env_vars
        monkeypatch.setenv("MY_MCP_TOKEN", "env-token")
        # multiplex off: legacy os.environ resolution
        assert _interpolate_env_vars("${MY_MCP_TOKEN}") == "env-token"


class TestSignalConfigUsesScope:
    """gateway.config.load_gateway_config resolves the Signal account through
    the secret scope, so a multiplexed secondary profile binds ITS OWN
    signal-cli account rather than inheriting the default profile's
    SIGNAL_ACCOUNT from os.environ (which would collide on the shared
    credential and get the adapter silently refused).
    """

    def _load(self, tmp_path, monkeypatch):
        # load_gateway_config reads config.yaml from HERMES_HOME (resolved
        # dynamically via get_hermes_home at call time), so pointing the env at
        # an empty temp home is enough — no module reload needed.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import gateway.config as gc
        return gc

    def test_scoped_account_wins_over_environ(self, tmp_path, monkeypatch):
        gc = self._load(tmp_path, monkeypatch)
        # os.environ holds the DEFAULT profile's Signal creds.
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://127.0.0.1:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15203148130")
        ss.set_multiplex_active(True)
        # Secondary profile scope carries a DISTINCT account.
        tok = ss.set_secret_scope({
            "SIGNAL_HTTP_URL": "http://127.0.0.1:8080",
            "SIGNAL_ACCOUNT": "+19293262783",
        })
        try:
            cfg = gc.load_gateway_config()
        finally:
            ss.reset_secret_scope(tok)
        from gateway.config import Platform
        assert cfg.platforms[Platform.SIGNAL].extra["account"] == "+19293262783"

    def test_unscoped_multiplex_off_reads_environ(self, tmp_path, monkeypatch):
        gc = self._load(tmp_path, monkeypatch)
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://127.0.0.1:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15203148130")
        # multiplex inactive, no scope: legacy os.environ behavior.
        cfg = gc.load_gateway_config()
        from gateway.config import Platform
        assert cfg.platforms[Platform.SIGNAL].extra["account"] == "+15203148130"


class TestProfilePathResolutionUnderMultiplexScope:
    """Profile-scoped paths must follow the per-turn _profile_runtime_scope.

    The multiplexed gateway (gateway.multiplex_profiles) serves every profile
    from ONE process, scoping each inbound turn with _profile_runtime_scope —
    the same in-process-many-profiles topology as the desktop tui_gateway. The
    profile-isolation fixes (per-call path resolution + thread context
    propagation) must therefore hold under THIS scope too, not just desktop.
    This is the regression guard proving reachability is not desktop-only.
    """

    def _profiles(self, tmp_path):
        prof_a = tmp_path / "profA"
        prof_b = tmp_path / "profB"
        for p in (prof_a, prof_b):
            (p / "skills").mkdir(parents=True, exist_ok=True)
            (p / "state").mkdir(parents=True, exist_ok=True)
        return prof_a, prof_b

    def test_skills_dir_follows_multiplex_scope(self, tmp_path):
        from gateway.run import _profile_runtime_scope
        import tools.skills_hub as sh

        prof_a, prof_b = self._profiles(tmp_path)
        with _profile_runtime_scope(prof_a):
            a_seen = Path(sh.SKILLS_DIR)
        with _profile_runtime_scope(prof_b):
            b_seen = Path(sh.SKILLS_DIR)

        assert a_seen == prof_a / "skills"
        assert b_seen == prof_b / "skills"

    def test_cache_dir_follows_multiplex_scope(self, tmp_path):
        from gateway.run import _profile_runtime_scope
        import gateway.platforms.base as gb

        _prof_a, prof_b = self._profiles(tmp_path)
        with _profile_runtime_scope(prof_b):
            seen = gb.get_image_cache_dir()
        assert str(seen).startswith(str(prof_b))

    def test_worker_thread_inherits_multiplex_scope(self, tmp_path):
        """A wrapped worker spawned inside the scope must see the right profile.

        The _profile_runtime_scope docstring relies on copy_context() carrying
        the override into the agent worker thread; this proves the M2 fix
        primitive delivers that under the multiplexer's scope.
        """
        import threading

        from gateway.run import _profile_runtime_scope
        from hermes_constants import get_hermes_home
        from tools.thread_context import propagate_context_to_thread

        _prof_a, prof_b = self._profiles(tmp_path)
        seen = {}

        def worker():
            seen["home"] = str(get_hermes_home())

        with _profile_runtime_scope(prof_b):
            t = threading.Thread(target=propagate_context_to_thread(worker))
            t.start()
            t.join()

        assert seen["home"] == str(prof_b)



class TestModelSwitchOpenRouterPathUsesScope:
    """`/model <alias>` for an OpenRouter-backed provider must run under the
    profile secret scope.

    Regression for the `/model glm` failure: switch_model ->
    resolve_runtime_provider hits the ``provider == "openrouter"`` branch, which
    reads ``OPENAI_BASE_URL`` (and ``OPENROUTER_BASE_URL``) via ``_getenv`` ->
    ``get_secret``. Under multiplexing an unscoped read fails closed with
    ``UnscopedSecretError`` instead of leaking another profile's value, so the
    slash-command dispatch (which the multiplexer does NOT wrap in the per-turn
    agent scope) raised. The fix installs the profile scope around the switch;
    these prove the resolution path fails closed unscoped and succeeds scoped.
    """

    def test_openrouter_resolution_fails_closed_unscoped(self, monkeypatch):
        from hermes_cli.runtime_provider import resolve_runtime_provider
        # A stale value in os.environ is exactly what fail-closed protects
        # against leaking to another profile's turn.
        monkeypatch.setenv("OPENAI_BASE_URL", "https://leaked.example/v1")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            resolve_runtime_provider(
                requested="openrouter",
                target_model="z-ai/glm-5.2",
            )

    def test_openrouter_resolution_succeeds_under_scope(self, monkeypatch):
        from hermes_cli.runtime_provider import resolve_runtime_provider
        ss.set_multiplex_active(True)
        # The profile scope carries this profile's OpenRouter key; the base_url
        # is absent (falls back to the OpenRouter default), which is the normal
        # case for an OpenRouter alias.
        tok = ss.set_secret_scope({"OPENROUTER_API_KEY": "sk-profileA-or"})
        try:
            runtime = resolve_runtime_provider(
                requested="openrouter",
                target_model="z-ai/glm-5.2",
            )
        finally:
            ss.reset_secret_scope(tok)
        assert "openrouter.ai" in runtime["base_url"]
        assert runtime["api_key"] == "sk-profileA-or"

    def test_switch_model_scoped_wrapper_installs_scope(self, tmp_path, monkeypatch):
        """The slash handler's ``_switch_model_scoped`` closure runs the switch
        inside ``_profile_runtime_scope`` under multiplexing, so the OpenRouter
        credential read sees a scope instead of failing closed.

        Exercises the exact wrapper the handler builds, without standing up a
        full gateway: a stand-in object providing the two attributes the closure
        touches (``config.multiplex_profiles`` and
        ``_resolve_profile_home_for_source``).
        """
        # Profile home with its own .env carrying the OpenRouter key.
        prof = tmp_path / "profileA"
        prof.mkdir()
        (prof / ".env").write_text("OPENROUTER_API_KEY=sk-fromA-env\n")

        monkeypatch.setenv("OPENAI_BASE_URL", "https://leaked.example/v1")
        ss.set_multiplex_active(True)

        from gateway.run import _profile_runtime_scope
        from hermes_cli.runtime_provider import resolve_runtime_provider

        # Mirror the closure the handler installs (see slash_commands.py
        # _handle_model_command._switch_model_scoped).
        multiplex_on = True

        def switch_scoped():
            if not multiplex_on:
                return resolve_runtime_provider(
                    requested="openrouter", target_model="z-ai/glm-5.2"
                )
            with _profile_runtime_scope(prof):
                return resolve_runtime_provider(
                    requested="openrouter", target_model="z-ai/glm-5.2"
                )

        runtime = switch_scoped()
        assert runtime["api_key"] == "sk-fromA-env"
        assert "openrouter.ai" in runtime["base_url"]


class TestModelSwitchPersistScopedToSourceProfile:
    """`/model <name>` config persist must target the REQUESTING profile.

    Regression for the P1 on the credential-scope fix: scoping only the
    `switch_model` resolver (secret read) but persisting `config.yaml` outside
    the profile scope meant a secondary profile's `/model glm` rewrote the
    DEFAULT profile's config.yaml (via the module-level home) and left the
    requesting profile unchanged. `_persist_switched_model` runs the config
    read+`save_config` under `_profile_runtime_scope`, so `get_hermes_home()`
    (which `save_config` honors) resolves to the requesting profile.

    This mirrors the closure the handler builds (see
    `_handle_model_command._persist_switched_model`) without standing up a full
    gateway.
    """

    def test_persist_writes_source_profile_not_default(self, tmp_path):
        import yaml

        from gateway.run import _profile_runtime_scope
        from hermes_cli.config import save_config
        from hermes_constants import get_hermes_home

        default_home = tmp_path / "default"
        default_home.mkdir()
        (default_home / "config.yaml").write_text(
            yaml.safe_dump({"model": {"default": "gpt-5.4", "provider": "openai-codex"}}),
            encoding="utf-8",
        )
        source_home = tmp_path / "profileB"
        source_home.mkdir()
        (source_home / "config.yaml").write_text(
            yaml.safe_dump({"model": {"default": "old-model", "provider": "openrouter"}}),
            encoding="utf-8",
        )

        ss.set_multiplex_active(True)

        # Mirror _do_persist() under the requesting profile's scope.
        def persist_scoped(new_model, provider):
            with _profile_runtime_scope(source_home):
                cfg_path = get_hermes_home() / "config.yaml"
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.setdefault("model", {})
                model_cfg["default"] = new_model
                model_cfg["provider"] = provider
                save_config(cfg)

        persist_scoped("z-ai/glm-5.2", "openrouter")

        # The requesting profile got the switch...
        src_cfg = yaml.safe_load((source_home / "config.yaml").read_text())
        assert src_cfg["model"]["default"] == "z-ai/glm-5.2"
        # ...and the default profile's config was NOT touched.
        def_cfg = yaml.safe_load((default_home / "config.yaml").read_text())
        assert def_cfg["model"]["default"] == "gpt-5.4"

    def test_current_config_read_scoped_to_source_profile(self, tmp_path):
        """The initial current-model/provider/custom-provider read must use the
        REQUESTING profile's config, not the default profile's.

        Regression for the follow-on P1: current_provider / user_provs /
        custom_provs feed switch_model's resolution. If read from the default
        profile's config, a secondary profile's `/model <name>` resolves against
        the wrong provider/custom-provider map. `_read_current_config` runs under
        `_profile_runtime_scope`, so `_load_gateway_config` (via get_hermes_home)
        reads the requesting profile.
        """
        import yaml

        from gateway.run import _profile_runtime_scope, _load_gateway_config

        default_home = tmp_path / "default"
        default_home.mkdir()
        (default_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": {"default": "gpt-5.4", "provider": "openai-codex"},
                    "custom_providers": [
                        {"name": "Default Endpoint", "base_url": "http://default/v1", "model": "d"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        source_home = tmp_path / "profileB"
        source_home.mkdir()
        (source_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": {"default": "z-ai/glm-5.2", "provider": "openrouter"},
                    "custom_providers": [
                        {"name": "B Endpoint", "base_url": "http://profileb/v1", "model": "b"}
                    ],
                }
            ),
            encoding="utf-8",
        )

        ss.set_multiplex_active(True)

        with _profile_runtime_scope(source_home):
            cfg = _load_gateway_config()

        # Read resolved the REQUESTING profile, not the default.
        assert cfg["model"]["provider"] == "openrouter"
        assert cfg["model"]["default"] == "z-ai/glm-5.2"
        names = [p["name"] for p in cfg.get("custom_providers", [])]
        assert names == ["B Endpoint"]

    def test_persist_default_resolved_under_source_profile(self, tmp_path):
        """A plain `/model <name>` (no --global/--session) must honor the
        REQUESTING profile's ``model.persist_switch_by_default``.

        Regression for the P2: ``resolve_persist_behavior`` reads
        ``persist_switch_by_default`` via ``load_config`` -> ``get_hermes_home``.
        Computed unscoped, a secondary profile that opted OUT of persistence
        (``persist_switch_by_default: false``) would still rewrite its config.yaml
        because the default profile's persist-on decision leaked in. Resolving it
        inside ``_profile_runtime_scope`` reads the requesting profile's value.
        """
        import yaml

        from gateway.run import _profile_runtime_scope
        from hermes_cli.model_switch import resolve_persist_behavior
        default_home = tmp_path / "default"
        default_home.mkdir()
        (default_home / "config.yaml").write_text(
            yaml.safe_dump({"model": {"persist_switch_by_default": True}}),
            encoding="utf-8",
        )
        source_home = tmp_path / "profileB"
        source_home.mkdir()
        (source_home / "config.yaml").write_text(
            yaml.safe_dump({"model": {"persist_switch_by_default": False}}),
            encoding="utf-8",
        )

        ss.set_multiplex_active(True)

        # Plain /model (no --global, no --session): persist_global should follow
        # the profile's persist_switch_by_default, resolved under its scope.
        with _profile_runtime_scope(source_home):
            persist_global = resolve_persist_behavior(is_global=False, is_session=False)
        # ProfileB opted out → no persist, despite the default profile opting in.
        assert persist_global is False

    def test_user_provider_key_ref_resolved_via_scope(self, monkeypatch):
        """`switch_model` user-provider branch (`api_key: ${VAR}` / `key_env`)
        must resolve the key through the profile secret scope, not os.environ.

        Regression for the P1: under multiplexing the wrapper installs the
        secret scope but does NOT mutate os.environ, so reading the user
        provider's `${VAR}` / `key_env` via os.environ.get would see the default
        profile's value (or nothing). Routing through get_secret reads the
        requesting profile's scoped .env.
        """
        from hermes_cli.model_switch import switch_model

        # os.environ carries the DEFAULT profile's value — must NOT be used.
        monkeypatch.setenv("MYPROV_KEY", "sk-default-leak")
        ss.set_multiplex_active(True)

        user_providers = {
            "myprov": {
                "base_url": "https://myprov.example/v1",
                "model": "myprov-model",
                "key_env": "MYPROV_KEY",
            }
        }

        # Scope carries the REQUESTING profile's key.
        tok = ss.set_secret_scope({"MYPROV_KEY": "sk-profileB"})
        try:
            result = switch_model(
                raw_input="myprov-model",
                current_provider="openrouter",
                current_model="old",
                is_global=False,
                explicit_provider="myprov",
                user_providers=user_providers,
            )
        finally:
            ss.reset_secret_scope(tok)

        assert result.success, result.error_message
        # The profile's scoped key won, not the os.environ leak.
        assert result.api_key == "sk-profileB"
        assert result.base_url == "https://myprov.example/v1"

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

    def test_list_scoped_runs_listing_under_source_profile(self, tmp_path, monkeypatch):
        """The bare `/model` listing must run under the requesting profile's
        scope so auth-store / config / credential-pool provider detection
        resolves the requesting profile, not the default.

        Mirrors the handler's `_list_scoped` wrapper: `_profile_runtime_scope`
        redirects `get_hermes_home()`, so a listing fn's `_load_auth_store` /
        `get_provider_auth_state` / config reads see the requesting profile.
        (The raw provider-env `os.environ` probes are a separate, tracked gap.)
        """
        from gateway.run import _profile_runtime_scope
        from hermes_constants import get_hermes_home

        default_home = tmp_path / "default"
        default_home.mkdir()
        source_home = tmp_path / "profileB"
        source_home.mkdir()

        ss.set_multiplex_active(True)

        seen = {}

        def _fake_listing(**kwargs):
            # A listing fn resolves the active home for auth.json / config reads.
            seen["home"] = str(get_hermes_home())
            return []

        # Mirror _list_scoped under multiplexing.
        with _profile_runtime_scope(source_home):
            _fake_listing()

        assert seen["home"] == str(source_home)

    def test_refresh_cache_clear_scoped_to_source_profile(self, tmp_path, monkeypatch):
        """`/model --refresh` must clear the REQUESTING profile's provider cache.

        Regression for the P2: the listing path reads the source profile's
        `provider_models_cache.json` under `_list_scoped`, but the `--refresh`
        cache clear ran before any scope was installed. Under multiplexing an
        unscoped clear wipes the DEFAULT profile's cache and then the listing
        reuses the requesting profile's stale entry — so refresh silently does
        nothing for a secondary profile. Clearing under the source-profile scope
        (`_profile_runtime_scope` redirects `get_hermes_home()`, which
        `_provider_models_cache_path()` honors) targets the right cache file.
        """
        from gateway.run import _profile_runtime_scope
        from hermes_cli.models import (
            _provider_models_cache_path,
            clear_provider_models_cache,
        )

        default_home = tmp_path / "default"
        default_home.mkdir()
        source_home = tmp_path / "profileB"
        source_home.mkdir()

        ss.set_multiplex_active(True)

        # Resolve each profile's cache path under its own scope, then seed both.
        with _profile_runtime_scope(default_home):
            default_cache = _provider_models_cache_path()
        with _profile_runtime_scope(source_home):
            source_cache = _provider_models_cache_path()

        assert default_cache != source_cache
        default_cache.parent.mkdir(parents=True, exist_ok=True)
        source_cache.parent.mkdir(parents=True, exist_ok=True)
        default_cache.write_text("{}", encoding="utf-8")
        source_cache.write_text("{}", encoding="utf-8")

        # Mirror the handler: clear under the requesting profile's scope.
        with _profile_runtime_scope(source_home):
            clear_provider_models_cache()

        # Only the requesting profile's cache was wiped; the default survives.
        assert not source_cache.exists()
        assert default_cache.exists()


class TestListProvidersEnvProbesUseScope:
    """Bare `/model` provider listing must probe env-var provider credentials
    through the profile secret scope, not the process `os.environ`.

    Regression for the display-only gap left after the `_list_scoped` fix:
    `_profile_runtime_scope` installs the profile secret scope but intentionally
    does NOT mutate `os.environ`, so `list_authenticated_providers`' raw
    `os.environ.get(...)` provider-credential probes still saw the DEFAULT
    profile's keys. Under multiplexing a secondary profile's `/model` therefore
    listed an env-var provider as available using the default profile's key.
    Routing those probes through `get_secret` reads the requesting profile's
    scope; with no scope + multiplexing off (CLI/TUI) it falls back to
    `os.environ`, so single-profile listing is unchanged.
    """

    def _list_deepseek(self, monkeypatch):
        from hermes_cli.model_switch import list_authenticated_providers

        # Isolate the section-1 direct env-var probe: mock the models.dev catalog
        # to a single api_key provider (deepseek), and disable the overlay /
        # canonical detection paths (which resolve credentials via the separate
        # credential-pool auto-seed, out of scope for this fix). Keep the listing
        # hermetic — no live model fetch.
        monkeypatch.setattr("agent.models_dev.fetch_models_dev",
                            lambda: {"deepseek": {"env": ["DEEPSEEK_API_KEY"]}})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr("hermes_cli.models.CANONICAL_PROVIDERS", [])
        monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids",
                            lambda *a, **kw: ["deepseek-chat"])
        providers = list_authenticated_providers(max_models=5)
        return [p for p in providers if p.get("slug") == "deepseek"]

    def test_scoped_profile_env_provider_detected_no_environ_leak(self, monkeypatch):
        """Profile B's scoped DEEPSEEK_API_KEY makes deepseek list even when
        os.environ has NO deepseek key (the default profile's env)."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        ss.set_multiplex_active(True)

        tok = ss.set_secret_scope({"DEEPSEEK_API_KEY": "sk-profileB-deepseek"})
        try:
            rows = self._list_deepseek(monkeypatch)
        finally:
            ss.reset_secret_scope(tok)

        assert rows, "deepseek should list from the profile's scoped key"

    def test_default_profile_environ_not_leaked_to_scoped_profile(self, monkeypatch):
        """os.environ carries the DEFAULT profile's deepseek key, but a
        secondary profile whose scope lacks it must NOT list deepseek."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-default-leak")
        ss.set_multiplex_active(True)

        # Profile B's scope has a different provider's key, not deepseek's.
        tok = ss.set_secret_scope({"OPENAI_API_KEY": "sk-profileB-openai"})
        try:
            rows = self._list_deepseek(monkeypatch)
        finally:
            ss.reset_secret_scope(tok)

        assert not rows, "os.environ deepseek key must not leak into profile B's listing"

    def test_scoped_miss_does_not_pool_seed_default_key_into_profile(
        self, monkeypatch, tmp_path
    ):
        """A named-profile scoped miss must stop before credential-pool seeding.

        Regression for the #100 P1: the Section-1 scoped env probe correctly
        missed (profile B has no DEEPSEEK_API_KEY), but the listing then fell
        through to ``load_pool()``, whose env auto-seed could persist the DEFAULT
        profile's process key into profile B's auth store. Simulate that unsafe
        fallback and prove the listing never reaches it.
        """
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        profile_home = tmp_path / "profiles" / "profileB"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-default-leak")
        ss.set_multiplex_active(True)

        # If the old pool fallback is reached, it would write the default
        # profile's env key into profile B's auth.json and mark the row present.
        def _unsafe_load_pool(_slug):
            import json

            auth_path = profile_home / "auth.json"
            auth_path.write_text(
                json.dumps(
                    {
                        "credential_pool": {
                            "deepseek": [
                                {
                                    "source": "env:DEEPSEEK_API_KEY",
                                    "access_token": "sk-default-leak",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            class _Pool:
                def has_credentials(self):
                    return True

            return _Pool()

        monkeypatch.setattr("agent.credential_pool.load_pool", _unsafe_load_pool)

        home_token = set_hermes_home_override(str(profile_home))
        scope_token = ss.set_secret_scope({"OPENAI_API_KEY": "sk-profileB-openai"})
        try:
            rows = self._list_deepseek(monkeypatch)
        finally:
            ss.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

        assert not rows, "scoped miss must not list via default-profile pool seed"
        assert not (profile_home / "auth.json").exists(), (
            "scoped miss must not persist the default profile's env key into "
            "the named profile auth store"
        )

    def test_single_profile_reads_environ_unchanged(self, monkeypatch):
        """Multiplex off, no scope (CLI/TUI): get_secret falls back to
        os.environ, so an env-var provider still lists exactly as before."""
        ss.set_multiplex_active(False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-cli-user")

        rows = self._list_deepseek(monkeypatch)

        assert rows, "single-profile CLI listing must still detect the env-var provider"



class TestScopedListingSkipsProcessGlobalCredentialFallbacks:
    """A NAMED-profile /model listing must NOT mark a provider available via a
    PROCESS-GLOBAL credential fallback (the credential-pool auto-seed — which
    for copilot runs `gh auth token` / reads COPILOT_GITHUB_TOKEN/GH_TOKEN/
    GITHUB_TOKEN — or the anthropic Claude-Code / Hermes-OAuth credential
    files). Those belong to the DEFAULT profile; attributing them to a named
    profile leaks another profile's identity into its picker.

    But the DEFAULT profile (the process owner) MUST keep those fallbacks even
    under multiplexing — its gh-auth / Claude-file / pool creds are its own.

    Regression for the Codex P1 follow-on on #100: the scoped env-var probe
    correctly left has_creds false, but the same block fell through to
    load_pool(hermes_slug), which auto-seeds copilot from the default process.
    And the Codex P2 follow-on: _profile_runtime_scope installs a scope for the
    DEFAULT profile too, so a bare listing must not drop the default's own creds.
    """

    @staticmethod
    def _named_profile_home(tmp_path):
        """Context that makes get_hermes_home() resolve to a NAMED profile
        (<home>/profiles/<name>), mirroring _profile_runtime_scope's home
        redirect for a secondary profile."""
        from hermes_constants import (
            set_hermes_home_override,
            reset_hermes_home_override,
        )
        import contextlib

        @contextlib.contextmanager
        def _cm():
            home = tmp_path / "profiles" / "profileB"
            home.mkdir(parents=True, exist_ok=True)
            tok = set_hermes_home_override(str(home))
            try:
                yield
            finally:
                reset_hermes_home_override(tok)

        return _cm()

    def _list_copilot(self, monkeypatch, *, pool_has_creds: bool):
        from hermes_cli.model_switch import list_authenticated_providers
        from hermes_cli.providers import HermesOverlay

        # Only copilot in the catalog; its scoped env vars are absent, the auth
        # store is empty, and the credential POOL reports credentials (as it
        # would after auto-seeding from the default profile's gh identity).
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.providers.HERMES_OVERLAYS",
            {"github-copilot": HermesOverlay(
                transport="openai_chat",
                extra_env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN"),
            )},
        )
        monkeypatch.setattr("hermes_cli.models.CANONICAL_PROVIDERS", [])
        monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.models.cached_provider_model_ids",
            lambda *a, **kw: ["gpt-4o-copilot"],
        )

        class _Pool:
            def has_credentials(self):
                return pool_has_creds

            def has_available(self):
                return pool_has_creds

        monkeypatch.setattr(
            "agent.credential_pool.load_pool", lambda slug: _Pool()
        )
        providers = list_authenticated_providers(max_models=5)
        return [p for p in providers if p.get("slug") in ("copilot", "github-copilot")]

    def test_scoped_profile_does_not_borrow_default_copilot_pool(
        self, monkeypatch, tmp_path
    ):
        """Under a scoped multiplex listing FOR A NAMED PROFILE, copilot must
        NOT list from the default profile's pool-seeded gh identity."""
        for ev in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(ev, raising=False)
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"OPENAI_API_KEY": "sk-profileB-openai"})
        try:
            with self._named_profile_home(tmp_path):
                rows = self._list_copilot(monkeypatch, pool_has_creds=True)
        finally:
            ss.reset_secret_scope(tok)
        assert not rows, "copilot must not list from the default profile's pool creds"

    def test_default_profile_under_multiplex_keeps_copilot_pool(
        self, monkeypatch, tmp_path
    ):
        """Codex P2 on #100: _profile_runtime_scope installs a secret scope for
        the DEFAULT profile too under multiplexing. A bare `/model` listing for
        the default (process-owner) profile must KEEP its own pool/gh fallback —
        the default profile's home is ~/.hermes (parent != 'profiles')."""
        from hermes_constants import (
            set_hermes_home_override,
            reset_hermes_home_override,
        )
        for ev in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(ev, raising=False)
        default_home = tmp_path / ".hermes"
        default_home.mkdir(parents=True, exist_ok=True)
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"OPENAI_API_KEY": "sk-default-openai"})
        home_tok = set_hermes_home_override(str(default_home))
        try:
            rows = self._list_copilot(monkeypatch, pool_has_creds=True)
        finally:
            reset_hermes_home_override(home_tok)
            ss.reset_secret_scope(tok)
        assert rows, (
            "default profile under multiplexing must still list copilot from its "
            "own pool creds"
        )

    def test_single_profile_still_lists_copilot_from_pool(self, monkeypatch):
        """Multiplex off / no scope (CLI/TUI): the pool fallback still runs, so
        copilot lists exactly as before."""
        ss.set_multiplex_active(False)
        rows = self._list_copilot(monkeypatch, pool_has_creds=True)
        assert rows, "single-profile listing must still detect copilot via the pool"

    # --- Canonical cross-check pass (section 2b) ---
    # The overlay pass above (section 2) and this canonical pass (section 2b,
    # the CANONICAL_PROVIDERS cross-check) each have their OWN load_pool()
    # fallback. Codex's P1 on #100 pointed specifically at the canonical
    # block: when a provider is absent from HERMES_OVERLAYS but present in
    # CANONICAL_PROVIDERS, listing falls through to load_pool(_cp.slug),
    # which auto-seeds copilot from the default process's gh identity. The
    # overlay-pass tests above empty CANONICAL_PROVIDERS, so they never
    # exercised this path.

    def _list_copilot_via_canonical(self, monkeypatch, *, pool_has_creds: bool):
        from hermes_cli.model_switch import list_authenticated_providers
        from hermes_cli.models import ProviderEntry

        # copilot reaches the canonical cross-check (section 2b) only: no
        # overlay match, no models.dev entry, empty auth store, scoped env
        # vars absent, and the credential POOL reports credentials (as it
        # would after auto-seeding from the default profile's gh identity).
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.models.CANONICAL_PROVIDERS",
            [ProviderEntry("copilot", "GitHub Copilot", "GitHub Copilot")],
        )
        monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.models.cached_provider_model_ids",
            lambda *a, **kw: ["gpt-4o-copilot"],
        )

        class _Pool:
            def has_credentials(self):
                return pool_has_creds

            def has_available(self):
                return pool_has_creds

        monkeypatch.setattr("agent.credential_pool.load_pool", lambda slug: _Pool())
        providers = list_authenticated_providers(max_models=5)
        return [p for p in providers if p.get("slug") == "copilot"]

    def test_scoped_profile_canonical_pass_does_not_borrow_default_pool(
        self, monkeypatch, tmp_path
    ):
        """Under a scoped multiplex listing FOR A NAMED PROFILE, the CANONICAL
        cross-check must NOT list copilot from the default profile's pool-seeded
        gh identity."""
        for ev in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(ev, raising=False)
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"OPENAI_API_KEY": "sk-profileB-openai"})
        try:
            with self._named_profile_home(tmp_path):
                rows = self._list_copilot_via_canonical(
                    monkeypatch, pool_has_creds=True
                )
        finally:
            ss.reset_secret_scope(tok)
        assert not rows, (
            "canonical pass must not list copilot from the default profile's "
            "pool creds under a scoped listing"
        )

    def test_single_profile_canonical_pass_still_lists_copilot_from_pool(
        self, monkeypatch
    ):
        """Multiplex off / no scope (CLI/TUI): the canonical pool fallback still
        runs, so copilot lists exactly as before."""
        ss.set_multiplex_active(False)
        rows = self._list_copilot_via_canonical(monkeypatch, pool_has_creds=True)
        assert rows, (
            "single-profile canonical listing must still detect copilot via the pool"
        )


class TestOpenAIDiscoveryUsesScope:
    """provider_model_ids / fingerprint for openai-api honor the profile scope.

    Regression for the Codex P2 on #100: the /model listing marks the
    openai-api row available from the requesting profile's scoped
    OPENAI_API_KEY, but downstream live discovery + the disk-cache
    fingerprint read os.getenv/os.environ directly, so profile B's picker
    could call /models and cache model availability using profile A's key.
    """

    def _capture_discovery_key(self, monkeypatch):
        """Return the api_key fetch_api_models is called with for openai-api."""
        from hermes_cli import models as m

        seen = {}

        def _fake_fetch(api_key, base_url, *a, **kw):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            # Return a model that survives the default-endpoint curated filter.
            return list(m._PROVIDER_MODELS.get("openai-api", [])) or ["gpt-5"]

        monkeypatch.setattr(m, "fetch_api_models", _fake_fetch)
        m.provider_model_ids("openai-api", force_refresh=True)
        return seen

    def test_discovery_uses_scoped_key_not_environ(self, monkeypatch):
        """Under multiplex, discovery must fetch with profile B's scoped key,
        never the default profile's OPENAI_API_KEY in os.environ."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-default-leak")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        ss.set_multiplex_active(True)

        tok = ss.set_secret_scope({"OPENAI_API_KEY": "sk-profileB"})
        try:
            seen = self._capture_discovery_key(monkeypatch)
        finally:
            ss.reset_secret_scope(tok)

        assert seen.get("api_key") == "sk-profileB"

    def test_discovery_resolves_profile_onepassword_reference(
        self, monkeypatch, tmp_path
    ):
        """The gateway-built scope must not pass a raw op:// ref to /models."""
        profile = tmp_path / "profiles" / "profileB"
        profile.mkdir(parents=True)
        (profile / ".env").write_text(
            "OPENAI_API_KEY=op://Private/ProfileB/key\n", encoding="utf-8"
        )
        (profile / ".op.env").write_text(
            "OP_SERVICE_ACCOUNT_TOKEN=ops-profileB\n"
            "OP_CONNECT_HOST=https://connect.profileb.test\n"
            "OP_CONNECT_TOKEN=connect-profileB\n",
            encoding="utf-8",
        )

        def _fake_fetch(**kwargs):
            assert kwargs["token_value"] == "ops-profileB"
            assert kwargs["include_process_auth"] is False
            assert kwargs["auth_env"] == {
                "OP_CONNECT_HOST": "https://connect.profileb.test",
                "OP_CONNECT_TOKEN": "connect-profileB",
            }
            assert kwargs["home_path"] == profile
            return {"OPENAI_API_KEY": "  sk-profileB-resolved  "}, []

        monkeypatch.setattr(
            "agent.secret_sources.onepassword.fetch_onepassword_secrets",
            _fake_fetch,
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ss.set_multiplex_active(True)

        tok = ss.set_secret_scope(ss.build_profile_secret_scope(profile))
        try:
            seen = self._capture_discovery_key(monkeypatch)
        finally:
            ss.reset_secret_scope(tok)

        assert seen.get("api_key") == "sk-profileB-resolved"

    def test_discovery_single_profile_reads_environ(self, monkeypatch):
        """Multiplex off, no scope (CLI/TUI): discovery falls back to
        os.environ, byte-identical to the legacy os.getenv behavior."""
        ss.set_multiplex_active(False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-cli-user")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        seen = self._capture_discovery_key(monkeypatch)

        assert seen.get("api_key") == "sk-cli-user"

    def test_fingerprint_isolated_between_profiles(self, monkeypatch):
        """The cache fingerprint must differ per scoped key so profile B
        never hits profile A's cached openai-api entry."""
        from hermes_cli.models import _credential_fingerprint

        monkeypatch.setenv("OPENAI_API_KEY", "sk-default-leak")
        ss.set_multiplex_active(True)

        tok_a = ss.set_secret_scope({"OPENAI_API_KEY": "sk-A"})
        try:
            fp_a = _credential_fingerprint("openai-api")
        finally:
            ss.reset_secret_scope(tok_a)

        tok_b = ss.set_secret_scope({"OPENAI_API_KEY": "sk-B"})
        try:
            fp_b = _credential_fingerprint("openai-api")
        finally:
            ss.reset_secret_scope(tok_b)

        assert fp_a != fp_b, "distinct scoped keys must produce distinct fingerprints"


class TestApiKeyProviderBaseUrlUsesScope:
    """resolve_api_key_provider_credentials resolves *_BASE_URL via the scope.

    Regression for the follow-on Codex P2 on #100: the scoped listing admits a
    scoped API-key provider (e.g. deepseek/stepfun with a scoped *_BASE_URL),
    then discovery calls cached_provider_model_ids -> provider_model_ids ->
    resolve_api_key_provider_credentials, which read the base URL with
    os.getenv(base_url_env_var). So /model could fetch the catalog from the
    default profile's base URL while using the secondary profile's api key.
    """

    def test_base_url_reads_scope_under_multiplex(self, monkeypatch):
        from hermes_cli.auth import resolve_api_key_provider_credentials

        # deepseek: api_key via DEEPSEEK_API_KEY, base_url via DEEPSEEK_BASE_URL.
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://default-profile.example/v1")
        ss.set_multiplex_active(True)

        tok = ss.set_secret_scope({
            "DEEPSEEK_API_KEY": "sk-profileB",
            "DEEPSEEK_BASE_URL": "https://profileB.example/v1",
        })
        try:
            creds = resolve_api_key_provider_credentials("deepseek")
        finally:
            ss.reset_secret_scope(tok)

        assert creds["base_url"] == "https://profileB.example/v1", (
            "scoped base URL must win over the default profile's os.environ value"
        )

    def test_base_url_single_profile_reads_environ(self, monkeypatch):
        from hermes_cli.auth import resolve_api_key_provider_credentials

        ss.set_multiplex_active(False)
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://cli-user.example/v1")

        creds = resolve_api_key_provider_credentials("deepseek")

        assert creds["base_url"] == "https://cli-user.example/v1", (
            "single-profile base URL must still read os.environ, byte-identical"
        )

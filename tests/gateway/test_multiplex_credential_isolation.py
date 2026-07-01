"""End-to-end credential isolation proof for multiplex mode (Workstream A).

These exercise the REAL resolution path (runtime_provider, secret scope, MCP
interpolation) rather than mocking it, proving the property that matters: two
profiles with different keys never see each other's, and an unscoped read in
multiplex mode fails closed instead of leaking.
"""
import pytest

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

"""`/credits` and `/usage` must read the REQUESTING profile's account/creds.

Regression for the partial-data gap follow-up to the `/model` credential-scope
fix (PR #97): slash-command dispatch runs OUTSIDE the multiplexer's per-turn
agent scope, so `build_credits_view` / `fetch_account_usage` / `nous_credits_lines`
resolve `get_hermes_home()` (auth.json) and `get_secret`-backed provider keys
against the DEFAULT profile — showing a secondary profile its own empty/partial
balance instead of its real account. The handlers now wrap those reads in
`_profile_secret_scope_for_source`, which installs `_profile_runtime_scope`
under multiplexing and is a no-op otherwise.

These exercise the shared scope helper the two handlers use (rather than
standing up a full gateway), proving profile-B reads resolve to profile B and
single-profile gateways get an unchanged no-op scope.
"""
from contextlib import nullcontext
from pathlib import Path

import pytest

from agent import secret_scope as ss
from gateway.slash_commands import GatewaySlashCommandsMixin


class _StubConfig:
    def __init__(self, multiplex: bool):
        self.multiplex_profiles = multiplex


class _StubRunner(GatewaySlashCommandsMixin):
    """Minimal object providing the two attributes the scope helper touches."""

    def __init__(self, multiplex: bool, profile_home: Path):
        self.config = _StubConfig(multiplex)
        self._profile_home = profile_home

    def _resolve_profile_home_for_source(self, source):
        return self._profile_home


@pytest.fixture(autouse=True)
def _reset():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def test_scope_noop_when_multiplex_off(tmp_path):
    runner = _StubRunner(multiplex=False, profile_home=tmp_path / "profB")
    scope = runner._profile_secret_scope_for_source(object())
    assert isinstance(scope, nullcontext)


def test_scope_redirects_home_to_requesting_profile(tmp_path):
    from hermes_constants import get_hermes_home

    prof_b = tmp_path / "profB"
    prof_b.mkdir()
    runner = _StubRunner(multiplex=True, profile_home=prof_b)
    ss.set_multiplex_active(True)

    with runner._profile_secret_scope_for_source(object()):
        assert str(get_hermes_home()) == str(prof_b)
    # Scope exits cleanly — home resolution is no longer pinned to profB.
    assert str(get_hermes_home()) != str(prof_b)


def test_scope_installs_profile_secret_scope(tmp_path):
    """Under the scope, the profile's own .env key wins over os.environ."""
    from agent.secret_scope import get_secret

    prof_b = tmp_path / "profB"
    prof_b.mkdir()
    (prof_b / ".env").write_text("OPENROUTER_API_KEY=sk-fromB-env\n")

    runner = _StubRunner(multiplex=True, profile_home=prof_b)
    ss.set_multiplex_active(True)

    with runner._profile_secret_scope_for_source(object()):
        assert get_secret("OPENROUTER_API_KEY") == "sk-fromB-env"

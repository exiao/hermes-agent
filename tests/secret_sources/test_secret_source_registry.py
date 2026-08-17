"""Tests for the secret-source contract + orchestrator.

Covers: registration gating (API version, name/scheme uniqueness, shape),
apply_all precedence (mapped beats bulk, first-wins, override_existing,
protected vars), conflict surfacing, timeout enforcement, provenance,
and Bitwarden's SecretSource adapter — plus the conformance kit run
against the bundled Bitwarden source.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources.base import (  # noqa: E402
    SECRET_SOURCE_API_VERSION,
    ErrorKind,
    FetchResult,
    SecretSource,
    is_valid_env_name,
    run_secret_cli,
    scrub_ansi,
)
from agent.secret_sources import registry as reg  # noqa: E402
from agent.secret_sources.bitwarden import BitwardenSource  # noqa: E402
from tests.secret_sources.conformance import SecretSourceConformance  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    """Each test starts with an empty registry and no builtin auto-load."""
    reg._reset_registry_for_tests()
    monkeypatch.setattr(reg, "_ensure_builtin_sources", lambda: None)
    yield
    reg._reset_registry_for_tests()


def _make_source(
    name="dummy",
    shape="mapped",
    secrets=None,
    error=None,
    error_kind=None,
    scheme=None,
    override=False,
    protected=(),
    api_version=SECRET_SOURCE_API_VERSION,
    fetch_fn=None,
):
    """Build a minimal conforming source for orchestrator tests."""

    class _Src(SecretSource):
        def fetch(self, cfg, home_path):
            if fetch_fn is not None:
                return fetch_fn(cfg, home_path)
            res = FetchResult()
            if error:
                res.error = error
                res.error_kind = error_kind or ErrorKind.INTERNAL
            else:
                res.secrets = dict(secrets or {})
            return res

        def override_existing(self, cfg):
            return override

        def protected_env_vars(self, cfg):
            return frozenset(protected)

    _Src.name = name
    _Src.label = name.title()
    _Src.shape = shape
    _Src.scheme = scheme
    _Src.api_version = api_version
    return _Src()


# ---------------------------------------------------------------------------
# Registration gating
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registers_conforming_source(self):
        assert reg.register_source(_make_source()) is True
        assert reg.get_source("dummy") is not None

    def test_rejects_non_secretsource_instance(self):
        assert reg.register_source(object()) is False

    def test_same_name_is_isolated_by_profile(self, tmp_path):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_a = str((tmp_path / "secrets-a").resolve())
        home_b = str((tmp_path / "secrets-b").resolve())
        source_a = _make_source(name="profile_secret", secrets={"A": "a"})
        source_b = _make_source(name="profile_secret", secrets={"B": "b"})
        assert reg.register_source(source_a, scope=home_a)
        assert reg.register_source(source_b, scope=home_b)

        token = set_hermes_home_override(home_a)
        try:
            assert reg.get_source("profile_secret") is source_a
            explicit_b_env = {}
            report = reg.apply_all(
                {"profile_secret": {"enabled": True}},
                Path(home_b),
                environ=explicit_b_env,
            )
            assert report.sources[0].result.secrets == {"B": "b"}
            assert explicit_b_env == {"B": "b"}
        finally:
            reset_hermes_home_override(token)
        token = set_hermes_home_override(home_b)
        try:
            assert reg.get_source("profile_secret") is source_b
        finally:
            reset_hermes_home_override(token)








# ---------------------------------------------------------------------------
# apply_all: precedence, conflicts, protection
# ---------------------------------------------------------------------------


class TestApplyAll:
    def test_disabled_sources_do_not_run(self, tmp_path):
        called = []

        def _fetch(cfg, home):
            called.append(True)
            return FetchResult(secrets={"A": "1"})

        reg.register_source(_make_source(fetch_fn=_fetch))
        env: dict = {}
        report = reg.apply_all({"dummy": {"enabled": False}}, tmp_path, environ=env)
        assert not called
        assert not report.sources
        assert env == {}

    def test_applies_secrets_and_records_provenance(self, tmp_path):
        reg.register_source(_make_source(secrets={"API_KEY": "v1"}))
        env: dict = {}
        report = reg.apply_all({"dummy": {"enabled": True}}, tmp_path, environ=env)
        assert env["API_KEY"] == "v1"
        assert report.provenance["API_KEY"].source == "dummy"
        assert report.provenance["API_KEY"].shape == "mapped"
        assert report.provenance["API_KEY"].overrode_env is False








    def test_failed_source_does_not_block_others(self, tmp_path):
        reg.register_source(
            _make_source(name="broken", error="boom", error_kind=ErrorKind.NETWORK)
        )
        reg.register_source(_make_source(name="works", secrets={"K": "v"}))
        env: dict = {}
        report = reg.apply_all(
            {"broken": {"enabled": True}, "works": {"enabled": True}},
            tmp_path, environ=env,
        )
        assert env["K"] == "v"
        broken = [s for s in report.sources if s.name == "broken"][0]
        assert broken.result.error_kind is ErrorKind.NETWORK




    def test_malformed_secrets_cfg_shapes_are_safe(self, tmp_path):
        reg.register_source(_make_source(secrets={"K": "v"}))
        for cfg in (None, [], "junk", {"dummy": "not-a-dict"}, {"sources": "junk"}):
            report = reg.apply_all(cfg, tmp_path, environ={})
            assert isinstance(report, reg.ApplyReport)


    def test_scoped_apply_fails_closed_on_legacy_source_without_environ(
        self, tmp_path
    ):
        """A profile-scoped apply must NOT run a legacy source whose fetch()
        lacks the 'environ' param — env-less it would read process os.environ
        (another profile's env under multiplexing). It fails closed instead."""
        # _make_source builds a fetch(self, cfg, home_path) — no 'environ' param.
        reg.register_source(_make_source(secrets={"K": "leaked"}))
        env: dict = {}
        report = reg.apply_all(
            {"dummy": {"enabled": True}}, tmp_path, environ=env, scoped=True
        )
        # Value never applied; source reported an error, not a silent env-read.
        assert "K" not in env
        assert report.sources[0].result.ok is False
        assert report.sources[0].result.error_kind is ErrorKind.NOT_CONFIGURED

    def test_scoped_apply_runs_environ_aware_source(self, tmp_path):
        """A source that DOES accept 'environ' still runs under a scoped apply
        and receives the scoped mapping (not process os.environ)."""
        seen = {}

        class _EnvSrc(SecretSource):
            def fetch(self, cfg, home_path, environ=None):
                seen["environ"] = environ
                res = FetchResult()
                res.secrets = {"K": "scoped-value"}
                return res

            def override_existing(self, cfg):
                return False

            def protected_env_vars(self, cfg):
                return frozenset()

        _EnvSrc.name = "envsrc"
        _EnvSrc.label = "EnvSrc"
        _EnvSrc.shape = "mapped"
        _EnvSrc.scheme = None
        _EnvSrc.api_version = SECRET_SOURCE_API_VERSION
        reg.register_source(_EnvSrc())

        env = {"BOOTSTRAP": "tok"}
        report = reg.apply_all(
            {"envsrc": {"enabled": True}}, tmp_path, environ=env, scoped=True
        )
        assert seen["environ"] is env  # scoped mapping, not os.environ
        assert env["K"] == "scoped-value"
        assert report.sources[0].result.ok is True

    def test_default_apply_still_runs_legacy_source_without_environ(self, tmp_path):
        """Non-scoped (single-profile) apply keeps backward compat: a legacy
        env-less source still runs so existing deployments are unaffected."""
        reg.register_source(_make_source(secrets={"K": "v"}))
        env: dict = {}
        report = reg.apply_all(
            {"dummy": {"enabled": True}}, tmp_path, environ=env, scoped=False
        )
        assert env["K"] == "v"
        assert report.sources[0].result.ok is True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_valid_env_name(self):
        assert is_valid_env_name("GOOD_NAME")
        assert is_valid_env_name("_LEADING")
        assert not is_valid_env_name("")
        assert not is_valid_env_name("1BAD")
        assert not is_valid_env_name("bad-name")
        assert not is_valid_env_name("has space")


    def test_run_secret_cli_minimal_env(self):
        proc = run_secret_cli(
            [sys.executable, "-c",
             "import os, json; print(json.dumps(sorted(os.environ)))"],
        )
        import json

        child_env = json.loads(proc.stdout)
        # No credential-bearing vars from the parent env leak through.
        assert not any(k.endswith(("_API_KEY", "_TOKEN", "_SECRET"))
                       for k in child_env)
        assert "NO_COLOR" in child_env





# ---------------------------------------------------------------------------
# Bitwarden adapter
# ---------------------------------------------------------------------------


class TestBitwardenSource:





    def test_fetch_delegates_to_fetch_bitwarden_secrets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
        import agent.secret_sources.bitwarden as bw

        monkeypatch.setattr(bw, "find_bws", lambda **kw: Path("/fake/bws"))
        captured = {}

        def _fake_fetch(**kwargs):
            captured.update(kwargs)
            return {"MY_KEY": "val"}, ["a warning"]

        monkeypatch.setattr(bw, "fetch_bitwarden_secrets", _fake_fetch)
        result = BitwardenSource().fetch(
            {"enabled": True, "project_id": "proj",
             "server_url": " https://vault.bitwarden.eu "},
            tmp_path,
        )
        assert result.ok
        assert result.secrets == {"MY_KEY": "val"}
        assert result.warnings == ["a warning"]
        assert captured["project_id"] == "proj"
        assert captured["server_url"] == "https://vault.bitwarden.eu"
        assert captured["home_path"] == tmp_path


    def test_e2e_through_orchestrator(self, tmp_path, monkeypatch):
        """Full path: registry → BitwardenSource → env, with fetch mocked."""
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
        import agent.secret_sources.bitwarden as bw

        monkeypatch.setattr(bw, "find_bws", lambda **kw: Path("/fake/bws"))
        monkeypatch.setattr(
            bw, "fetch_bitwarden_secrets",
            lambda **kw: ({"ANTHROPIC_API_KEY": "sk-ant", "BWS_ACCESS_TOKEN": "steal"}, []),
        )
        reg.register_source(BitwardenSource())
        env = {"BWS_ACCESS_TOKEN": "0.token"}
        report = reg.apply_all(
            {"bitwarden": {"enabled": True, "project_id": "proj"}},
            tmp_path, environ=env,
        )
        assert env["ANTHROPIC_API_KEY"] == "sk-ant"
        # The bootstrap token is protected even though BSM carried it.
        assert env["BWS_ACCESS_TOKEN"] == "0.token"
        assert report.provenance["ANTHROPIC_API_KEY"].source == "bitwarden"


# ---------------------------------------------------------------------------
# Conformance kit applied to the bundled source
# ---------------------------------------------------------------------------


class TestBitwardenConformance(SecretSourceConformance):
    @pytest.fixture
    def source(self, monkeypatch):
        # Never hit the network / auto-install path in conformance runs.
        import agent.secret_sources.bitwarden as bw

        monkeypatch.setattr(bw, "find_bws", lambda **kw: None)
        monkeypatch.delenv("BWS_ACCESS_TOKEN", raising=False)
        return BitwardenSource()


# ---------------------------------------------------------------------------
# 1Password adapter
# ---------------------------------------------------------------------------


class TestOnePasswordSource:







    def test_mapped_op_beats_bulk_bitwarden_through_orchestrator(
        self, tmp_path, monkeypatch
    ):
        """The headline multi-source scenario: both vaults claim the same var."""
        import agent.secret_sources.bitwarden as bw
        import agent.secret_sources.onepassword as op

        monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token")
        monkeypatch.setattr(bw, "find_bws", lambda **kw: Path("/fake/bws"))
        monkeypatch.setattr(
            bw, "fetch_bitwarden_secrets",
            lambda **kw: ({"SHARED_KEY": "from-bitwarden",
                           "BW_ONLY": "bw-val"}, []),
        )
        monkeypatch.setattr(op, "find_op", lambda *_a, **_kw: Path("/fake/op"))
        monkeypatch.setattr(
            op, "fetch_onepassword_secrets",
            lambda **kw: ({"SHARED_KEY": "from-1password"}, []),
        )
        reg.register_source(bw.BitwardenSource())
        reg.register_source(op.OnePasswordSource())
        env = {"BWS_ACCESS_TOKEN": "0.token"}
        report = reg.apply_all(
            {
                # bitwarden listed FIRST — mapped 1Password must still win.
                "sources": ["bitwarden", "onepassword"],
                "bitwarden": {"enabled": True, "project_id": "proj"},
                "onepassword": {"enabled": True,
                                "env": {"SHARED_KEY": "op://V/I/F"}},
            },
            tmp_path, environ=env,
        )
        assert env["SHARED_KEY"] == "from-1password"
        assert env["BW_ONLY"] == "bw-val"
        assert report.provenance["SHARED_KEY"].source == "onepassword"
        assert report.provenance["BW_ONLY"].source == "bitwarden"
        assert report.conflicts  # the shadowed bitwarden claim is surfaced


class TestOnePasswordConformance(SecretSourceConformance):
    @pytest.fixture
    def source(self, monkeypatch):
        import agent.secret_sources.onepassword as op

        monkeypatch.setattr(op, "find_op", lambda *_a, **_kw: None)
        monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
        return op.OnePasswordSource()


class TestApplyAllForwardsRawEnviron:
    """apply_all must forward the RAW ``environ`` (None-preserving) to each
    source's fetch(), not the materialized os.environ.

    Regression: apply_all passed the materialized ``env`` (which is os.environ
    when environ=None) into the fetch call, so OnePasswordSource saw a non-None
    environ and switched to isolated mode (include_process_auth=False) even on
    the default load_hermes_dotenv path. That dropped an interactive `op`
    session's process auth at startup. The fetch must receive None on the
    default path and the explicit dict only for profile-scoped builds.
    """

    def _recording_source(self):
        seen = {}

        class _Src(SecretSource):
            name = "recorder"
            shape = "bulk"

            def fetch(self, cfg, home_path, *, environ=None):
                seen["environ"] = environ
                return FetchResult()

            def override_existing(self, cfg):
                return False

            def protected_env_vars(self, cfg):
                return frozenset()

        return _Src(), seen

    def test_default_path_forwards_none(self, tmp_path, monkeypatch):
        src, seen = self._recording_source()
        monkeypatch.setattr(
            reg, "_ordered_enabled_sources", lambda cfg, **_: [src]
        )
        # environ omitted → default load_hermes_dotenv path.
        reg.apply_all({"recorder": {"enabled": True}}, tmp_path)
        assert seen["environ"] is None

    def test_scoped_path_forwards_the_dict(self, tmp_path, monkeypatch):
        src, seen = self._recording_source()
        monkeypatch.setattr(
            reg, "_ordered_enabled_sources", lambda cfg, **_: [src]
        )
        scoped = {"ANTHROPIC_API_KEY": "sk-scoped"}
        reg.apply_all({"recorder": {"enabled": True}}, tmp_path, environ=scoped)
        assert seen["environ"] is scoped


class TestFetchInheritsProfileContext:
    """A source's fetch() runs in a ThreadPoolExecutor worker; the profile
    contextvars (HERMES_HOME override + secret scope) installed by
    _profile_runtime_scope must propagate into that worker via copy_context().

    Regression for the #100 P2: without copy_context, a source that consults
    get_hermes_home() internally (e.g. Bitwarden's find_bws -> managed
    <hermes_home>/bin/bws lookup) resolves the DEFAULT profile's path in the
    worker thread, so a named profile could miss its own binary / use the
    default profile's copy.
    """

    def _home_recording_source(self):
        seen = {}

        class _Src(SecretSource):
            name = "homerec"
            shape = "bulk"

            def fetch(self, cfg, home_path, *, environ=None):
                # Read the HERMES_HOME contextvar override from INSIDE the
                # worker thread — exactly what find_bws() does via
                # _hermes_bin_dir() -> get_hermes_home().
                from hermes_constants import get_hermes_home

                seen["home"] = str(get_hermes_home())
                return FetchResult()

            def override_existing(self, cfg):
                return False

            def protected_env_vars(self, cfg):
                return frozenset()

        return _Src(), seen

    def test_fetch_worker_sees_hermes_home_override(self, tmp_path, monkeypatch):
        from hermes_constants import (
            set_hermes_home_override,
            reset_hermes_home_override,
        )

        src, seen = self._home_recording_source()
        monkeypatch.setattr(reg, "_ordered_enabled_sources", lambda cfg, **_: [src])

        profile_home = tmp_path / ".hermes" / "profiles" / "coder"
        profile_home.mkdir(parents=True)

        # Install the override contextvar in THIS thread, exactly like
        # _profile_runtime_scope does, then run apply_all (which fetches on a
        # worker thread). The worker must see the override, not the default.
        token = set_hermes_home_override(str(profile_home))
        try:
            reg.apply_all({"homerec": {"enabled": True}}, profile_home)
        finally:
            reset_hermes_home_override(token)

        assert seen["home"] == str(profile_home), (
            "fetch worker thread must inherit the HERMES_HOME override "
            "contextvar via copy_context, not fall back to the default profile"
        )

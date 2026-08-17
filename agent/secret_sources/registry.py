"""Secret-source registry + apply orchestrator.

This module owns everything that must be uniform across secret backends
so no individual source can get it wrong:

* registration (name/scheme uniqueness, API-version gating)
* per-source wall-clock timeout enforcement around ``fetch()``
* precedence: mapped sources beat bulk sources; within a shape,
  ``secrets.sources`` order (or registration order) decides; first
  claim wins — later sources never silently clobber an earlier one
* ``override_existing`` semantics (may beat .env/shell, never another
  secret source, never a protected var)
* cross-source conflict warnings (shadowed claims are always surfaced)
* provenance: which source supplied every applied var

The single entry point for startup is :func:`apply_all`, called from
``hermes_cli.env_loader._apply_external_secret_sources()``.

Plugins register additional sources via
``PluginContext.register_secret_source()`` which lands in
:func:`register_source`.  In-tree sources are registered lazily by
:func:`_ensure_builtin_sources` — the set of bundled sources is
deliberately closed (Bitwarden, and 1Password once it lands); new
third-party backends ship as standalone plugin repos implementing
:class:`agent.secret_sources.base.SecretSource`.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import inspect
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional

from agent.secret_sources.base import (
    SECRET_SOURCE_API_VERSION,
    ErrorKind,
    FetchResult,
    SecretSource,
    is_valid_env_name,
    reset_source_environment,
    set_source_environment,
)
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

# Ordered registry: name → source instance.  Python dicts preserve
# insertion order, which doubles as the default apply order. Origin is
# recorded beside each source so consumers never infer ownership from names.
_SOURCES: Dict[str, SecretSource] = {}
_SOURCE_ORIGINS: Dict[str, str] = {}
_SCOPED_SOURCES: Dict[str, Dict[str, SecretSource]] = {}
_BUILTINS_LOADED = False
_REGISTRY_LOCK = threading.RLock()


@dataclass
class AppliedVar:
    """Provenance record for one env var the orchestrator set."""

    name: str
    source: str          # SecretSource.name
    shape: str           # "mapped" | "bulk"
    overrode_env: bool   # replaced a pre-existing .env/shell value


@dataclass
class SourceReport:
    """One source's outcome within an :class:`ApplyReport`."""

    name: str
    label: str
    result: FetchResult
    applied: List[str] = field(default_factory=list)
    skipped_existing: List[str] = field(default_factory=list)   # .env/shell won
    skipped_claimed: List[str] = field(default_factory=list)    # earlier source won
    skipped_protected: List[str] = field(default_factory=list)  # bootstrap-auth guard
    skipped_invalid: List[str] = field(default_factory=list)    # bad env-var name


@dataclass
class ApplyReport:
    """Merged outcome of one orchestrated apply pass."""

    sources: List[SourceReport] = field(default_factory=list)
    provenance: Dict[str, AppliedVar] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)  # human-readable warnings

    @property
    def applied_any(self) -> bool:
        return bool(self.provenance)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_source(
    source: SecretSource,
    *,
    replace: bool = False,
    builtin: bool = False,
    scope: Optional[str] = None,
) -> bool:
    """Register a secret source.  Returns True on success.

    Rejections are logged, never raised — a bad plugin must not take
    down startup.  ``replace`` allows tests / user plugins to override
    a bundled source of the same name (last-writer-wins like model
    providers), but scheme collisions across *different* names are
    always rejected.
    """
    if not isinstance(source, SecretSource):
        logger.warning(
            "Ignoring secret source %r: does not inherit from SecretSource",
            source,
        )
        return False
    name = getattr(source, "name", "") or ""
    if not name or not name.replace("_", "").isalnum() or name != name.lower():
        logger.warning("Ignoring secret source with invalid name %r", name)
        return False
    if getattr(source, "api_version", None) != SECRET_SOURCE_API_VERSION:
        logger.warning(
            "Ignoring secret source '%s': built against secret-source API v%s, "
            "this Hermes speaks v%s",
            name, getattr(source, "api_version", "?"), SECRET_SOURCE_API_VERSION,
        )
        return False
    if getattr(source, "shape", None) not in ("mapped", "bulk"):
        logger.warning(
            "Ignoring secret source '%s': shape must be 'mapped' or 'bulk', got %r",
            name, getattr(source, "shape", None),
        )
        return False
    with _REGISTRY_LOCK:
        effective = dict(_SOURCES)
        if scope is not None:
            effective.update(_SCOPED_SOURCES.get(scope, {}))
        if name in effective and not replace:
            logger.warning(
                "Secret source '%s' already registered; ignoring duplicate", name
            )
            return False
        scheme = getattr(source, "scheme", None)
        if scheme:
            for other_name, other in effective.items():
                if other_name != name and getattr(other, "scheme", None) == scheme:
                    logger.warning(
                        "Ignoring secret source '%s': scheme '%s://' is already "
                        "owned by source '%s'",
                        name,
                        scheme,
                        other_name,
                    )
                    return False
        target = _SOURCES if scope is None else _SCOPED_SOURCES.setdefault(scope, {})
        target[name] = source
        if scope is None:
            _SOURCE_ORIGINS[name] = "builtin" if builtin else "plugin"
    return True


def get_source(name: str, *, scope: Optional[str] = None) -> Optional[SecretSource]:
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        return _SCOPED_SOURCES.get(scope or hermes_home_key(), {}).get(
            name
        ) or _SOURCES.get(name)


def snapshot_registration(
    name: str, *, scope: Optional[str] = None
) -> Optional[SecretSource]:
    """Return the registration owned by exactly one registry layer."""
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        target = _SOURCES if scope is None else _SCOPED_SOURCES.get(scope, {})
        return target.get(name)


def restore_registration(
    name: str,
    current: SecretSource,
    previous: Optional[SecretSource],
    *,
    scope: Optional[str] = None,
) -> bool:
    """Restore a host-owned source registration if it is still current."""
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        target = _SOURCES if scope is None else _SCOPED_SOURCES.setdefault(scope, {})
        if target.get(name) is not current:
            return False
        if previous is None:
            target.pop(name, None)
        else:
            target[name] = previous
        if scope is not None and not target:
            _SCOPED_SOURCES.pop(scope, None)
    return True


def list_sources(*, scope: Optional[str] = None) -> List[SecretSource]:
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        merged = dict(_SOURCES)
        merged.update(_SCOPED_SOURCES.get(scope or hermes_home_key(), {}))
        return list(merged.values())


def list_plugin_sources() -> List[SecretSource]:
    """Return sources registered outside the bundled bootstrap set.

    Includes both legacy global plugin registrations (``_SOURCE_ORIGINS ==
    "plugin"``) and the current scope's profile-keyed registrations — every
    scoped entry is plugin-registered by definition, since bundled sources
    register with ``scope=None`` (#64229 profile isolation).
    """
    _ensure_builtin_sources()
    with _REGISTRY_LOCK:
        merged: Dict[str, SecretSource] = {
            name: source
            for name, source in _SOURCES.items()
            if _SOURCE_ORIGINS.get(name) == "plugin"
        }
        merged.update(_SCOPED_SOURCES.get(hermes_home_key(), {}))
        return list(merged.values())


def _ensure_builtin_sources() -> None:
    """Idempotently register the bundled sources.

    Lazy so importing this module stays cheap and so a broken bundled
    source can never break registration of the others.
    """
    global _BUILTINS_LOADED
    with _REGISTRY_LOCK:
        if _BUILTINS_LOADED:
            return
        _BUILTINS_LOADED = True
        try:
            from agent.secret_sources.bitwarden import BitwardenSource

            register_source(BitwardenSource(), builtin=True)
        except Exception:  # noqa: BLE001 — never block startup
            logger.warning(
                "Failed to register bundled Bitwarden secret source",
                exc_info=True,
            )
        try:
            from agent.secret_sources.onepassword import OnePasswordSource

            register_source(OnePasswordSource(), builtin=True)
        except Exception:  # noqa: BLE001 — never block startup
            logger.warning(
                "Failed to register bundled 1Password secret source",
                exc_info=True,
            )
        try:
            from agent.secret_sources.command import CommandSource

            register_source(CommandSource(), builtin=True)
        except Exception:  # noqa: BLE001 — never block startup
            logger.warning(
                "Failed to register bundled command secret source",
                exc_info=True,
            )


def _reset_registry_for_tests() -> None:
    global _BUILTINS_LOADED
    with _REGISTRY_LOCK:
        _SOURCES.clear()
        _SOURCE_ORIGINS.clear()
        _SCOPED_SOURCES.clear()
        _BUILTINS_LOADED = False


# ---------------------------------------------------------------------------
# Orchestrated apply
# ---------------------------------------------------------------------------


# Sentinel: ``None`` is a MEANINGFUL value for the environ forwarded to
# fetch() (it selects process-auth mode), so "argument omitted" needs its own
# marker distinct from None.
_RAW_ENVIRON_UNSET = object()


def _fetch_with_timeout(
    source: SecretSource, cfg: dict, home_path: Path,
    environ: MutableMapping[str, str],
    scoped: bool = False,
    raw_environ: Any = _RAW_ENVIRON_UNSET,
) -> FetchResult:
    """Run source.fetch() under a wall-clock budget; never raises.

    ``environ`` is the MATERIALIZED mapping (``os.environ`` on the default
    path) used for the source-environment contextvar. ``raw_environ`` is the
    caller's None-preserving value and is what gets forwarded to ``fetch()``:
    OnePasswordSource treats any non-None ``environ`` as isolated mode
    (``include_process_auth=False``), so forwarding the materialized
    ``os.environ`` on the default path would run ``op`` with an isolated HOME
    and drop an interactive 1Password session. Defaults to ``environ`` so
    existing callers keep their behaviour.

    ``scoped`` marks a genuine profile-isolation apply (a named profile under
    ``gateway.multiplex_profiles``): a source whose ``fetch()`` cannot consume
    ``environ`` is failed closed rather than run against the process
    ``os.environ`` (which under multiplexing is another profile's env). The
    generic ``apply_all`` path (``scoped=False``) keeps the legacy env-less
    call so pre-``environ`` sources still work for single-profile deployments.

    The budget is enforced with a daemon worker thread: a source that
    blows its budget is reported as ``TIMEOUT`` and its (eventual)
    result is discarded.  The thread itself may linger until process
    exit — acceptable for a startup-only path, and strictly better than
    an unbounded hang on every ``hermes`` invocation.
    """
    timeout = source.fetch_timeout_seconds(cfg)
    # Copy the caller's context so the worker thread inherits the profile
    # contextvars (HERMES_HOME override + active secret scope) installed by
    # ``_profile_runtime_scope``. Without this, a source whose fetch() consults
    # ``get_hermes_home()`` internally (e.g. Bitwarden's ``find_bws`` → managed
    # ``<hermes_home>/bin/bws`` lookup) resolves the DEFAULT profile's path in
    # the plain ThreadPoolExecutor worker, since ContextVars do not propagate to
    # pool threads. Mirrors the copy_context() pattern in account_usage.py.
    ctx = contextvars.copy_context()
    # None-preserving value forwarded to fetch(); see the docstring. _UNSET (not
    # None) marks "caller didn't distinguish", because None is itself meaningful
    # here — it is what keeps the default path in process-auth mode.
    fetch_environ = environ if raw_environ is _RAW_ENVIRON_UNSET else raw_environ
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"secret-src-{source.name}"
    )
    try:
        # Fail closed BEFORE dispatching: under a profile-scoped apply (a named
        # profile in gateway.multiplex_profiles) a source whose fetch() cannot
        # take ``environ`` would read bootstrap credentials from the process
        # ``os.environ`` — another profile's env — and populate this profile
        # from another profile's vault. Refuse rather than leak. Upstream's
        # contextvar handoff below covers sources that DO take ``environ``.
        fetch_params = inspect.signature(source.fetch).parameters
        if scoped and "environ" not in fetch_params:
            res = FetchResult()
            res.error = (
                f"secret source '{source.name}' cannot consume a scoped "
                "credential environment (its fetch() lacks the 'environ' "
                "parameter) — skipped in profile-scoped mode to prevent "
                "cross-profile credential leakage. Update the source to the "
                "current secret-source API to use it under multiplexing."
            )
            res.error_kind = ErrorKind.NOT_CONFIGURED
            return res

        def _fetch() -> FetchResult:
            token = set_source_environment(environ)
            try:
                if "environ" in fetch_params:
                    return source.fetch(cfg, home_path, environ=fetch_environ)
                return source.fetch(cfg, home_path)
            finally:
                reset_source_environment(token)

        future = executor.submit(ctx.run, _fetch)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            res = FetchResult()
            res.error = (
                f"fetch exceeded {timeout:.0f}s budget — startup continued "
                "without this source (raise secrets."
                f"{source.name}.timeout_seconds if the backend is just slow)"
            )
            res.error_kind = ErrorKind.TIMEOUT
            return res
        except Exception as exc:  # noqa: BLE001 — contract violation, contain it
            res = FetchResult()
            res.error = f"fetch raised {type(exc).__name__}: {exc}"
            res.error_kind = ErrorKind.INTERNAL
            return res
    finally:
        executor.shutdown(wait=False)

    if not isinstance(result, FetchResult):
        res = FetchResult()
        res.error = (
            f"fetch returned {type(result).__name__} instead of FetchResult"
        )
        res.error_kind = ErrorKind.INTERNAL
        return res
    return result


def _ordered_enabled_sources(
    secrets_cfg: dict, *, scope: Optional[str] = None
) -> List[SecretSource]:
    """Resolve which sources run, in which order.

    Order: the optional ``secrets.sources`` list wins; sources not named
    there follow in registration order.  Enabled = the source's own
    ``is_enabled`` says so for its config section.  Mapped-vs-bulk
    precedence is applied on top of this order by :func:`apply_all`.
    """
    sources = {source.name: source for source in list_sources(scope=scope)}

    explicit = secrets_cfg.get("sources")
    order: List[str] = []
    if isinstance(explicit, list):
        for entry in explicit:
            if isinstance(entry, str) and entry in sources and entry not in order:
                order.append(entry)
        unknown = [e for e in explicit
                   if isinstance(e, str) and e not in sources]
        if unknown:
            logger.warning(
                "secrets.sources names unknown source(s): %s (known: %s)",
                ", ".join(unknown), ", ".join(sources) or "none",
            )
    for name in sources:
        if name not in order:
            order.append(name)

    enabled: List[SecretSource] = []
    for name in order:
        source = sources[name]
        cfg = secrets_cfg.get(name)
        cfg = cfg if isinstance(cfg, dict) else {}
        try:
            if source.is_enabled(cfg):
                enabled.append(source)
        except Exception:  # noqa: BLE001
            logger.warning("Secret source '%s' is_enabled() raised; skipping",
                           name, exc_info=True)
    return enabled


def _active_profile_name(home_path: Optional[Path]) -> str:
    """Best-effort active profile name for profile-scoped secret aliases.

    A named profile's HERMES_HOME is ``~/.hermes/profiles/<name>``; the
    default profile (``~/.hermes``) returns "".
    """
    if home_path is not None:
        resolved = Path(home_path)
        if resolved.parent.name == "profiles" and resolved.name:
            return resolved.name
    for env_name in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        value = os.environ.get(env_name, "").strip()
        if value and value != "default":
            return value
    return ""


# Only credential-shaped names get auto-aliased — a random profile-suffixed
# var should not silently hydrate an unsuffixed name.
_ALIAS_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY", "_PASSWORD")


def _profile_alias_target(var: str, profile: str) -> Optional[str]:
    """Map ``FOO_<PROFILE>`` to ``FOO`` for the active profile when safe."""
    if not profile:
        return None
    suffix = "_" + profile.replace("-", "_").upper()
    if not var.endswith(suffix):
        return None
    alias = var[: -len(suffix)]
    if not alias or not is_valid_env_name(alias):
        return None
    if not any(alias.endswith(s) for s in _ALIAS_SUFFIXES):
        return None
    return alias


def apply_all(secrets_cfg: dict, home_path: Path,
              environ: Optional[MutableMapping[str, str]] = None,
              scoped: bool = False) -> ApplyReport:
    """Fetch from every enabled source and apply the merged result to env.

    ``environ`` defaults to ``os.environ``; injectable for tests.

    ``scoped`` marks a genuine profile-isolation build (a named profile under
    ``gateway.multiplex_profiles``). When true, a source whose ``fetch()``
    cannot consume ``environ`` is failed closed instead of run against the
    process environment, so a named profile can never be populated from another
    profile's vault. Leave ``scoped=False`` for the default single-profile
    apply path.

    Precedence per env var (most-specific intent wins):

    1. ``secrets.preserve_existing`` names — a pre-existing env value always
       wins for these, even against a source with ``override_existing: true``
       (escape hatch for profile-local platform secrets, #58073).
    2. Pre-existing env (.env / shell) — unless the winning source has
       ``override_existing: true``.
    3. Mapped sources, in configured order.
    4. Bulk sources, in configured order.

    First claim wins.  A later source that also carries the var gets a
    ``skipped_claimed`` entry and a conflict warning — never a silent
    clobber, and ``override_existing`` never applies across sources.

    Profile aliasing (#51447): when running under a named profile, an applied
    var ``FOO_<PROFILE>`` (credential-shaped suffixes only) also hydrates the
    canonical ``FOO`` so platform adapters and plugins that read fixed env
    names see the profile's value.  The alias obeys the same protected /
    preserve / claimed / override guards and is disabled with
    ``secrets.profile_alias: false``.
    """
    import os as _os

    # ``env`` is the materialized mapping the merge/precedence phase below reads
    # and writes (env.get / env[var] = ...). It must NOT be forwarded to the
    # sources' fetch(): OnePasswordSource treats any non-None ``environ`` as
    # isolated mode (include_process_auth=False), so passing the materialized
    # os.environ on the default load_hermes_dotenv path (environ=None) would run
    # ``op`` with an isolated HOME and drop an interactive 1Password session. The
    # fetch call forwards the RAW ``environ`` (None-preserving) so process-auth
    # mode is kept on the default path and isolated only for explicit
    # profile-scoped builds that pass a real dict.
    env = environ if environ is not None else _os.environ
    report = ApplyReport()

    secrets_cfg = secrets_cfg if isinstance(secrets_cfg, dict) else {}
    enabled = _ordered_enabled_sources(
        secrets_cfg, scope=hermes_home_key(home_path)
    )
    if not enabled:
        return report

    preserve_raw = secrets_cfg.get("preserve_existing")
    preserve: frozenset = frozenset(
        n.strip() for n in preserve_raw if isinstance(n, str) and n.strip()
    ) if isinstance(preserve_raw, list) else frozenset()

    alias_enabled = bool(secrets_cfg.get("profile_alias", True))
    profile = _active_profile_name(home_path) if alias_enabled else ""

    # Mapped sources outrank bulk sources regardless of list order:
    # an explicit VAR→ref binding is stronger intent than a project dump.
    ordered = ([s for s in enabled if s.shape == "mapped"]
               + [s for s in enabled if s.shape == "bulk"])

    # Fetch phase.
    fetches: List[tuple[SecretSource, dict, FetchResult]] = []
    protected: Dict[str, str] = {}  # var → source that protects it
    for source in ordered:
        cfg = secrets_cfg.get(source.name)
        cfg = cfg if isinstance(cfg, dict) else {}
        result = _fetch_with_timeout(
            source, cfg, home_path, env, scoped=scoped, raw_environ=environ
        )
        fetches.append((source, cfg, result))
        try:
            for var in source.protected_env_vars(cfg):
                protected.setdefault(var, source.name)
        except Exception:  # noqa: BLE001
            pass

    # Every var any source supplies directly — an alias never shadows a
    # var that some source will (or tried to) claim by its real name.
    supplied_directly: set = set()
    for _, _, result in fetches:
        if result.ok:
            supplied_directly.update(
                v for v in result.secrets if isinstance(v, str)
            )

    # Apply phase — sequential, first-wins, fully attributed.
    claimed: Dict[str, str] = {}  # var → source name that won it
    for source, cfg, result in fetches:
        sr = SourceReport(name=source.name,
                          label=source.label or source.name,
                          result=result)
        report.sources.append(sr)
        if not result.ok:
            continue

        try:
            override = source.override_existing(cfg)
        except Exception:  # noqa: BLE001
            override = False

        def _try_apply(var: str, value: str, *, is_alias: bool = False) -> bool:
            """Apply one var through the shared guard chain. True = applied."""
            if not is_valid_env_name(var):
                sr.skipped_invalid.append(var)
                return False
            if var in protected:
                sr.skipped_protected.append(var)
                return False
            if var in claimed:
                sr.skipped_claimed.append(var)
                report.conflicts.append(
                    f"{var}: kept value from {claimed[var]}; "
                    f"{source.name} also supplies it (first source wins — "
                    "remove one binding or reorder secrets.sources)"
                )
                return False
            existed = bool(env.get(var))
            if existed and var in preserve:
                sr.skipped_existing.append(var)
                return False
            if existed and not override:
                sr.skipped_existing.append(var)
                return False
            env[var] = value
            claimed[var] = source.name
            sr.applied.append(var)
            report.provenance[var] = AppliedVar(
                name=var,
                source=source.name,
                shape=source.shape,
                overrode_env=existed,
            )
            return True

        for var, value in result.secrets.items():
            if not isinstance(var, str) or not isinstance(value, str):
                continue
            applied = _try_apply(var, value)

            if not applied or not profile:
                continue
            alias = _profile_alias_target(var, profile)
            if alias and alias not in supplied_directly and alias not in claimed:
                if _try_apply(alias, value, is_alias=True):
                    result.warnings.append(
                        f"applied profile-scoped {var} as {alias} "
                        f"(active profile {profile!r})"
                    )

    return report

"""Context-local state for delegate_task child execution.

A Hermes process may itself be a Kanban dispatcher worker with HERMES_KANBAN_* in
os.environ. In-process delegate_task children and cron jobs fired via
``cronjob(action="run")`` are NOT dispatcher-owned, so identity gates must fail
closed for them without mutating the process-global environment.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterator, Mapping, MutableMapping

_DELEGATED_CHILD_CONTEXT: ContextVar[bool] = ContextVar("hermes_delegated_child_context", default=False)
# Any in-process execution that is NOT the dispatcher-owned worker (cron jobs). Kept separate
# so delegate_task-specific behaviour (subprocess env scrubbing, its error strings) is unchanged.
_NON_DISPATCHER_OWNED_CONTEXT: ContextVar[bool] = ContextVar("hermes_non_dispatcher_owned_context", default=False)
_DELEGATED_CHILD_SURFACE: ContextVar[str] = ContextVar("hermes_delegated_child_surface", default="normal")
_AUDIT_EVIDENCE_PATHS: ContextVar[tuple[str, ...]] = ContextVar("hermes_audit_evidence_paths", default=())

DELEGATED_CHILD_ENV_MARKER = "HERMES_DELEGATED_CHILD_CONTEXT"
AUDIT_CHILD_SURFACE = "audit"
AUDIT_ALLOWED_TOOLS = frozenset({"audit_read_file"})
_EXTERNAL_ISOLATED_BACKENDS = frozenset({"singularity", "modal", "daytona", "vercel_sandbox"})

KANBAN_ENV_KEYS: tuple[str, ...] = (
    "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB",
)


def canonicalize_evidence_paths(paths: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve parent-supplied evidence paths before an audit child can read them."""
    if not paths:
        return ()
    resolved: list[str] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            path = Path(raw).expanduser().resolve(strict=True)
            if path.is_file() or path.is_dir():
                resolved.append(str(path))
        except OSError:
            continue
    return tuple(dict.fromkeys(resolved))


@contextmanager
def delegated_child_context(
    session_id: str | None = None,
    *,
    surface: str = "normal",
    evidence_paths: list[str] | tuple[str, ...] | None = None,
) -> Iterator[None]:
    """Mark child execution and isolate its task-local session identity."""
    if surface not in {"normal", AUDIT_CHILD_SURFACE}:
        raise ValueError(f"unknown delegated child surface: {surface!r}")
    child_token = _DELEGATED_CHILD_CONTEXT.set(True)
    surface_token = _DELEGATED_CHILD_SURFACE.set(surface)
    evidence_token = _AUDIT_EVIDENCE_PATHS.set(canonicalize_evidence_paths(evidence_paths))
    try:
        from gateway.session_context import scoped_current_session_id  # lazy: it calls is_delegated_child_context()

        with scoped_current_session_id(session_id):
            yield
    finally:
        _AUDIT_EVIDENCE_PATHS.reset(evidence_token)
        _DELEGATED_CHILD_SURFACE.reset(surface_token)
        _DELEGATED_CHILD_CONTEXT.reset(child_token)


def is_delegated_child_context() -> bool:
    """Return True while code is running for a delegate_task child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get())


def is_audit_child_context() -> bool:
    """Return True while the child is restricted to supplied evidence reads."""
    return is_delegated_child_context() and _DELEGATED_CHILD_SURFACE.get() == AUDIT_CHILD_SURFACE


def audit_evidence_paths() -> tuple[str, ...]:
    """Return canonical evidence roots granted to the current audit child."""
    return _AUDIT_EVIDENCE_PATHS.get()


def audit_tool_allowed(name: str) -> bool:
    """Tool-dispatch gate for the evidence-only audit surface."""
    return not is_audit_child_context() or name in AUDIT_ALLOWED_TOOLS


def audit_tool_rejection(name: str) -> str:
    """Stable refusal returned when an audit child requests another capability."""
    return (
        f"{name} refused: audit children may only read parent-supplied evidence "
        "through audit_read_file."
    )


def command_child_isolation_available() -> bool:
    """Whether a command-enabled delegated child has an external filesystem boundary."""
    if is_audit_child_context():
        return False
    # A parent dispatcher worker may evaluate this before entering the child
    # ContextVar. Ordinary user-created children without a claim keep legacy behavior.
    if not is_delegated_child_context() and not os.environ.get("HERMES_KANBAN_TASK"):
        return True
    # User-created delegated children outside a dispatcher worker do not inherit
    # a board claim. Preserve their existing coding behavior; worker children need
    # an external boundary because same-user local execution is not one.
    if not os.environ.get("HERMES_KANBAN_TASK"):
        return True
    try:
        from tools.terminal_tool import _get_env_config
        config = _get_env_config()
        env_type = config.get("env_type")
        from tools.terminal_tool_backends import _REQUIREMENT_CHECKERS
        if env_type in _EXTERNAL_ISOLATED_BACKENDS:
            checker = _REQUIREMENT_CHECKERS.get(env_type)
            return bool(checker(config)) if checker is not None else False
        if env_type != "docker":
            return False
        if config.get("host_cwd") or config.get("docker_mount_cwd_to_workspace"):
            return False
        if config.get("docker_volumes"):
            return False
        protected_env = {"HERMES_HOME", "HERMES_KANBAN_DB", "HERMES_KANBAN_TASK", "HERMES_KANBAN_BOARD"}
        forwarded = {str(name) for name in config.get("docker_forward_env", [])}
        explicit_env = config.get("docker_env", {})
        if not isinstance(explicit_env, Mapping):
            return False
        if forwarded & protected_env or set(explicit_env) & protected_env:
            return False
        # Extra args can add an equivalent bind mount or environment pass-through,
        # so an unreviewed ad hoc Docker command is not treated as an isolation boundary.
        if config.get("docker_extra_args"):
            return False
        checker = _REQUIREMENT_CHECKERS.get(env_type)
        return bool(checker(config)) if checker is not None else False
    except Exception:
        return False


def enter_non_dispatcher_owned_context() -> Token[bool]:
    """Token form of :func:`non_dispatcher_owned_context` for long try/finally scopes."""
    return _NON_DISPATCHER_OWNED_CONTEXT.set(True)


def exit_non_dispatcher_owned_context(token: Token[bool]) -> None:
    """Restore the flag saved by :func:`enter_non_dispatcher_owned_context`."""
    _NON_DISPATCHER_OWNED_CONTEXT.reset(token)


@contextmanager
def non_dispatcher_owned_context() -> Iterator[None]:
    """Mark in-process execution that does NOT own the dispatcher's Kanban task; without it
    a cron agent run inside a worker is misread as that worker (kanban toolset force-added,
    ``kanban_complete`` defaulting to its task). ContextVar-scoped rather than clearing
    os.environ, which the worker's claim heartbeat and concurrent readers share."""
    token = enter_non_dispatcher_owned_context()
    try:
        yield
    finally:
        exit_non_dispatcher_owned_context(token)


def is_dispatcher_owned_worker_context() -> bool:
    """The single predicate every ``HERMES_KANBAN_*`` identity gate should use."""
    return not (_DELEGATED_CHILD_CONTEXT.get() or _NON_DISPATCHER_OWNED_CONTEXT.get())


def is_delegated_child_process_context() -> bool:
    """Return True in this process or a subprocess spawned by a child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get()) or bool(os.environ.get(DELEGATED_CHILD_ENV_MARKER))


def scrub_kanban_env(env: Mapping[str, str] | MutableMapping[str, str]) -> dict[str, str]:
    """Return *env* with dispatcher-only Kanban variables removed and the lineage marker set."""
    cleaned = {k: v for k, v in env.items() if k not in KANBAN_ENV_KEYS}
    cleaned[DELEGATED_CHILD_ENV_MARKER] = "1"
    return cleaned


def delegated_child_subprocess_env(
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Env override only when delegated-child lineage must cross fork: preserves ``env=None``
    inherit semantics for non-delegated calls; in a child, a scrubbed env carrying the marker."""
    if not is_delegated_child_process_context():
        return None if env is None else dict(env)
    return scrub_kanban_env(os.environ if env is None else env)

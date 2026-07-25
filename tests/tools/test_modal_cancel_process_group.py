"""Cancelling a Modal command must kill the command, not the sandbox.

Regression coverage for the wedge that took every Modal-backed worker lane
down: `cancel_fn` called `sandbox.terminate()`, so one command timeout
destroyed the sandbox and every later exec returned exit 1 with empty output
for the rest of the session.
"""

import shlex
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    import tools.terminal_tool  # noqa: F401
except ImportError:
    pytest.skip("hermes-agent tools not importable (missing deps)", allow_module_level=True)

from tools.environments.modal import (  # noqa: E402
    _MODAL_CANCEL_GRACE_SECONDS,
    _MODAL_PGID_DIR,
    ModalEnvironment,
    _wrap_for_group_cancel,
)


class _FakeSandbox:
    """Records exec'd scripts; flags any attempt to terminate it."""

    def __init__(self):
        self.commands: list[str] = []
        self.terminated = False

        outer = self

        class _Exec:
            async def aio(self, *args, **kwargs):
                outer.commands.append(args[-1])
                return _FakeProc()

        class _Terminate:
            async def aio(self, *args, **kwargs):
                outer.terminated = True

        self.exec = _Exec()
        self.terminate = _Terminate()


class _FakeProc:
    def __init__(self):
        class _Empty:
            async def aio(self_inner):
                return ""

        class _Zero:
            async def aio(self_inner):
                return 0

        self.stdout = type("S", (), {"read": _Empty()})()
        self.stderr = type("S", (), {"read": _Empty()})()
        self.wait = _Zero()


class _FakeWorker:
    def __init__(self):
        self.stopped = False

    def run_coroutine(self, coro, timeout=600):
        import asyncio

        return asyncio.new_event_loop().run_until_complete(coro)

    def stop(self):
        self.stopped = True


def _env_with_fakes():
    env = ModalEnvironment.__new__(ModalEnvironment)
    env._sandbox = _FakeSandbox()
    env._worker = _FakeWorker()
    return env


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


def test_wrapper_runs_command_in_its_own_process_group():
    """setsid gives the command a process group cancel() can signal."""
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/.hermes-pgid/abc")

    assert "setsid" in wrapped
    assert shlex.quote("echo hi") in wrapped
    # the group id is published by the setsid'd shell reading its OWN pgid,
    # not by the parent guessing $! before setsid(2) has run
    assert "ps -o pgid= -p $$" in wrapped
    assert "echo G:$__pgid" in wrapped
    assert "/tmp/.hermes-pgid/abc" in wrapped


def test_group_id_is_not_published_from_the_parents_bang_pid():
    """$! names the forked process before setsid(2) creates the group.

    Publishing G:$! would let a cancel in that window signal a group that does
    not exist yet (ESRCH) and leave the command running.
    """
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x")

    assert "echo G:$__hermes_pid" not in wrapped


def test_publication_falls_back_to_a_pid_when_ps_is_unavailable():
    """An image with setsid but no working ps must not record a bare "G:".

    A target-less record resolves to "-", signals nothing, and the command
    survives cancellation.
    """
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x")

    assert 'if [ -n "$__pgid" ]; then echo G:$__pgid' in wrapped
    assert "else echo P:$$ >" in wrapped


def test_wrapper_propagates_the_real_exit_code():
    """A wrapped command must not mask the command's own status."""
    wrapped = _wrap_for_group_cancel("false", "/tmp/x")

    assert "wait $__hermes_pid" in wrapped
    assert "exit $__hermes_rc" in wrapped


def test_wrapper_falls_back_when_setsid_is_absent():
    """No setsid still runs the command rather than failing the exec."""
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x")

    assert "command -v setsid" in wrapped
    assert "else bash -c" in wrapped


def test_fallback_records_a_pid_target_not_a_group_target():
    """Without setsid the child shares our group; -pid would hit nothing.

    Signalling -<pid> in the fallback targets a process group that does not
    exist (ESRCH), silently leaving the command running on every cancel.
    """
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x")

    assert "echo P:$__hermes_pid > /tmp/x" in wrapped


def test_wrapper_refuses_to_start_when_already_cancelled():
    """A cancel that lands while the exec is queued must not be lost."""
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x")

    # checked before launching...
    assert "if [ -e /tmp/x.cancel ]; then rm -f /tmp/x.cancel" in wrapped
    assert f"exit {128 + 15}" in wrapped


def test_wrapper_kills_the_child_when_cancel_races_publication():
    """Cancel landing after fork but before the pgid is published still kills.

    The child pid is valid the moment fork returns, unlike the group, so the
    post-launch marker check signals it directly and escalates.
    """
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x")

    assert 'kill -TERM "$__hermes_pid"' in wrapped
    assert f"seq 1 {_MODAL_CANCEL_GRACE_SECONDS}" in wrapped
    assert 'kill -KILL "$__hermes_pid"' in wrapped


def test_wrapper_honors_login_shell():
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x", login=True)

    assert "bash -l -c" in wrapped


def test_wrapper_quotes_the_command():
    """A command containing shell metacharacters must not break the wrapper."""
    nasty = "echo 'a b'; rm -rf /nope & $(whoami)"
    wrapped = _wrap_for_group_cancel(nasty, "/tmp/x")

    assert shlex.quote(nasty) in wrapped


def test_wrapper_cleans_up_its_bookkeeping_files():
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x")

    assert "rm -f /tmp/x /tmp/x.cancel" in wrapped


def test_pgid_files_live_outside_synced_and_snapshotted_paths():
    """Cancellation bookkeeping must not leak into user files or snapshots."""
    assert _MODAL_PGID_DIR.startswith("/tmp/")


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------


def _cancel_scripts(env):
    """The cancel execs, found by content (exec order is thread-dependent)."""
    return [c for c in env._sandbox.commands if "rec=$(cat " in c]


def _cancel_script(env):
    scripts = _cancel_scripts(env)
    assert scripts, f"cancel never exec'd a kill; saw {env._sandbox.commands}"
    return scripts[-1]


def test_cancel_signals_the_process_group_and_spares_the_sandbox():
    """The core regression: cancel must never terminate the sandbox."""
    env = _env_with_fakes()

    handle = env._run_bash("sleep 300")
    handle.kill()

    assert env._sandbox.terminated is False, "cancel tore down the sandbox"
    cancel_script = _cancel_script(env)
    # resolves the recorded tag into a group (-pid) or bare-pid target
    assert "G:*) target=-${rec#G:}" in cancel_script
    assert "P:*) target=${rec#P:}" in cancel_script
    assert 'kill -TERM "$target"' in cancel_script
    assert 'kill -KILL "$target"' in cancel_script


def test_cancel_waits_for_a_late_published_pid():
    """A missing pid file means "not yet", not "nothing to kill"."""
    env = _env_with_fakes()

    handle = env._run_bash("sleep 300")
    handle.kill()

    script = _cancel_script(env)
    # retries the read instead of bailing on the first miss
    assert f"for _ in $(seq 1 {_MODAL_CANCEL_GRACE_SECONDS}); do" in script
    assert "rec=$(cat " in script


def test_cancel_marks_before_reading_so_a_queued_command_still_dies():
    """Cancel arriving before the command starts must not be silently dropped."""
    env = _env_with_fakes()

    handle = env._run_bash("sleep 300")
    handle.kill()

    script = _cancel_script(env)
    marker = script.index("touch ")
    read = script.index("rec=$(cat ")
    assert marker < read, "cancel read the pid file before marking the intent"


def test_cancel_escalates_to_sigkill_after_a_grace_period():
    """A command that ignores SIGTERM still dies."""
    env = _env_with_fakes()

    handle = env._run_bash("trap '' TERM; sleep 300")
    handle.kill()

    script = _cancel_script(env)
    assert f"seq 1 {_MODAL_CANCEL_GRACE_SECONDS}" in script
    # stops escalating early once the target is gone
    assert 'kill -0 "$target"' in script


def test_cancel_is_a_noop_when_the_command_already_finished():
    """No pid file means nothing to kill; must exit clean, not error."""
    env = _env_with_fakes()

    handle = env._run_bash("echo hi")
    handle.kill()

    script = _cancel_script(env)
    assert "|| exit 0" in script


def test_cancel_failure_does_not_destroy_the_sandbox():
    """Even when the cancel exec itself fails, the sandbox survives."""
    env = _env_with_fakes()

    def _boom(coro, timeout=600):
        if hasattr(coro, "close"):
            coro.close()
        raise RuntimeError("modal is having a bad day")

    handle = env._run_bash("sleep 300")
    env._worker.run_coroutine = _boom

    handle.kill()  # must swallow, not raise

    assert env._sandbox.terminated is False


def test_each_command_gets_its_own_pid_file():
    """Two concurrent commands must not cancel each other."""
    env = _env_with_fakes()

    env._run_bash("sleep 300").kill()
    env._run_bash("sleep 300").kill()

    cancels = _cancel_scripts(env)
    assert len(cancels) == 2
    assert cancels[0] != cancels[1], "commands shared a pid file"

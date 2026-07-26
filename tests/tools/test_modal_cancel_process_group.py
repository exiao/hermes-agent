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
    _MODAL_EXEC_TIMEOUT_HEADROOM,
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
    """Job control gives the command a process group cancel() can signal."""
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/.hermes-pgid/abc")

    assert "set -m" in wrapped
    assert shlex.quote("echo hi") in wrapped
    # under `set -m` the backgrounded job leads a new group whose id IS $!,
    # so the pid recorded for cancel() is a valid group target immediately
    assert "echo $__hermes_pgid > /tmp/.hermes-pgid/abc" in wrapped


def test_group_id_needs_no_external_binaries_or_publication_window():
    """setsid/ps derivations added a window where the pgid was not yet valid.

    Job control assigns the group at fork, so there is no window and no
    dependency on binaries that a slim image may not ship.
    """
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x")

    assert "setsid" not in wrapped
    assert "ps -o pgid=" not in wrapped




def test_wrapper_propagates_the_real_exit_code():
    """A wrapped command must not mask the command's own status."""
    wrapped = _wrap_for_group_cancel("false", "/tmp/x")

    assert "wait $__hermes_pgid" in wrapped
    assert "exit $__hermes_rc" in wrapped




def test_wrapper_refuses_to_start_when_already_cancelled():
    """A cancel that lands while the exec is queued must not be lost."""
    wrapped = _wrap_for_group_cancel("echo hi", "/tmp/x")

    # refuses to launch when the marker is already there...
    assert f"if [ -e /tmp/x.cancel ]; then exit {128 + 15}; fi" in wrapped
    # ...and honours a marker that appears while the job was starting
    assert 'kill -TERM -"$__hermes_pgid"' in wrapped



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
    return [c for c in env._sandbox.commands if "pgid=$(cat " in c]


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
    # signals the GROUP (-pgid), reaching every descendant
    assert 'kill -TERM -"$pgid"' in cancel_script
    assert 'kill -KILL -"$pgid"' in cancel_script



def test_cancel_marks_before_reading_so_a_queued_command_still_dies():
    """Cancel arriving before the command starts must not be silently dropped."""
    env = _env_with_fakes()

    handle = env._run_bash("sleep 300")
    handle.kill()

    script = _cancel_script(env)
    marker = script.index("touch ")
    read = script.index("pgid=$(cat ")
    assert marker < read, "cancel read the pid file before marking the intent"


def test_cancel_escalates_to_sigkill_after_a_grace_period():
    """A command that ignores SIGTERM still dies."""
    env = _env_with_fakes()

    handle = env._run_bash("trap '' TERM; sleep 300")
    handle.kill()

    script = _cancel_script(env)
    assert f"seq 1 {_MODAL_CANCEL_GRACE_SECONDS}" in script
    # stops escalating early once the group is gone
    assert 'kill -0 -"$pgid"' in script


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


def test_sdk_exec_deadline_has_headroom_over_the_local_deadline():
    """The local deadline must fire first so cancellation actually runs.

    _wait_for_process kills at exactly `timeout`. If the SDK's exec deadline
    were also `timeout`, Modal could reap the outer bash first while the real
    command, a background job in its own process group, kept running: the
    handle would report completion and cancel() would never reap the group.
    """
    env = _env_with_fakes()
    captured = {}

    class _Exec:
        async def aio(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return _FakeProc()

    env._sandbox.exec = _Exec()

    handle = env._run_bash("echo hi", timeout=120)
    handle.wait(timeout=5)

    assert captured["timeout"] > 120, "SDK deadline must not race the local one"
    assert captured["timeout"] == 120 + _MODAL_EXEC_TIMEOUT_HEADROOM


def test_each_command_gets_its_own_pid_file():
    """Two concurrent commands must not cancel each other."""
    env = _env_with_fakes()

    env._run_bash("sleep 300").kill()
    env._run_bash("sleep 300").kill()

    cancels = _cancel_scripts(env)
    assert len(cancels) == 2
    assert cancels[0] != cancels[1], "commands shared a pid file"

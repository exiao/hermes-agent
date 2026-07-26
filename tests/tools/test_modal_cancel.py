"""Cancellation contract for the Modal backend.

These assert BEHAVIOR (what cancel does to the sandbox), not the text of the
kill script. The live tests at the bottom drive real Modal sandboxes and skip
unless Modal credentials and the SDK are present.
"""

import os
import shlex

import pytest

modal_env = pytest.importorskip("tools.environments.modal")


def test_cancellable_command_preserves_the_original_command():
    tagged = modal_env._cancellable_command("echo hi && ls", "/tmp/.hermes-cancel/x")
    assert tagged.endswith("echo hi && ls")


def test_cancellable_command_records_the_shell_pid_not_an_env_var():
    """Regression: an exported marker is clobbered by the env snapshot.

    ``_wrap_command`` sources the session's ``export -p`` snapshot, so any
    variable this prefix exports is replaced by the bootstrap value on every
    later command — cancel would then match nothing. ``$$`` is written by the
    shell itself and is immune.
    """
    tagged = modal_env._cancellable_command("sleep 300", "/tmp/.hermes-cancel/x")
    assert "echo $$ >" in tagged
    assert "export HERMES_CANCEL_TOKEN" not in tagged


def test_cancellable_command_quotes_the_pidfile():
    nasty = "/tmp/a b; rm -rf /"
    tagged = modal_env._cancellable_command("true", nasty)
    assert shlex.quote(nasty) in tagged
    assert "; rm -rf /" not in tagged.replace(shlex.quote(nasty), "")


def test_pidfile_write_cannot_break_the_command():
    """A read-only /tmp must not turn every command into a failure."""
    tagged = modal_env._cancellable_command("echo hi", "/tmp/.hermes-cancel/x")
    assert "|| true" in tagged.split("echo hi")[0]


class _FakeProc:
    def __init__(self):
        self.wait = _Aio(0)


class _Aio:
    """Mimics Modal's ``obj.method.aio(...)`` shape."""

    def __init__(self, result):
        self._result = result

    async def aio(self, *args, **kwargs):
        return self._result


class _SandboxExec:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    async def aio(self, *args, **kwargs):
        self._sandbox.exec_calls.append(args)
        return _FakeProc()


class _SandboxTerminate:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    async def aio(self, *args, **kwargs):
        self._sandbox.terminated = True


class _RecordingSandbox:
    def __init__(self):
        self.exec_calls = []
        self.terminated = False
        self.exec = _SandboxExec(self)
        self.terminate = _SandboxTerminate(self)


class _InlineWorker:
    def run_coroutine(self, coro, timeout=600):
        import asyncio

        return asyncio.run(coro)


def _make_env(sandbox):
    env = modal_env.ModalEnvironment.__new__(modal_env.ModalEnvironment)
    env._sandbox = sandbox
    env._worker = _InlineWorker()
    return env


def test_cancel_does_not_terminate_the_sandbox():
    """The regression: cancelling one command used to destroy the session."""
    sandbox = _RecordingSandbox()
    env = _make_env(sandbox)

    handle = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    handle.kill()

    assert sandbox.terminated is False, "cancel must not tear down the sandbox"


def test_cancel_issues_a_kill_through_the_same_sandbox():
    sandbox = _RecordingSandbox()
    env = _make_env(sandbox)

    handle = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    handle.kill()

    kill_calls = [c for c in sandbox.exec_calls if modal_env._CANCEL_SCRIPT in c]
    assert len(kill_calls) == 1, f"expected one cancel exec, got {sandbox.exec_calls}"
    assert kill_calls[0][-1].startswith(modal_env._CANCEL_DIR)


def test_each_command_gets_a_distinct_pidfile():
    """Cancelling command A must not be able to signal command B."""
    sandbox = _RecordingSandbox()
    env = _make_env(sandbox)

    a = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    b = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    a.kill()
    b.kill()

    pidfiles = [c[-1] for c in sandbox.exec_calls if modal_env._CANCEL_SCRIPT in c]
    assert len(pidfiles) == 2
    assert pidfiles[0] != pidfiles[1]


def test_cancel_failure_is_swallowed_and_leaves_the_sandbox_alone():
    class _FailingExec:
        async def aio(self, *args, **kwargs):
            raise RuntimeError("transport down")

    sandbox = _RecordingSandbox()
    sandbox.exec = _FailingExec()  # type: ignore[assignment]
    env = _make_env(sandbox)

    handle = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    handle.kill()  # must not raise

    assert sandbox.terminated is False


# ---------------------------------------------------------------------------
# Live E2E — real Modal sandboxes, real cancellation.
# ---------------------------------------------------------------------------

_LIVE = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
_live_only = pytest.mark.skipif(not _LIVE, reason="requires live Modal credentials")


def _live_cancel_cycle(inner_command: str, name: str):
    """Run ``inner_command``, cancel it, and report (rc, sandbox_alive, state)."""
    import asyncio

    modal = pytest.importorskip("modal")

    async def _run():
        app = await modal.App.lookup.aio("hermes-cancel-e2e", create_if_missing=True)
        sandbox = await modal.Sandbox.create.aio(
            "sleep", "infinity", image=modal.Image.debian_slim(), app=app, timeout=300,
        )
        try:
            seed = await sandbox.exec.aio("bash", "-c", "echo persisted > /tmp/state")
            await seed.wait.aio()

            pidfile = modal_env._cancel_pidfile(name)
            target = await sandbox.exec.aio(
                "bash", "-c", modal_env._cancellable_command(inner_command, pidfile),
                timeout=300,
            )
            await asyncio.sleep(3)

            killer = await sandbox.exec.aio(
                "bash", "-c", modal_env._CANCEL_SCRIPT, "--", pidfile, timeout=30,
            )
            kill_out = await killer.stdout.read.aio()
            await killer.wait.aio()

            rc = await asyncio.wait_for(target.wait.aio(), timeout=30)

            after = await sandbox.exec.aio("bash", "-c", "cat /tmp/state")
            state = await after.stdout.read.aio()
            await after.wait.aio()
            return rc, state, await sandbox.poll.aio(), kill_out
        finally:
            await sandbox.terminate.aio()

    return asyncio.run(_run())


@_live_only
def test_live_cancel_keeps_the_sandbox_and_its_state_usable():
    rc, state, poll, kill_out = _live_cancel_cycle("sleep 300", "live-sleep")

    assert "cancelled=1" in kill_out
    assert rc != 0, "cancelled command should report a signal exit"
    assert "persisted" in state, "sandbox filesystem must survive cancellation"
    assert poll is None, "sandbox must still be running after a cancel"


@_live_only
def test_live_cancel_reaches_a_shell_builtin_only_command():
    """Regression: a command that never forks has no tagged child process."""
    rc, state, poll, kill_out = _live_cancel_cycle("while :; do :; done", "live-builtin")

    assert "cancelled=1" in kill_out
    assert rc != 0
    assert poll is None


@_live_only
def test_live_cancel_survives_the_env_snapshot_being_sourced():
    """Regression: the real path sources an ``export -p`` snapshot first."""
    inner = (
        "echo 'export HERMES_CANCEL_TOKEN=STALE' > /tmp/snap; "
        "source /tmp/snap; sleep 300"
    )
    rc, state, poll, kill_out = _live_cancel_cycle(inner, "live-snapshot")

    assert "cancelled=1" in kill_out
    assert rc != 0
    assert poll is None


@_live_only
def test_live_cancel_kills_the_whole_child_tree():
    rc, state, poll, kill_out = _live_cancel_cycle(
        "sleep 300 & sleep 300 & wait", "live-tree"
    )

    assert "cancelled=1" in kill_out
    assert rc != 0
    assert poll is None

"""Cancellation contract for the Modal backend.

The unit tests here assert BEHAVIOR (what cancel does to the sandbox), not the
text of the kill script. The live test at the bottom drives a real Modal
sandbox and is skipped unless Modal credentials and the SDK are present.
"""

import os
import shlex

import pytest

modal_env = pytest.importorskip("tools.environments.modal")


def test_cancellable_command_tags_and_preserves_the_original():
    tagged = modal_env._cancellable_command("echo hi && ls", "tok-1")
    assert tagged.endswith("echo hi && ls")
    assert f"export {modal_env._CANCEL_TOKEN_ENV}=tok-1;" in tagged


def test_cancellable_command_quotes_the_token():
    """A token is interpolated into a shell string, so it must be quoted."""
    nasty = "a b; rm -rf /"
    tagged = modal_env._cancellable_command("true", nasty)
    assert tagged.startswith(
        f"export {modal_env._CANCEL_TOKEN_ENV}={shlex.quote(nasty)};"
    )


class _FakeProc:
    def __init__(self):
        self.wait = _Aio(0)


class _Aio:
    """Mimics Modal's ``obj.method.aio(...)`` shape."""

    def __init__(self, result):
        self._result = result

    async def aio(self, *args, **kwargs):
        return self._result

    def __call__(self, *args, **kwargs):
        return self._result


class _RecordingSandbox:
    def __init__(self):
        self.exec_calls = []
        self.terminated = False
        self.exec = _SandboxExec(self)
        self.terminate = _SandboxTerminate(self)


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


def test_cancel_issues_a_token_scoped_kill_through_the_same_sandbox():
    sandbox = _RecordingSandbox()
    env = _make_env(sandbox)

    handle = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    handle.kill()

    kill_calls = [c for c in sandbox.exec_calls if modal_env._CANCEL_SCRIPT in c]
    assert len(kill_calls) == 1, f"expected one cancel exec, got {sandbox.exec_calls}"
    token = kill_calls[0][-1]
    assert token.startswith("hermes-")


def test_each_command_gets_a_distinct_cancel_token():
    """Cancelling command A must not be able to signal command B."""
    sandbox = _RecordingSandbox()
    env = _make_env(sandbox)

    a = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    b = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    a.kill()
    b.kill()

    tokens = [c[-1] for c in sandbox.exec_calls if modal_env._CANCEL_SCRIPT in c]
    assert len(tokens) == 2
    assert tokens[0] != tokens[1]


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
# Live E2E — real Modal sandbox, real cancellation.
# ---------------------------------------------------------------------------

_LIVE = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))


@pytest.mark.skipif(not _LIVE, reason="requires live Modal credentials")
def test_live_cancel_keeps_the_sandbox_and_its_state_usable():
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

            token = "live-token-1"
            target = await sandbox.exec.aio(
                "bash", "-c", modal_env._cancellable_command("sleep 300", token),
                timeout=300,
            )
            await asyncio.sleep(3)

            killer = await sandbox.exec.aio(
                "bash", "-c", modal_env._CANCEL_SCRIPT, "--", token, timeout=30,
            )
            await killer.wait.aio()

            rc = await asyncio.wait_for(target.wait.aio(), timeout=30)

            after = await sandbox.exec.aio("bash", "-c", "cat /tmp/state")
            out = await after.stdout.read.aio()
            await after.wait.aio()
            return rc, out, await sandbox.poll.aio()
        finally:
            await sandbox.terminate.aio()

    rc, state, poll = asyncio.run(_run())

    assert rc != 0, "cancelled command should report a signal exit"
    assert "persisted" in state, "sandbox filesystem must survive cancellation"
    assert poll is None, "sandbox must still be running after a cancel"

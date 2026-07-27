"""Cancellation contract for the Modal backend.

These assert BEHAVIOR (what cancel does to the sandbox), not the text of the
kill script. The live tests at the bottom drive real Modal sandboxes and skip
unless Modal credentials and the SDK are present.
"""

import asyncio
import os
import shlex
import time
import subprocess
import threading

import pytest

from tools.environments.base import BaseEnvironment

modal_env = pytest.importorskip("tools.environments.modal")


def test_cancellable_command_preserves_the_original_command(tmp_path):
    """The wrapped command must still DO what the bare command does."""
    import subprocess

    pidfile = tmp_path / "cmd.pid"
    marker = tmp_path / "ran"
    inner = f"echo hi && echo second > {marker}"

    bare = subprocess.run(["bash", "-c", inner], capture_output=True, text=True)
    marker.unlink(missing_ok=True)
    wrapped = subprocess.run(
        ["bash", "-c", modal_env._cancellable_command(inner, str(pidfile))],
        capture_output=True, text=True,
    )

    assert wrapped.stdout == bare.stdout, "wrapper changed the command's output"
    assert wrapped.returncode == bare.returncode
    assert marker.read_text() == "second\n", "the command's side effects must run"


def test_cancellable_command_records_the_shell_pid_not_an_env_var(tmp_path):
    """Regression: an exported marker is clobbered by the env snapshot.

    ``_wrap_command`` sources the session's ``export -p`` snapshot, so any
    variable this prefix exports is replaced by the bootstrap value on every
    later command — cancel would then match nothing. ``$$`` is written by the
    shell itself and is immune.
    """
    import subprocess

    pidfile = tmp_path / "cmd.pid"
    # A shell BUILTIN never forks, so the recorded pid must be the shell's own
    # and must still be live while the command runs.
    tagged = modal_env._cancellable_command(
        f'read -r _ < {tmp_path / "gate"} 2>/dev/null; :', str(pidfile)
    )
    (tmp_path / "gate").write_text("go\n")
    proc = subprocess.run(["bash", "-c", tagged + f'; echo "$$" > {tmp_path / "self"}'],
                          capture_output=True, text=True)

    assert proc.returncode == 0, proc
    # The file records "<pid> <pgid>"; the first field is the contract the
    # cancel script reads. (On Linux the pgid is present; on platforms without
    # /proc it is empty, which the cancel script re-derives.)
    recorded = pidfile.read_text().split()[0]
    actual = (tmp_path / "self").read_text().strip()
    assert recorded == actual, (
        "recorded pid %r is not the command shell's own pid %r" % (recorded, actual)
    )


def test_cancellable_command_quotes_the_pidfile(tmp_path):
    """A hostile pidfile path must be data, never executable shell."""
    import subprocess

    canary = tmp_path / "canary"
    canary.write_text("intact")
    nasty = str(tmp_path / f"a b; rm -f {canary}")

    proc = subprocess.run(
        ["bash", "-c", modal_env._cancellable_command("echo ok", nasty)],
        capture_output=True, text=True,
    )

    assert "ok" in proc.stdout, proc
    assert canary.read_text() == "intact", "injected command executed"
    # The whole hostile string must have been passed as ONE literal argument to
    # the redirect, so bash complains about that exact path rather than running
    # the `rm` after a `;`. Either it wrote the file or it failed on the literal
    # path; both prove the injection never became shell syntax.
    assert os.path.exists(nasty) or nasty in proc.stderr, (
        "hostile pidfile path was not treated as one literal argument: %r" % (proc,)
    )


def test_pidfile_write_cannot_break_the_command(tmp_path):
    """An unwritable pid-file location must not fail an otherwise fine command.

    The guard matters under ``set -e`` and inside ``&&`` chains, which real
    commands use: there a failed redirect aborts the whole command rather than
    printing a warning. Both shapes are exercised.
    """
    import subprocess

    readonly = tmp_path / "ro"
    readonly.mkdir()
    readonly.chmod(0o500)
    pidfile = str(readonly / "cmd.pid")
    try:
        plain = subprocess.run(
            ["bash", "-c", modal_env._cancellable_command("echo hi", pidfile)],
            capture_output=True, text=True,
        )
        errexit = subprocess.run(
            ["bash", "-ec", modal_env._cancellable_command("echo hi", pidfile)],
            capture_output=True, text=True,
        )
        chained = subprocess.run(
            ["bash", "-c", modal_env._cancellable_command("echo hi && echo two", pidfile)],
            capture_output=True, text=True,
        )
    finally:
        readonly.chmod(0o700)

    for label, proc in (("plain", plain), ("set -e", errexit), ("&&", chained)):
        assert proc.returncode == 0, (
            "unwritable pid path broke the command under %s: %r" % (label, proc)
        )
        assert "hi" in proc.stdout, (label, proc)


def test_cancellable_command_preserves_exit_status(tmp_path):
    pidfile = tmp_path / "command.pid"
    tagged = modal_env._cancellable_command("exit 37", str(pidfile))

    result = subprocess.run(["bash", "-c", tagged], capture_output=True, text=True)

    assert result.returncode == 37


def test_cancellable_command_preserves_dollar_dollar_semantics(tmp_path):
    """Regression: a subshell wrapper made `$$` point at the wrapper, not self.

    Bash keeps `$$` as the parent shell's PID inside `( ... )`, so a command
    that signals itself via `kill -TERM $$` hit the wrapper instead and came
    back 143 rather than its own trapped 42.
    """
    pidfile = tmp_path / "command.pid"
    tagged = modal_env._cancellable_command(
        "trap 'exit 42' TERM; kill -TERM $$; sleep 3", str(pidfile)
    )

    result = subprocess.run(["bash", "-c", tagged], capture_output=True, text=True)

    assert result.returncode == 42, (
        "the command's own $$ trap must still fire; got %s" % result.returncode
    )


class _Reader:
    def __init__(self, value=""):
        self.read = _Aio(value)


class _FakeProc:
    def __init__(self):
        self.stdout = _Reader()
        self.stderr = _Reader()
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


def _kill_after_launch(handle, sandbox):
    """Wait until this command's exec RPC has returned, then cancel.

    ``_run_bash`` deliberately defers a cancel that arrives before the command
    is up (the launcher replays it instead), so a test that wants to observe
    the immediate-cancel path has to let the launch happen first.
    """
    import time

    before = len(sandbox.exec_calls)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(sandbox.exec_calls) == before:
        time.sleep(0.01)
    handle.kill()


def test_cancel_issues_a_kill_through_the_same_sandbox():
    sandbox = _RecordingSandbox()
    env = _make_env(sandbox)

    handle = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    _kill_after_launch(handle, sandbox)

    kill_calls = [c for c in sandbox.exec_calls if modal_env._CANCEL_SCRIPT in c]
    assert len(kill_calls) >= 1, f"expected a cancel exec, got {sandbox.exec_calls}"
    assert kill_calls[0][-1].startswith(modal_env._CANCEL_DIR)


def test_each_command_gets_a_distinct_pidfile():
    """Cancelling command A must not be able to signal command B."""
    sandbox = _RecordingSandbox()
    env = _make_env(sandbox)

    a = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    _kill_after_launch(a, sandbox)
    b = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    _kill_after_launch(b, sandbox)

    pidfiles = [c[-1] for c in sandbox.exec_calls if modal_env._CANCEL_SCRIPT in c]
    assert len(pidfiles) >= 2
    assert pidfiles[0] != pidfiles[-1]


def test_cancel_before_launch_is_replayed_not_dropped():
    """An interrupt beating the exec RPC must still reach the command."""
    sandbox = _RecordingSandbox()
    env = _make_env(sandbox)

    handle = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    handle.kill()  # may land before or after launch; either way it must apply
    handle.wait(timeout=5)

    kill_calls = [c for c in sandbox.exec_calls if modal_env._CANCEL_SCRIPT in c]
    assert kill_calls, f"cancel was dropped entirely: {sandbox.exec_calls}"
    assert sandbox.terminated is False


def test_cancel_during_a_slow_exec_rpc_is_still_applied():
    """The launch race, isolated: cancel arrives while exec is in flight.

    A shell-side poll for the PID file cannot fix this — the RPC that would
    create it hasn't returned yet, and it may take longer than any fixed
    timeout. The launcher must replay the pending cancel once it is up.
    """
    import asyncio
    import threading

    launched = threading.Event()

    class _SlowExec:
        def __init__(self, sandbox):
            self._sandbox = sandbox

        async def aio(self, *args, **kwargs):
            self._sandbox.exec_calls.append(args)
            if modal_env._CANCEL_SCRIPT not in args:
                # The target command's RPC is slow to return.
                await asyncio.sleep(0.5)
                launched.set()
            return _FakeProc()

    sandbox = _RecordingSandbox()
    sandbox.exec = _SlowExec(sandbox)  # type: ignore[assignment]
    env = _make_env(sandbox)

    handle = modal_env.ModalEnvironment._run_bash(env, "sleep 300", timeout=5)
    handle.kill()  # lands while the exec RPC above is still in flight
    handle.wait(timeout=10)

    assert launched.is_set(), "the target command never launched"
    kill_calls = [c for c in sandbox.exec_calls if modal_env._CANCEL_SCRIPT in c]
    assert kill_calls, (
        "a cancel that arrived before launch was dropped: %r" % (sandbox.exec_calls,)
    )
    assert sandbox.terminated is False


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


class _DynamicWait:
    def __init__(self, was_cancelled):
        self._was_cancelled = was_cancelled

    async def aio(self):
        return 143 if self._was_cancelled() else 0


class _TargetProc:
    def __init__(self, was_cancelled):
        self.stdout = _Reader()
        self.stderr = _Reader()
        self.wait = _DynamicWait(was_cancelled)


class _StartupBlockingSandbox:
    """Delays target exec startup until the test asks it to complete."""

    def __init__(self):
        self.target_starting = threading.Event()
        self.release_target = threading.Event()
        self.target_returned = threading.Event()
        self.cancelled = threading.Event()
        self.cancel_after_target_start = False
        self.exec = _StartupBlockingExec(self)

    async def exec_aio(self, *args, **kwargs):
        if modal_env._CANCEL_SCRIPT in args:
            self.cancel_after_target_start = self.target_returned.is_set()
            self.cancelled.set()
            return _FakeProc()

        self.target_starting.set()
        await asyncio.to_thread(self.release_target.wait, 2)
        self.target_returned.set()
        return _TargetProc(self.cancelled.is_set)


class _StartupBlockingExec:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    async def aio(self, *args, **kwargs):
        return await self._sandbox.exec_aio(*args, **kwargs)


def test_execute_cancels_after_delayed_modal_exec_startup():
    """A pending interrupt must reach a command only after its exec startup ends."""
    sandbox = _StartupBlockingSandbox()
    env = _make_env(sandbox)
    BaseEnvironment.__init__(env, cwd="/root", timeout=5)
    env._before_execute = lambda: None

    def wait_then_cancel(handle, *, timeout, bounded_capture):
        assert sandbox.target_starting.wait(2), "target exec never entered startup"
        handle.kill()
        assert not sandbox.cancelled.is_set(), "cancel must wait for target startup"
        sandbox.release_target.set()
        handle.wait(timeout=2)
        return {"output": handle.stdout.read(), "returncode": handle.returncode}

    env._wait_for_process = wait_then_cancel

    result = env.execute("sleep 300")

    assert sandbox.cancel_after_target_start
    assert result["returncode"] == 143


def test_execute_drains_target_when_pending_cancel_replay_fails():
    """A replay transport error must not replace the target's result with 1."""
    class _FailingReplaySandbox(_StartupBlockingSandbox):
        async def exec_aio(self, *args, **kwargs):
            if modal_env._CANCEL_SCRIPT in args:
                self.cancel_after_target_start = self.target_returned.is_set()
                raise RuntimeError("cancel transport down")
            return await super().exec_aio(*args, **kwargs)

    sandbox = _FailingReplaySandbox()
    env = _make_env(sandbox)
    BaseEnvironment.__init__(env, cwd="/root", timeout=5)
    env._before_execute = lambda: None

    def wait_then_cancel(handle, *, timeout, bounded_capture):
        assert sandbox.target_starting.wait(2), "target exec never entered startup"
        handle.kill()
        sandbox.release_target.set()
        handle.wait(timeout=2)
        return {"output": handle.stdout.read(), "returncode": handle.returncode}

    env._wait_for_process = wait_then_cancel

    result = env.execute("sleep 300")

    assert sandbox.cancel_after_target_start
    assert result["returncode"] == 0


# ---------------------------------------------------------------------------
# Live E2E — real Modal sandboxes, real cancellation.
# ---------------------------------------------------------------------------

_LIVE = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
_live_only = pytest.mark.skipif(not _LIVE, reason="requires live Modal credentials")


def _live_cancel_cycle(inner_command: str, name: str):
    """Interrupt ``inner_command`` through the REAL execute() path.

    Drives production wiring end to end: ``set_interrupt`` ->
    ``_wait_for_process`` -> ``proc.kill()`` -> ``_run_bash``'s cancel
    callback (including its pending-cancel coordination) ->
    ``_ThreadedProcessHandle``. Nothing here calls the cancel helpers
    directly, so a break anywhere in that chain fails the test.

    Returns ``(result, still_usable, state, elapsed)``.
    """
    from tools.interrupt import set_interrupt

    pytest.importorskip("modal")

    env = modal_env.ModalEnvironment(
        image="python:3.11-slim",
        persistent_filesystem=False,
        task_id="cancel-e2e-" + name,
    )
    main_tid = threading.current_thread().ident

    def _interrupt_soon():
        time.sleep(4)
        set_interrupt(True, thread_id=main_tid)

    try:
        seed = env.execute(
            "mkdir -p /workspace && echo persisted > /workspace/state", timeout=60
        )
        assert seed["returncode"] == 0, seed

        threading.Thread(target=_interrupt_soon, daemon=True).start()
        started = time.time()
        result = env.execute(inner_command, timeout=180)
        elapsed = time.time() - started
        set_interrupt(False, thread_id=main_tid)

        # The sandbox must still be usable and session state intact - the
        # whole point of not terminating it in order to cancel a command.
        still_usable = env.execute("echo alive", timeout=60)
        state = env.execute("cat /workspace/state", timeout=60)
        return result, still_usable, state, elapsed
    finally:
        set_interrupt(False, thread_id=main_tid)
        env.cleanup()


@_live_only
def test_live_cancel_keeps_the_sandbox_and_its_state_usable():
    result, still_usable, state, elapsed = _live_cancel_cycle("sleep 300", "sleep")

    assert result["returncode"] == 130, "interrupt path did not fire: %r" % (result,)
    assert elapsed < 60, "command not stopped promptly: %.1fs" % elapsed
    assert still_usable["returncode"] == 0, "SANDBOX BRICKED: %r" % (still_usable,)
    assert "alive" in still_usable["output"], still_usable
    assert "persisted" in state["output"], "session state lost: %r" % (state,)


@_live_only
def test_live_cancel_reaches_a_shell_builtin_only_command():
    """Regression: a command that never forks has no tagged child process."""
    result, still_usable, state, elapsed = _live_cancel_cycle(
        "while :; do :; done", "builtin"
    )

    assert result["returncode"] == 130, "interrupt path did not fire: %r" % (result,)
    assert elapsed < 60, "builtin loop not cancelled promptly: %.1fs" % elapsed
    assert still_usable["returncode"] == 0, "SANDBOX BRICKED: %r" % (still_usable,)
    assert "persisted" in state["output"], state


@_live_only
def test_live_cancel_survives_the_env_snapshot_being_sourced():
    """Regression: the real path sources an ``export -p`` snapshot first."""
    inner = (
        "echo 'export HERMES_CANCEL_TOKEN=STALE' > /tmp/snap; "
        "source /tmp/snap; sleep 300"
    )
    result, still_usable, state, elapsed = _live_cancel_cycle(inner, "snapshot")

    assert result["returncode"] == 130, "interrupt path did not fire: %r" % (result,)
    assert elapsed < 60, "not cancelled promptly: %.1fs" % elapsed
    assert still_usable["returncode"] == 0, "SANDBOX BRICKED: %r" % (still_usable,)
    assert "persisted" in state["output"], state


@_live_only
def test_live_cancel_kills_the_whole_child_tree():
    result, still_usable, state, elapsed = _live_cancel_cycle(
        "sleep 300 & sleep 300 & wait", "tree"
    )

    assert result["returncode"] == 130, "interrupt path did not fire: %r" % (result,)
    assert elapsed < 60, "child tree not cancelled promptly: %.1fs" % elapsed
    assert still_usable["returncode"] == 0, "SANDBOX BRICKED: %r" % (still_usable,)
    assert "persisted" in state["output"], state


@_live_only
def test_live_cancel_reaches_a_descendant_that_left_the_process_group():
    """Regression: a signal to the group alone misses a detached child.

    A command that calls ``setsid`` puts its child in a NEW process group, so
    a group-only kill terminates the wrapper, reports success, and leaves the
    child running. Cancellation walks the /proc parent tree as well.
    """
    import asyncio

    modal = pytest.importorskip("modal")

    marker = "hermes_detached_child"
    count = (
        'n=0; self=$$; for d in /proc/[0-9]*; do p=${d#/proc/}; '
        '[ "$p" = "$self" ] && continue; '
        'c=$(tr "\\0" " " < "$d/cmdline" 2>/dev/null); '
        f'case "$c" in *{marker}*) n=$((n+1));; esac; done; echo "n=$n"'
    )

    async def _run():
        app = await modal.App.lookup.aio("hermes-cancel-e2e", create_if_missing=True)
        sandbox = await modal.Sandbox.create.aio(
            "sleep", "infinity", image=modal.Image.debian_slim(), app=app, timeout=300,
        )
        try:
            pidfile = modal_env._cancel_pidfile("live-detached")
            inner = (
                f"setsid bash -c 'exec -a {marker} sleep 900'; sleep 300"
            )
            target = await sandbox.exec.aio(
                "bash", "-c", modal_env._cancellable_command(inner, pidfile), timeout=300,
            )
            await asyncio.sleep(3)

            before = await sandbox.exec.aio("bash", "-c", count)
            before_out = await before.stdout.read.aio()
            await before.wait.aio()

            killer = await sandbox.exec.aio(
                "bash", "-c", modal_env._CANCEL_SCRIPT, "--", pidfile, timeout=30,
            )
            await killer.wait.aio()
            try:
                await asyncio.wait_for(target.wait.aio(), timeout=25)
            except Exception:
                pass
            await asyncio.sleep(1.5)

            after = await sandbox.exec.aio("bash", "-c", count)
            after_out = await after.stdout.read.aio()
            await after.wait.aio()
            return before_out, after_out
        finally:
            await sandbox.terminate.aio()

    before_out, after_out = asyncio.run(_run())

    assert "n=0" not in before_out, "the detached child never started"
    assert "n=0" in after_out, f"detached child survived cancel: {after_out!r}"


@_live_only
def test_live_cancel_reaches_a_descendant_that_ignores_term():
    """Regression: gating the KILL escalation on the wrapper PID leaks children.

    The wrapper bash exits on TERM, so a ``/proc/$pid`` check right after would
    be false and the KILL would be skipped — while a child that trapped TERM
    keeps running. Escalating to the process group unconditionally reaches it.
    """
    import asyncio

    modal = pytest.importorskip("modal")

    marker = "hermes_term_ignorer"
    count = (
        'n=0; self=$$; for d in /proc/[0-9]*; do p=${d#/proc/}; '
        '[ "$p" = "$self" ] && continue; '
        'c=$(tr "\\0" " " < "$d/cmdline" 2>/dev/null); '
        f'case "$c" in *{marker}*) n=$((n+1));; esac; done; echo "n=$n"'
    )

    async def _run():
        app = await modal.App.lookup.aio("hermes-cancel-e2e", create_if_missing=True)
        sandbox = await modal.Sandbox.create.aio(
            "sleep", "infinity", image=modal.Image.debian_slim(), app=app, timeout=300,
        )
        try:
            pidfile = modal_env._cancel_pidfile("live-term-ignorer")
            inner = f"bash -c 'trap \"\" TERM; exec -a {marker} sleep 900' & wait"
            target = await sandbox.exec.aio(
                "bash", "-c", modal_env._cancellable_command(inner, pidfile), timeout=300,
            )
            await asyncio.sleep(3)

            before = await sandbox.exec.aio("bash", "-c", count)
            before_out = await before.stdout.read.aio()
            await before.wait.aio()

            killer = await sandbox.exec.aio(
                "bash", "-c", modal_env._CANCEL_SCRIPT, "--", pidfile, timeout=30,
            )
            await killer.wait.aio()
            try:
                await asyncio.wait_for(target.wait.aio(), timeout=25)
            except Exception:
                pass
            await asyncio.sleep(1.5)

            after = await sandbox.exec.aio("bash", "-c", count)
            after_out = await after.stdout.read.aio()
            await after.wait.aio()
            return before_out, after_out
        finally:
            await sandbox.terminate.aio()

    before_out, after_out = asyncio.run(_run())

    assert "n=0" not in before_out, "the TERM-ignoring child never started"
    assert "n=0" in after_out, f"TERM-ignoring descendant survived cancel: {after_out!r}"


@_live_only
def test_live_cancel_that_arrives_before_the_command_registers():
    """Regression: an interrupt can beat the PID file into existence.

    ``/stop`` landing while ``sandbox.exec`` is still starting the command
    used to make cancel report ``cancelled=0`` and return, leaving the command
    running remotely until its SDK timeout. Cancel now waits briefly for the
    PID file to appear.
    """
    import asyncio

    modal = pytest.importorskip("modal")

    marker = "hermes_late_starter"

    async def _run():
        app = await modal.App.lookup.aio("hermes-cancel-e2e", create_if_missing=True)
        sandbox = await modal.Sandbox.create.aio(
            "sleep", "infinity", image=modal.Image.debian_slim(), app=app, timeout=300,
        )
        try:
            pidfile = modal_env._cancel_pidfile("live-race")
            inner = f"sleep 2; exec -a {marker} sleep 900"

            # Fire the canceller FIRST, so it arrives before the PID file.
            killer = await sandbox.exec.aio(
                "bash", "-c", modal_env._CANCEL_SCRIPT, "--", pidfile, timeout=60,
            )
            await asyncio.sleep(0.5)
            target = await sandbox.exec.aio(
                "bash", "-c", modal_env._cancellable_command(inner, pidfile), timeout=300,
            )

            kill_out = await killer.stdout.read.aio()
            await killer.wait.aio()
            try:
                await asyncio.wait_for(target.wait.aio(), timeout=30)
            except Exception:
                pass
            await asyncio.sleep(1.5)

            count = (
                'n=0; self=$$; for d in /proc/[0-9]*; do p=${d#/proc/}; '
                '[ "$p" = "$self" ] && continue; '
                'c=$(tr "\\0" " " < "$d/cmdline" 2>/dev/null); '
                f'case "$c" in *{marker}*) n=$((n+1));; esac; done; echo "n=$n"'
            )
            chk = await sandbox.exec.aio("bash", "-c", count)
            out = await chk.stdout.read.aio()
            await chk.wait.aio()
            return kill_out, out
        finally:
            await sandbox.terminate.aio()

    kill_out, after_out = asyncio.run(_run())

    assert "cancelled=1" in kill_out, f"cancel gave up before the command registered: {kill_out!r}"
    assert "n=0" in after_out, f"command survived an early cancel: {after_out!r}"


@_live_only
def test_live_stale_pidfiles_are_swept_through_the_real_run_bash_path():
    """Cleanup happens as an age sweep at command start, driven end to end.

    A completed command's PID file is inert, so it is not removed immediately;
    it is swept once it is older than the max age. This drives the real
    ``execute()`` path: it ages a planted file past the cutoff, runs a normal
    command, and asserts the sweep took the stale file and left the live one.
    """
    pytest.importorskip("modal")

    env = modal_env.ModalEnvironment(
        image="python:3.11-slim",
        persistent_filesystem=False,
        task_id="cancel-pidfile-sweep",
    )
    try:
        # A command that installs and clears its own EXIT trap, exactly like
        # FileOperations._atomic_write, must still behave normally.
        result = env.execute(
            "trap 'echo user_cleanup' EXIT; trap - EXIT; echo wrote", timeout=60
        )
        assert "wrote" in result["output"], result
        assert result["returncode"] == 0, result

        # Plant and verify in ONE command: the sweep runs at command start,
        # so a file created afterwards survives that command by construction.
        stale = f"{modal_env._CANCEL_DIR}/stale-probe"
        before = env.execute(
            f"mkdir -p {modal_env._CANCEL_DIR} && echo 999 > {stale} && "
            f"touch -d '2 days ago' {stale} && "
            f"{{ [ -e {stale} ] && echo present || echo gone; }}",
            timeout=60,
        )
        assert "present" in before["output"], before

        # Any subsequent command sweeps it as part of its own prefix.
        env.execute("echo sweep_runs_here", timeout=60)
        after = env.execute(
            f"[ -e {stale} ] && echo present || echo gone", timeout=60
        )
    finally:
        env.cleanup()

    assert "gone" in after["output"], f"stale pid file was not swept: {after}"


@_live_only
def test_live_cancel_parses_the_pgid_when_comm_contains_a_space():
    """Regression: ``cut -f5`` on /proc/pid/stat breaks on a spaced comm.

    ``comm`` is the executable basename and sits in parentheses as field 2, so
    a space inside it shifts every later field: ``cut -f5`` then yields the
    PPID instead of the PGID and cancel signals the wrong group.
    """
    import asyncio

    modal = pytest.importorskip("modal")

    async def _run():
        app = await modal.App.lookup.aio("hermes-cancel-e2e", create_if_missing=True)
        sandbox = await modal.Sandbox.create.aio(
            "sleep", "infinity", image=modal.Image.debian_slim(), app=app, timeout=300,
        )
        try:
            # A stat line with a spaced comm, ppid=7, pgid=42 — distinguishable.
            line = "100 (we ird) S 7 42 42 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 99 0 0"
            script = (
                f"line='{line}'; "
                "echo \"parsed=$(echo \"$line\" | sed 's/^.*) //' | cut -d' ' -f3)\""
            )
            proc = await sandbox.exec.aio("bash", "-c", script, timeout=60)
            out = await proc.stdout.read.aio()
            await proc.wait.aio()
            return out
        finally:
            await sandbox.terminate.aio()

    out = asyncio.run(_run())

    assert "parsed=42" in out, f"pgid misparsed on a spaced comm: {out!r}"
    assert "parsed=7" not in out, "parser returned the PPID instead of the PGID"


def test_cancel_script_reads_the_pgid_correctly_for_any_comm(tmp_path):
    """Behavior contract: the parser must yield the real PGID, not a field index.

    ``comm`` is the executable basename, sits parenthesized as field 2 of
    /proc/pid/stat, and may contain spaces — which shifts every later field.
    This runs the script's actual parser over synthetic stat lines instead of
    asserting which shell tools it uses, so any correct implementation passes.
    """
    import re
    import subprocess

    # Lift the parser expression out of the script and run it against fixtures.
    parser = next(
        line.strip() for line in modal_env._CANCEL_SCRIPT.splitlines()
        if line.strip().startswith("pgid=$(")
    )
    assert "/proc/$pid/stat" in parser, parser
    parser = parser.replace('"/proc/$pid/stat"', '"$STATFILE"')

    cases = [
        # (comm, ppid, pgid)
        ("sleep", "7", "42"),
        ("we ird", "7", "42"),
        ("many  spaces here", "3", "99"),
        ("(nested)", "3", "99"),
    ]
    for comm, ppid, pgid in cases:
        statfile = tmp_path / re.sub(r"\W+", "_", comm)
        statfile.write_text(
            f"100 ({comm}) S {ppid} {pgid} {pgid} 0 -1 4194304 "
            "0 0 0 0 0 0 0 0 20 0 1 0 99 0 0\n"
        )
        out = subprocess.run(
            ["bash", "-c", f'STATFILE={statfile}; {parser}; echo "$pgid"'],
            capture_output=True, text=True,
        )
        got = out.stdout.strip()
        assert got == pgid, (
            "comm %r: parsed pgid %r, expected %r (ppid was %r)"
            % (comm, got, pgid, ppid)
        )


def test_pidfile_cleanup_does_not_touch_the_command_or_its_exit(tmp_path):
    """Behavior contract for the class of post-command-cleanup bugs.

    Four findings came from cleaning up a PID file AFTER its command: an EXIT
    trap the command clobbers, a subshell that breaks ``$$``, an awaited
    reaper that extends the caller's timed window, and a fire-and-forget
    reaper that races ``cleanup()``. Rather than forbidding identifiers, this
    asserts the observable properties any correct implementation must have:
    running the wrapper must not alter the command's exit status, its ``$$``,
    or its own EXIT trap, and must not outlive the command.
    """
    import subprocess

    pidfile = tmp_path / "cmd.pid"

    # 1. Exit status passes through untouched.
    exited = subprocess.run(
        ["bash", "-c", modal_env._cancellable_command("exit 37", str(pidfile))],
        capture_output=True, text=True,
    )
    assert exited.returncode == 37, exited

    # 2. `$$` still refers to the command's own shell, so a command that
    #    signals itself gets its own trap rather than killing the wrapper.
    selfsignal = subprocess.run(
        ["bash", "-c", modal_env._cancellable_command(
            "trap 'exit 42' TERM; kill -TERM $$; sleep 3", str(pidfile))],
        capture_output=True, text=True,
    )
    assert selfsignal.returncode == 42, selfsignal

    # 3. A command's own EXIT trap still runs and is not displaced by ours.
    trapped = subprocess.run(
        ["bash", "-c", modal_env._cancellable_command(
            "trap 'echo MY_TRAP_RAN' EXIT; echo body", str(pidfile))],
        capture_output=True, text=True,
    )
    assert "body" in trapped.stdout, trapped
    assert "MY_TRAP_RAN" in trapped.stdout, "the command's own EXIT trap was lost"

    # 4. Cleanup leaves nothing running once the command returns: no stray
    #    background job can still be mutating the filesystem afterwards.
    jobs = subprocess.run(
        ["bash", "-c", modal_env._cancellable_command(
            "jobs -r | wc -l", str(pidfile))],
        capture_output=True, text=True,
    )
    assert jobs.stdout.strip().splitlines()[-1].strip() == "0", (
        "cleanup left a background job running alongside the command: %r" % (jobs,)
    )

    # 5. Cleanup behaves IDENTICALLY however the command terminates. This is
    #    what actually separates a sweep from the broken variants: an EXIT
    #    trap cleans up on a normal exit but leaks when the command clears
    #    the trap, and trailing `rm` cleans up on a normal exit but leaks when
    #    the command calls `exit` directly. Path-dependent cleanup is the bug;
    #    which specific mechanism replaces it is not this test's business.
    outcomes = set()
    for i, body in enumerate([
        "echo plain",                      # ordinary completion
        "trap 'echo user' EXIT; trap - EXIT; echo cleared",  # clears traps
        "exit 0",                          # exits directly
        "kill -TERM $$",                   # dies on a signal
    ]):
        pf = tmp_path / ("mode%d.pid" % i)
        subprocess.run(
            ["bash", "-c", modal_env._cancellable_command(body, str(pf))],
            capture_output=True, text=True,
        )
        outcomes.add(pf.exists())

    assert len(outcomes) == 1, (
        "pid-file cleanup depends on how the command exits, so some exit "
        "paths leak and others do not: %r" % (outcomes,)
    )


def test_stale_pidfile_sweep_only_targets_old_files_in_its_own_dir(tmp_path):
    """Run the sweep: it must take only old files, only in its own directory."""
    import subprocess

    old_age = time.time() - (modal_env._PIDFILE_MAX_AGE_MINUTES + 60) * 60

    fresh = tmp_path / "fresh.pid"
    fresh.write_text("1")
    stale = tmp_path / "stale.pid"
    stale.write_text("2")
    os.utime(stale, (old_age, old_age))

    nested = tmp_path / "sub"
    nested.mkdir()
    nested_stale = nested / "stale.pid"
    nested_stale.write_text("3")
    os.utime(nested_stale, (old_age, old_age))
    os.utime(nested, (old_age, old_age))

    sweep = modal_env._stale_pidfile_sweep().replace(
        shlex.quote(modal_env._CANCEL_DIR), shlex.quote(str(tmp_path))
    )
    subprocess.run(["bash", "-c", sweep], check=False)

    assert fresh.exists(), "a live command's pid file was swept"
    assert not stale.exists(), "a stale pid file was not swept"
    assert nested_stale.exists(), "the sweep recursed out of its own directory"
    assert nested.is_dir(), "the sweep deleted a directory"


def test_stale_pidfile_sweep_cannot_fail_the_command(tmp_path):
    """A sweep over a missing/unreadable dir must still exit 0."""
    import subprocess

    sweep = modal_env._stale_pidfile_sweep().replace(
        shlex.quote(modal_env._CANCEL_DIR), shlex.quote(str(tmp_path / "nope"))
    )
    proc = subprocess.run(["bash", "-c", sweep], capture_output=True, text=True)

    assert proc.returncode == 0, "a failing sweep would break every command: %r" % (proc,)


def test_stale_pidfile_sweep_leaves_a_live_commands_file_alone(tmp_path):
    """A running command's own PID file must survive the next command's sweep."""
    import subprocess

    fresh = tmp_path / "fresh.pid"
    fresh.write_text("123")
    old = tmp_path / "old.pid"
    old.write_text("456")
    old_age = time.time() - (modal_env._PIDFILE_MAX_AGE_MINUTES + 60) * 60
    os.utime(old, (old_age, old_age))

    sweep = modal_env._stale_pidfile_sweep().replace(
        shlex.quote(modal_env._CANCEL_DIR), shlex.quote(str(tmp_path))
    )
    subprocess.run(["bash", "-c", sweep], check=False)

    assert fresh.exists(), "a live command's pid file was swept"
    assert not old.exists(), "a stale pid file was not swept"


@_live_only
def test_live_cancel_reaches_a_child_that_outlives_the_wrapper():
    """Regression: the wrapper can exit while its background child runs on.

    ``sleep 300 &`` returns the wrapper immediately, but the child inherits
    stdout so the exec RPC stays pending. Cancellation then arrives after the
    recorded PID is gone. Deriving the PGID from ``/proc/$pid`` at cancel time
    fails there and returned ``cancelled=0`` while the child kept running, so
    the PGID is recorded alongside the PID while the wrapper is still alive.
    """
    import asyncio

    modal = pytest.importorskip("modal")

    marker = "hermes_bg_survivor"
    count = (
        'n=0; self=$$; for d in /proc/[0-9]*; do p=${d#/proc/}; '
        '[ "$p" = "$self" ] && continue; '
        'c=$(tr "\\\\0" " " < "$d/cmdline" 2>/dev/null); '
        f'case "$c" in *{marker}*) n=$((n+1));; esac; done; echo "n=$n"'
    )

    async def _run():
        app = await modal.App.lookup.aio("hermes-cancel-e2e", create_if_missing=True)
        sandbox = await modal.Sandbox.create.aio(
            "sleep", "infinity", image=modal.Image.debian_slim(), app=app, timeout=300,
        )
        try:
            pidfile = modal_env._cancel_pidfile("live-bgchild")
            # The wrapper exits after ~2s; the child must still be reachable.
            inner = f"bash -c 'exec -a {marker} sleep 900' & sleep 2"
            await sandbox.exec.aio(
                "bash", "-c", modal_env._cancellable_command(inner, pidfile), timeout=120,
            )
            await asyncio.sleep(6)

            before = await sandbox.exec.aio("bash", "-c", count)
            before_out = await before.stdout.read.aio()
            await before.wait.aio()

            killer = await sandbox.exec.aio(
                "bash", "-c", modal_env._CANCEL_SCRIPT, "--", pidfile, timeout=30,
            )
            kill_out = await killer.stdout.read.aio()
            await killer.wait.aio()
            await asyncio.sleep(1.5)

            after = await sandbox.exec.aio("bash", "-c", count)
            after_out = await after.stdout.read.aio()
            await after.wait.aio()
            return before_out, after_out, kill_out
        finally:
            await sandbox.terminate.aio()

    before_out, after_out, kill_out = asyncio.run(_run())

    assert "n=0" not in before_out, "the background child never started"
    assert "cancelled=1" in kill_out, (
        "cancel gave up because the wrapper had exited: %r" % (kill_out,)
    )
    assert "n=0" in after_out, f"background child survived cancel: {after_out!r}"

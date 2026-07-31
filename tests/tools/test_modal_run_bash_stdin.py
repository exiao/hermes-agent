"""Modal ``_run_bash`` must feed ``stdin_data`` to the remote process."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from tools.environments import modal as modal_env


def _make_env():
    env = object.__new__(modal_env.ModalEnvironment)
    env._sandbox = MagicMock()
    env._worker = MagicMock()
    env._persistent = False
    env._task_id = "test"
    env._sync_manager = None
    return env


def _make_stdin():
    stdin = MagicMock()
    written = []
    stdin.write = written.append
    stdin.write_eof = MagicMock()
    stdin.drain = MagicMock()
    stdin.drain.aio = AsyncMock()
    stdin._written = written
    return stdin


def _wire(env):
    stdin = _make_stdin()

    async def mock_exec(*args, **kwargs):
        proc = MagicMock()
        proc.stdin = stdin
        proc.stdout = MagicMock()
        proc.stdout.read = MagicMock()
        proc.stdout.read.aio = AsyncMock(return_value="")
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock()
        proc.stderr.read.aio = AsyncMock(return_value="")
        proc.wait = MagicMock()
        proc.wait.aio = AsyncMock(return_value=0)
        return proc

    env._sandbox.exec = MagicMock()
    env._sandbox.exec.aio = mock_exec

    def run_coroutine(coro, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    env._worker.run_coroutine = run_coroutine
    return stdin


class TestModalRunBashStdin:
    def test_execute_uses_modal_pipe_mode(self):
        assert modal_env.ModalEnvironment._stdin_mode == "pipe"

    def test_stdin_data_is_written_and_closed(self):
        env = _make_env()
        stdin = _wire(env)

        handle = env._run_bash('cat > "$tmp"', stdin_data="hello body")
        handle.wait()

        assert "".join(stdin._written) == "hello body"
        stdin.write_eof.assert_called_once()

    def test_empty_stdin_data_still_closes_the_pipe(self):
        env = _make_env()
        stdin = _wire(env)

        handle = env._run_bash('cat > "$tmp"', stdin_data="")
        handle.wait()

        assert stdin._written == []
        stdin.write_eof.assert_called_once()

    def test_large_stdin_is_chunked_under_the_sdk_cap(self):
        env = _make_env()
        stdin = _wire(env)

        payload = "x" * (modal_env.ModalEnvironment._STDIN_CHUNK_SIZE * 2 + 7)
        handle = env._run_bash('cat > "$tmp"', stdin_data=payload)
        handle.wait()

        assert "".join(stdin._written) == payload
        assert len(stdin._written) == 3
        assert all(
            len(chunk) <= modal_env.ModalEnvironment._STDIN_CHUNK_SIZE
            for chunk in stdin._written
        )
        stdin.write_eof.assert_called_once()

    def test_no_stdin_data_leaves_the_pipe_untouched(self):
        env = _make_env()
        stdin = _wire(env)

        handle = env._run_bash("echo hi")
        handle.wait()

        assert stdin._written == []
        stdin.write_eof.assert_not_called()

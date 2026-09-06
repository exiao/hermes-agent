"""Modal cloud execution environment using the native Modal SDK directly
(``Sandbox.create()`` + ``Sandbox.exec()``) with persistent snapshots across sessions."""

import asyncio
import base64
import io
import itertools
import logging
import os
import shlex
import tarfile
import threading
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from tools.environments.base import BaseEnvironment, _load_json_store, _save_json_store
from tools.environments.base_output import _ThreadedProcessHandle
from tools.environments.file_sync import (
    FileSyncManager, iter_sync_files, quoted_mkdir_command, quoted_purge_command,
    quoted_rm_command, synced_subtree_roots, unique_parent_dirs)
from tools.environments.remote_common import bash_argv, ensure_lazy_dep

logger = logging.getLogger(__name__)


def _sandbox_supports(param: str) -> bool:
    try:
        import inspect
        import modal
        return param in inspect.signature(modal.Sandbox.create).parameters
    except Exception:
        return False


_CANCEL_DIR = "/tmp/.hermes-cancel"
_CANCEL_TIMEOUT_SECONDS = 20
_cancel_id_counter = itertools.count()
_PIDFILE_MAX_AGE_MINUTES = 720
_CANCEL_SCRIPT = r"""
pidfile="$1"
waited=0
while [ ! -s "$pidfile" ] && [ "$waited" -lt 20 ]; do
  sleep 0.1
  waited=$((waited+1))
done
[ -s "$pidfile" ] || { echo "cancelled=0"; exit 0; }
read -r pid pgid < "$pidfile"
rm -f "$pidfile" 2>/dev/null
if [ -z "$pgid" ] && [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
  pgid=$(sed 's/^.*) //' "/proc/$pid/stat" 2>/dev/null | cut -d' ' -f3)
fi
[ -n "$pgid" ] && [ "$pgid" != "0" ] || pgid=""
[ -n "$pgid" ] || [ -d "/proc/$pid" ] || { echo "cancelled=0"; exit 0; }
collect_tree() {
  echo "$1"
  for d in /proc/[0-9]*; do
    child=${d#/proc/}
    [ "$child" = "$1" ] && continue
    cppid=$(sed 's/^.*) //' "$d/stat" 2>/dev/null | cut -d' ' -f2)
    [ "$cppid" = "$1" ] && collect_tree "$child"
  done
}
targets=$([ -d "/proc/$pid" ] && collect_tree "$pid")
for sig in TERM KILL; do
  [ -n "$pgid" ] && kill -"$sig" -"$pgid" 2>/dev/null
  for target in $targets; do kill -"$sig" "$target" 2>/dev/null; done
  [ "$sig" = TERM ] && sleep 0.3
done
echo "cancelled=1"
"""


def _cancel_pidfile(command_id: str) -> str:
    return f"{_CANCEL_DIR}/{command_id}"


def _stale_pidfile_sweep() -> str:
    return (
        f"find {shlex.quote(_CANCEL_DIR)} -maxdepth 1 -type f "
        f"-mmin +{_PIDFILE_MAX_AGE_MINUTES} -delete 2>/dev/null || true")


def _cancellable_command(cmd_string: str, pidfile: str) -> str:
    quoted = shlex.quote(pidfile)
    return (
        f"mkdir -p {shlex.quote(_CANCEL_DIR)} 2>/dev/null; {_stale_pidfile_sweep()}; "
        f"echo \"$$ $(sed 's/^.*) //' /proc/$$/stat 2>/dev/null | cut -d' ' -f3)\" "
        f"> {quoted} 2>/dev/null || true; {cmd_string}")

_SNAPSHOT_STORE = get_hermes_home() / "modal_snapshots.json"


def _load_snapshots() -> dict:
    return _load_json_store(_SNAPSHOT_STORE)


def _save_snapshots(data: dict) -> None:
    _save_json_store(_SNAPSHOT_STORE, data)


def _get_snapshot_restore_candidate(task_id: str) -> tuple[str | None, bool]:
    """Return (snapshot_id, from_legacy_key); the namespaced key wins over the legacy bare task id."""
    snapshots = _load_snapshots()
    for key, legacy in ((f"direct:{task_id}", False), (task_id, True)):
        snapshot_id = snapshots.get(key)
        if isinstance(snapshot_id, str) and snapshot_id:
            return snapshot_id, legacy
    return None, False


def _store_direct_snapshot(task_id: str, snapshot_id: str) -> None:
    snapshots = _load_snapshots()
    snapshots[f"direct:{task_id}"] = snapshot_id
    snapshots.pop(task_id, None)
    _save_snapshots(snapshots)


def _delete_direct_snapshot(task_id: str, snapshot_id: str | None = None) -> None:
    snapshots = _load_snapshots()
    stale = [k for k in (f"direct:{task_id}", task_id)
             if snapshots.get(k) is not None and snapshot_id in (None, snapshots[k])]
    for key in stale:
        snapshots.pop(key)
    if stale:
        _save_snapshots(snapshots)


def _resolve_modal_image(image_spec: Any) -> Any:
    """Convert registry references or snapshot ids into Modal image objects. Registry images
    get pip repaired (ensurepip) before Modal's bootstrap; ubuntu/debian also get python3."""
    ensure_lazy_dep("terminal.modal")
    import modal as _modal

    if not isinstance(image_spec, str):
        return image_spec
    if image_spec.startswith("im-"):
        return _modal.Image.from_id(image_spec)
    setup_commands = [
        "RUN rm -rf /usr/local/lib/python*/site-packages/pip* 2>/dev/null; "
        "python -m ensurepip --upgrade --default-pip 2>/dev/null || true"]
    if any(base in image_spec.lower() for base in ("ubuntu", "debian")):
        setup_commands.insert(0,
            "RUN apt-get update -qq && apt-get install -y -qq python3 python3-venv > /dev/null 2>&1 || true")
    return _modal.Image.from_registry(image_spec, setup_dockerfile_commands=setup_commands)


async def _stream_stdin(proc, payload: str, chunk_size: int) -> None:
    """Write UTF-8 chunks that stay below Modal's byte-size transport cap."""
    start = 0
    chunk_bytes = 0
    for index, char in enumerate(payload):
        char_bytes = len(char.encode("utf-8"))
        if chunk_bytes and chunk_bytes + char_bytes > chunk_size:
            proc.stdin.write(payload[start:index])
            await proc.stdin.drain.aio()
            start, chunk_bytes = index, 0
        chunk_bytes += char_bytes
    if start < len(payload):
        proc.stdin.write(payload[start:])
        await proc.stdin.drain.aio()
    proc.stdin.write_eof()
    await proc.stdin.drain.aio()


def _as_text(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


class _AsyncWorker:
    """Background thread with its own event loop for async-safe Modal calls."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self):
        def _run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._started.set()
            self._loop.run_forever()
        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=30)

    def run_coroutine(self, coro, timeout=600):
        from agent.async_utils import safe_schedule_threadsafe
        # safe_schedule_threadsafe closes the coroutine and returns None for a missing/closed loop.
        future = safe_schedule_threadsafe(coro, self._loop)
        if future is None:
            raise RuntimeError("AsyncWorker loop is not running")
        return future.result(timeout=timeout)

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10)


class ModalEnvironment(BaseEnvironment):
    """Modal cloud execution via native Modal sandboxes: spawn-per-call via _ThreadedProcessHandle
    wrapping async SDK calls; cancellation signals only the running command."""

    _stdin_mode = "pipe"
    _snapshot_timeout = 60  # Modal cold starts can be slow
    # Modal SDK stdin buffer limit: the command-router path allows 16 MB but the legacy server
    # path caps at 2 MB, so chunks stay under 2 MB and each is flushed individually via drain().
    _STDIN_CHUNK_SIZE = 1 * 1024 * 1024

    def __init__(self, image: str, cwd: str = "/root", timeout: int = 60,
                 modal_sandbox_kwargs: Optional[dict[str, Any]] = None,
                 persistent_filesystem: bool = True, task_id: str = "default"):
        super().__init__(cwd=cwd, timeout=timeout)
        self._persistent, self._task_id = persistent_filesystem, task_id
        self._sandbox = self._app = None
        self._worker = _AsyncWorker()
        self._sync_manager: FileSyncManager | None = None  # initialized after sandbox creation
        restored_snapshot_id, restored_from_legacy_key = (
            _get_snapshot_restore_candidate(self._task_id) if self._persistent else (None, False))
        if restored_snapshot_id:
            logger.info("Modal: restoring from snapshot %s", restored_snapshot_id[:20])
        ensure_lazy_dep("terminal.modal")
        import modal as _modal
        credential_mounts = []
        initial_credential_remote_paths: set[str] = set()
        late_credential_remote_paths: set[str] = set()
        try:
            from tools.credential_files import get_credential_file_mounts
            sync_roots = synced_subtree_roots("/root/.hermes")
            for entry in get_credential_file_mounts():
                remote_path = entry["container_path"]
                if any(remote_path == root or remote_path.startswith(root + "/") for root in sync_roots):
                    late_credential_remote_paths.add(remote_path)
                    logger.warning("Modal: not mounting credential below sync root: %s", remote_path)
                    continue
                credential_mounts.append(
                    _modal.Mount.from_local_file(entry["host_path"], remote_path=remote_path))
                initial_credential_remote_paths.add(remote_path)
        except Exception as e:
            logger.debug("Modal: could not load credential file mounts: %s", e)
        self._worker.start()
        self._initial_credential_remote_paths = initial_credential_remote_paths
        self._late_credential_remote_paths = late_credential_remote_paths

        def _create(image_spec: Any) -> None:
            async def _create_sandbox():
                app = await _modal.App.lookup.aio("hermes-agent", create_if_missing=True)
                create_kwargs = dict(modal_sandbox_kwargs or {})
                if credential_mounts:
                    create_kwargs["mounts"] = list(create_kwargs.pop("mounts", [])) + credential_mounts
                if "idle_timeout" not in create_kwargs:
                    try:
                        idle = max(0, int(os.getenv("TERMINAL_CONTAINER_IDLE_TIMEOUT", "0")))
                    except (TypeError, ValueError):
                        idle = 0
                    if idle:
                        hard_lifetime = int(create_kwargs.get("timeout", 3600))
                        create_kwargs["idle_timeout"] = min(max(idle, 600), hard_lifetime)
                if create_kwargs.get("idle_timeout") and not _sandbox_supports("idle_timeout"):
                    create_kwargs.pop("idle_timeout")
                sandbox = await _modal.Sandbox.create.aio(
                    "sleep", "infinity", image=image_spec, app=app,
                    timeout=int(create_kwargs.pop("timeout", 3600)), **create_kwargs)
                return app, sandbox
            self._app, self._sandbox = self._worker.run_coroutine(_create_sandbox(), timeout=300)
        try:
            try:
                _create(_resolve_modal_image(restored_snapshot_id or image))
            except Exception as exc:
                if not restored_snapshot_id:
                    raise
                logger.warning("Modal: failed to restore snapshot %s, retrying with base image: %s",
                               restored_snapshot_id[:20], exc)
                _delete_direct_snapshot(self._task_id, restored_snapshot_id)
                _create(_resolve_modal_image(image))
            else:
                if restored_snapshot_id and restored_from_legacy_key:
                    _store_direct_snapshot(self._task_id, restored_snapshot_id)
        except Exception:
            self._worker.stop()
            raise
        logger.info("Modal: sandbox created (task=%s)", self._task_id)
        try:
            if restored_snapshot_id:
                self._purge_synced_subtrees()
            self._sync_manager = FileSyncManager(
                get_files_fn=lambda: iter_sync_files("/root/.hermes"),
                upload_fn=self._modal_upload, delete_fn=self._modal_delete,
                bulk_upload_fn=self._modal_bulk_upload, bulk_download_fn=self._modal_bulk_download)
            self._sync_manager.sync(force=True, raise_on_error=True)
            self.init_session()
        except Exception:
            self._terminate_sandbox_quietly()
            self._worker.stop()
            raise

    def _exec(self, cmd: str, *, timeout: int, stdin: str | None = None, fail_label: str | None = None,
              capture: bool = False):
        """Run ``bash -c cmd`` in the sandbox. ``stdin`` is streamed in chunks; ``capture`` returns
        stdout; ``fail_label`` turns a non-zero exit into RuntimeError (with stderr unless capturing)."""
        async def _run():
            proc = await self._sandbox.exec.aio("bash", "-c", cmd)
            if stdin is not None:
                await _stream_stdin(proc, stdin, self._STDIN_CHUNK_SIZE)
            data = await proc.stdout.read.aio() if capture else None
            exit_code = await proc.wait.aio()
            if fail_label and exit_code != 0:
                detail = "" if capture else f": {await proc.stderr.read.aio()}"
                raise RuntimeError(f"Modal {fail_label} failed (exit {exit_code}){detail}")
            return data
        return self._worker.run_coroutine(_run(), timeout=timeout)

    def _modal_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via base64 piped through stdin."""
        cmd = f"mkdir -p {shlex.quote(str(Path(remote_path).parent))} && base64 -d > {shlex.quote(remote_path)}"
        self._exec(cmd, stdin=base64.b64encode(Path(host_path).read_bytes()).decode("ascii"), timeout=30)

    def _modal_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files as one in-memory gzipped tar streamed through stdin
        into ``base64 -d | tar xzf -``, avoiding the SDK's 64 KB exec-arg limit."""
        if not files:
            return
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for host_path, remote_path in files:
                tar.add(host_path, arcname=remote_path.lstrip("/"))
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        cmd = f"{quoted_mkdir_command(unique_parent_dirs(files))} && base64 -d | tar xzf - -C /"
        self._exec(cmd, stdin=payload, timeout=120, fail_label="bulk upload")

    def _modal_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive (sandboxes run as root, so /root/.hermes)."""
        data = self._exec("tar cf - -C / root/.hermes", timeout=120, fail_label="bulk download", capture=True)
        dest.write_bytes(data.encode() if isinstance(data, str) else data)

    def _terminate_sandbox_quietly(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            self._worker.run_coroutine(sandbox.terminate.aio(), timeout=30)
        except Exception as exc:
            logger.warning("Modal: could not terminate sandbox after failure: %s", exc)
        finally:
            self._sandbox = None

    def _purge_synced_subtrees(self) -> None:
        self._exec(
            quoted_purge_command("/root/.hermes"), timeout=30,
            fail_label="sync-owned subtree purge")

    def _modal_delete(self, remote_paths: list[str]) -> None:
        self._exec(quoted_rm_command(remote_paths), timeout=15)

    def _remove_late_credential_files(self) -> None:
        if self._late_credential_remote_paths:
            self._exec(
                quoted_rm_command(sorted(self._late_credential_remote_paths)),
                timeout=30, fail_label="late credential removal")

    def _before_execute(self) -> None:
        try:
            from tools.credential_files import get_credential_file_mounts
            self._late_credential_remote_paths.update(
                entry["container_path"] for entry in get_credential_file_mounts()
                if entry["container_path"] not in self._initial_credential_remote_paths)
        except Exception as exc:
            logger.debug("Modal: could not track late credential mounts: %s", exc)
        self._sync_manager.sync()  # rate-limited internally

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120, stdin_data: str | None = None):
        sandbox, worker = self._sandbox, self._worker
        if sandbox is None:
            raise RuntimeError("Modal sandbox is not initialized")
        pidfile = _cancel_pidfile(f"{id(self):x}-{next(_cancel_id_counter)}")
        tagged_command = _cancellable_command(cmd_string, pidfile)
        execution_lock = threading.Lock()
        execution_started = False
        cancel_requested = False
        cancel_dispatched = False

        async def _do_cancel():
            process = await sandbox.exec.aio(
                "bash", "-c", _CANCEL_SCRIPT, "--", pidfile,
                timeout=_CANCEL_TIMEOUT_SECONDS)
            await process.wait.aio()

        def _claim_cancel_dispatch() -> bool:
            nonlocal cancel_dispatched
            with execution_lock:
                if not execution_started or cancel_dispatched:
                    return False
                cancel_dispatched = True
                return True

        def cancel():
            nonlocal cancel_requested
            with execution_lock:
                cancel_requested = True
            if _claim_cancel_dispatch():
                try:
                    worker.run_coroutine(_do_cancel(), timeout=_CANCEL_TIMEOUT_SECONDS + 10)
                except Exception as exc:
                    logger.warning("Modal: could not cancel remote command: %s", exc)

        def exec_fn() -> tuple[str, int]:
            async def _do():
                nonlocal execution_started
                process = await sandbox.exec.aio(*bash_argv(tagged_command, login), timeout=timeout)
                with execution_lock:
                    execution_started = True
                    pending_cancel = cancel_requested
                if pending_cancel and _claim_cancel_dispatch():
                    try:
                        await _do_cancel()
                    except Exception as exc:
                        logger.warning("Modal: could not cancel remote command: %s", exc)
                if stdin_data is not None:
                    await _stream_stdin(process, stdin_data, self._STDIN_CHUNK_SIZE)
                stdout = _as_text(await process.stdout.read.aio())
                stderr = _as_text(await process.stderr.read.aio())
                exit_code = await process.wait.aio()
                return "\n".join(part for part in (stdout, stderr) if part), exit_code
            return worker.run_coroutine(_do(), timeout=timeout + 30)
        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        """Snapshot the filesystem (if persistent) then stop the sandbox."""
        if self._sandbox is None:
            return
        if self._sync_manager:
            logger.info("Modal: syncing files from sandbox...")
            self._sync_manager.sync_back()
        if self._persistent:
            credentials_removed = True
            try:
                self._remove_late_credential_files()
            except Exception as exc:
                credentials_removed = False
                logger.warning("Modal: refusing to snapshot unsanitized credentials: %s", exc)
            async def _snapshot():
                return (await self._sandbox.snapshot_filesystem.aio()).object_id
            if credentials_removed:
                try:
                    snapshot_id = self._worker.run_coroutine(_snapshot(), timeout=60)
                except Exception:
                    snapshot_id = None  # snapshot errors are non-fatal; sandbox still terminates
            else:
                snapshot_id = None
            if snapshot_id:
                try:
                    _store_direct_snapshot(self._task_id, snapshot_id)
                    logger.info("Modal: saved filesystem snapshot %s for task %s", snapshot_id[:20], self._task_id)
                except Exception as e:
                    logger.warning("Modal: filesystem snapshot failed: %s", e)
        try:
            self._worker.run_coroutine(self._sandbox.terminate.aio(), timeout=15)
        except Exception:
            pass
        finally:
            self._worker.stop()
            self._sandbox = self._app = None

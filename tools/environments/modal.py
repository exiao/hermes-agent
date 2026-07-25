"""Modal cloud execution environment using the native Modal SDK directly.

Uses ``Sandbox.create()`` + ``Sandbox.exec()`` instead of the older runtime
wrapper, while preserving Hermes' persistent snapshot behavior across sessions.
"""

import asyncio
import base64
import io
import logging
import shlex
import tarfile
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from tools.environments.base import (
    BaseEnvironment,
    _ThreadedProcessHandle,
    _load_json_store,
    _save_json_store,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = get_hermes_home() / "modal_snapshots.json"
_DIRECT_SNAPSHOT_NAMESPACE = "direct"

# Where per-command process-group id files live inside the sandbox. Under /tmp
# so they never land in a filesystem snapshot or a synced directory.
_MODAL_PGID_DIR = "/tmp/.hermes-pgid"
# Seconds a cancelled command gets to exit on SIGTERM before SIGKILL.
_MODAL_CANCEL_GRACE_SECONDS = 5
# Headroom on the SDK exec deadline so the LOCAL deadline in
# _wait_for_process always fires first and cancellation runs. If Modal's
# deadline won, it would kill the outer bash while the setsid'd command kept
# running, and no group cancellation would ever be issued.
_MODAL_EXEC_TIMEOUT_HEADROOM = 15


def _wrap_for_group_cancel(cmd_string: str, pid_file: str, *, login: bool = False) -> str:
    """Run cmd_string in its own process group, recording how to signal it.

    The Modal SDK's ContainerProcess exposes no kill, so cancellation used to
    terminate the entire sandbox. Instead, start the command with setsid so it
    leads a new process group, and record that group's id where cancel() can
    read it. Signalling -<pgid> then reaches the command and every descendant
    it spawned, leaving the sandbox untouched.

    The recorded value is tagged with how it must be signalled: ``G:<pgid>``
    when setsid gave the command its own process group, ``P:<pid>`` for the
    no-setsid fallback, where the child shares the wrapper's group and only the
    child pid itself can be signalled (signalling -<pid> there would target a
    nonexistent group and leave the command running).

    The group id is published by the setsid'd shell ITSELF: ``setsid`` makes
    that shell a session AND process-group leader, so its pgid is exactly its
    own ``$$``, no ``ps`` required. Publishing ``$!`` from the parent instead
    would name the forked process before ``setsid(2)`` had necessarily
    executed, so a cancel landing in that window would signal a group that does
    not exist yet, get ESRCH, and let the command establish itself unkilled.

    A cancel can also arrive before the command has even started, while the
    exec is still queued. cancel() drops a cancel marker first, so the wrapper
    refuses to start when the marker is already present, and re-checks after
    the child exists. That post-check signals the direct child pid (valid the
    moment fork returns, unlike the group) and escalates to SIGKILL, so a
    cancel racing publication still kills the command.

    The exit status of the wrapped command is propagated unchanged, so callers
    (and the agent) see the real exit code, stdout, and stderr.
    """
    flags = "-l -c" if login else "-c"
    quoted = shlex.quote(cmd_string)
    cancel_file = f"{pid_file}.cancel"
    # Runs inside the setsid'd shell, which leads its own process group, so
    # $$ IS the pgid. Publish it, then become the command.
    inner = f"echo G:$$ > {pid_file}; exec bash {flags} {quoted}"
    return (
        f"mkdir -p {_MODAL_PGID_DIR} 2>/dev/null; "
        # Cancelled before we got scheduled: never start the command.
        f"if [ -e {cancel_file} ]; then rm -f {cancel_file} 2>/dev/null; "
        f"exit {128 + 15}; fi; "
        f"if command -v setsid >/dev/null 2>&1; then "
        f"setsid bash -c {shlex.quote(inner)} & __hermes_pid=$!; "
        f"else bash {flags} {quoted} & __hermes_pid=$!; "
        f"echo P:$__hermes_pid > {pid_file}; fi; "
        # Cancelled while we were starting up: kill the child directly (its pid
        # is valid immediately, unlike the group) and escalate.
        f"if [ -e {cancel_file} ]; then "
        f'kill -TERM "$__hermes_pid" 2>/dev/null; '
        f"for _ in $(seq 1 {_MODAL_CANCEL_GRACE_SECONDS}); do "
        f'kill -0 "$__hermes_pid" 2>/dev/null || break; sleep 1; done; '
        f'kill -KILL "$__hermes_pid" 2>/dev/null; fi; '
        f"wait $__hermes_pid; __hermes_rc=$?; "
        f"rm -f {pid_file} {cancel_file} 2>/dev/null; exit $__hermes_rc"
    )


def _load_snapshots() -> dict:
    return _load_json_store(_SNAPSHOT_STORE)


def _save_snapshots(data: dict) -> None:
    _save_json_store(_SNAPSHOT_STORE, data)


def _direct_snapshot_key(task_id: str) -> str:
    return f"{_DIRECT_SNAPSHOT_NAMESPACE}:{task_id}"


def _get_snapshot_restore_candidate(task_id: str) -> tuple[str | None, bool]:
    snapshots = _load_snapshots()
    namespaced_key = _direct_snapshot_key(task_id)
    snapshot_id = snapshots.get(namespaced_key)
    if isinstance(snapshot_id, str) and snapshot_id:
        return snapshot_id, False
    legacy_snapshot_id = snapshots.get(task_id)
    if isinstance(legacy_snapshot_id, str) and legacy_snapshot_id:
        return legacy_snapshot_id, True
    return None, False


def _store_direct_snapshot(task_id: str, snapshot_id: str) -> None:
    snapshots = _load_snapshots()
    snapshots[_direct_snapshot_key(task_id)] = snapshot_id
    snapshots.pop(task_id, None)
    _save_snapshots(snapshots)


def _delete_direct_snapshot(task_id: str, snapshot_id: str | None = None) -> None:
    snapshots = _load_snapshots()
    updated = False
    for key in (_direct_snapshot_key(task_id), task_id):
        value = snapshots.get(key)
        if value is None:
            continue
        if snapshot_id is None or value == snapshot_id:
            snapshots.pop(key, None)
            updated = True
    if updated:
        _save_snapshots(snapshots)


def _ensure_modal_sdk() -> None:
    """Lazy-install modal on demand. Idempotent — fast no-op once installed."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("terminal.modal", prompt=False)
    except ImportError:
        pass
    except Exception as e:
        raise ImportError(str(e))


def _resolve_modal_image(image_spec: Any) -> Any:
    """Convert registry references or snapshot ids into Modal image objects.

    Includes add_python support for ubuntu/debian images (absorbed from PR 4511).
    """
    _ensure_modal_sdk()
    import modal as _modal

    if not isinstance(image_spec, str):
        return image_spec

    if image_spec.startswith("im-"):
        return _modal.Image.from_id(image_spec)

    # PR 4511: add python to ubuntu/debian images that don't have it
    lower = image_spec.lower()
    add_python = any(base in lower for base in ("ubuntu", "debian"))

    setup_commands = [
        "RUN rm -rf /usr/local/lib/python*/site-packages/pip* 2>/dev/null; "
        "python -m ensurepip --upgrade --default-pip 2>/dev/null || true",
    ]
    if add_python:
        setup_commands.insert(0,
            "RUN apt-get update -qq && apt-get install -y -qq python3 python3-venv > /dev/null 2>&1 || true"
        )

    return _modal.Image.from_registry(
        image_spec,
        setup_dockerfile_commands=setup_commands,
    )


class _AsyncWorker:
    """Background thread with its own event loop for async-safe Modal calls."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started.wait(timeout=30)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run_coroutine(self, coro, timeout=600):
        from agent.async_utils import safe_schedule_threadsafe
        if self._loop is None or self._loop.is_closed():
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("AsyncWorker loop is not running")
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
    """Modal cloud execution via native Modal sandboxes.

    Spawn-per-call via _ThreadedProcessHandle wrapping async SDK calls.
    cancel_fn wired to sandbox.terminate for interrupt support.
    """

    _stdin_mode = "heredoc"
    _snapshot_timeout = 60  # Modal cold starts can be slow

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        modal_sandbox_kwargs: Optional[dict[str, Any]] = None,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._sandbox = None
        self._app = None
        self._worker = _AsyncWorker()
        self._sync_manager: FileSyncManager | None = None  # initialized after sandbox creation

        sandbox_kwargs = dict(modal_sandbox_kwargs or {})

        restored_snapshot_id = None
        restored_from_legacy_key = False
        if self._persistent:
            restored_snapshot_id, restored_from_legacy_key = _get_snapshot_restore_candidate(
                self._task_id
            )
            if restored_snapshot_id:
                logger.info("Modal: restoring from snapshot %s", restored_snapshot_id[:20])

        _ensure_modal_sdk()
        import modal as _modal

        cred_mounts = []
        try:
            from tools.credential_files import (
                get_credential_file_mounts,
                iter_skills_files,
                iter_cache_files,
            )

            for mount_entry in get_credential_file_mounts():
                cred_mounts.append(
                    _modal.Mount.from_local_file(
                        mount_entry["host_path"],
                        remote_path=mount_entry["container_path"],
                    )
                )
            for entry in iter_skills_files():
                cred_mounts.append(
                    _modal.Mount.from_local_file(
                        entry["host_path"],
                        remote_path=entry["container_path"],
                    )
                )
            cache_files = iter_cache_files()
            for entry in cache_files:
                cred_mounts.append(
                    _modal.Mount.from_local_file(
                        entry["host_path"],
                        remote_path=entry["container_path"],
                    )
                )
        except Exception as e:
            logger.debug("Modal: could not load credential file mounts: %s", e)

        self._worker.start()

        async def _create_sandbox(image_spec: Any):
            app = await _modal.App.lookup.aio("hermes-agent", create_if_missing=True)
            create_kwargs = dict(sandbox_kwargs)
            if cred_mounts:
                existing_mounts = list(create_kwargs.pop("mounts", []))
                existing_mounts.extend(cred_mounts)
                create_kwargs["mounts"] = existing_mounts
            sandbox = await _modal.Sandbox.create.aio(
                "sleep", "infinity",
                image=image_spec,
                app=app,
                timeout=int(create_kwargs.pop("timeout", 3600)),
                **create_kwargs,
            )
            return app, sandbox

        try:
            target_image_spec = restored_snapshot_id or image
            try:
                effective_image = _resolve_modal_image(target_image_spec)
                self._app, self._sandbox = self._worker.run_coroutine(
                    _create_sandbox(effective_image), timeout=300,
                )
            except Exception as exc:
                if not restored_snapshot_id:
                    raise
                logger.warning(
                    "Modal: failed to restore snapshot %s, retrying with base image: %s",
                    restored_snapshot_id[:20], exc,
                )
                _delete_direct_snapshot(self._task_id, restored_snapshot_id)
                base_image = _resolve_modal_image(image)
                self._app, self._sandbox = self._worker.run_coroutine(
                    _create_sandbox(base_image), timeout=300,
                )
            else:
                if restored_snapshot_id and restored_from_legacy_key:
                    _store_direct_snapshot(self._task_id, restored_snapshot_id)
        except Exception:
            self._worker.stop()
            raise

        logger.info("Modal: sandbox created (task=%s)", self._task_id)

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files("/root/.hermes"),
            upload_fn=self._modal_upload,
            delete_fn=self._modal_delete,
            bulk_upload_fn=self._modal_bulk_upload,
            bulk_download_fn=self._modal_bulk_download,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    def _modal_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via base64 piped through stdin."""
        content = Path(host_path).read_bytes()
        b64 = base64.b64encode(content).decode("ascii")
        container_dir = str(Path(remote_path).parent)
        cmd = (
            f"mkdir -p {shlex.quote(container_dir)} && "
            f"base64 -d > {shlex.quote(remote_path)}"
        )

        async def _write():
            proc = await self._sandbox.exec.aio("bash", "-c", cmd)
            offset = 0
            chunk_size = self._STDIN_CHUNK_SIZE
            while offset < len(b64):
                proc.stdin.write(b64[offset:offset + chunk_size])
                await proc.stdin.drain.aio()
                offset += chunk_size
            proc.stdin.write_eof()
            await proc.stdin.drain.aio()
            await proc.wait.aio()

        self._worker.run_coroutine(_write(), timeout=30)

    # Modal SDK stdin buffer limit (legacy server path).  The command-router
    # path allows 16 MB, but we must stay under the smaller 2 MB cap for
    # compatibility.  Chunks are written below this threshold and flushed
    # individually via drain().
    _STDIN_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB — safe for both transport paths

    def _modal_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files via tar archive piped through stdin.

        Builds a gzipped tar archive in memory and streams it into a
        ``base64 -d | tar xzf -`` pipeline via the process's stdin,
        avoiding the Modal SDK's 64 KB ``ARG_MAX_BYTES`` exec-arg limit.
        """
        if not files:
            return

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for host_path, remote_path in files:
                tar.add(host_path, arcname=remote_path.lstrip("/"))
        payload = base64.b64encode(buf.getvalue()).decode("ascii")

        parents = unique_parent_dirs(files)
        mkdir_part = quoted_mkdir_command(parents)
        cmd = f"{mkdir_part} && base64 -d | tar xzf - -C /"

        async def _bulk():
            proc = await self._sandbox.exec.aio("bash", "-c", cmd)

            # Stream payload through stdin in chunks to stay under the
            # SDK's per-write buffer limit (2 MB legacy / 16 MB router).
            offset = 0
            chunk_size = self._STDIN_CHUNK_SIZE
            while offset < len(payload):
                proc.stdin.write(payload[offset:offset + chunk_size])
                await proc.stdin.drain.aio()
                offset += chunk_size

            proc.stdin.write_eof()
            await proc.stdin.drain.aio()

            exit_code = await proc.wait.aio()
            if exit_code != 0:
                stderr_text = await proc.stderr.read.aio()
                raise RuntimeError(
                    f"Modal bulk upload failed (exit {exit_code}): {stderr_text}"
                )

        self._worker.run_coroutine(_bulk(), timeout=120)

    def _modal_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive.

        Modal sandboxes always run as root, so /root/.hermes is hardcoded
        (consistent with iter_sync_files call on line 269).
        """
        async def _download():
            proc = await self._sandbox.exec.aio(
                "bash", "-c", "tar cf - -C / root/.hermes"
            )
            data = await proc.stdout.read.aio()
            exit_code = await proc.wait.aio()
            if exit_code != 0:
                raise RuntimeError(f"Modal bulk download failed (exit {exit_code})")
            return data

        tar_bytes = self._worker.run_coroutine(_download(), timeout=120)
        if isinstance(tar_bytes, str):
            tar_bytes = tar_bytes.encode()
        dest.write_bytes(tar_bytes)

    def _modal_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via exec."""
        rm_cmd = quoted_rm_command(remote_paths)

        async def _rm():
            proc = await self._sandbox.exec.aio("bash", "-c", rm_cmd)
            await proc.wait.aio()

        self._worker.run_coroutine(_rm(), timeout=15)

    def _before_execute(self) -> None:
        """Sync files to sandbox via FileSyncManager (rate-limited internally)."""
        self._sync_manager.sync()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping an async Modal sandbox exec."""
        sandbox = self._sandbox
        worker = self._worker

        # Run the command in its own process group and record that group's
        # leader PID, so cancel() can signal the command (and every descendant
        # it spawned) without touching the sandbox itself.
        pid_file = f"{_MODAL_PGID_DIR}/{uuid.uuid4().hex}"
        wrapped = _wrap_for_group_cancel(cmd_string, pid_file, login=login)

        def cancel():
            # Signal the command's process GROUP, not the sandbox. Terminating
            # the sandbox here used to wedge the whole session: every later exec
            # hit a dead sandbox and returned exit 1 with empty output forever.
            # SIGTERM first so the command can clean up, then SIGKILL for
            # anything that ignores or blocks TERM.
            #
            # The cancel marker is written FIRST so a cancel that arrives before
            # the command starts (exec still queued) is not lost: the wrapper
            # checks for it both before launching and once the child exists, and
            # kills the child itself in that window. A missing pid file here
            # therefore means "the wrapper will honour the marker", not "nothing
            # to do". The recorded pid is tagged G: (own process group, signal
            # the group) or P: (no setsid, signal the pid itself).
            script = (
                f"mkdir -p {_MODAL_PGID_DIR} 2>/dev/null; "
                f"touch {pid_file}.cancel 2>/dev/null; "
                f"rec=''; "
                # The pgid is published by the setsid'd shell a moment after the
                # exec starts; wait briefly for it rather than giving up. If it
                # never appears the wrapper's own marker check does the killing.
                f"for _ in $(seq 1 {_MODAL_CANCEL_GRACE_SECONDS}); do "
                f"rec=$(cat {pid_file} 2>/dev/null) && [ -n \"$rec\" ] && break; "
                "sleep 1; done; "
                '[ -n "$rec" ] || exit 0; '
                'case "$rec" in '
                'G:*) target=-${rec#G:} ;; '
                'P:*) target=${rec#P:} ;; '
                '*) exit 0 ;; '
                'esac; '
                'kill -TERM "$target" 2>/dev/null; '
                f"for _ in $(seq 1 {_MODAL_CANCEL_GRACE_SECONDS}); do "
                'kill -0 "$target" 2>/dev/null || exit 0; sleep 1; done; '
                'kill -KILL "$target" 2>/dev/null; exit 0'
            )

            async def _cancel():
                process = await sandbox.exec.aio("bash", "-c", script)
                await process.wait.aio()

            try:
                worker.run_coroutine(
                    _cancel(), timeout=_MODAL_CANCEL_GRACE_SECONDS + 15
                )
            except Exception as exc:
                # A failed cancel leaves a runaway command, which is bad but
                # recoverable. Tearing down the sandbox is worse.
                logger.warning("Modal: could not cancel command: %s", exc)

        def exec_fn() -> tuple[str, int]:
            async def _do():
                args = ["bash", "-c", wrapped]
                # Give the SDK's own deadline headroom over the local one in
                # _wait_for_process, which is exactly `timeout`. If Modal fired
                # first it would kill the outer bash while the real command,
                # now a background child in its own setsid session, kept
                # running: the handle would report completion and cancel()
                # would never be invoked to reap the group. The local deadline
                # must always win so cancellation is what stops a command.
                process = await sandbox.exec.aio(
                    *args, timeout=timeout + _MODAL_EXEC_TIMEOUT_HEADROOM
                )
                stdout = await process.stdout.read.aio()
                stderr = await process.stderr.read.aio()
                exit_code = await process.wait.aio()
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                output = stdout
                if stderr:
                    output = f"{stdout}\n{stderr}" if stdout else stderr
                return output, exit_code

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
            try:
                async def _snapshot():
                    img = await self._sandbox.snapshot_filesystem.aio()
                    return img.object_id

                try:
                    snapshot_id = self._worker.run_coroutine(_snapshot(), timeout=60)
                except Exception:
                    snapshot_id = None

                if snapshot_id:
                    _store_direct_snapshot(self._task_id, snapshot_id)
                    logger.info(
                        "Modal: saved filesystem snapshot %s for task %s",
                        snapshot_id[:20], self._task_id,
                    )
            except Exception as e:
                logger.warning("Modal: filesystem snapshot failed: %s", e)

        try:
            self._worker.run_coroutine(self._sandbox.terminate.aio(), timeout=15)
        except Exception:
            pass
        finally:
            self._worker.stop()
            self._sandbox = None
            self._app = None

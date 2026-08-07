"""Modal cloud execution environment using the native Modal SDK directly.

Uses ``Sandbox.create()`` + ``Sandbox.exec()`` instead of the older runtime
wrapper, while preserving Hermes' persistent snapshot behavior across sessions.
"""

import asyncio
import base64
import io
import itertools
import logging
import shlex
import tarfile
import threading
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
    quoted_purge_command,
    synced_subtree_roots,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = get_hermes_home() / "modal_snapshots.json"
_DIRECT_SNAPSHOT_NAMESPACE = "direct"

# Cancellation
# ------------
# Modal's ``ContainerProcess`` exposes only poll/wait/stdout/stderr — there is
# no kill. The obvious workaround, ``sandbox.terminate()``, cancels the command
# by destroying the entire sandbox: ``poll()`` then returns an exit code and
# every later ``exec`` raises ``NotFoundError``, so one interrupted command
# bricks the session (measured against modal==1.3.4).
#
# The sandbox itself, however, still accepts ``exec``. So we cancel a command
# the way a shell would: have the command record its own PID, then run a
# second exec that signals that process group. The sandbox, its filesystem,
# and every other running command survive.
#
# Why a PID file rather than an environment marker: the command runs under
# ``_wrap_command``, which sources the session's ``export -p`` snapshot. Any
# variable we export is therefore overwritten by the snapshot's value on every
# later command, and a shell-builtin-only command (``while :; do :; done``)
# never spawns a child carrying it at all. ``$$`` is written by the shell
# itself, so it is correct in both cases. Both verified against live Modal.
_CANCEL_DIR = "/tmp/.hermes-cancel"
_CANCEL_TIMEOUT_SECONDS = 20
_cancel_id_counter = itertools.count()
# Age after which a PID file cannot belong to a live command and is swept.
_PIDFILE_MAX_AGE_MINUTES = 720


def _is_provider_reaped_error(exc: BaseException | str) -> bool:
    """Identify Modal errors that mean the cached sandbox no longer exists."""
    error_type = type(exc).__name__.lower().replace("_", "") if isinstance(exc, BaseException) else ""
    message = str(exc).lower()
    if "notfounderror" in error_type or "notfounderror" in message.replace("_", ""):
        return True
    return "sandbox" in message and any(
        marker in message
        for marker in ("not found", "terminated", "does not exist", "stopped")
    )

# TERM the process group first so the command can run traps, then KILL
# whatever ignored it. The KILL escalation is NOT conditional on the recorded
# PID still existing: a descendant that traps TERM outlives the wrapper bash,
# so gating on ``/proc/$pid`` would skip the escalation and leak it (measured:
# a trapping child survived the guarded version and died under this one).
#
# The PID file may lag the exec RPC by a few ms, so poll briefly for it. This
# only smooths that small window; the launch race proper (a cancel arriving
# before ``sandbox.exec`` returns at all) is closed in Python by _run_bash,
# which replays a pending cancel once the command is up. A file that never
# appears means the command never started, and a no-op is correct.
#
# The PGID is read by skipping past the parenthesized ``comm`` field rather
# than with ``cut -f5``: ``comm`` is the executable's basename and may contain
# spaces, which shifts every positional field. Measured on a stat line whose
# comm is ``(we ird)``, ``cut -f5`` returned the PPID (7) where the PGID was
# 42; the sed form returns 42 for both spaced and unspaced comms.
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
# The wrapper may already be gone while its children run on (``sleep 300 &``
# exits the wrapper immediately but the child keeps the stdout handle open, so
# the exec RPC is still pending). The recorded PGID therefore has to survive
# the wrapper: re-derive it from /proc only as a fallback for an older file.
if [ -z "$pgid" ] && [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
  pgid=$(sed 's/^.*) //' "/proc/$pid/stat" 2>/dev/null | cut -d' ' -f3)
fi
[ -n "$pgid" ] && [ "$pgid" != "0" ] || pgid=""
[ -n "$pgid" ] || [ -d "/proc/$pid" ] || { echo "cancelled=0"; exit 0; }

# Collect the recorded shell and every descendant by walking /proc parent
# links. The process group alone is not enough: a command that calls setsid
# (or otherwise leaves the group) keeps running while a group-only signal
# reports success. Walking the tree catches those; signalling the group as
# well catches anything that reparented away from us.
collect_tree() {
  echo "$1"
  for d in /proc/[0-9]*; do
    child=${d#/proc/}
    [ "$child" = "$1" ] && continue
    cppid=$(sed 's/^.*) //' "$d/stat" 2>/dev/null | cut -d' ' -f2)
    [ "$cppid" = "$1" ] && collect_tree "$child"
  done
}
if [ -d "/proc/$pid" ]; then
  targets=$(collect_tree "$pid")
else
  targets=""
fi

for sig in TERM KILL; do
  [ -n "$pgid" ] && kill -"$sig" -"$pgid" 2>/dev/null
  for target in $targets; do
    kill -"$sig" "$target" 2>/dev/null
  done
  [ "$sig" = TERM ] && sleep 0.3
done
echo "cancelled=1"
"""


def _cancel_pidfile(command_id: str) -> str:
    return f"{_CANCEL_DIR}/{command_id}"


def _stale_pidfile_sweep() -> str:
    """Shell to drop PID files older than a command could plausibly still hold.

    Cleanup is a sweep, not a per-command teardown. Every attempt at the
    latter created a new failure of the same class: an in-shell ``EXIT`` trap
    is clobbered by the command's own; a subshell to survive that breaks
    ``$$``; an awaited reaper extends the caller's timed window; a
    fire-and-forget reaper then races ``cleanup()``'s snapshot. Each fix was
    real and each spawned the next, which is the signal to change the shape
    rather than patch again.

    A PID file is inert metadata under ``/tmp``: nothing reads it except
    ``cancel()``, which targets one exact path. So it does not need timely
    removal at all — it needs to not accumulate. Sweeping files older than
    ``_PIDFILE_MAX_AGE_MINUTES`` at command START has no completion path to
    miss, nothing to await, and no task to race, and it bounds the directory
    for any command that outlives its own cleanup (a cancel, a crash, a
    sandbox reset).
    """
    return (
        f"find {shlex.quote(_CANCEL_DIR)} -maxdepth 1 -type f "
        f"-mmin +{_PIDFILE_MAX_AGE_MINUTES} -delete 2>/dev/null || true"
    )


def _cancellable_command(cmd_string: str, pidfile: str) -> str:
    """Prefix ``cmd_string`` so it records its own PID for cancellation.

    ``$$`` is the shell's own PID, written before the command runs, so this
    works for shell builtins that never fork and survives the env snapshot
    that ``_wrap_command`` sources. Writing the file must never break the
    command, hence the ``|| true``.

    Nothing is appended after the command, deliberately. Two attempts at
    in-shell cleanup each changed the command's semantics:

    - an ``EXIT`` trap is silently discarded by any command that installs its
      own (``FileOperations._atomic_write`` sets one, then ``trap - EXIT``),
      so the file leaked on the normal file-write path;
    - wrapping in ``( ... )`` restores cleanup but breaks ``$$``: bash keeps
      ``$$`` as the *parent* shell's PID inside a subshell, so a command doing
      ``trap 'exit 42' TERM; kill -TERM $$`` signalled the wrapper instead of
      itself and returned 143 where it used to return 42.

    Stale files are reaped by the age sweep prefixed here, which runs before
    the command and so cannot interact with how it exits.
    """
    quoted = shlex.quote(pidfile)
    return (
        f"mkdir -p {shlex.quote(_CANCEL_DIR)} 2>/dev/null; "
        f"{_stale_pidfile_sweep()}; "
        f"echo \"$$ $(sed 's/^.*) //' /proc/$$/stat 2>/dev/null "
        f"| cut -d' ' -f3)\" > {quoted} 2>/dev/null || true; "
        f"{cmd_string}"
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
    cancel_fn signals the running command by token (see _CANCEL_SCRIPT) so an
    interrupt or timeout never destroys the sandbox.
    """

    _stdin_mode = "pipe"
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

        credential_mounts = []
        initial_credential_remote_paths: set[str] = set()
        late_credential_remote_paths: set[str] = set()
        try:
            from tools.credential_files import get_credential_file_mounts

            for mount_entry in get_credential_file_mounts():
                remote_path = mount_entry["container_path"]
                if any(
                    remote_path == root or remote_path.startswith(root + "/")
                    for root in synced_subtree_roots("/root/.hermes")
                ):
                    # Modal mounts are read-only. A recursive purge of a
                    # sync-owned root would therefore fail if a credential
                    # mount lived below it. Keep this credential writable via
                    # the normal sync path and scrub it before snapshotting.
                    late_credential_remote_paths.add(remote_path)
                    logger.warning(
                        "Modal: not mounting credential below sync root: %s",
                        remote_path,
                    )
                    continue
                credential_mounts.append(
                    _modal.Mount.from_local_file(
                        mount_entry["host_path"],
                        remote_path=remote_path,
                    )
                )
                initial_credential_remote_paths.add(remote_path)
        except Exception as e:
            logger.debug("Modal: could not load credential file mounts: %s", e)

        self._worker.start()
        self._initial_credential_remote_paths = initial_credential_remote_paths
        self._late_credential_remote_paths = late_credential_remote_paths

        async def _create_sandbox(image_spec: Any):
            app = await _modal.App.lookup.aio("hermes-agent", create_if_missing=True)
            create_kwargs = dict(sandbox_kwargs)
            # Only credentials are mounted. Skills, plans and caches arrive via
            # the forced sync below, which runs on every construction and is the
            # single source of truth for those paths -- mounting them here too
            # would upload the same files twice and, being read-only, would also
            # block the restore purge.
            if credential_mounts:
                existing_mounts = list(create_kwargs.pop("mounts", []))
                existing_mounts.extend(credential_mounts)
                create_kwargs["mounts"] = existing_mounts
            # ``timeout`` is a hard MAXIMUM LIFETIME, not an inactivity window:
            # it kills the sandbox mid-command, so it cannot be lowered to reap
            # leaked sandboxes without truncating legitimate long builds.
            # ``idle_timeout`` is the inactivity reaper, and it is the backstop
            # for a sandbox whose owning process died before ``cleanup()`` ran
            # (SIGKILL, OOM, ``os._exit`` past the cleanup hook). Without it a
            # leaked sandbox bills for the full ``timeout`` window.
            create_timeout = int(create_kwargs.pop("timeout", 3600))
            idle_timeout = create_kwargs.pop("idle_timeout", None)
            extra: dict[str, Any] = {}
            if idle_timeout:
                # Never exceed the hard lifetime, and only pass the kwarg when
                # the installed modal SDK actually supports it.
                idle_seconds = max(1, min(int(idle_timeout), create_timeout))
                try:
                    import inspect as _inspect
                    if "idle_timeout" in _inspect.signature(
                        _modal.Sandbox.create
                    ).parameters:
                        extra["idle_timeout"] = idle_seconds
                    else:
                        logger.debug(
                            "Modal: installed SDK has no idle_timeout support; "
                            "leaked sandboxes will reap at timeout=%ss",
                            create_timeout,
                        )
                except Exception:  # pragma: no cover - defensive
                    pass
            sandbox = await _modal.Sandbox.create.aio(
                "sleep", "infinity",
                image=image_spec,
                app=app,
                timeout=create_timeout,
                **extra,
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
                # The fallback sandbox is a pristine base image, so there is no
                # prior-instance state to reconcile.
                restored_snapshot_id = None
        except Exception:
            self._worker.stop()
            raise

        logger.info("Modal: sandbox created (task=%s)", self._task_id)

        try:
            if restored_snapshot_id and restored_from_legacy_key:
                _store_direct_snapshot(self._task_id, restored_snapshot_id)
            if restored_snapshot_id:
                # A restored snapshot still holds the previous instance's synced
                # files, but the fresh FileSyncManager below starts with an empty
                # map and so computes no deletions for them.  Clear the sync-owned
                # subtrees; the forced sync re-uploads everything still on the host.
                self._purge_synced_subtrees()

            self._sync_manager = FileSyncManager(
                get_files_fn=lambda: iter_sync_files("/root/.hermes"),
                upload_fn=self._modal_upload,
                delete_fn=self._modal_delete,
                bulk_upload_fn=self._modal_bulk_upload,
                bulk_download_fn=self._modal_bulk_download,
            )
            self._sync_manager.sync(force=True, raise_on_error=True)
            self.init_session()
        except Exception:
            # The sandbox is live by now, so it outlives the constructor unless
            # we tear it down here. Without this the caller sees an exception
            # while a paid sandbox and its worker thread keep running until the
            # sandbox timeout.
            self._terminate_sandbox_quietly()
            self._worker.stop()
            raise

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

    @staticmethod
    def _iter_stdin_chunks(payload: str, max_bytes: int):
        """Yield payload chunks whose UTF-8 encoding fits the transport cap."""
        start = 0
        chunk_bytes = 0
        for index, char in enumerate(payload):
            char_bytes = len(char.encode("utf-8"))
            if chunk_bytes and chunk_bytes + char_bytes > max_bytes:
                yield payload[start:index]
                start = index
                chunk_bytes = 0
            chunk_bytes += char_bytes
        if start < len(payload):
            yield payload[start:]

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

    def _terminate_sandbox_quietly(self) -> None:
        """Best-effort sandbox teardown for a failed construction path."""
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            async def _terminate():
                await sandbox.terminate.aio()

            self._worker.run_coroutine(_terminate(), timeout=30)
        except Exception as exc:
            logger.warning("Modal: could not terminate sandbox after failure: %s", exc)
        finally:
            self._sandbox = None

    def _purge_synced_subtrees(self) -> None:
        """Clear sync-owned directories in a restored sandbox.

        Raises on a nonzero exit. ``sync(force=True)`` only uploads files that
        still exist on the host, so it cannot discover -- let alone delete --
        remote paths a previous instance left behind. A silently failed purge
        therefore leaves the sandbox serving the very files this reconciliation
        exists to remove, with nothing downstream able to notice.
        """
        cmd = quoted_purge_command("/root/.hermes")

        async def _purge():
            proc = await self._sandbox.exec.aio("bash", "-c", cmd)
            return await proc.wait.aio()

        exit_code = self._worker.run_coroutine(_purge(), timeout=30)
        if exit_code != 0:
            raise RuntimeError(
                f"Modal: failed to purge sync-owned subtrees (exit {exit_code}); "
                "refusing to run on unreconciled snapshot state"
            )

    def _modal_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via exec."""
        rm_cmd = quoted_rm_command(remote_paths)

        async def _rm():
            proc = await self._sandbox.exec.aio("bash", "-c", rm_cmd)
            await proc.wait.aio()

        self._worker.run_coroutine(_rm(), timeout=15)

    def _remove_late_credential_files(self) -> None:
        """Remove credentials uploaded after construction before snapshotting.

        Credential mounts are excluded from the restored-subtree purge because
        Modal mounts remain available without being copied into snapshots.  A
        credential registered after sandbox creation has no such mount, so the
        recurring sync would otherwise copy it into the filesystem snapshot.
        """
        if not self._late_credential_remote_paths:
            return

        cmd = quoted_rm_command(sorted(self._late_credential_remote_paths))

        async def _remove():
            proc = await self._sandbox.exec.aio("bash", "-c", cmd)
            return await proc.wait.aio()

        exit_code = self._worker.run_coroutine(_remove(), timeout=30)
        if exit_code != 0:
            raise RuntimeError(
                "Modal: failed to remove late-synced credentials before snapshot "
                f"(exit {exit_code})"
            )

    def _before_execute(self) -> None:
        """Sync files to sandbox via FileSyncManager (rate-limited internally)."""
        try:
            from tools.credential_files import get_credential_file_mounts

            self._late_credential_remote_paths.update(
                entry["container_path"]
                for entry in get_credential_file_mounts()
                if entry["container_path"] not in self._initial_credential_remote_paths
            )
        except Exception as exc:
            logger.debug("Modal: could not track late credential mounts: %s", exc)
        self._sync_manager.sync()

    def execute(self, *args: Any, **kwargs: Any) -> dict:
        """Evict this instance when Modal has already reaped its sandbox."""
        try:
            result = super().execute(*args, **kwargs)
        except Exception as exc:
            if _is_provider_reaped_error(exc):
                try:
                    from tools.terminal_tool import _evict_cached_environment
                    _evict_cached_environment(self)
                except ImportError:
                    pass
                self._sandbox = None
                self._worker.stop()
            raise
        if result.get("returncode") == 1 and _is_provider_reaped_error(
            result.get("output", "")
        ):
            try:
                from tools.terminal_tool import _evict_cached_environment
                _evict_cached_environment(self)
            except ImportError:
                pass
            self._sandbox = None
            self._worker.stop()
        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None):
        """Return a _ThreadedProcessHandle wrapping an async Modal sandbox exec."""
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Modal sandbox is not initialized")
        worker = self._worker
        pidfile = _cancel_pidfile(f"{id(self):x}-{next(_cancel_id_counter)}")
        tagged_command = _cancellable_command(cmd_string, pidfile)

        execution_lock = threading.Lock()
        execution_started = False
        cancel_requested = False
        cancel_dispatched = False

        async def _do_cancel() -> None:
            proc = await sandbox.exec.aio(
                "bash", "-c", _CANCEL_SCRIPT, "--", pidfile,
                timeout=_CANCEL_TIMEOUT_SECONDS,
            )
            await proc.wait.aio()

        def _claim_cancel_dispatch() -> bool:
            nonlocal cancel_dispatched
            with execution_lock:
                if not execution_started or cancel_dispatched:
                    return False
                cancel_dispatched = True
                return True

        def _cancel_after_startup() -> None:
            try:
                worker.run_coroutine(_do_cancel(), timeout=_CANCEL_TIMEOUT_SECONDS + 10)
            except Exception as exc:
                # Best-effort: the caller has already stopped waiting on this
                # command. Leaving a stray process behind is strictly better
                # than tearing down a live sandbox.
                logger.warning("Modal: could not cancel remote command: %s", exc)

        def cancel():
            """Signal only this command, leaving the sandbox usable.

            An interrupt may arrive while ``sandbox.exec.aio`` is still
            creating the target. Remember it until that startup completes, so
            cancellation cannot race ahead of the PID-file registration.
            """
            nonlocal cancel_requested
            if sandbox is None:
                return
            with execution_lock:
                cancel_requested = True
            if _claim_cancel_dispatch():
                _cancel_after_startup()

        def exec_fn() -> tuple[str, int]:
            async def _do():
                nonlocal execution_started
                args = ["bash"]
                if login:
                    args.extend(["-l", "-c", tagged_command])
                else:
                    args.extend(["-c", tagged_command])
                process = await sandbox.exec.aio(*args, timeout=timeout)
                with execution_lock:
                    execution_started = True
                    pending_cancel = cancel_requested
                if pending_cancel and _claim_cancel_dispatch():
                    try:
                        await _do_cancel()
                    except Exception as exc:
                        # Match the post-start cancellation path: a transient
                        # cancellation transport failure must not prevent this
                        # handle from draining and waiting for its target.
                        logger.warning("Modal: could not cancel remote command: %s", exc)
                # Feed stdin before draining stdout. File writes pass their
                # body through this pipe, and an explicit EOF is required for
                # the remote `cat` to finish. Chunk writes to stay below the
                # SDK's per-write buffer cap.
                if stdin_data is not None:
                    chunk_size = self._STDIN_CHUNK_SIZE
                    for chunk in self._iter_stdin_chunks(stdin_data, chunk_size):
                        process.stdin.write(chunk)
                        await process.stdin.drain.aio()
                    process.stdin.write_eof()
                    await process.stdin.drain.aio()
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
                self._remove_late_credential_files()

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

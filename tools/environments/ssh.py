
"""SSH remote execution environment with ControlMaster connection persistence."""

import contextlib
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from hermes_constants import hermes_home_key
from tools.environments.base import BaseEnvironment, EnvironmentConnectionError
from tools.environments.base_output import _popen_bash
from tools.environments.file_sync import (
    FileSyncManager, iter_sync_files, quoted_mkdir_command, quoted_rm_command, unique_parent_dirs)
from tools.environments.remote_common import (
    bash_argv, client_env_with, load_hermes_env_vars, prepend_unset, resolve_passthrough_env, run_capture)

logger = logging.getLogger(__name__)

_TAR_CONCURRENT_CHANGE_WARNINGS = (
    "file changed as we read it", "file removed before we read it")
_TAR_SUMMARY_SUFFIXES = (
    "exiting with failure status due to previous errors", "error exit delayed from previous errors.")
_BULK_UPLOAD_MIN_TIMEOUT = 120
_BULK_UPLOAD_MAX_TIMEOUT = 1800
_BULK_UPLOAD_BYTES_PER_SEC = 2_000_000
_SYNC_BACK_EXCLUDE_DIRS = ("venvs",)


def _tar_stderr_is_only_concurrent_change(stderr: str) -> bool:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return False
    return all(
        line.lower().endswith(_TAR_SUMMARY_SUFFIXES)
        or any(warning in line.lower() for warning in _TAR_CONCURRENT_CHANGE_WARNINGS)
        for line in lines)


def _bulk_upload_timeout(files: list[tuple[str, str]]) -> int:
    total = 0
    for host_path, _ in files:
        try:
            total += os.path.getsize(host_path)
        except OSError:
            pass
    return min(int(total / _BULK_UPLOAD_BYTES_PER_SEC) + _BULK_UPLOAD_MIN_TIMEOUT,
               _BULK_UPLOAD_MAX_TIMEOUT)

# Windows OpenSSH has no Unix-socket ControlMaster: ControlPath/ControlMaster options
# fail the connection outright ('getsockname failed: Not a socket'). Skip multiplexing there.
# Skip multiplexing there; each command pays a fresh connection but the backend works. See #73927.
_SSH_MULTIPLEX = os.name != "nt"

# Module-level binding: tests patch ``ssh._load_hermes_env_vars`` to fake the .env file.
_load_hermes_env_vars = load_hermes_env_vars


def _ensure_ssh_available() -> None:
    """Fail fast with a clear error when the SSH client is unavailable."""
    for tool in ("ssh", "scp"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool.upper()} is not installed or not in PATH. "
                               "Install OpenSSH client: apt install openssh-client")


def _sync_error(reason: str, subject: str, what: str = "the SSH connection") -> EnvironmentConnectionError:
    return EnvironmentConnectionError(
        reason, retry_hint=f"{subject} failed — verify {what} is healthy, then retry.")


class SSHEnvironment(BaseEnvironment):
    """Run commands on a remote machine over SSH.

    Spawn-per-call: every execute() spawns a fresh ``ssh ... bash -c`` process.
    Session snapshot preserves env vars across calls; CWD persists via in-band
    stdout markers. Uses SSH ControlMaster for connection reuse.
    """

    # Passthrough values are re-forwarded on every command (see _run_bash), so like docker/local
    # they stay out of the remote snapshot under multiplex.
    _profile_scoped_passthrough = True

    def __init__(self, host: str, user: str, cwd: str = "~",
                 timeout: int = 60, port: int = 22, key_path: str = "",
                 probe_only: bool = False):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host, self.user, self.port, self.key_path = host, user, port, key_path
        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        # Short, deterministic socket name: the path must stay under macOS's 104-byte sun_path
        # limit (raw user@host:port + SSH's 16-byte suffix under a deep $TMPDIR exceeds it), and
        # stability across reconnects keeps ControlMaster reuse working. A probe gets its own
        # per-instance socket so its cleanup() can never close the agent's shared master.
        socket_key = f"{user}@{host}:{port}"
        if probe_only:
            socket_key = f"{socket_key}:probe:{self._session_id}"
        _socket_id = hashlib.sha256(socket_key.encode()).hexdigest()[:16]
        self.control_socket = self.control_dir / f"{_socket_id}.sock"
        _ensure_ssh_available()
        self._establish_connection()
        if probe_only:
            self._sync_manager = None
            return
        self._remote_home = self._detect_remote_home()
        # Profile-scoped remote root + persisted manifest: the PR's whole
        # point. Base still syncs every profile into one shared
        # "<home>/.hermes", which is what lets two profiles overwrite each
        # other's trees and forces a full re-upload after eviction.
        manifest_scope = hashlib.sha256(
            hermes_home_key().encode()
        ).hexdigest()
        self._remote_hermes_home = f"{self._remote_home}/.hermes/profiles/{manifest_scope}"
        self._sync_manifest_path = f"{self._remote_hermes_home}/.sync-manifest.json"
        self._sync_manifest_key = (
            f"{hermes_home_key()}|{self.user}@{self.host}:{self.port}|"
            f"{self._remote_hermes_home}"
        )

        self._ensure_remote_dirs()
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(self._remote_hermes_home),
            upload_fn=self._scp_upload,
            delete_fn=self._ssh_delete,
            bulk_upload_fn=self._ssh_bulk_upload,
            bulk_download_fn=self._ssh_bulk_download,
            manifest_load_fn=self._load_sync_manifest,
            manifest_save_fn=self._save_sync_manifest,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    def _control_socket_for(self, send_env: tuple[str, ...]) -> Path:
        """One ControlMaster per SendEnv name-set, beside the plain target socket: a mux master only
        relays the env names it was itself started with and silently drops the rest, so a passthrough
        command must ride a master that knows its names. scp/sync/probes keep the plain socket."""
        plain = Path(self.control_socket)
        if not send_env:
            return plain
        # <target-id[:8]><names-hash[:8]>.sock: same length as the plain socket (macOS's 104-byte
        # sun_path cap) and prefix-globbable so cleanup() finds every sibling without extra state.
        digest = hashlib.sha256(" ".join(send_env).encode()).hexdigest()[:8]
        return plain.with_name(f"{plain.stem[:8]}{digest}.sock")

    def _control_sockets(self) -> list[Path]:
        """The plain socket plus every SendEnv-set sibling (shared 8-char target prefix)."""
        plain = Path(self.control_socket)
        siblings = sorted(plain.parent.glob(f"{plain.stem[:8]}*.sock")) if plain.parent.is_dir() else []
        return [plain, *(s for s in siblings if s != plain)]

    def _target_flags(self, port_flag: str) -> list:
        """Port/key flags shared by ssh (``-p``) and scp (``-P``)."""
        flags = [port_flag, str(self.port)] if self.port != 22 else []
        return flags + (["-i", self.key_path] if self.key_path else [])

    def _build_ssh_command(self, extra_args: list | None = None, send_env: Iterable[str] = ()) -> list:
        send_env = tuple(sorted(send_env))
        cmd = ["ssh"]
        if _SSH_MULTIPLEX:
            cmd.extend(["-o", f"ControlPath={self._control_socket_for(send_env)}",
                        "-o", "ControlMaster=auto", "-o", "ControlPersist=300"])
        cmd.extend(["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"])
        # Names only; values ride the ssh client's own environment (never the remote command text).
        cmd.extend(arg for name in send_env for arg in ("-o", f"SendEnv={name}"))
        cmd.extend(self._target_flags("-p"))
        cmd.extend(extra_args or [])
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def _run_ssh(self, remote_cmd: str, timeout: float) -> subprocess.CompletedProcess:
        """Run one remote shell command over the multiplexed connection, capturing output."""
        return run_capture(self._build_ssh_command() + [remote_cmd], timeout=timeout)

    def _run_ssh_checked(self, remote_cmd: str, timeout: float, reason: str, subject: str) -> None:
        result = self._run_ssh(remote_cmd, timeout=timeout)
        if result.returncode != 0:
            raise _sync_error(f"{reason}: {result.stderr.strip()}", subject)

    def _establish_connection(self):
        try:
            result = self._run_ssh("echo 'SSH connection established'", timeout=15)
        except subprocess.TimeoutExpired:
            raise EnvironmentConnectionError(
                f"SSH connection to {self.user}@{self.host} timed out",
                retry_hint=(f"Check network connectivity to {self.host}:{self.port} "
                            "and that sshd is accepting connections, then retry."))
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise EnvironmentConnectionError(
                f"SSH connection failed: {error_msg}",
                retry_hint=(f"Verify {self.user}@{self.host}:{self.port} is reachable "
                            "(host up, sshd running, key/agent auth working), then "
                            "retry — the connection is re-established automatically."))

    def _detect_remote_home(self) -> str:
        """Detect the remote user's home directory."""
        with contextlib.suppress(Exception):
            result = self._run_ssh("echo $HOME", timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                logger.debug("SSH: remote home = %s", result.stdout.strip())
                return result.stdout.strip()
        return "/root" if self.user == "root" else f"/home/{self.user}"

    def _load_sync_manifest(self) -> tuple[dict[str, tuple[float, int]], dict[str, str]] | None:
        path = shlex.quote(self._sync_manifest_path)
        cmd = self._build_ssh_command()
        cmd.append(f"if test -f {path}; then cat {path}; else exit 3; fi")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 3:
            return None
        if result.returncode != 0:
            raise RuntimeError(
                f"remote sync manifest read failed: {result.stderr.strip()}"
            )
        try:
            payload = json.loads(result.stdout)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != 1 or payload.get("key") != self._sync_manifest_key:
            return None
        files = payload.get("files")
        if not isinstance(files, dict):
            return None

        parsed: dict[str, tuple[float, int]] = {}
        for remote_path, file_key in files.items():
            if not isinstance(remote_path, str) or not isinstance(file_key, list):
                return None
            if len(file_key) != 2:
                return None
            try:
                mtime, size = float(file_key[0]), int(file_key[1])
            except (TypeError, ValueError, OverflowError):
                return None
            if size < 0:
                return None
            parsed[remote_path] = (mtime, size)
        raw_hashes = payload.get("hashes", {})
        if not isinstance(raw_hashes, dict):
            raw_hashes = {}
        hashes = {
            remote_path: digest
            for remote_path, digest in raw_hashes.items()
            if isinstance(remote_path, str)
            and isinstance(digest, str)
            and len(digest) == 64
        }
        return parsed, hashes

    def _save_sync_manifest(
        self,
        files: dict[str, tuple[float, int]],
        hashes: dict[str, str],
    ) -> None:
        payload = json.dumps(
            {
                "version": 1,
                "key": self._sync_manifest_key,
                "files": {
                    remote_path: [mtime, size]
                    for remote_path, (mtime, size) in files.items()
                },
                "hashes": hashes,
            },
            sort_keys=True,
        )
        manifest_path = shlex.quote(self._sync_manifest_path)
        manifest_dir = shlex.quote(self._sync_manifest_path.rsplit("/", 1)[0])
        cmd = self._build_ssh_command()
        cmd.append(
            f"tmp=$(mktemp {manifest_dir}/.sync-manifest.XXXXXX) "
            f"&& cat > \"$tmp\" && chmod 600 \"$tmp\" && mv \"$tmp\" {manifest_path}"
        )
        result = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"remote sync manifest write failed: {result.stderr.strip()}"
            )

    def _ensure_remote_dirs(self) -> None:
        """Create base ~/.hermes directory tree on remote in one SSH call."""
        # Profile-scoped base (this PR) via base's _run_ssh helper.
        base = self._remote_hermes_home
        self._run_ssh(quoted_mkdir_command([base, f"{base}/skills", f"{base}/credentials", f"{base}/cache"]),
                      timeout=10)

    # _get_sync_files provided via iter_sync_files in FileSyncManager init

    def _scp_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via scp over ControlMaster."""
        self._run_ssh(f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}", timeout=10)
        scp_cmd = ["scp"] + (["-o", f"ControlPath={self.control_socket}"] if _SSH_MULTIPLEX else [])
        scp_cmd += self._target_flags("-P") + [host_path, f"{self.user}@{self.host}:{remote_path}"]
        result = run_capture(scp_cmd, timeout=30)
        if result.returncode != 0:
            raise _sync_error(f"scp failed: {result.stderr.strip()}", f"File sync to {self.user}@{self.host}")

    def _ssh_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in one tar-over-SSH stream: local ``tar c`` piped through one SSH
        connection to remote ``tar x``, after a single batched ``mkdir -p``."""
        if not files:
            return

        base = self._remote_hermes_home
        parents = unique_parent_dirs(files)
        if parents:
            self._run_ssh_checked(quoted_mkdir_command(parents), 30, "remote mkdir failed",
                                  f"Remote directory setup on {self.host}")

        # Symlink staging avoids fragile GNU tar --transform rules. On Windows
        # without Developer Mode symlink creation raises OSError winerror 1314;
        # only that case falls back to a plain copy, other OSErrors re-raise.
        with tempfile.TemporaryDirectory(prefix="hermes-ssh-bulk-") as staging:
            for host_path, remote_path in files:
                try:
                    rel_remote = os.path.relpath(remote_path, base)
                except ValueError as exc:
                    raise RuntimeError(f"remote path {remote_path!r} is not under sync base {base!r}") from exc
                if rel_remote == "." or rel_remote.startswith("../"):
                    raise RuntimeError(f"remote path {remote_path!r} escapes sync base {base!r}")
                staged = os.path.join(staging, rel_remote)
                os.makedirs(os.path.dirname(staged), exist_ok=True)
                try:
                    os.symlink(os.path.abspath(host_path), staged)
                except OSError as e:
                    if getattr(e, "winerror", None) != 1314:
                        raise
                    shutil.copy2(host_path, staged)

            # --no-overwrite-dir keeps tar from stamping the staging dir's mode onto
            # existing dirs (e.g. /home/<user>); a umask-002 0775 home breaks sshd StrictModes.
            ssh_cmd = self._build_ssh_command() + [
                f"tar xf - --no-overwrite-dir -C {shlex.quote(base)}"]
            tar_env = {**os.environ, "COPYFILE_DISABLE": "1"}
            tar_proc = subprocess.Popen(
                ["tar", "-chf", "-", "-C", staging, "."],
                env=tar_env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                ssh_proc = subprocess.Popen(ssh_cmd, stdin=tar_proc.stdout,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                tar_proc.kill()
                tar_proc.wait()
                raise
            tar_proc.stdout.close()  # let tar_proc receive SIGPIPE if ssh_proc exits early
            try:
                _, ssh_stderr = ssh_proc.communicate(timeout=_bulk_upload_timeout(files))
                # communicate() (not wait()) drains stderr so tar can't deadlock on >PIPE_BUF errors.
                if tar_proc.poll() is None:
                    _, tar_stderr_raw = tar_proc.communicate(timeout=10)
                else:
                    tar_stderr_raw = tar_proc.stderr.read() if tar_proc.stderr else b""
            except subprocess.TimeoutExpired:
                for proc in (tar_proc, ssh_proc):
                    proc.kill()
                for proc in (tar_proc, ssh_proc):
                    proc.wait()  # kill both first, then reap: never wait on one while the other blocks
                raise EnvironmentConnectionError(
                    "SSH bulk upload timed out",
                    retry_hint=f"Bulk file sync to {self.host} timed out — check the connection and retry.")
            if tar_proc.returncode != 0:
                raise RuntimeError(f"tar create failed (rc={tar_proc.returncode}): "
                                   f"{tar_stderr_raw.decode(errors='replace').strip()}")
            if ssh_proc.returncode != 0:
                raise _sync_error(f"tar extract over SSH failed (rc={ssh_proc.returncode}): "
                                  f"{ssh_stderr.decode(errors='replace').strip()}",
                                  f"File sync over SSH to {self.host}", what="the connection")
        logger.debug("SSH: bulk-uploaded %d file(s) via tar pipe", len(files))

    def _remote_supports_gzip(self) -> bool:
        cmd = self._build_ssh_command() + ["command -v gzip >/dev/null 2>&1"]
        try:
            result = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def _ssh_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        # Tar from / with the full path so archive entries keep absolute paths
        # (home/user/.hermes/skills/f.py), matching _pushed_hashes keys.
        # Profile-scoped base (this PR) so sync-back only pulls THIS profile's
        # tree instead of every profile sharing one remote directory.
        rel_base = self._remote_hermes_home.lstrip("/")
        # The remote tree is live: a running agent writes cache and log files
        # while tar reads them, which tar reports as "file changed as we read
        # it" and exits 1. That is a warning about a file we will pick up on
        # the next sync, not a transfer failure. --warning= is GNU-only and the
        # remote may run BSD/libarchive tar, so tolerate the exit code below
        # instead of passing a flag that would make every download fail there.
        # Never pull back remote-only build artifacts. `venvs/` is created ON
        # the remote by tooling and is never uploaded (the push side sends an
        # explicit file list), so every byte of it is one-way junk that the
        # diff below discards anyway. Measured on the Hetzner QA box: it is
        # 204 MB of the 247 MB tree and takes sync_back from 6.6s to 56s, so
        # three retries overrun the 120s shutdown watchdog and the gateway is
        # SIGKILLed mid-cleanup. It also contains an absolute symlink
        # (venvs/*/bin/python), which makes Python's tarfile extraction raise
        # "is a link to an absolute path" and fail the attempt outright.
        # Excluding it fixes both the timeout and the hard error.
        exclude_args = " ".join(
            f"--exclude={shlex.quote(f'{rel_base}/{directory}')}"
            for directory in _SYNC_BACK_EXCLUDE_DIRS)
        ssh_cmd = self._build_ssh_command() + [
            f"tar cf - -C / {exclude_args} {shlex.quote(rel_base)}"]
        with open(dest, "wb") as f:
            result = subprocess.run(
                ssh_cmd, stdin=subprocess.DEVNULL, stdout=f,
                stderr=subprocess.PIPE, timeout=_BULK_UPLOAD_MAX_TIMEOUT)
        stderr = (result.stderr or b"").decode(errors="replace").strip()
        if result.returncode != 0 and (
                result.returncode != 1 or not _tar_stderr_is_only_concurrent_change(stderr)):
            raise _sync_error(f"SSH bulk download failed: {stderr}",
                              f"File sync from {self.host}")

    def _ssh_delete(self, remote_paths: list[str]) -> None:
        self._run_ssh_checked(quoted_rm_command(remote_paths), 10, "remote rm failed",
                              f"Remote file cleanup on {self.host}")

    def _before_execute(self) -> None:
        if self._sync_manager is not None:
            self._sync_manager.sync()  # rate-limited internally

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Forward the passthrough allowlist (skill ``required_environment_variables`` +
        ``terminal.env_passthrough``) the way docker does: ``SendEnv`` carries the names, the ssh
        client's env carries the values, so secrets never enter the remote ``bash -c`` argv. The
        remote sshd must ``AcceptEnv`` them (#14091). Profile-scoped names missing from the active
        scope are unset remotely so a shared host cannot serve another profile's value.

        HERMES_HOME is exported inline (this PR) rather than sent via SendEnv: it is a path, not a
        secret, so the argv is a safe place for it, and sending it would need another AcceptEnv
        entry on every remote host. Without it the remote agent writes to the unscoped default and
        the profile isolation this PR adds is lost on the remote side."""
        values, unset_names = resolve_passthrough_env(hermes_env_loader=_load_hermes_env_vars)
        cmd_string = f"export HERMES_HOME={shlex.quote(self._remote_hermes_home)}; {cmd_string}"
        cmd = self._build_ssh_command(send_env=values) + bash_argv(shlex.quote(prepend_unset(cmd_string, unset_names)), login)
        client_env = client_env_with(values)
        return _popen_bash(cmd, stdin_data, env=client_env) if client_env is not None else _popen_bash(cmd, stdin_data)

    def cleanup(self):
        if self._sync_manager:
            logger.info("SSH: syncing files from sandbox...")
            self._sync_manager.sync_back()
        for socket in self._control_sockets():
            if not socket.exists():
                continue
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                cmd = ["ssh", "-o", f"ControlPath={socket}", "-O", "exit", f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
            with contextlib.suppress(OSError):
                socket.unlink()

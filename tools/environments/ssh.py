"""SSH remote execution environment with ControlMaster connection persistence."""

import hashlib
import logging
import os

# Windows OpenSSH has no Unix-domain-socket ControlMaster support —
# passing ControlPath/ControlMaster options fails the connection outright
# ('getsockname failed: Not a socket', #73927). Skip multiplexing there;
# each command pays a fresh connection but the backend works.
_SSH_MULTIPLEX = os.name != "nt"
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.environments.base import (
    BaseEnvironment,
    EnvironmentConnectionError,
    _popen_bash,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)


# Diagnostics that mean "the tree moved under me while I read it", not a
# transfer failure. The remote tree is live: agents write cache and log files
# while tar reads them. The archive is complete apart from those entries and
# the next sync picks them up.
_TAR_CONCURRENT_CHANGE_WARNINGS = (
    "file changed as we read it",
    "file removed before we read it",
)

# Both tars print a summary line after the real diagnostics; it carries no
# information of its own and must not decide the outcome.
_TAR_SUMMARY_SUFFIXES = (
    "exiting with failure status due to previous errors",
    "error exit delayed from previous errors.",
)


def _tar_stderr_is_only_concurrent_change(stderr: str) -> bool:
    """True if every tar diagnostic line is a known concurrent-change warning.

    Requiring *every* line to match is the point: a benign warning printed
    alongside a genuine read error must not launder a truncated archive into
    an accepted sync-back. Empty stderr returns False, since exit 1 with no
    explanation is not attributable to a warning.
    """
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    if not lines:
        return False
    for line in lines:
        low = line.lower()
        if low.endswith(_TAR_SUMMARY_SUFFIXES):
            continue
        if not any(w in low for w in _TAR_CONCURRENT_CHANGE_WARNINGS):
            return False
    return True


_BULK_UPLOAD_MIN_TIMEOUT = 120
_BULK_UPLOAD_MAX_TIMEOUT = 1800

# Remote-only directories excluded from sync_back. These are created on the
# remote and never uploaded, so pulling them back is pure cost: the diff
# always discards them. Keep this list to genuinely remote-generated build
# artifacts — anything an agent might legitimately author must sync back.
_SYNC_BACK_EXCLUDE_DIRS = ("venvs",)
# Deliberately pessimistic: a cold sync of a large skills tree is thousands of
# small files, where per-file overhead dominates raw link speed.
_BULK_UPLOAD_BYTES_PER_SEC = 2_000_000


def _bulk_upload_timeout(files: list[tuple[str, str]]) -> int:
    """Return a transfer budget in seconds scaled to the payload size.

    Sizing off the payload keeps a cold first sync (every skill and cache
    file at once) from being killed mid-stream while still failing fast on a
    genuinely wedged connection during small incremental syncs.
    """
    total = 0
    for host_path, _ in files:
        try:
            total += os.path.getsize(host_path)
        except OSError:
            # An unreadable file is skipped by tar too; it costs no transfer
            # time, so leaving it out of the estimate is correct.
            continue
    scaled = int(total / _BULK_UPLOAD_BYTES_PER_SEC) + _BULK_UPLOAD_MIN_TIMEOUT
    return min(scaled, _BULK_UPLOAD_MAX_TIMEOUT)


def _ensure_ssh_available() -> None:
    """Fail fast with a clear error when the SSH client is unavailable."""
    if not shutil.which("ssh"):
        raise RuntimeError(
            "SSH is not installed or not in PATH. Install OpenSSH client: apt install openssh-client"
        )
    if not shutil.which("scp"):
        raise RuntimeError(
            "SCP is not installed or not in PATH. Install OpenSSH client: apt install openssh-client"
        )


class SSHEnvironment(BaseEnvironment):
    """Run commands on a remote machine over SSH.

    Spawn-per-call: every execute() spawns a fresh ``ssh ... bash -c`` process.
    Session snapshot preserves env vars across calls.
    CWD persists via in-band stdout markers.
    Uses SSH ControlMaster for connection reuse.
    """

    def __init__(self, host: str, user: str, cwd: str = "~",
                 timeout: int = 60, port: int = 22, key_path: str = ""):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        # Keep the socket filename short and deterministic so the full path
        # stays under the 104-byte sun_path limit that macOS enforces on
        # Unix domain sockets. A raw ``user@host:port`` — especially with an
        # IPv6 host — plus the 16-byte random suffix SSH appends in
        # ControlMaster mode easily exceeds the limit under macOS's
        # deeply-nested $TMPDIR (e.g. /var/folders/xx/yy/T/). Hashing the
        # triple keeps the path stable across reconnects so ControlMaster
        # reuse still works.
        _socket_id = hashlib.sha256(
            f"{user}@{host}:{port}".encode()
        ).hexdigest()[:16]
        self.control_socket = self.control_dir / f"{_socket_id}.sock"
        _ensure_ssh_available()
        self._establish_connection()
        self._remote_home = self._detect_remote_home()

        self._ensure_remote_dirs()
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._scp_upload,
            delete_fn=self._ssh_delete,
            bulk_upload_fn=self._ssh_bulk_upload,
            bulk_download_fn=self._ssh_bulk_download,
        )
        self._sync_manager.sync(force=True)

        self.init_session()

    def _build_ssh_command(self, extra_args: list | None = None) -> list:
        cmd = ["ssh"]
        if _SSH_MULTIPLEX:
            cmd.extend(["-o", f"ControlPath={self.control_socket}"])
            cmd.extend(["-o", "ControlMaster=auto"])
            cmd.extend(["-o", "ControlPersist=300"])
        cmd.extend(["-o", "BatchMode=yes"])
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        cmd.extend(["-o", "ConnectTimeout=10"])
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def _establish_connection(self):
        cmd = self._build_ssh_command()
        cmd.append("echo 'SSH connection established'")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=15,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                raise EnvironmentConnectionError(
                    f"SSH connection failed: {error_msg}",
                    retry_hint=(
                        f"Verify {self.user}@{self.host}:{self.port} is reachable "
                        "(host up, sshd running, key/agent auth working), then "
                        "retry — the connection is re-established automatically."
                    ),
                )
        except subprocess.TimeoutExpired:
            raise EnvironmentConnectionError(
                f"SSH connection to {self.user}@{self.host} timed out",
                retry_hint=(
                    f"Check network connectivity to {self.host}:{self.port} "
                    "and that sshd is accepting connections, then retry."
                ),
            )

    def _detect_remote_home(self) -> str:
        """Detect the remote user's home directory."""
        try:
            cmd = self._build_ssh_command()
            cmd.append("echo $HOME")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
            home = result.stdout.strip()
            if home and result.returncode == 0:
                logger.debug("SSH: remote home = %s", home)
                return home
        except Exception:
            pass
        if self.user == "root":
            return "/root"
        return f"/home/{self.user}"

    # ------------------------------------------------------------------
    # File sync (via FileSyncManager)
    # ------------------------------------------------------------------

    def _ensure_remote_dirs(self) -> None:
        """Create base ~/.hermes directory tree on remote in one SSH call."""
        base = f"{self._remote_home}/.hermes"
        dirs = [base, f"{base}/skills", f"{base}/credentials", f"{base}/cache"]
        cmd = self._build_ssh_command()
        cmd.append(quoted_mkdir_command(dirs))
        subprocess.run(
            cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=10,
            stdin=subprocess.DEVNULL,
        )

    # _get_sync_files provided via iter_sync_files in FileSyncManager init

    def _scp_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via scp over ControlMaster."""
        parent = str(Path(remote_path).parent)
        mkdir_cmd = self._build_ssh_command()
        mkdir_cmd.append(f"mkdir -p {shlex.quote(parent)}")
        subprocess.run(
            mkdir_cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=10,
            stdin=subprocess.DEVNULL,
        )

        scp_cmd = ["scp"]
        if _SSH_MULTIPLEX:
            scp_cmd.extend(["-o", f"ControlPath={self.control_socket}"])
        if self.port != 22:
            scp_cmd.extend(["-P", str(self.port)])
        if self.key_path:
            scp_cmd.extend(["-i", self.key_path])
        scp_cmd.extend([host_path, f"{self.user}@{self.host}:{remote_path}"])
        result = subprocess.run(
            scp_cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise EnvironmentConnectionError(
                f"scp failed: {result.stderr.strip()}",
                retry_hint=(
                    f"File sync to {self.user}@{self.host} failed — verify the "
                    "SSH connection is healthy, then retry."
                ),
            )

    def _ssh_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in a single tar-over-SSH stream.

        Pipes ``tar c`` on the local side through an SSH connection to
        ``tar x`` on the remote, transferring all files in one TCP stream
        instead of spawning a subprocess per file.  Directory creation is
        batched into a single ``mkdir -p`` call beforehand.

        Typical improvement: ~580 files goes from O(N) scp round-trips
        to a single streaming transfer.
        """
        if not files:
            return

        base = f"{self._remote_home}/.hermes"
        parents = unique_parent_dirs(files)
        if parents:
            cmd = self._build_ssh_command()
            cmd.append(quoted_mkdir_command(parents))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=30,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise EnvironmentConnectionError(
                    f"remote mkdir failed: {result.stderr.strip()}",
                    retry_hint=(
                        f"Remote directory setup on {self.host} failed — verify "
                        "the SSH connection is healthy, then retry."
                    ),
                )

        # Symlink staging avoids fragile GNU tar --transform rules.
        # On Windows without Developer Mode, symlink creation raises
        # OSError with winerror 1314 (privilege not held).  Catch only
        # that specific error and fall back to a plain copy; all other
        # OSErrors (e.g. disk full, bad path) are re-raised as normal.
        with tempfile.TemporaryDirectory(prefix="hermes-ssh-bulk-") as staging:
            for host_path, remote_path in files:
                try:
                    rel_remote = os.path.relpath(remote_path, base)
                except ValueError as exc:
                    raise RuntimeError(
                        f"remote path {remote_path!r} is not under sync base {base!r}"
                    ) from exc

                if rel_remote == "." or rel_remote.startswith("../"):
                    raise RuntimeError(
                        f"remote path {remote_path!r} escapes sync base {base!r}"
                    )

                staged = os.path.join(staging, rel_remote)
                os.makedirs(os.path.dirname(staged), exist_ok=True)
                try:
                    os.symlink(os.path.abspath(host_path), staged)
                except OSError as e:
                    # WinError 1314: symlink privilege not held (Windows without Dev Mode)
                    if getattr(e, "winerror", None) == 1314:
                        shutil.copy2(host_path, staged)
                    else:
                        raise

            compressed = self._remote_supports_gzip()
            tar_cmd = ["tar", "-czhf" if compressed else "-chf", "-", "-C", staging, "."]
            ssh_cmd = self._build_ssh_command()
            # --no-overwrite-dir prevents tar from overwriting the mode of
            # existing directories (e.g. /home/<user>) with the staging
            # directory's mode.  Without this, a umask 002 produces 0775
            # dirs which breaks sshd StrictModes (refuses authorized_keys).
            extract_flags = "xzf" if compressed else "xf"
            ssh_cmd.append(f"tar {extract_flags} - --no-overwrite-dir -C {shlex.quote(base)}")

            tar_proc = subprocess.Popen(
                tar_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                ssh_proc = subprocess.Popen(
                    ssh_cmd, stdin=tar_proc.stdout, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception:
                tar_proc.kill()
                tar_proc.wait()
                raise

            # Allow tar_proc to receive SIGPIPE if ssh_proc exits early
            tar_proc.stdout.close()

            # A fixed ceiling fails on payload, not on health: the first sync
            # of a fresh remote pushes every skill and cache file at once
            # (measured: 237 MB / 12k files), which cannot finish in 120s on a
            # normal uplink, so every cold start died mid-transfer and rolled
            # back. Scale the budget with the bytes actually being sent and
            # keep a floor for small incremental syncs.
            upload_timeout = _bulk_upload_timeout(files)
            try:
                _, ssh_stderr = ssh_proc.communicate(timeout=upload_timeout)
                # Use communicate() instead of wait() to drain stderr and
                # avoid deadlock if tar produces more than PIPE_BUF of errors.
                tar_stderr_raw = b""
                if tar_proc.poll() is None:
                    _, tar_stderr_raw = tar_proc.communicate(timeout=10)
                else:
                    tar_stderr_raw = tar_proc.stderr.read() if tar_proc.stderr else b""
            except subprocess.TimeoutExpired:
                tar_proc.kill()
                ssh_proc.kill()
                tar_proc.wait()
                ssh_proc.wait()
                raise EnvironmentConnectionError(
                    "SSH bulk upload timed out",
                    retry_hint=(
                        f"Bulk file sync to {self.host} timed out — check the "
                        "connection and retry."
                    ),
                )

            if tar_proc.returncode != 0:
                raise RuntimeError(
                    f"tar create failed (rc={tar_proc.returncode}): "
                    f"{tar_stderr_raw.decode(errors='replace').strip()}"
                )
            if ssh_proc.returncode != 0:
                raise EnvironmentConnectionError(
                    f"tar extract over SSH failed (rc={ssh_proc.returncode}): "
                    f"{ssh_stderr.decode(errors='replace').strip()}",
                    retry_hint=(
                        f"File sync over SSH to {self.host} failed — verify the "
                        "connection is healthy, then retry."
                    ),
                )

        logger.debug("SSH: bulk-uploaded %d file(s) via tar pipe", len(files))

    def _remote_supports_gzip(self) -> bool:
        """Return whether the remote tar can invoke gzip for archive extraction."""
        cmd = self._build_ssh_command()
        cmd.append("command -v gzip >/dev/null 2>&1")
        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def _ssh_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        # Tar from / with the full path so archive entries preserve absolute
        # paths (e.g. home/user/.hermes/skills/f.py), matching _pushed_hashes keys.
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        ssh_cmd = self._build_ssh_command()
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
            f"--exclude={shlex.quote(f'{rel_base}/{d}')}"
            for d in _SYNC_BACK_EXCLUDE_DIRS
        )
        ssh_cmd.append(f"tar cf - -C / {exclude_args} {shlex.quote(rel_base)}")
        with open(dest, "wb") as f:
            result = subprocess.run(
                ssh_cmd,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.PIPE,
                timeout=_BULK_UPLOAD_MAX_TIMEOUT,
            )
        # GNU and BSD tar both use exit 1 for a concurrent file-change warning,
        # but BSD tar also uses it for ordinary errors. Accept only when EVERY
        # diagnostic line is a known warning: a real error printed alongside a
        # warning would otherwise pass and suppress the retry on a partial
        # archive.
        stderr = (result.stderr or b"").decode(errors="replace").strip()
        if result.returncode != 0 and (
            result.returncode != 1
            or not _tar_stderr_is_only_concurrent_change(stderr)
        ):
            raise EnvironmentConnectionError(
                f"SSH bulk download failed: {stderr}",
                retry_hint=(
                    f"File sync from {self.host} failed — verify the SSH "
                    "connection is healthy, then retry."
                ),
            )

    def _ssh_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files in one SSH call."""
        cmd = self._build_ssh_command()
        cmd.append(quoted_rm_command(remote_paths))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise EnvironmentConnectionError(
                f"remote rm failed: {result.stderr.strip()}",
                retry_hint=(
                    f"Remote file cleanup on {self.host} failed — verify the "
                    "SSH connection is healthy, then retry."
                ),
            )

    def _before_execute(self) -> None:
        """Sync files to remote via FileSyncManager (rate-limited internally)."""
        self._sync_manager.sync()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn an SSH process that runs bash on the remote host."""
        cmd = self._build_ssh_command()
        if login:
            cmd.extend(["bash", "-l", "-c", shlex.quote(cmd_string)])
        else:
            cmd.extend(["bash", "-c", shlex.quote(cmd_string)])

        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
        if self._sync_manager:
            logger.info("SSH: syncing files from sandbox...")
            self._sync_manager.sync_back()

        if self.control_socket.exists():
            try:
                cmd = ["ssh", "-o", f"ControlPath={self.control_socket}",
                       "-O", "exit", f"{self.user}@{self.host}"]
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=5,
                    stdin=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                self.control_socket.unlink()
            except OSError:
                pass

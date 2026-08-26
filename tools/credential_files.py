"""File passthrough registry for remote terminal backends.

Remote backends (Docker, Modal, SSH) create sandboxes with no host files.
This module ensures that credential files, skill directories, and host-side
cache directories (documents, images, audio, screenshots) are mounted or
synced into those sandboxes so the agent can access them.

**Credentials and skills** — session-scoped registry fed by skill declarations
(``required_credential_files``) and user config (``terminal.credential_files``).

**Cache directories** — gateway-cached uploads, browser screenshots, TTS
audio, and processed images.  Mounted read-only so the remote terminal can
reference files the host side created (e.g. ``unzip`` an uploaded archive).

Remote backends call :func:`get_credential_file_mounts`,
:func:`get_skills_directory_mount` / :func:`iter_skills_files`, and
:func:`get_cache_directory_mounts` / :func:`iter_cache_files` at sandbox
creation time and before each command (for resync on Modal).
"""

from __future__ import annotations

import logging
import os
import posixpath
from contextvars import ContextVar
from pathlib import Path
from typing import Dict, List, Optional
from hermes_cli.config import cfg_get

try:  # pragma: no cover - exercised via the fail-closed test below
    from agent.file_safety import get_read_block_error
except ImportError:  # noqa: F401 - sentinel consumed in register_credential_file
    get_read_block_error = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Session-scoped list of credential files to mount.
# Backed by ContextVar to prevent cross-session data bleed in the gateway pipeline.
_registered_files_var: ContextVar[Dict[str, str]] = ContextVar("_registered_files")


def _get_registered() -> Dict[str, str]:
    """Get or create the registered credential files dict for the current context/session."""
    try:
        return _registered_files_var.get()
    except LookupError:
        val: Dict[str, str] = {}
        _registered_files_var.set(val)
        return val


# Cache for config-based file list (loaded once per process).
_config_files: List[Dict[str, str]] | None = None


def _resolve_hermes_home() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def register_credential_file(
    relative_path: str,
    container_base: str = "/root/.hermes",
) -> bool:
    """Register a credential file for mounting into remote sandboxes.

    *relative_path* is relative to ``HERMES_HOME`` (e.g. ``google_token.json``).
    Returns True if the file exists on the host and was registered.

    Security: rejects absolute paths and path traversal sequences (``..``).
    The resolved host path must remain inside HERMES_HOME so that a malicious
    skill cannot declare ``required_credential_files: ['../../.ssh/id_rsa']``
    and exfiltrate sensitive host files into a container sandbox.

    Containment alone is not sufficient, because HERMES_HOME is exactly where
    the MASTER credential stores live. A skill legitimately needs its own
    service token (``google_token.json``); it never needs ``.env`` (every
    provider key), ``auth.json`` (all provider tokens and OAuth grants),
    ``mcp-tokens/`` or the Bitwarden plaintext cache. Those are refused via
    the canonical read deny-list (``agent.file_safety.get_read_block_error``)
    — the same guard that stops the agent reading them with ``read_file``, so
    the mount surface cannot hand a skill what the read surface denies it.
    """
    hermes_home = _resolve_hermes_home()

    # Reject absolute paths — they bypass the HERMES_HOME sandbox entirely.
    if os.path.isabs(relative_path):
        logger.warning(
            "credential_files: rejected absolute path %r (must be relative to HERMES_HOME)",
            relative_path,
        )
        return False

    host_path = hermes_home / relative_path

    # Resolve symlinks and normalise ``..`` before the containment check so
    # that traversal like ``../. ssh/id_rsa`` cannot escape HERMES_HOME.
    from tools.path_security import validate_within_dir

    containment_error = validate_within_dir(host_path, hermes_home)
    if containment_error:
        logger.warning(
            "credential_files: rejected path traversal %r (%s)",
            relative_path,
            containment_error,
        )
        return False

    resolved = host_path.resolve()
    if not resolved.is_file():
        logger.debug("credential_files: skipping %s (not found)", resolved)
        return False

    # Master credential stores are never mountable, even though they sit
    # inside HERMES_HOME and therefore pass the containment check above.
    # Fails CLOSED: if the canonical guard can't be consulted we refuse the
    # mount rather than risk bind-mounting auth.json into a sandbox. The
    # import lives at module top (no circular-import concern — file_safety is
    # stdlib-only); the sentinel + logger.exception keep guard failures
    # debuggable instead of silently swallowed (#67665).
    if get_read_block_error is None:
        logger.error(
            "credential_files: refusing %r — agent.file_safety could not be "
            "imported, so the master-store deny-list cannot be consulted",
            relative_path,
        )
        return False
    try:
        denied = get_read_block_error(str(resolved))
    except Exception:
        logger.exception(
            "credential_files: refusing %r — read guard raised", relative_path
        )
        return False
    if denied:
        logger.warning(
            "credential_files: refused %r — it is a credential store the agent "
            "is denied from reading; a skill may mount its own service token, "
            "not the master key files",
            relative_path,
        )
        return False

    container_path = f"{container_base.rstrip('/')}/{relative_path}"
    _get_registered()[container_path] = str(resolved)
    logger.debug("credential_files: registered %s -> %s", resolved, container_path)
    return True


def register_credential_files(
    entries: list,
    container_base: str = "/root/.hermes",
) -> List[str]:
    """Register multiple credential files from skill frontmatter entries.

    Each entry is either a string (relative path) or a dict with a ``path``
    key.  Returns the list of relative paths that were NOT found on the host
    (i.e. missing files).
    """
    missing = []
    for entry in entries:
        if isinstance(entry, str):
            rel_path = entry.strip()
        elif isinstance(entry, dict):
            rel_path = (entry.get("path") or entry.get("name") or "").strip()
        else:
            continue
        if not rel_path:
            continue
        if not register_credential_file(rel_path, container_base):
            missing.append(rel_path)
    return missing


def _load_config_files() -> List[Dict[str, str]]:
    """Load ``terminal.credential_files`` from config.yaml (cached)."""
    global _config_files
    if _config_files is not None:
        return _config_files

    result: List[Dict[str, str]] = []
    try:
        from hermes_cli.config import read_raw_config
        hermes_home = _resolve_hermes_home()
        cfg = read_raw_config()
        cred_files = cfg_get(cfg, "terminal", "credential_files")
        if isinstance(cred_files, list):
            from tools.path_security import validate_within_dir

            for item in cred_files:
                if isinstance(item, str) and item.strip():
                    rel = item.strip()
                    if os.path.isabs(rel):
                        logger.warning(
                            "credential_files: rejected absolute config path %r", rel,
                        )
                        continue
                    host_path = hermes_home / rel
                    containment_error = validate_within_dir(host_path, hermes_home)
                    if containment_error:
                        logger.warning(
                            "credential_files: rejected config path traversal %r (%s)",
                            rel, containment_error,
                        )
                        continue
                    resolved_path = host_path.resolve()
                    if resolved_path.is_file():
                        container_path = f"/root/.hermes/{rel}"
                        result.append({
                            "host_path": str(resolved_path),
                            "container_path": container_path,
                        })
    except Exception as e:
        logger.warning("Could not read terminal.credential_files from config: %s", e)

    _config_files = result
    return _config_files


def get_credential_file_mounts() -> List[Dict[str, str]]:
    """Return all credential files that should be mounted into remote sandboxes.

    Each item has ``host_path`` and ``container_path`` keys.
    Combines skill-registered files and user config.
    """
    mounts: Dict[str, str] = {}

    # Skill-registered files
    for container_path, host_path in _get_registered().items():
        # Re-check existence (file may have been deleted since registration)
        if Path(host_path).is_file():
            mounts[container_path] = host_path

    # Config-based files
    for entry in _load_config_files():
        cp = entry["container_path"]
        if cp not in mounts and Path(entry["host_path"]).is_file():
            mounts[cp] = entry["host_path"]

    return [
        {"host_path": hp, "container_path": cp}
        for cp, hp in mounts.items()
    ]


def get_skills_directory_mount(
    container_base: str = "/root/.hermes",
) -> list[Dict[str, str]]:
    """Return mount info for all skill directories (local + external).

    Skills may include ``scripts/``, ``templates/``, and ``references/``
    subdirectories that the agent needs to execute inside remote sandboxes.

    **Security:** Bind mounts follow symlinks, so a malicious symlink inside
    the skills tree could expose arbitrary host files to the container.  When
    symlinks are detected, this function creates a sanitized copy (regular
    files only) in a temp directory and returns that path instead.  When no
    symlinks are present (the common case), the original directory is returned
    directly with zero overhead.

    Returns a list of dicts with ``host_path`` and ``container_path`` keys.
    The local skills dir mounts at ``<container_base>/skills``, external dirs
    at ``<container_base>/external_skills/<index>``.
    """
    mounts = []
    hermes_home = _resolve_hermes_home()
    skills_dir = hermes_home / "skills"
    if skills_dir.is_dir():
        host_path = _safe_skills_path(skills_dir)
        mounts.append({
            "host_path": host_path,
            "container_path": f"{container_base.rstrip('/')}/skills",
        })

    # Mount external skill dirs
    try:
        from agent.skill_utils import get_external_skills_dirs, get_project_skills_dirs
        for idx, ext_dir in enumerate(get_external_skills_dirs()):
            if ext_dir.is_dir():
                host_path = _safe_skills_path(ext_dir)
                mounts.append({
                    "host_path": host_path,
                    "container_path": f"{container_base.rstrip('/')}/external_skills/{idx}",
                })
        # Trusted project-local skill dirs (repo checkouts). Separate
        # namespace so container paths stay stable if external_dirs change.
        for idx, proj_dir in enumerate(get_project_skills_dirs()):
            if proj_dir.is_dir():
                host_path = _safe_skills_path(proj_dir)
                mounts.append({
                    "host_path": host_path,
                    "container_path": f"{container_base.rstrip('/')}/project_skills/{idx}",
                })
    except ImportError:
        pass

    return mounts


_safe_skills_tempdir: Path | None = None


def _safe_skills_path(skills_dir: Path) -> str:
    """Return *skills_dir* if symlink-free, else a sanitized temp copy."""
    global _safe_skills_tempdir

    symlinks = [p for p in skills_dir.rglob("*") if p.is_symlink()]
    if not symlinks:
        return str(skills_dir)

    for link in symlinks:
        logger.warning("credential_files: skipping symlink in skills dir: %s -> %s",
                       link, os.readlink(link))

    import atexit
    import shutil
    import tempfile

    # Reuse the same temp dir across calls to avoid accumulation.
    if _safe_skills_tempdir and _safe_skills_tempdir.is_dir():
        shutil.rmtree(_safe_skills_tempdir, ignore_errors=True)

    safe_dir = Path(tempfile.mkdtemp(prefix="hermes-skills-safe-"))
    _safe_skills_tempdir = safe_dir

    for item in skills_dir.rglob("*"):
        if item.is_symlink():
            continue
        rel = item.relative_to(skills_dir)
        target = safe_dir / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(target))

    def _cleanup():
        if safe_dir.is_dir():
            shutil.rmtree(safe_dir, ignore_errors=True)

    atexit.register(_cleanup)
    logger.info("credential_files: created symlink-safe skills copy at %s", safe_dir)
    return str(safe_dir)


def _walk_skill_tree(root: Path, container_root: str) -> List[Dict[str, str]]:
    """Enumerate every regular file under *root*, following safe symlinks.

    A profile lane does not copy shared skills, ``hermes profile`` symlinks
    them into the root ``~/.hermes/skills`` tree (41 of the dev lane's 96
    skills are links).  ``Path.rglob`` does not descend a symlinked directory,
    so a blanket ``is_symlink()`` filter yielded one dead entry per linked
    skill and uploaded none of its files.  A Modal worker told to "load the
    pr-text skill" then found no such directory and improvised.

    Symlinks are followed only while the target stays inside a SKILLS tree:
    the tree being walked, or the shared root ``~/.hermes/skills``.  The whole
    Hermes home is too wide a boundary -- it also holds every other profile's
    ``SOUL.md``, memories, state DBs and config, none of which a skill link
    should be able to smuggle into a sandbox.  Loops are bounded by the
    ancestor set on the current descent path.
    """
    from hermes_constants import get_default_hermes_root

    entries: List[Dict[str, str]] = []
    # A profile lane's links point OUT of its own skills dir and into the
    # shared root skills tree, so the walked tree alone is too tight.  Both
    # skills trees, and nothing else under the Hermes home.
    boundaries: List[Path] = []
    for candidate in (root, get_default_hermes_root(_resolve_hermes_home()) / "skills"):
        try:
            boundaries.append(candidate.resolve())
        except (OSError, RuntimeError):
            continue
    try:
        walk_root = root.resolve()
    except (OSError, RuntimeError):
        return entries
    if not boundaries:
        return entries

    def walk(directory: Path, rel_prefix: str, ancestors: frozenset) -> None:
        try:
            real = directory.resolve()
        except (OSError, RuntimeError):
            # RuntimeError: a true cyclic chain (a -> b -> a); OSError: dangling.
            return
        # Loop guard scoped to the CURRENT descent path, not the whole walk: a
        # global visited-set would silently drop the second of two skills that
        # link to the same shared target.
        if real in ancestors:
            return
        ancestors = ancestors | {real}
        try:
            children = sorted(directory.iterdir())
        except OSError:
            return
        for item in children:
            rel = f"{rel_prefix}{item.name}"
            try:
                item_real = item.resolve()
            except (OSError, RuntimeError):
                continue
            if item.is_symlink() and not any(
                item_real.is_relative_to(b) for b in boundaries
            ):
                logger.warning(
                    "skills sync: skipping symlink out of the skills trees: %s", item,
                )
                continue
            if item_real.is_dir():
                walk(item, f"{rel}/", ancestors)
            elif item_real.is_file():
                # Containment is not enough: the Hermes home is exactly where
                # the master credential stores live, so a skill holding
                # ``reference.txt -> ~/.hermes/.env`` would pass the boundary
                # check above and upload the secret under an innocent path.
                # Same guard the mount surface uses; fails CLOSED.
                if get_read_block_error is None:
                    logger.error(
                        "skills sync: refusing %s -- agent.file_safety could not "
                        "be imported, so the deny-list cannot be consulted", item,
                    )
                    continue
                try:
                    denied = get_read_block_error(str(item_real))
                except Exception:
                    logger.exception("skills sync: deny-list check failed for %s", item)
                    continue
                if denied:
                    logger.warning(
                        "skills sync: skipping denied file %s (%s)", item, denied,
                    )
                    continue
                entry = {
                    # The RESOLVED path: the Modal uploader tars host_path, and
                    # tar.add() archives a symlink rather than dereferencing it,
                    # so a linked SKILL.md extracted as a dangling link to a
                    # host-only target the container cannot read.
                    "host_path": str(item_real),
                    "container_path": f"{container_root}/{rel}",
                }
                # A followed link leaves the walked tree only when it points at
                # the OTHER profile's skills tree.  Resolving host_path there
                # makes sync_back's mapping land on that profile's file, so a
                # remote worker editing its copy would silently rewrite another
                # profile's skill.  Read-availability was the whole point of
                # following the link: mark it upload-only.
                if not item_real.is_relative_to(walk_root):
                    entry["upload_only"] = "1"
                entries.append(entry)

    walk(root, "", frozenset())
    return entries


def iter_skills_files(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Yield individual (host_path, container_path) entries for skills files.

    Includes both the local skills dir and any external dirs configured via
    skills.external_dirs.  Skips symlinks entirely.  Preferred for backends
    that upload files individually (Daytona, Modal) rather than mounting a
    directory.
    """
    result: List[Dict[str, str]] = []

    hermes_home = _resolve_hermes_home()
    skills_dir = hermes_home / "skills"
    if skills_dir.is_dir():
        container_root = f"{container_base.rstrip('/')}/skills"
        result.extend(_walk_skill_tree(skills_dir, container_root))

    # Include third-party skill dirs
    try:
        from agent.skill_utils import get_external_skills_dirs, get_project_skills_dirs
        for idx, ext_dir in enumerate(get_external_skills_dirs()):
            if not ext_dir.is_dir():
                continue
            container_root = f"{container_base.rstrip('/')}/external_skills/{idx}"
            result.extend(_walk_skill_tree(ext_dir, container_root))
        for idx, proj_dir in enumerate(get_project_skills_dirs()):
            if not proj_dir.is_dir():
                continue
            container_root = f"{container_base.rstrip('/')}/project_skills/{idx}"
            result.extend(_walk_skill_tree(proj_dir, container_root))
    except ImportError:
        pass

    return result


# ---------------------------------------------------------------------------
# Cache directory mounts (documents, images, audio, videos, screenshots)
# ---------------------------------------------------------------------------

# The cache subdirectories that should be mirrored into remote backends.
# Each tuple is (new_subpath, old_name) matching hermes_constants.get_hermes_dir().
_CACHE_DIRS: list[tuple[str, str]] = [
    ("cache/documents", "document_cache"),
    ("cache/images", "image_cache"),
    ("cache/audio", "audio_cache"),
    ("cache/videos", "video_cache"),
    ("cache/screenshots", "browser_screenshots"),
    ("cache/web", "web_cache"),
    ("cache/delegation", "delegation_cache"),
    # Oversized tool results (tools/tool_result_storage.py). Host-side is the
    # single canonical location; mounting/syncing it lets remote backends
    # read spilled results at the translated path instead of needing a
    # separate in-sandbox copy.
    ("cache/spillover", "cache/spillover"),
    # Desktop/clipboard/PDF uploads land in the flat top-level ``images/`` dir
    # (tui_gateway attach RPCs), not under ``cache/``. Mount it so vision can
    # reach uploads inside sandbox containers (#69575). No legacy alias exists,
    # so both tuple slots are ``images``.
    ("images", "images"),
    # Desktop non-image file attachments (tui_gateway ``file.attach`` staging)
    # land in the flat top-level ``attachments/`` dir. Mount it so the agent's
    # file tools can read dropped binaries (zip/pdf/...) from inside sandbox
    # containers instead of dangling host paths (#76577).
    ("attachments", "attachments"),
]


def get_cache_directory_mounts(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Return mount entries for each cache directory that exists on disk.

    Used by Docker to create bind mounts.  Each entry has ``host_path`` and
    ``container_path`` keys.  The host path is resolved via
    ``get_hermes_dir()`` for backward compatibility with old directory layouts.
    """
    from hermes_constants import get_hermes_dir

    mounts: List[Dict[str, str]] = []
    for new_subpath, old_name in _CACHE_DIRS:
        host_dir = get_hermes_dir(new_subpath, old_name)
        if not host_dir.is_dir():
            # Create missing staging dirs instead of skipping them: Docker
            # snapshots this mount list at container CREATION, so a dir that
            # appears later (first desktop attachment, first clipboard image)
            # would dangle for the whole life of a persistent container
            # (#76577). An empty bind-mounted dir costs nothing; a missing
            # mount costs the feature. get_hermes_dir() already resolved
            # new-vs-legacy layout, so creating its answer cannot shadow a
            # populated legacy dir.
            try:
                host_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue  # unwritable home (tests, RO mounts) — skip as before
        # Always map to the *new* container layout regardless of host layout.
        container_path = f"{container_base.rstrip('/')}/{new_subpath}"
        mounts.append({
            "host_path": str(host_dir),
            "container_path": container_path,
        })
    return mounts


def map_cache_path_to_container(
    host_path: str,
    container_base: str = "/root/.hermes",
) -> Optional[str]:
    """Map a host cache path to its mounted path under *container_base*.

    Returns the POSIX container path when *host_path* lives under one of the
    auto-mounted cache directories, otherwise ``None``.  Backend-agnostic: the
    caller decides which ``container_base`` applies (Docker ``/root/.hermes``,
    SSH ``<remote_home>/.hermes``, etc.) and whether translation is wanted.
    Always joins with ``posixpath`` because container/remote paths are POSIX
    regardless of the host OS.
    """
    path = Path(host_path)
    for mount in get_cache_directory_mounts(container_base=container_base):
        host_dir = Path(mount["host_path"])
        try:
            rel = path.relative_to(host_dir)
        except ValueError:
            continue
        return posixpath.join(mount["container_path"], rel.as_posix())
    return None


def from_agent_visible_cache_path(
    container_path: str,
    container_base: str = "/root/.hermes",
) -> str:
    """Translate a sandbox/container cache path back to its host path.

    Inverse of :func:`to_agent_visible_cache_path`. Returns the input unchanged
    when the active backend is not Docker, or when the path is not under any
    auto-mounted cache directory — the caller then treats a still-container
    path as "no host file" and falls back to an in-container read.
    """
    if os.environ.get("TERMINAL_ENV", "local") != "docker":
        return container_path

    path = Path(container_path)
    for mount in get_cache_directory_mounts(container_base=container_base):
        try:
            rel = path.relative_to(mount["container_path"])
        except ValueError:
            continue
        return str(Path(mount["host_path"]) / rel)
    return container_path


def to_agent_visible_cache_path(
    host_path: str,
    container_base: str = "/root/.hermes",
) -> str:
    """Translate a host cache path to its mounted path inside the sandbox.

    Returns the input unchanged if it is not under any auto-mounted cache
    directory, or if the active terminal backend does not require path
    translation (local).

    Per-backend base (mirrors ``_agent_cache_base_for_env`` in
    tools/image_generation_tool.py, the proven heuristics for where each
    backend's Hermes cache lands):

    * docker / modal — bind-mounted (docker) or per-file-synced (modal) at
      ``/root/.hermes`` (the *container_base* default).
    * ssh / daytona / vercel_sandbox — file-synced under the remote user's
      home; ``~/.hermes`` is shell-expanded by the remote shell, so tool
      commands resolve it regardless of the actual remote home. Previously
      these backends synced the bytes but still rendered the dangling host
      path (#76577 gap).
    * singularity — NOT translated: Apptainer auto-binds the host home, so
      the host path is directly readable and translation would dangle
      (cache dirs are not remapped into that sandbox).

    Backend is identified by TERMINAL_ENV (same env var
    tools/terminal_tool.py reads in _get_environment_config).
    """
    backend = (os.environ.get("TERMINAL_ENV") or "local").strip().lower()
    if backend in ("docker", "modal"):
        pass  # /root/.hermes default
    elif backend in ("ssh", "daytona", "vercel_sandbox"):
        container_base = "~/.hermes"
    else:
        return host_path  # local, singularity, unknown: host path is correct

    mapped = map_cache_path_to_container(host_path, container_base=container_base)
    return mapped if mapped is not None else host_path


# Raster diagrams a card cites with a ``Diagram:`` line.  Restricted to the
# ``diagrams/`` subtree and size-capped: the plans root also accumulates
# full-page screenshots and run artifacts, and shipping those is what the
# markdown-only filter was protecting against.
_DIAGRAM_ROOT = "diagrams"
_DIAGRAM_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_DIAGRAM_MAX_BYTES = 2 * 1024 * 1024


def _is_synced_diagram(item: Path, rel: Path) -> bool:
    """Whether a non-text plan file is a card-citable diagram worth syncing."""
    if not rel.parts or rel.parts[0] != _DIAGRAM_ROOT:
        return False
    if item.suffix.lower() not in _DIAGRAM_SUFFIXES:
        return False
    try:
        return item.stat().st_size <= _DIAGRAM_MAX_BYTES
    except OSError:
        return False


def iter_plans_files(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Yield individual (host_path, container_path) entries for plan files.

    Kanban card briefs routinely link a plan by host path
    (``~/.hermes/plans/<task>.md``), so a worker on a remote backend could not
    read the plan it was told to follow and blocked instead.  Skips symlinks;
    markdown/text plus size-capped ``diagrams/`` images, and skips archived
    subtrees, so a stray artifact in plans/ can't bloat every sandbox.

    Cards cite a rendered diagram alongside the plan (``Diagram:
    ~/.hermes/plans/diagrams/<name>/out.png``).  Text-only sync left every one
    of those paths dead on a remote lane.
    """
    result: List[Dict[str, str]] = []

    hermes_home = _resolve_hermes_home()
    plans_dir = hermes_home / "plans"
    # Do not follow a symlinked plans root: the per-file symlink filter below
    # cannot protect the tree once rglob() has followed the root out of
    # HERMES_HOME.
    if plans_dir.is_symlink() or not plans_dir.is_dir():
        return result

    container_root = f"{container_base.rstrip('/')}/plans"
    allowed_suffixes = {".md", ".txt", ".json", ".yaml", ".yml", ".svg"}
    # Subtrees a worker never needs: archived/superseded plans and rescued run
    # artifacts.  Without this the sync is ~4.5k files / 44MB on every start.
    skip_roots = {"archive", "wt-reaper-rescued", "shelved", "babysit-split-backup"}
    for item in plans_dir.rglob("*"):
        if item.is_symlink() or not item.is_file():
            continue
        rel = item.relative_to(plans_dir)
        if rel.parts and rel.parts[0] in skip_roots:
            continue
        if item.suffix.lower() not in allowed_suffixes and not _is_synced_diagram(
            item, rel
        ):
            continue
        result.append({
            "host_path": str(item),
            "container_path": f"{container_root}/{rel.as_posix()}",
        })

    return result


def iter_cache_files(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Return individual (host_path, container_path) entries for cache files.

    Used by Modal to upload files individually and resync before each command.
    Skips symlinks.  The container paths use the new ``cache/<subdir>`` layout.
    """
    from hermes_constants import get_hermes_dir

    result: List[Dict[str, str]] = []
    for new_subpath, old_name in _CACHE_DIRS:
        host_dir = get_hermes_dir(new_subpath, old_name)
        if not host_dir.is_dir():
            continue
        container_root = f"{container_base.rstrip('/')}/{new_subpath}"
        for item in host_dir.rglob("*"):
            if item.is_symlink() or not item.is_file():
                continue
            rel = item.relative_to(host_dir)
            result.append({
                "host_path": str(item),
                "container_path": f"{container_root}/{rel}",
            })
    return result


def clear_credential_files() -> None:
    """Reset the skill-scoped registry (e.g. on session reset)."""
    _get_registered().clear()



"""A restored Modal snapshot must not resurrect files deleted from the host.

``FileSyncManager`` is constructed fresh per environment with an empty
``_synced_files`` map, so it computes deletions only for files uploaded within
its own lifetime.  When a persistent sandbox is restored from a snapshot, files
uploaded by the *previous* instance are still on disk and no delete is ever
issued for them.  The environment therefore clears the sync-owned subtrees on
restore, and the initial forced sync re-uploads whatever still exists on the
host.
"""

import shlex

from tools.environments.file_sync import quoted_purge_command, synced_subtree_roots


def test_synced_subtree_roots_cover_every_sync_source():
    """Every non-credential source in iter_sync_files lives under a purge root."""
    from tools.credential_files import _CACHE_DIRS

    roots = synced_subtree_roots("/root/.hermes")

    assert "/root/.hermes/skills" in roots
    assert "/root/.hermes/external_skills" in roots
    for subpath, _ in _CACHE_DIRS:
        assert f"/root/.hermes/{subpath}" in roots


def test_synced_subtree_roots_honour_container_base():
    """Daytona/SSH homes are not /root, so the roots must follow container_base."""
    roots = synced_subtree_roots("/home/daytona/.hermes")

    assert all(r.startswith("/home/daytona/.hermes/") for r in roots)
    assert "/home/daytona/.hermes/skills" in roots


def test_purge_command_is_recursive_and_shell_quoted():
    cmd = quoted_purge_command("/root/.hermes")

    assert cmd.startswith("rm -rf ")
    targets = shlex.split(cmd)[2:]
    assert set(targets) == set(synced_subtree_roots("/root/.hermes"))


def test_purge_never_targets_hermes_home_itself():
    """A bare-root purge would destroy credentials and runtime state."""
    for base in ("/root/.hermes", "/root/.hermes/", "/home/user/.hermes"):
        roots = synced_subtree_roots(base)
        assert base.rstrip("/") not in roots
        assert all(len(r) > len(base.rstrip("/")) for r in roots)


def test_purge_excludes_credential_paths():
    """Credentials interleave with runtime state; their parents are not sync-owned."""
    from tools.credential_files import get_credential_file_mounts

    roots = synced_subtree_roots("/root/.hermes")
    for entry in get_credential_file_mounts():
        remote = entry["container_path"]
        assert not any(remote.startswith(root + "/") for root in roots), remote

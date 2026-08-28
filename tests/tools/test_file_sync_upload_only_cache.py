"""The upload-only path set must not re-walk the skills tree on every sync.

`_sync_transaction` needs the upload-only set on EVERY sync, and computing it
walks the entire skills tree a second time on top of `_get_files_fn()`. Two
full walks per synced command: measured at ~3s each on an 8k-file skill
collection, paid before a single byte moves, on SSH, Modal and Daytona alike.

Behavior contracts asserted here (not a snapshot of any timing value):
  1. Repeated syncs inside the TTL walk the tree exactly once.
  2. The cache expires, so a newly added upload-only path is picked up.
  3. The memo is per-manager, so one profile's paths never leak into another.
"""

import io
import tarfile
from unittest.mock import MagicMock

import tools.environments.file_sync as fs
from tools.environments.file_sync import FileSyncManager


def _install_counting_stubs(monkeypatch, skills):
    """Point the lazily-imported helpers at counting stubs."""
    calls = {"skills": 0}

    def fake_iter_skills_files(*_a, **_kw):
        calls["skills"] += 1
        return list(skills)

    import tools.credential_files as cf
    monkeypatch.setattr(cf, "iter_skills_files", fake_iter_skills_files)
    monkeypatch.setattr(cf, "get_credential_file_mounts", lambda *_a, **_kw: [])
    return calls


def _manager(files=()):
    return FileSyncManager(
        get_files_fn=lambda: list(files),
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
    )


def test_repeated_syncs_walk_skills_tree_once(monkeypatch, tmp_path):
    """Ten syncs in a row must not mean ten skills walks."""
    linked = tmp_path / "other-profile-skill.md"
    linked.write_text("x")
    calls = _install_counting_stubs(
        monkeypatch, [{"host_path": str(linked), "upload_only": True}]
    )

    mgr = _manager()
    for _ in range(10):
        mgr.sync(force=True)

    assert calls["skills"] == 1, (
        f"skills tree walked {calls['skills']}x for 10 syncs; the per-sync "
        "re-walk is the whole cost this memo exists to remove"
    )
    assert str(linked.resolve()) in mgr._upload_only_host_paths


def test_cache_expires_and_sees_new_paths(monkeypatch, tmp_path):
    """A stale memo must not outlive the TTL, or a new skill stays unprotected."""
    a = tmp_path / "a.md"
    a.write_text("x")
    b = tmp_path / "b.md"
    b.write_text("x")

    skills = [{"host_path": str(a), "upload_only": True}]
    calls = _install_counting_stubs(monkeypatch, skills)

    mgr = _manager()
    mgr.sync(force=True)
    assert str(a.resolve()) in mgr._upload_only_host_paths
    assert calls["skills"] == 1

    # A new upload-only skill appears; the memo is still warm.
    skills.append({"host_path": str(b), "upload_only": True})
    mgr.sync(force=True)
    assert str(b.resolve()) not in mgr._upload_only_host_paths

    # Age the memo past its TTL.
    mgr._upload_only_cache_time -= fs._UPLOAD_ONLY_TTL_SECONDS + 1
    mgr.sync(force=True)
    assert str(b.resolve()) in mgr._upload_only_host_paths
    assert calls["skills"] == 2


def test_memo_is_per_manager_not_shared(monkeypatch, tmp_path):
    """Two managers (two profiles) must not share one memo."""
    a = tmp_path / "profile-a.md"
    a.write_text("x")
    calls = _install_counting_stubs(
        monkeypatch, [{"host_path": str(a), "upload_only": True}]
    )

    first = _manager()
    first.sync(force=True)
    assert calls["skills"] == 1

    # A second manager must compute its own set, not inherit a warm cache.
    second = _manager()
    second.sync(force=True)
    assert calls["skills"] == 2, (
        "a module-global memo would leak one profile's credential paths "
        "into another profile's sync"
    )


def test_empty_result_is_cached(monkeypatch):
    """A profile with no credentials and no linked skills must still cache.

    Guarding on the SET being truthy meant a validly-empty result was never
    cached, so every sync re-walked the whole skills tree -- exactly the cost
    this memo removes, still paid by every credential-free profile.
    """
    calls = _install_counting_stubs(monkeypatch, [])  # nothing upload-only

    mgr = _manager()
    for _ in range(5):
        mgr.sync(force=True)

    assert mgr._upload_only_host_paths == set()
    assert calls["skills"] == 1, (
        f"skills tree walked {calls['skills']}x for a validly-empty "
        "upload-only set; an empty answer is still an answer"
    )


def test_new_upload_only_path_is_protected_during_sync_back(monkeypatch, tmp_path):
    """A path added during a warm memo remains protected through teardown."""
    regular = tmp_path / "regular.md"
    linked = tmp_path / "shared-skill.md"
    regular.write_text("regular")
    linked.write_text("host version")
    regular_remote = "/root/.hermes/skills/regular.md"
    linked_remote = "/root/.hermes/skills/shared-skill.md"
    files = [(str(regular), regular_remote)]
    current_upload_only: set[str] = set()
    monkeypatch.setattr(fs, "_credential_host_paths", lambda: set(current_upload_only))

    def download_changed_file(destination):
        with tarfile.open(destination, "w") as tar:
            data = b"remote version"
            info = tarfile.TarInfo("root/.hermes/skills/shared-skill.md")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    mgr = FileSyncManager(
        get_files_fn=lambda: list(files),
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
        bulk_download_fn=download_changed_file,
    )
    mgr.sync(force=True)

    # The cache is warm when a new cross-profile link appears.
    files.append((str(linked), linked_remote))
    current_upload_only.add(str(linked.resolve()))
    mgr.sync(force=True)
    assert str(linked.resolve()) in mgr._upload_only_host_paths

    # The link can disappear before teardown; the protection at upload time
    # must still win over a fresh, now-empty discovery result.
    current_upload_only.clear()
    mgr.sync_back(hermes_home=tmp_path / "home")
    assert linked.read_text() == "host version"


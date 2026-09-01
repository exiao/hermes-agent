"""Sweep for sync temp files orphaned by SIGKILL (untrappable reaper kills)."""

import os
import time

import pytest

from tools.environments import file_sync


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(file_sync.tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path


def _age(path, hours):
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_removes_old_tars_and_staging_dirs(temp_root):
    tar = temp_root / f"{file_sync._SYNC_BACK_TAR_PREFIX}abc.tar"
    tar.write_bytes(b"x")
    staging = temp_root / "hermes-ssh-bulk-xyz"
    staging.mkdir()
    (staging / "nested.txt").write_text("x")
    back = temp_root / "hermes-sync-back-1"
    back.mkdir()
    for p in (tar, staging, back):
        _age(p, 5)

    assert file_sync.cleanup_file_sync_artifacts(max_age_hours=1) == 3
    assert not tar.exists() and not staging.exists() and not back.exists()


def test_keeps_fresh_artifacts(temp_root):
    tar = temp_root / f"{file_sync._SYNC_BACK_TAR_PREFIX}fresh.tar"
    tar.write_bytes(b"x")
    assert file_sync.cleanup_file_sync_artifacts(max_age_hours=1) == 0
    assert tar.exists()


def test_ignores_foreign_temp_files(temp_root):
    """A bare tmp*.tar glob would delete other apps' files. Ours must not."""
    other = temp_root / "tmpb4nana.tar"
    other.write_bytes(b"x")
    _age(other, 99)
    assert file_sync.cleanup_file_sync_artifacts(max_age_hours=1) == 0
    assert other.exists()


def test_sync_back_tar_is_named(temp_root):
    """The tar must carry our prefix, or the sweep can't find it."""
    seen = {}

    class Stop(Exception):
        pass

    def fake_download(path):
        seen["name"] = os.path.basename(str(path))
        raise Stop

    mgr = object.__new__(file_sync.FileSyncManager)
    mgr._bulk_download_fn = fake_download
    mgr._get_files_fn = lambda: []
    with pytest.raises(Stop):
        mgr._sync_back_impl()
    assert seen["name"].startswith(file_sync._SYNC_BACK_TAR_PREFIX)
    assert seen["name"].endswith(".tar")

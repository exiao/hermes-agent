"""A cloud worker must be able to read the plan its card told it to follow.

Kanban card briefs link a plan by host path (``~/.hermes/plans/<task>.md``).
Remote backends see no host filesystem, so workers on those lanes blocked
instead of reading the plan. Plans ride the existing credential/skill sync
path, so change detection, deletion, and retry come from FileSyncManager
rather than a second mechanism.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "plans").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_plan_files_are_enumerated_for_the_sandbox(hermes_home):
    from tools.credential_files import iter_plans_files

    (hermes_home / "plans" / "task.md").write_text("# the plan\n")

    entries = iter_plans_files()

    assert entries == [{
        "host_path": str(hermes_home / "plans" / "task.md"),
        "container_path": "/root/.hermes/plans/task.md",
    }]


def test_container_base_is_honoured(hermes_home):
    """Daytona and SSH homes are not /root."""
    from tools.credential_files import iter_plans_files

    (hermes_home / "plans" / "task.md").write_text("# the plan\n")

    entries = iter_plans_files(container_base="/home/daytona/.hermes")

    assert entries[0]["container_path"] == "/home/daytona/.hermes/plans/task.md"


def test_archived_and_binary_plans_are_skipped(hermes_home):
    """~4.5k files / 44MB of dead weight would otherwise ship every start."""
    from tools.credential_files import iter_plans_files

    plans = hermes_home / "plans"
    (plans / "keep.md").write_text("# keep\n")
    (plans / "archive").mkdir()
    (plans / "archive" / "old.md").write_text("# old\n")
    (plans / "screenshot.png").write_bytes(b"\x89PNG")

    names = {Path(e["host_path"]).name for e in iter_plans_files()}

    assert names == {"keep.md"}


def test_symlinks_are_not_followed(hermes_home, tmp_path):
    """A symlinked plan (or plans root) must not read outside HERMES_HOME."""
    from tools.credential_files import iter_plans_files

    secret = tmp_path / "secret.md"
    secret.write_text("not a plan\n")
    os.symlink(secret, hermes_home / "plans" / "link.md")

    assert iter_plans_files() == []


def test_missing_plans_dir_is_not_an_error(hermes_home):
    from tools.credential_files import iter_plans_files

    (hermes_home / "plans").rmdir()

    assert iter_plans_files() == []


def test_plans_join_the_recurring_sync(hermes_home):
    """Not creation-time only: a plan edited later must still reach the sandbox.

    Being in iter_sync_files() is what gives plans change detection, deletion,
    and rollback from FileSyncManager instead of a parallel mechanism.
    """
    from tools.environments.file_sync import iter_sync_files

    (hermes_home / "plans" / "task.md").write_text("# the plan\n")

    remote_paths = [remote for _host, remote in iter_sync_files()]

    assert "/root/.hermes/plans/task.md" in remote_paths


def test_plans_are_never_synced_back_to_the_host(hermes_home, tmp_path):
    """Push down, never pull back.

    sync_back() resolves remote files to host paths and copies them over. A
    cloud worker editing its own plan would otherwise overwrite the
    authoritative plan and patch-note files on the host.
    """
    import tarfile
    from tools.environments.file_sync import FileSyncManager

    plan = hermes_home / "plans" / "task.md"
    plan.write_text("# the authoritative plan\n")

    # A remote tar in which the worker edited its own plan.
    remote_tar = tmp_path / "remote.tar"
    edited = tmp_path / "task.md"
    edited.write_text("# the worker overwrote this\n")
    with tarfile.open(remote_tar, "w") as tar:
        tar.add(edited, arcname="root/.hermes/plans/task.md")

    def _download(dest: Path) -> None:
        Path(dest).write_bytes(remote_tar.read_bytes())

    manager = FileSyncManager(
        get_files_fn=lambda: [(str(plan), "/root/.hermes/plans/task.md")],
        upload_fn=lambda host, remote: None,
        delete_fn=lambda paths: None,
        bulk_download_fn=_download,
    )
    manager.sync(force=True)

    manager.sync_back(hermes_home=hermes_home)

    assert plan.read_text() == "# the authoritative plan\n", (
        "a sandbox must never overwrite the host's plan"
    )

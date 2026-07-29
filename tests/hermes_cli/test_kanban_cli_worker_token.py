"""Worker-token authorization for ``hermes kanban complete``."""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban import run_slash


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    return home


def test_cli_complete_requires_run_token_only_in_worker_context(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="x", assignee="worker")
        kb.claim_task(conn, task_id)
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    assert "run token" in run_slash(f"complete {task_id} --summary review-only")

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "running"
        rejected = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "completion_rejected"
        ]
        assert rejected[-1].payload["reason"] == "missing_worker_run_id"
    finally:
        conn.close()

    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert "Completed" in run_slash(f"complete {task_id} --summary operator-close")


def test_cli_worker_cannot_complete_sibling_task(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        worker_task = kb.create_task(conn, title="worker task", assignee="worker")
        sibling_task = kb.create_task(conn, title="sibling task", assignee="worker")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", worker_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")
    assert "scoped to task" in run_slash(f"complete {sibling_task} --summary forged")

    conn = kb.connect()
    try:
        sibling = kb.get_task(conn, sibling_task)
        assert sibling is not None and sibling.status == "ready"
    finally:
        conn.close()

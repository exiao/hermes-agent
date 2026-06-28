"""Create-time guard against spec-less placeholder tasks.

A task with an empty body and a bare placeholder title (the literal
``"<assignee> task"``, ``"untitled"``, the assignee name, etc.) carries no work
a worker could act on. It must never reach a worker lane in ``ready`` — it is
routed to ``triage`` so a specifier can flesh out the spec (or archive it).
This is the durable fix for the phantom "dev task" tickets that blocked the dev
lane.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# --- pure predicate -------------------------------------------------------


@pytest.mark.parametrize(
    "title,body,assignee",
    [
        ("dev task", None, "dev"),          # the exact phantom-ticket fingerprint
        ("designer task", "", "designer"),  # other lane, empty (not null) body
        ("dev", None, "dev"),               # title == assignee
        ("untitled", None, "dev"),          # bare placeholder
        ("Untitled", "   ", "dev"),         # case-insensitive + whitespace body
        ("task", None, None),               # bare "task", no assignee
        ("TBD", None, "qa"),
    ],
)
def test_is_spec_less_true(title, body, assignee):
    assert kb._is_spec_less(title, body, assignee) is True


@pytest.mark.parametrize(
    "title,body,assignee",
    [
        ("dev task", "Fix the login bug in auth.py", "dev"),  # placeholder title but real body
        ("Refactor the dispatcher", None, "dev"),             # real title, no body
        ("review", None, "reviewer"),                         # not "<assignee> task" / != assignee
        ("dev tasks", None, "dev"),                           # plural — not the placeholder
        ("parent", None, "worker"),                           # ordinary short title
    ],
)
def test_is_spec_less_false(title, body, assignee):
    assert kb._is_spec_less(title, body, assignee) is False


# --- create-time behaviour ------------------------------------------------


def test_spec_less_task_routed_to_triage(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dev task", assignee="dev")
        task = kb.get_task(conn, tid)
        assert task is not None
        # Never reaches a worker lane in ``ready``.
        assert task.status == "triage", task.status
    finally:
        conn.close()


def test_well_specified_task_still_ready(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Fix CI-blocking ruff error on live-config",
            body="run_agent.py:107 has an invalid noqa; reword the comment.",
            assignee="dev",
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready", task.status
    finally:
        conn.close()


def test_explicit_triage_still_honoured(kanban_home):
    """A normal task created with triage=True stays in triage (no regression)."""
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="Real work", body="do the thing", assignee="dev", triage=True
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "triage", task.status
    finally:
        conn.close()


def test_default_assignee_placeholder_routed_to_triage(kanban_home, monkeypatch):
    """No explicit assignee but kanban.default_assignee resolves the placeholder.

    Regression for the P2: ``create_task(title="default task")`` with no
    assignee must still be caught — otherwise the dispatcher auto-assigns it to
    ``kanban.default_assignee`` and spawns a spec-less worker. We resolve the
    default assignee inside the guard and key the placeholder check on it.
    """
    monkeypatch.setattr(kb, "_default_assignee", lambda: "default")
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="default task")  # no assignee
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "triage", task.status
    finally:
        conn.close()


def test_decompose_spec_less_child_parked_in_triage(kanban_home):
    """A malformed decomposer result with a spec-less child must not reach a
    worker lane. The child path INSERTs directly (bypassing create_task), so
    the guard is applied there too: spec-less children land in ``triage``,
    well-specified ones promote to ``ready`` as usual.
    """
    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="rough idea", triage=True)
        children = [
            {"title": "dev task", "body": "", "assignee": "dev"},  # spec-less
            {"title": "Wire the API client", "body": "use httpx", "assignee": "dev"},
        ]
        ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
        assert ids is not None and len(ids) == 2
        spec_less, real = kb.get_task(conn, ids[0]), kb.get_task(conn, ids[1])
        assert spec_less.status == "triage", spec_less.status
        assert real.status == "ready", real.status
    finally:
        conn.close()

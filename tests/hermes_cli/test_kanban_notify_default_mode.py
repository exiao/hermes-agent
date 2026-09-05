"""Notification subscriptions wake the destination agent by default.

Passive 'notify' sends a one-line message and runs no agent turn, so a card
completed and the chat that filed it stayed dead. The kanban_create tool and
/kanban create already passed 'notify+wake' explicitly; the CLI subscribe verb
and the dashboard home-subscribe route inherited passive delivery by accident.
The default now lives in add_notify_sub, so every caller inherits it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb


@pytest.fixture()
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    yield conn
    conn.close()


def _mode(conn, task_id):
    subs = kb.list_notify_subs(conn, task_id)
    return subs[0]["delivery_mode"] if subs else None


def test_fresh_sub_defaults_to_notify_plus_wake(board):
    tid = kb.create_task(board, title="t", assignee="dev")
    kb.add_notify_sub(board, task_id=tid, platform="signal",
                      chat_id="group:abc", chat_type="group")
    assert _mode(board, tid) == "notify+wake"


def test_tui_keeps_passive_notify(board):
    """The TUI poller posts into the running session; it has no wake path."""
    tid = kb.create_task(board, title="t", assignee="dev")
    kb.add_notify_sub(board, task_id=tid, platform="tui", chat_id="sess-key")
    assert _mode(board, tid) == "notify"


def test_explicit_notify_still_wins(board):
    """An operator can still choose passive delivery."""
    tid = kb.create_task(board, title="t", assignee="dev")
    kb.add_notify_sub(board, task_id=tid, platform="signal",
                      chat_id="group:abc", delivery_mode="notify")
    assert _mode(board, tid) == "notify"


def test_unknown_mode_falls_back_to_the_platform_default(board):
    """A typo must not silently pick passive delivery."""
    tid = kb.create_task(board, title="t", assignee="dev")
    kb.add_notify_sub(board, task_id=tid, platform="signal",
                      chat_id="group:abc", delivery_mode="wakeup")
    assert _mode(board, tid) == "notify+wake"


def test_resubscribe_without_a_mode_leaves_an_existing_row_alone(board):
    """The default applies to fresh rows only, never as a silent upgrade."""
    tid = kb.create_task(board, title="t", assignee="dev")
    kb.add_notify_sub(board, task_id=tid, platform="signal",
                      chat_id="group:abc", delivery_mode="notify")
    kb.add_notify_sub(board, task_id=tid, platform="signal",
                      chat_id="group:abc")
    assert _mode(board, tid) == "notify"


def test_cli_notify_subscribe_wakes_by_default(board):
    """The reported symptom: the CLI verb, driven through real argparse.

    A hand-built Namespace would pass against the broken code, because the
    bug sat between the parser default and what the DB layer inserted.
    """
    tid = kb.create_task(board, title="t", assignee="dev")
    top = argparse.ArgumentParser(prog="hermes")
    parser = kanban_cli.build_parser(top.add_subparsers(dest="command"))
    args = parser.parse_args([
        "notify-subscribe", tid,
        "--platform", "signal", "--chat-id", "group:abc",
        "--chat-type", "group",
    ])
    assert kanban_cli.kanban_command(args) == 0
    assert _mode(board, tid) == "notify+wake"


def test_cli_notify_subscribe_without_chat_type_stays_passive(board):
    """A route-less CLI subscription must not wake a guessed DM session."""
    tid = kb.create_task(board, title="t", assignee="dev")
    top = argparse.ArgumentParser(prog="hermes")
    parser = kanban_cli.build_parser(top.add_subparsers(dest="command"))
    args = parser.parse_args([
        "notify-subscribe", tid,
        "--platform", "signal", "--chat-id", "group:abc",
    ])
    assert kanban_cli.kanban_command(args) == 0
    assert _mode(board, tid) == "notify"


def test_cli_resubscribe_without_chat_type_keeps_existing_mode(board):
    """Route-less re-subscribe must preserve an intentional wake mode."""
    tid = kb.create_task(board, title="t", assignee="dev")
    kb.add_notify_sub(board, task_id=tid, platform="signal",
                      chat_id="group:abc", chat_type="group",
                      delivery_mode="notify+wake")
    top = argparse.ArgumentParser(prog="hermes")
    parser = kanban_cli.build_parser(top.add_subparsers(dest="command"))
    args = parser.parse_args([
        "notify-subscribe", tid,
        "--platform", "signal", "--chat-id", "group:abc",
    ])
    assert kanban_cli.kanban_command(args) == 0
    assert _mode(board, tid) == "notify+wake"


def test_child_inherits_the_parents_mode_not_the_default(board):
    """Inheritance copies the parent row; a passive parent stays passive."""
    parent = kb.create_task(board, title="p", assignee="dev")
    kb.add_notify_sub(board, task_id=parent, platform="signal",
                      chat_id="group:abc", delivery_mode="notify")
    child = kb.create_task(board, title="c", assignee="dev", parents=[parent])
    assert _mode(board, child) == "notify"

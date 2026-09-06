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
        "--chat-type", "group", "--user-id", "user-1",
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


def test_cli_group_subscribe_without_participant_stays_passive(board):
    """A group type without the participant key must not wake a shared guess."""
    tid = kb.create_task(board, title="t", assignee="dev")
    top = argparse.ArgumentParser(prog="hermes")
    parser = kanban_cli.build_parser(top.add_subparsers(dest="command"))
    args = parser.parse_args([
        "notify-subscribe", tid,
        "--platform", "signal", "--chat-id", "group:abc",
        "--chat-type", "group",
    ])
    assert kanban_cli.kanban_command(args) == 0
    assert _mode(board, tid) == "notify"


def test_cli_thread_subscribe_without_participant_stays_passive(board):
    """A per-user thread wake needs the participant used in its session key."""
    tid = kb.create_task(board, title="t", assignee="dev")
    top = argparse.ArgumentParser(prog="hermes")
    parser = kanban_cli.build_parser(top.add_subparsers(dest="command"))
    args = parser.parse_args([
        "notify-subscribe", tid,
        "--platform", "signal", "--chat-id", "group:abc",
        "--chat-type", "thread", "--thread-id", "thread-1",
    ])
    assert kanban_cli.kanban_command(args) == 0
    assert _mode(board, tid) == "notify"


def test_cli_api_server_subscribe_wakes_raw_session(board):
    """The API adapter wakes by self-posting to chat_id as its raw session id."""
    tid = kb.create_task(board, title="t", assignee="dev")
    top = argparse.ArgumentParser(prog="hermes")
    parser = kanban_cli.build_parser(top.add_subparsers(dest="command"))
    args = parser.parse_args([
        "notify-subscribe", tid,
        "--platform", "api_server", "--chat-id", "raw-session-1",
    ])
    assert kanban_cli.kanban_command(args) == 0
    assert _mode(board, tid) == "notify+wake"


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


def _cli_subscribe(tid, argv):
    top = argparse.ArgumentParser(prog="hermes")
    parser = kanban_cli.build_parser(top.add_subparsers(dest="command"))
    return kanban_cli.kanban_command(
        parser.parse_args(["notify-subscribe", tid, *argv])
    )


def test_slack_group_with_participant_still_stays_passive(board):
    """Slack keys sessions on the workspace scope, which the CLI cannot persist.

    A participant id makes this look route-complete, and on every other
    platform it is. Slack's session key also carries scope_id, recoverable
    only from delivery_metadata or the adapter's ephemeral channel map, so
    after a gateway restart the wake would land in an unscoped, context-free
    session.
    """
    tid = kb.create_task(board, title="t", assignee="dev")
    assert _cli_subscribe(tid, [
        "--platform", "slack", "--chat-id", "C123",
        "--chat-type", "group", "--user-id", "U456",
    ]) == 0
    assert _mode(board, tid) == "notify"


def test_slack_thread_with_participant_still_stays_passive(board):
    """Same scope gap on the per-user thread route."""
    tid = kb.create_task(board, title="t", assignee="dev")
    assert _cli_subscribe(tid, [
        "--platform", "slack", "--chat-id", "C123",
        "--chat-type", "thread", "--thread-id", "1700000.1",
        "--user-id", "U456",
    ]) == 0
    assert _mode(board, tid) == "notify"


def test_slack_dm_still_wakes(board):
    """A Slack DM keys on the chat id, so it is route-complete without scope."""
    tid = kb.create_task(board, title="t", assignee="dev")
    assert _cli_subscribe(tid, [
        "--platform", "slack", "--chat-id", "D123", "--chat-type", "dm",
    ]) == 0
    assert _mode(board, tid) == "notify+wake"


def test_non_slack_group_with_participant_still_wakes(board):
    """The Slack carve-out must not gate every other platform's groups."""
    tid = kb.create_task(board, title="t", assignee="dev")
    assert _cli_subscribe(tid, [
        "--platform", "signal", "--chat-id", "group:abc",
        "--chat-type", "group", "--user-id", "u1",
    ]) == 0
    assert _mode(board, tid) == "notify+wake"


def test_explicit_wake_still_overrides_the_slack_guard(board):
    """The guard only sets an implicit default; an operator can still opt in."""
    tid = kb.create_task(board, title="t", assignee="dev")
    assert _cli_subscribe(tid, [
        "--platform", "slack", "--chat-id", "C123",
        "--chat-type", "group", "--user-id", "U456",
        "--delivery-mode", "notify+wake",
    ]) == 0
    assert _mode(board, tid) == "notify+wake"

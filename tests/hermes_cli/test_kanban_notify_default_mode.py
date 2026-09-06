"""Notification subscriptions wake the destination agent by default.

Passive 'notify' sends a one-line message and runs no agent turn, so a card
completed and the chat that filed it stayed dead. The kanban_create tool and
/kanban create already passed 'notify+wake' explicitly; the CLI subscribe verb
and the dashboard home-subscribe route inherited passive delivery by accident.
The default now lives in add_notify_sub, so every caller inherits it.

Two invariants cover this, driven by parameter tables rather than one test per
routing permutation:

1. A fresh subscription wakes IFF the caller supplied enough routing identity
   to rebuild the originating session key. Slack is the exception at every
   chat type, because its key also carries a workspace scope the CLI cannot
   persist.
2. An explicit delivery_mode always wins, and a re-subscribe never changes an
   existing row's mode.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db_notify as kbn
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture()
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    yield conn
    conn.close()


def _mode(conn, task_id):
    subs = kbn.list_notify_subs(conn, task_id)
    return subs[0]["delivery_mode"] if subs else None


def _subscribe(task_id, argv):
    """Drive the real argparse dispatcher, not the handler in isolation.

    The original bug lived between the parser default and what the DB layer
    inserted, so a hand-built Namespace would pass against the broken code.
    """
    top = argparse.ArgumentParser(prog="hermes")
    parser = kanban_cli.build_parser(top.add_subparsers(dest="command"))
    return kanban_cli.kanban_command(
        parser.parse_args(["notify-subscribe", task_id, *argv])
    )


# (name, extra CLI args, expected mode). One row per ROUTING RULE, not per
# permutation: a wake needs a rebuildable session key, and Slack can never
# rebuild one from the CLI because its key carries a workspace scope.
ROUTING = [
    # Enough identity to rebuild the key -> wake.
    ("signal dm", ["--platform", "signal", "--chat-id", "+1555",
                   "--chat-type", "dm"], "notify+wake"),
    ("signal group with participant", ["--platform", "signal", "--chat-id", "group:abc",
                                       "--chat-type", "group", "--user-id", "u1"], "notify+wake"),
    # api_server uses chat_id as the raw session id and self-posts.
    ("api_server raw session", ["--platform", "api_server",
                                "--chat-id", "raw-session-1"], "notify+wake"),
    # Missing identity -> passive, rather than waking a guessed DM session.
    ("no chat_type", ["--platform", "signal", "--chat-id", "group:abc"], "notify"),
    ("group without participant", ["--platform", "signal", "--chat-id", "group:abc",
                                   "--chat-type", "group"], "notify"),
    ("thread without participant", ["--platform", "signal", "--chat-id", "group:abc",
                                    "--chat-type", "thread",
                                    "--thread-id", "t1"], "notify"),
    # Slack keys on a workspace scope the CLI cannot persist: passive at EVERY
    # chat type, even when the participant makes it look route-complete.
    ("slack group", ["--platform", "slack", "--chat-id", "C1",
                     "--chat-type", "group", "--user-id", "U1"], "notify"),
    ("slack thread", ["--platform", "slack", "--chat-id", "C1", "--chat-type", "thread",
                      "--thread-id", "1.1", "--user-id", "U1"], "notify"),
    ("slack dm", ["--platform", "slack", "--chat-id", "D1",
                  "--chat-type", "dm"], "notify"),
    ("slack mixed case", ["--platform", "  SlAcK  ", "--chat-id", "C1",
                          "--chat-type", "group", "--user-id", "U1"], "notify"),
]


@pytest.mark.parametrize("name,argv,expected", ROUTING, ids=[r[0] for r in ROUTING])
def test_fresh_subscription_wakes_only_with_rebuildable_routing(board, name, argv, expected):
    tid = kb.create_task(board, title=name, assignee="dev")
    assert _subscribe(tid, argv) == 0
    assert _mode(board, tid) == expected


# (name, how the row is created, expected mode). The default must never
# override an explicit choice, in either direction.
EXPLICIT = [
    ("explicit notify beats the wake default",
     dict(platform="signal", chat_id="group:abc", delivery_mode="notify"), "notify"),
    ("tui stays passive: its poller has no wake path",
     dict(platform="tui", chat_id="sess-key"), "notify"),
    ("an unknown mode falls back to the platform default, never to silence",
     dict(platform="signal", chat_id="group:abc", delivery_mode="wakeup"), "notify+wake"),
]


@pytest.mark.parametrize("name,kwargs,expected", EXPLICIT, ids=[r[0][:32] for r in EXPLICIT])
def test_explicit_mode_wins_over_the_default(board, name, kwargs, expected):
    tid = kb.create_task(board, title=name, assignee="dev")
    kbn.add_notify_sub(board, task_id=tid, **kwargs)
    assert _mode(board, tid) == expected


@pytest.mark.parametrize("existing", ["notify", "notify+wake"])
def test_resubscribe_never_changes_an_existing_rows_mode(board, existing):
    """The default applies to fresh rows only. A route-less re-subscribe must
    neither upgrade a passive row nor downgrade a deliberate wake."""
    tid = kb.create_task(board, title="t", assignee="dev")
    kbn.add_notify_sub(board, task_id=tid, platform="signal", chat_id="group:abc",
                       chat_type="group", delivery_mode=existing)
    assert _subscribe(tid, ["--platform", "signal", "--chat-id", "group:abc"]) == 0
    assert _mode(board, tid) == existing


def test_child_inherits_the_parents_mode_not_the_default(board):
    """Inheritance copies the parent row; a passive parent stays passive."""
    parent = kb.create_task(board, title="p", assignee="dev")
    kbn.add_notify_sub(board, task_id=parent, platform="signal",
                       chat_id="group:abc", delivery_mode="notify")
    child = kb.create_task(board, title="c", assignee="dev", parents=[parent])
    assert _mode(board, child) == "notify"

@pytest.mark.parametrize('stored', ['Slack', '  SLACK  '])
def test_legacy_unnormalized_row_is_matched_not_duplicated(board, stored):
    """Rows created before platform normalization stored --platform verbatim.
    A re-subscribe must find them case-insensitively; matching only the
    normalized key would insert a SECOND logical row and the notifier, which
    lowercases both, would then deliver twice."""
    tid = kb.create_task(board, title='legacy', assignee='dev')
    kbn.add_notify_sub(board, task_id=tid, platform=stored, chat_id='C1',
                       chat_type='group', user_id='U1', delivery_mode='notify+wake')

    assert _subscribe(tid, ['--platform', 'slack', '--chat-id', 'C1',
                            '--chat-type', 'group', '--user-id', 'U1']) == 0

    subs = kbn.list_notify_subs(board, tid)
    assert len(subs) == 1, f're-subscribe duplicated the legacy row: {subs}'
    assert subs[0]['delivery_mode'] == 'notify+wake'

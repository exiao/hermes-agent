from plugins.kanban.dashboard import plugin_api
from hermes_cli import kanban_db_notify as kbn
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _home(monkeypatch, tmp_path):
    """Point the dashboard endpoint at a real board file."""
    monkeypatch.setattr(
        plugin_api,
        "_configured_home_channels",
        lambda: [{"platform": "signal", "chat_id": "group:abc", "thread_id": ""}],
    )
    monkeypatch.setattr(plugin_api, "_resolve_board", lambda board: board)
    monkeypatch.setattr(plugin_api, "_conn",
                        lambda board=None: kbc.connect(db_path=tmp_path / "board.db"))
    monkeypatch.setattr(plugin_api, "_active_profile_name", lambda: "default")


def _mode(tmp_path, task_id):
    conn = kbc.connect(db_path=tmp_path / "board.db")
    try:
        return kbn.list_notify_subs(conn, task_id)[0]["delivery_mode"]
    finally:
        conn.close()


def test_home_subscribe_is_passive_but_never_downgrades(monkeypatch, tmp_path):
    """A home subscription carries no chat_type/user_id, so it cannot rebuild the
    originating session: a fresh one must stay passive. A re-subscribe must NOT
    downgrade a row someone deliberately set to notify+wake.

    Driven against a real board file rather than a patched helper. The previous
    version patched ``plugin_api.kanban_db.add_notify_sub`` while the endpoint
    calls ``plugin_api.kbn.add_notify_sub``, so the assertion never observed the
    endpoint at all.
    """
    conn = kbc.connect(db_path=tmp_path / "board.db")
    fresh = kb.create_task(conn, title="fresh", assignee="dev")
    existing = kb.create_task(conn, title="existing", assignee="dev")
    kbn.add_notify_sub(
        conn, task_id=existing, platform="signal", chat_id="group:abc",
        chat_type="group", user_id="user-1", delivery_mode="notify+wake",
    )
    conn.close()

    _home(monkeypatch, tmp_path)

    assert plugin_api.subscribe_home(fresh, "signal")["ok"] is True
    assert _mode(tmp_path, fresh) == "notify"

    assert plugin_api.subscribe_home(existing, "signal")["ok"] is True
    assert _mode(tmp_path, existing) == "notify+wake"

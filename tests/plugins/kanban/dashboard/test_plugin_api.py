from unittest.mock import MagicMock

from plugins.kanban.dashboard import plugin_api
from hermes_cli import kanban_db as kb


def test_home_subscription_stays_passive_without_session_routing(monkeypatch):
    captured = {}
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    monkeypatch.setattr(
        plugin_api,
        "_configured_home_channels",
        lambda: [{"platform": "signal", "chat_id": "group:abc", "thread_id": ""}],
    )
    monkeypatch.setattr(plugin_api, "_resolve_board", lambda board: board)
    monkeypatch.setattr(plugin_api, "_conn", lambda board: conn)
    monkeypatch.setattr(plugin_api.kanban_db, "get_task", lambda conn, task_id: object())
    monkeypatch.setattr(
        plugin_api.kanban_db,
        "add_notify_sub",
        lambda conn, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(plugin_api, "_active_profile_name", lambda: "default")

    assert plugin_api.subscribe_home("task-1", "signal") == {
        "ok": True,
        "task_id": "task-1",
        "home_channel": {"platform": "signal", "chat_id": "group:abc", "thread_id": ""},
    }
    assert captured["delivery_mode"] == "notify"


def test_home_resubscribe_preserves_existing_delivery_mode(monkeypatch, tmp_path):
    conn = kb.connect(db_path=tmp_path / "board.db")
    task_id = kb.create_task(conn, title="task", assignee="dev")
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="signal",
        chat_id="group:abc",
        chat_type="group",
        user_id="user-1",
        delivery_mode="notify+wake",
    )
    conn.close()

    monkeypatch.setattr(
        plugin_api,
        "_configured_home_channels",
        lambda: [{"platform": "signal", "chat_id": "group:abc", "thread_id": ""}],
    )
    monkeypatch.setattr(plugin_api, "_resolve_board", lambda board: board)
    monkeypatch.setattr(plugin_api, "_conn", lambda board=None: kb.connect(db_path=tmp_path / "board.db"))
    monkeypatch.setattr(plugin_api, "_active_profile_name", lambda: "default")

    assert plugin_api.subscribe_home(task_id, "signal")["ok"] is True

    check = kb.connect(db_path=tmp_path / "board.db")
    try:
        assert kb.list_notify_subs(check, task_id)[0]["delivery_mode"] == "notify+wake"
    finally:
        check.close()

from plugins.kanban.dashboard import plugin_api


def test_home_subscription_stays_passive_without_session_routing(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        plugin_api,
        "_configured_home_channels",
        lambda: [{"platform": "signal", "chat_id": "group:abc", "thread_id": ""}],
    )
    monkeypatch.setattr(plugin_api, "_resolve_board", lambda board: board)
    monkeypatch.setattr(plugin_api, "_conn", lambda board: type("Conn", (), {"close": lambda self: None})())
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

"""Tests for the standalone Kanban card-drop receiver.

Two layers:
  * Unit: create_card / comment_card / auth logic with the CLI shell-out
    monkeypatched (no board, no subprocess).
  * Integration: boot the real ThreadingHTTPServer and drive it with urllib,
    still monkeypatching the CLI so we assert wire behavior (auth 403, dedupe
    pass-through, goal flag) without mutating a real board.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# Load the receiver module by path (it lives outside the package tree).
_MOD_PATH = Path(__file__).with_name("kanban_receiver.py")
_spec = importlib.util.spec_from_file_location("kanban_receiver", _MOD_PATH)
assert _spec and _spec.loader
kr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kr)


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _clear_secret(monkeypatch):
    monkeypatch.delenv("KANBAN_RECEIVER_SECRET", raising=False)
    monkeypatch.delenv("CRON_SECRET", raising=False)


# --------------------------------------------------------------------------
# create_card
# --------------------------------------------------------------------------

def test_create_card_happy_path(monkeypatch):
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return _FakeProc(stdout=json.dumps({"id": "t_abc123", "status": "ready"}))

    monkeypatch.setattr(kr, "_run_hermes_kanban", fake_run)
    status, body = kr.create_card(
        {"assignee": "equity-analyst", "title": "AVGO deep dive", "body": "the ask"}
    )
    assert status == 200
    assert body == {"id": "t_abc123"}
    a = captured["args"]
    assert a[0] == "create"
    assert "--assignee" in a and "equity-analyst" in a
    assert "--json" in a


def test_create_card_requires_assignee_and_title(monkeypatch):
    monkeypatch.setattr(kr, "_run_hermes_kanban", lambda args: _FakeProc())
    status, body = kr.create_card({"title": "no assignee"})
    assert status == 400
    status, body = kr.create_card({"assignee": "dev"})
    assert status == 400


def test_create_card_rejects_unknown_assignee(monkeypatch):
    monkeypatch.setattr(kr, "_run_hermes_kanban", lambda args: _FakeProc())
    status, body = kr.create_card({"assignee": "root", "title": "x"})
    assert status == 400
    assert "not permitted" in body["error"]


def test_create_card_passes_dedupe_key(monkeypatch):
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return _FakeProc(stdout='{"id": "t_1"}')

    monkeypatch.setattr(kr, "_run_hermes_kanban", fake_run)
    kr.create_card({"assignee": "dev", "title": "x", "dedupe_key": "chat:run42"})
    a = captured["args"]
    assert "--idempotency-key" in a
    assert a[a.index("--idempotency-key") + 1] == "chat:run42"


def test_create_card_goal_flag(monkeypatch):
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return _FakeProc(stdout='{"id": "t_1"}')

    monkeypatch.setattr(kr, "_run_hermes_kanban", fake_run)
    kr.create_card(
        {"assignee": "equity-analyst", "title": "x", "goal": True, "goal_max_turns": 15}
    )
    a = captured["args"]
    assert "--goal" in a
    assert "--goal-max-turns" in a
    assert a[a.index("--goal-max-turns") + 1] == "15"


def test_create_card_bad_priority(monkeypatch):
    monkeypatch.setattr(kr, "_run_hermes_kanban", lambda args: _FakeProc())
    status, body = kr.create_card({"assignee": "dev", "title": "x", "priority": "high"})
    assert status == 400


def test_create_card_cli_failure(monkeypatch):
    monkeypatch.setattr(
        kr, "_run_hermes_kanban", lambda args: _FakeProc(returncode=1, stderr="boom")
    )
    status, body = kr.create_card({"assignee": "dev", "title": "x"})
    assert status == 502


def test_create_card_timeout(monkeypatch):
    def raise_timeout(args):
        raise subprocess.TimeoutExpired(cmd="hermes", timeout=30)

    monkeypatch.setattr(kr, "_run_hermes_kanban", raise_timeout)
    status, body = kr.create_card({"assignee": "dev", "title": "x"})
    assert status == 504


# --------------------------------------------------------------------------
# comment_card
# --------------------------------------------------------------------------

def test_comment_happy_path(monkeypatch):
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return _FakeProc(stdout="ok")

    monkeypatch.setattr(kr, "_run_hermes_kanban", fake_run)
    status, body = kr.comment_card({"card_id": "t_abc", "text": "a follow-up"})
    assert status == 200
    assert body["id"] == "t_abc"
    a = captured["args"]
    assert a[0] == "comment"
    assert "t_abc" in a and "a follow-up" in a
    # positionals come after the `--` guard
    assert a[a.index("--") + 1] == "t_abc"


def test_comment_requires_fields(monkeypatch):
    monkeypatch.setattr(kr, "_run_hermes_kanban", lambda args: _FakeProc())
    assert kr.comment_card({"text": "hi"})[0] == 400
    assert kr.comment_card({"card_id": "t_1"})[0] == 400


def test_comment_unknown_card_is_404(monkeypatch):
    monkeypatch.setattr(
        kr, "_run_hermes_kanban",
        lambda args: _FakeProc(returncode=1, stderr="task not found: t_x"),
    )
    status, body = kr.comment_card({"card_id": "t_x", "text": "hi"})
    assert status == 404


# --------------------------------------------------------------------------
# secret resolution
# --------------------------------------------------------------------------

def test_secret_unset_is_none():
    assert kr._secret() is None


def test_secret_prefers_receiver_var(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "cron")
    monkeypatch.setenv("KANBAN_RECEIVER_SECRET", "recv")
    assert kr._secret() == "recv"


def test_secret_falls_back_to_cron(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "cron")
    assert kr._secret() == "cron"


# --------------------------------------------------------------------------
# Integration over real HTTP (CLI still mocked)
# --------------------------------------------------------------------------

@pytest.fixture
def server(monkeypatch):
    """Boot the receiver on an ephemeral port; yield its base URL."""
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), kr.Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(url, payload, headers=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_health_no_auth(server):
    with urllib.request.urlopen(f"{server}/health", timeout=5) as resp:
        assert resp.status == 200
        assert json.loads(resp.read())["ok"] is True


def test_http_fail_closed_when_secret_unset(server):
    # No secret in env -> every write is 403.
    status, body = _post(f"{server}/kanban/card-drop",
                         {"assignee": "dev", "title": "x"})
    assert status == 403


def test_http_403_wrong_secret(server, monkeypatch):
    monkeypatch.setenv("KANBAN_RECEIVER_SECRET", "right")
    status, body = _post(f"{server}/kanban/card-drop",
                         {"assignee": "dev", "title": "x"},
                         headers={"X-Cron-Secret": "wrong"})
    assert status == 403


def test_http_card_drop_with_secret(server, monkeypatch):
    monkeypatch.setenv("KANBAN_RECEIVER_SECRET", "right")
    monkeypatch.setattr(
        kr, "_run_hermes_kanban", lambda args: _FakeProc(stdout='{"id": "t_live1"}')
    )
    status, body = _post(f"{server}/kanban/card-drop",
                         {"assignee": "equity-analyst", "title": "AVGO"},
                         headers={"X-Cron-Secret": "right"})
    assert status == 200
    assert body["id"] == "t_live1"


def test_http_comment_with_secret(server, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "s")
    monkeypatch.setattr(kr, "_run_hermes_kanban", lambda args: _FakeProc(stdout="ok"))
    status, body = _post(f"{server}/kanban/comment",
                         {"card_id": "t_live1", "text": "hi"},
                         headers={"X-Cron-Secret": "s"})
    assert status == 200
    assert body["commented"] is True


def test_http_unknown_route_404(server, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "s")
    status, body = _post(f"{server}/nope", {}, headers={"X-Cron-Secret": "s"})
    assert status == 404

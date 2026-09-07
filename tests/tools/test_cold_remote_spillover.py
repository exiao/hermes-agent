"""Cold remote spillover regressions through executor, storage, and read_file."""

import base64
import json
import re
from types import SimpleNamespace

from agent.tool_executor import _ToolCallRef, _commit_tool_result, _finalize_tool_batch
from tools.budget_config import BudgetConfig
import tools.file_tools as file_tools
import tools.terminal_tool as terminal_tool
import tools.terminal_tool_lifecycle as lifecycle


class FakeRemoteEnvironment:
    cwd = "/root"
    _sync_manager = None

    def __init__(self):
        self.commands = []
        self.files = {}


    def get_temp_dir(self):
        return "/tmp"

    def execute(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command.startswith("test -r"):
            return {"output": "", "returncode": 1}
        if "cat >" in command:
            path_match = re.search(r"cat > (?:'([^']+)'|(\S+))", command)
            path = path_match.group(1) or path_match.group(2)
            self.files[path] = kwargs.get("stdin_data", "")
            return {"output": "", "returncode": 0}
        if "__HERMES_RF_" not in command:
            return {"output": "", "returncode": 0}

        path_match = re.search(r"if \[ -f '([^']+)' \]", command)
        path = path_match.group(1) if path_match else ""
        content = self.files.get(path)
        if content is None:
            self.last_output = "__hermes_missing__\\n"
            return {"output": self.last_output, "returncode": 0}

        sentinel = re.search(r"echo (__HERMES_RF_[A-Za-z0-9]+__)", command).group(1)
        page_match = re.search(r"sed -n '(\d+),(\d+)p'", command)
        start, end = (int(page_match.group(1)), int(page_match.group(2)))
        lines = content.splitlines()
        page = "\n".join(lines[start - 1:end])
        output = "\n".join([
            str(len(content.encode("utf-8"))), sentinel,
            base64.b64encode(content.encode("utf-8")[:1000]).decode(), sentinel,
            page, sentinel, str(len(lines)), sentinel,
            "1" if content.endswith("\n") else "0", sentinel, "0 0",
        ])
        return {"output": output + "\n", "returncode": 0}


class FakeAgent:
    _current_tool = None
    verbose_logging = False
    tool_progress_callback = None
    _subdirectory_hints = SimpleNamespace(check_tool_call=lambda *args: "")
    _tool_guardrails = SimpleNamespace(record_persisted_result=lambda *args: None)

    def _touch_activity(self, *args):
        pass

    def _tool_result_content_for_active_model(self, _name, result):
        return result

    def _flush_messages_to_session_db(self, _messages):
        return True

    def _apply_pending_steer_to_tool_results(self, _messages, _num_tools):
        pass


def _configure_remote(monkeypatch, tmp_path, factory):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "synthetic.invalid")
    monkeypatch.setenv("TERMINAL_SSH_USER", "synthetic")
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_create_configured_env", factory)
    monkeypatch.setattr(lifecycle, "_create_configured_env", factory)
    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()
    file_tools.clear_file_ops_cache()


def _commit(agent, task_id, tool_name, content, budget):
    messages = []
    _commit_tool_result(
        agent,
        messages,
        _ToolCallRef(tool_name, {}, task_id, f"call-{task_id}", []),
        content,
        budget=budget,
        tool_duration=0,
        is_error=False,
        blocked=True,
        effect_disposition=None,
    )
    return messages[0]["content"]


def _saved_path(result):
    return result.split("Full output saved to: ", 1)[1].splitlines()[0]


def test_cold_executor_spill_is_readable_by_real_file_route(monkeypatch, tmp_path):
    remote = FakeRemoteEnvironment()
    _configure_remote(monkeypatch, tmp_path, lambda *args, **kwargs: remote)
    content = "FIRST-MARKER\n" + ("middle\n" * 12_000) + "FINAL-MARKER\n"

    result = _commit(
        FakeAgent(), "cold-session", "synthetic_tool", content,
        BudgetConfig(tool_overrides={"synthetic_tool": 1_000}),
    )
    path = _saved_path(result)
    first = json.loads(file_tools.read_file_tool(path, offset=1, limit=2_000, task_id="cold-session"))
    final = json.loads(file_tools.read_file_tool(path, offset=12_002, limit=2, task_id="cold-session"))

    assert path.startswith("/tmp/hermes-results/")
    assert first["content"].startswith("1|FIRST-MARKER")
    assert "FINAL-MARKER" in final["content"]
    assert len(remote.files) == 1
    assert len(remote.commands) >= 3


def test_cold_executor_uses_the_session_cwd_for_environment_creation(monkeypatch, tmp_path):
    remote = FakeRemoteEnvironment()
    created = []

    def factory(*args, **kwargs):
        created.append(kwargs)
        return remote

    _configure_remote(monkeypatch, tmp_path, factory)
    terminal_tool.record_session_cwd("cwd-session", "/session/workspace")

    _commit(
        FakeAgent(), "cwd-session", "synthetic_tool", "x" * 4_000,
        BudgetConfig(tool_overrides={"synthetic_tool": 1_000}),
    )

    assert created[0]["cwd"] == "/session/workspace"


def test_aggregate_overflow_uses_the_cold_remote_file_route(monkeypatch, tmp_path):
    remote = FakeRemoteEnvironment()
    _configure_remote(monkeypatch, tmp_path, lambda *args, **kwargs: remote)
    messages = [
        {"role": "tool", "tool_call_id": "aggregate-one", "content": "A" * 70_000},
        {"role": "tool", "tool_call_id": "aggregate-two", "content": "B" * 70_000},
    ]
    agent = FakeAgent()

    _finalize_tool_batch(
        agent, messages, "aggregate-session", 2,
        BudgetConfig(turn_budget=100_000),
    )

    persisted = [message["content"] for message in messages if "Full output saved to: " in message["content"]]
    assert len(persisted) == 1
    path = _saved_path(persisted[0])
    read = json.loads(file_tools.read_file_tool(path, offset=1, limit=2, task_id="aggregate-session"))
    assert read["content"].startswith("1|A") or read["content"].startswith("1|B")
    assert path.startswith("/tmp/hermes-results/")


def test_small_cold_result_does_not_create_remote_environment(monkeypatch, tmp_path):
    created = []
    _configure_remote(monkeypatch, tmp_path, lambda *args, **kwargs: created.append(True))

    result = _commit(
        FakeAgent(), "small-session", "synthetic_tool", "small",
        BudgetConfig(tool_overrides={"synthetic_tool": 1_000}),
    )

    assert result == "small"
    assert created == []
    assert terminal_tool._active_environments == {}


def test_remote_setup_failure_does_not_advertise_host_path(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic remote setup failure")

    _configure_remote(monkeypatch, tmp_path, fail)
    result = _commit(
        FakeAgent(), "failed-session", "synthetic_tool", "x" * 4_000,
        BudgetConfig(tool_overrides={"synthetic_tool": 1_000}),
    )

    assert "Full output saved to:" not in result
    assert "Full output could not be saved to sandbox" in result
    assert (tmp_path / ".hermes" / "cache" / "spillover" / "call-failed-session.txt").exists()


def test_terminal_config_failure_keeps_canonical_copy_fail_closed(monkeypatch, tmp_path):
    _configure_remote(monkeypatch, tmp_path, lambda *args, **kwargs: None)
    monkeypatch.setenv("TERMINAL_SSH_PORT", "not-a-port")

    result = _commit(
        FakeAgent(), "config-failure-session", "synthetic_tool", "x" * 4_000,
        BudgetConfig(tool_overrides={"synthetic_tool": 1_000}),
    )

    assert "Full output saved to:" not in result
    assert "Full output could not be saved to sandbox" in result
    assert (tmp_path / ".hermes" / "cache" / "spillover" / "call-config-failure-session.txt").exists()

"""Per-job reasoning_effort override for cron jobs.

A job pinned to a shared model must be able to set its own effort without
moving the global agent.reasoning_effort for every other surface.
"""

from unittest.mock import MagicMock, patch

import cron.scheduler as cron_scheduler
from hermes_constants import parse_reasoning_effort, resolve_reasoning_config


def _resolve(job: dict, cfg: dict, model: str):
    """Mirror of the resolution in cron/scheduler.py."""
    result = parse_reasoning_effort(job.get("reasoning_effort"))
    if result is None:
        result = resolve_reasoning_config(cfg, model)
    return result


class TestPerJobReasoningEffort:
    def test_job_override_beats_global(self):
        cfg = {"agent": {"reasoning_effort": "low"}}
        job = {"reasoning_effort": "high"}
        assert _resolve(job, cfg, "gpt-5.6-luna") == {"enabled": True, "effort": "high"}

    def test_absent_job_field_falls_back_to_global(self):
        cfg = {"agent": {"reasoning_effort": "low"}}
        assert _resolve({}, cfg, "gpt-5.6-luna") == {"enabled": True, "effort": "low"}

    def test_job_override_beats_per_model_override(self):
        cfg = {
            "agent": {
                "reasoning_effort": "low",
                "reasoning_overrides": {"gpt-5.6-luna": "medium"},
            }
        }
        job = {"reasoning_effort": "high"}
        assert _resolve(job, cfg, "gpt-5.6-luna") == {"enabled": True, "effort": "high"}

    def test_job_can_disable_thinking(self):
        cfg = {"agent": {"reasoning_effort": "high"}}
        assert _resolve({"reasoning_effort": "none"}, cfg, "gpt-5.6-luna") == {
            "enabled": False
        }

    def test_unrecognized_job_value_falls_back(self):
        cfg = {"agent": {"reasoning_effort": "low"}}
        job = {"reasoning_effort": "turbo"}
        assert _resolve(job, cfg, "gpt-5.6-luna") == {"enabled": True, "effort": "low"}


def _run_job_and_capture_reasoning(job, tmp_path):
    """Run the scheduler path with provider and inference calls isolated."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("agent:\n  reasoning_effort: low\n")
    fake_db = MagicMock()
    fake_agent = MagicMock()
    fake_agent.run_conversation.return_value = {"final_response": "ok"}

    with patch.object(cron_scheduler, "_hermes_home", tmp_path), \
         patch.object(cron_scheduler, "_resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openrouter",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
         patch("run_agent.AIAgent", return_value=fake_agent) as agent_cls:
        result = cron_scheduler.run_job(job)

    return result, agent_cls.call_args.kwargs["reasoning_config"]


def test_run_job_passes_per_job_reasoning_override_to_agent(tmp_path):
    result, reasoning_config = _run_job_and_capture_reasoning(
        {
            "id": "reasoning-override",
            "name": "reasoning override",
            "prompt": "hello",
            "model": "gpt-5.6-luna",
            "reasoning_effort": "high",
        },
        tmp_path,
    )

    assert result[0] is True
    assert result[3] is None
    assert reasoning_config == {"enabled": True, "effort": "high"}


def test_run_job_passes_resolved_reasoning_fallback_to_agent(tmp_path):
    result, reasoning_config = _run_job_and_capture_reasoning(
        {
            "id": "reasoning-fallback",
            "name": "reasoning fallback",
            "prompt": "hello",
            "model": "gpt-5.6-luna",
        },
        tmp_path,
    )

    assert result[0] is True
    assert result[3] is None
    assert reasoning_config == {"enabled": True, "effort": "low"}

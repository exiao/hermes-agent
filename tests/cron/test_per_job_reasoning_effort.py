"""Per-job reasoning_effort override for cron jobs.

A job pinned to a shared model must be able to set its own effort without
moving the global agent.reasoning_effort for every other surface.
"""

from unittest.mock import MagicMock, patch

from hermes_constants import parse_reasoning_effort, resolve_reasoning_config


def _run_job_with_agent_capture(job, tmp_path):
    """Run the real scheduler path while capturing AIAgent construction."""
    fake_db = MagicMock()
    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._get_hermes_home", return_value=tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=fake_db),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            },
        ),
        patch("run_agent.AIAgent") as mock_agent_cls,
    ):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent

        from cron.scheduler import run_job

        result = run_job(job)

    return result, mock_agent_cls


def _resolve(job: dict, cfg: dict, model: str):
    """Mirror of the resolution in cron/scheduler.py."""
    result = parse_reasoning_effort(job.get("reasoning_effort"))
    if result is None:
        result = resolve_reasoning_config(cfg, model)
    return result


class TestPerJobReasoningEffort:
    def test_job_override_reaches_scheduler_agent(self, tmp_path):
        result, mock_agent_cls = _run_job_with_agent_capture(
            {
                "id": "reasoning-test",
                "name": "reasoning test",
                "prompt": "hello",
                "model": "gpt-5.6-luna",
                "reasoning_effort": "high",
            },
            tmp_path,
        )

        assert result[0] is True
        assert mock_agent_cls.call_args.kwargs["reasoning_config"] == {
            "enabled": True,
            "effort": "high",
        }

    def test_resolved_fallback_reaches_scheduler_agent(self, tmp_path):
        (tmp_path / "config.yaml").write_text("agent:\n  reasoning_effort: low\n")
        result, mock_agent_cls = _run_job_with_agent_capture(
            {
                "id": "reasoning-fallback",
                "name": "reasoning fallback",
                "prompt": "hello",
                "model": "gpt-5.6-luna",
            },
            tmp_path,
        )

        assert result[0] is True
        assert mock_agent_cls.call_args.kwargs["reasoning_config"] == {
            "enabled": True,
            "effort": "low",
        }

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

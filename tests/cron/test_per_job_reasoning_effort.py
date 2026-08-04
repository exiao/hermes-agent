"""Per-job reasoning_effort override for cron jobs.

A job pinned to a shared model must be able to set its own effort without
moving the global agent.reasoning_effort for every other surface.
"""

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

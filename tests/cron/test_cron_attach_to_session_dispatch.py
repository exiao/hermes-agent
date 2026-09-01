"""`attach_to_session` must survive the trip from the tool schema to the store.

The flag was advertised in CRONJOB_SCHEMA, declared by cronjob(), normalized by
create_job() and honoured by the scheduler — but the registry handler lambda
never extracted it from the model's args, so every agent-created job stored the
field absent and silently inherited the global ``cron.mirror_delivery``.

Calling cronjob() directly passes with or without that bug, so these tests drive
the REGISTERED HANDLER. The False case is the load-bearing one: it is the only
assertion that distinguishes a working per-job override from the global being on.
"""

from __future__ import annotations

import json

import pytest

from cron.jobs import load_jobs
from cron.scheduler import _cron_mirror_delivery_enabled

# Properties the handler deliberately does NOT forward from the model's args.
# model/provider/base_url: per-job inference pins are user-owned, so a model
# must not be able to point unattended spend elsewhere. include_disabled is a
# read filter with its own default. See the comment in the handler lambda.
INTENTIONALLY_NOT_FORWARDED = {"model", "provider", "base_url", "reasoning_effort"}


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate the cron store (same pattern as tests/cron/test_jobs.py)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path / "cron"


def _handler():
    import tools.cronjob_tools as mod

    return mod.registry._tools["cronjob"].handler


def _create_via_tool(**extra):
    args = {"action": "create", "prompt": "daily digest", "schedule": "every 1h"}
    args.update(extra)
    out = json.loads(_handler()(args))
    assert out["success"] is True, out
    return load_jobs()[0]


class TestAttachToSessionReachesTheStore:
    def test_true_is_persisted(self, tmp_cron_dir):
        assert _create_via_tool(attach_to_session=True)["attach_to_session"] is True

    def test_false_is_persisted_and_overrides_global_mirror(self, tmp_cron_dir):
        """The case the flag exists for: one fire-and-forget job that must NOT
        mirror, while the global cron.mirror_delivery is on."""
        job = _create_via_tool(attach_to_session=False)
        assert job["attach_to_session"] is False
        assert _cron_mirror_delivery_enabled(job, {"cron": {"mirror_delivery": True}}) is False

    def test_absent_leaves_the_job_following_the_global(self, tmp_cron_dir):
        job = _create_via_tool()
        assert "attach_to_session" not in job
        assert _cron_mirror_delivery_enabled(job, {"cron": {"mirror_delivery": True}}) is True


def test_every_schema_property_is_forwarded_or_documented(monkeypatch):
    """Class guard: a property the model can send must either reach cronjob()
    through the handler, or be on the explicit exclusion list above. This is
    what catches the next parameter dropped between schema and function.

    Runs the registered handler with a unique sentinel per property and reads
    the keyword arguments cronjob() actually received, so a refactor of the
    handler is free as long as the values still arrive.
    """
    import tools.cronjob_tools as mod

    received: dict = {}
    monkeypatch.setattr(
        mod, "cronjob", lambda **kwargs: received.update(kwargs) or "{}"
    )

    properties = list(mod.CRONJOB_SCHEMA["parameters"]["properties"])
    args = {name: f"sentinel-{name}" for name in properties}
    args["action"] = "list"
    _handler()(args)

    dropped = sorted(
        name
        for name in properties
        if name not in INTENTIONALLY_NOT_FORWARDED
        and received.get(name) != args[name]
    )
    assert not dropped, f"schema properties never reaching cronjob(): {dropped}"

"""Modal entrypoint for the isolated Kanban ``memo-evaluator`` lane.

Run only through ``hermes_cli.kanban_modal``. The local shim owns every Kanban
DB lifecycle transition; this app receives a serialized brief and returns a
structured result only.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal

APP_NAME = "hermes-kanban-memo-evaluator"
PROFILE_SOURCE = Path(
    os.environ.get(
        "HERMES_MODAL_MEMO_EVALUATOR_PROFILE",
        "~/.hermes/profiles/memo-evaluator",
    )
).expanduser()
SOUL_SOURCE = PROFILE_SOURCE / "SOUL.md"
SKILLS_SOURCE = PROFILE_SOURCE / "skills"
PROXY_BASE_URL = "https://proxy.getbloom.app"
PROXY_SECRET = modal.Secret.from_name(
    "bloom-llm-proxy", required_keys=["OPENAI_API_KEY"]
)

if not SOUL_SOURCE.is_file() or not SKILLS_SOURCE.is_dir():
    raise RuntimeError(
        "Set HERMES_MODAL_MEMO_EVALUATOR_PROFILE to a profile containing SOUL.md and skills/."
    )

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("hermes-agent>=0.18.2,<0.19")
    .add_local_file(SOUL_SOURCE, "/opt/memo-evaluator/SOUL.md")
    .add_local_dir(SKILLS_SOURCE, "/opt/memo-evaluator/skills")
)
app = modal.App(APP_NAME)


def _block(reason: str, *, kind: str = "transient") -> dict[str, Any]:
    return {"outcome": "block", "reason": reason, "kind": kind}


def _parse_worker_result(text: str) -> dict[str, Any]:
    result = json.loads(text.strip())
    if not isinstance(result, dict):
        raise ValueError("worker result must be a JSON object")
    outcome = result.get("outcome")
    if outcome == "complete" and isinstance(result.get("summary"), str):
        return result
    if outcome == "block" and isinstance(result.get("reason"), str):
        return result
    raise ValueError("worker result must declare a complete or block outcome")


@app.function(
    image=image,
    secrets=[PROXY_SECRET],
    timeout=3600,
    env={
        "HERMES_HOME": "/opt/memo-evaluator",
        "HERMES_INFERENCE_PROVIDER": "custom",
        "OPENAI_BASE_URL": PROXY_BASE_URL,
    },
)
def evaluate_memo(request_json: str) -> str:
    """Evaluate a brief remotely without a Kanban DB or local workspace mount."""
    try:
        request = json.loads(request_json)
        brief = request["brief"]
        if not isinstance(brief, str) or not brief.strip():
            return json.dumps(_block("Modal request is missing a task brief.", kind="capability"))
    except (KeyError, TypeError, json.JSONDecodeError):
        return json.dumps(_block("Modal request is malformed.", kind="capability"))

    prompt = """You are the memo-evaluator lane. Evaluate the supplied Kanban task.
You are running remotely with no Kanban database or local workspace mount. Do not
claim, complete, block, or edit a Kanban card yourself. Return exactly one JSON
object and no markdown: either
{"outcome":"complete","summary":"...","metadata":{...}}
or {"outcome":"block","reason":"...","kind":"needs_input|capability|transient"}.

Task brief follows:
""" + brief
    completed = subprocess.run(
        ["hermes", "--cli", "chat", "-q", prompt],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return json.dumps(_block("Remote memo evaluator failed; inspect the Modal call logs."))
    try:
        return json.dumps(_parse_worker_result(completed.stdout))
    except (ValueError, json.JSONDecodeError):
        return json.dumps(
            _block("Remote memo evaluator returned an invalid structured result.", kind="transient")
        )


@app.local_entrypoint()
def main(request_json: str) -> str:
    """Return the remote result plus the FunctionCall audit handle to the shim."""
    call = evaluate_memo.spawn(request_json)
    result = json.loads(call.get())
    result["modal_call_id"] = call.object_id
    result["modal_log_url"] = call.get_dashboard_url()
    return json.dumps(result)

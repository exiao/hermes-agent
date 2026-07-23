"""Modal entrypoint for the isolated Kanban ``memo-evaluator`` lane.

Run only through ``hermes_cli.kanban_modal``. The local shim owns every Kanban
DB lifecycle transition; this app receives a serialized brief and returns a
structured result only.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

APP_NAME = "hermes-kanban-memo-evaluator"


def _default_profile_source() -> Path:
    """Resolve the memo-evaluator profile dir, honoring HERMES_HOME.

    Anchored to the profiles root via ``get_profile_dir`` so custom / Docker /
    profile-isolated ``HERMES_HOME`` layouts (e.g. ``/opt/data``) resolve
    correctly instead of a hardcoded ``~/.hermes`` that would not exist there.
    Only ever called under ``modal.is_local()`` (the local launcher), so the
    ``hermes_cli`` import is always available.
    """
    from hermes_cli.profiles import get_profile_dir

    return get_profile_dir("memo-evaluator")


# Model the memo-evaluator profile is pinned to (see its config.yaml). Passed
# explicitly because the image bakes only SOUL.md + skills/, not a config.yaml,
# so there is no on-disk model default inside the container.
MEMO_EVALUATOR_MODEL = "claude-fable-5"
MEMO_EVALUATOR_PROVIDER = "anthropic"

# The memo-evaluator runs a Claude model through the shared billing proxy, which
# only speaks the Anthropic Messages API. The proxy hostname and its inbound gate
# key both live in the existing ``research-proxy`` Modal secret (ANTHROPIC_BASE_URL
# + ANTHROPIC_API_KEY, where the key equals the proxy gate key — Hermes sends it as
# ``x-api-key`` and the proxy strips it and injects the real OAuth subscription).
PROXY_SECRET = modal.Secret.from_name(
    "research-proxy",
    required_keys=["ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"],
)

# The profile source (SOUL.md + skills/) is only present on the machine that
# builds/launches the app; inside the Modal container the module is re-imported
# with only the baked ``/opt/memo-evaluator`` payload, so these local paths do
# not exist. Guard the source resolution, check, and the ``add_local_*`` mounts
# on ``modal.is_local()`` — resolving/validating the source at container-import
# time crashes every remote run.
if modal.is_local():
    _profile_source = _default_profile_source()
    _soul_source = _profile_source / "SOUL.md"
    _skills_source = _profile_source / "skills"
    if not _soul_source.is_file() or not _skills_source.is_dir():
        raise RuntimeError(
            f"memo-evaluator profile at {_profile_source} is missing SOUL.md "
            "and skills/; create it with `hermes profile create memo-evaluator`."
        )

_base_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    # The worker runs Hermes with ``--provider anthropic``; the native
    # Anthropic SDK is an optional extra, so install it here or the adapter
    # raises ImportError before evaluating any brief.
    "hermes-agent[anthropic]>=0.18.2,<0.19"
)
if modal.is_local():
    image = _base_image.add_local_file(
        _soul_source, "/opt/memo-evaluator/SOUL.md"
    ).add_local_dir(_skills_source, "/opt/memo-evaluator/skills")
else:
    image = _base_image
app = modal.App(APP_NAME)


def _block(reason: str, *, kind: str = "transient") -> dict[str, Any]:
    return {"outcome": "block", "reason": reason, "kind": kind}


def _extract_last_json_object(text: str) -> str:
    """Return the last balanced top-level JSON object in ``text``.

    Quiet-mode ``hermes chat -Q`` writes only the final response to stdout, but
    a stray startup line (an interpreter warning, a security-scanner notice) can
    still precede it. Scanning for the last balanced ``{...}`` recovers the model
    verdict without assuming the whole stream is pure JSON.
    """
    depth = 0
    start = -1
    candidates: list[str] = []
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
    if not candidates:
        raise ValueError("no JSON object found in worker output")
    return candidates[-1]


def _parse_worker_result(text: str) -> dict[str, Any]:
    result = json.loads(_extract_last_json_object(text))
    if not isinstance(result, dict):
        raise ValueError("worker result must be a JSON object")
    outcome = result.get("outcome")
    if outcome == "complete" and isinstance(result.get("summary"), str):
        return result
    if outcome == "block" and isinstance(result.get("reason"), str):
        return result
    raise ValueError("worker result must declare a complete or block outcome")


# Modal caps a function timeout at 24h; use that as the ceiling for an
# uncapped ``memo-evaluator`` task. The default here only applies when the shim
# doesn't override it via ``with_options`` (see ``main``).
_MODAL_MAX_TIMEOUT = 24 * 60 * 60
_DEFAULT_TIMEOUT = 3600


def _resolve_function_timeout(max_runtime: Any) -> int | dict[str, Any]:
    """Map a task's ``max_runtime_seconds`` to a Modal function timeout.

    Returns an int timeout to apply, or a ``block`` result dict when the task's
    runtime cannot be honored remotely. A runtime above Modal's hard 24h cap is
    rejected (not silently clamped): clamping would kill a longer task
    mid-evaluation and requeue it as a transient failure forever, so surface it
    as a capability block up front — before any paid remote spawn — so a human
    reroutes it to a backend without the cap. ``None`` / unparseable means
    uncapped and uses Modal's 24h ceiling.
    """
    if max_runtime is not None:
        try:
            max_runtime = int(max_runtime)
        except (TypeError, ValueError):
            max_runtime = None
    if max_runtime is not None and max_runtime > _MODAL_MAX_TIMEOUT:
        return _block(
            f"Task max_runtime_seconds ({max_runtime}s) exceeds Modal's "
            f"{_MODAL_MAX_TIMEOUT}s function-timeout cap; run it on a backend "
            "without the 24h limit.",
            kind="capability",
        )
    return _MODAL_MAX_TIMEOUT if max_runtime is None else min(max_runtime, _MODAL_MAX_TIMEOUT)


@app.function(
    image=image,
    secrets=[PROXY_SECRET],
    timeout=_DEFAULT_TIMEOUT,
    env={
        "HERMES_HOME": "/opt/memo-evaluator",
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
        [
            "hermes",
            "--cli",
            "chat",
            "-Q",
            "-q",
            prompt,
            "-m",
            MEMO_EVALUATOR_MODEL,
            "--provider",
            MEMO_EVALUATOR_PROVIDER,
        ],
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
def main() -> str:
    """Return the remote result plus the FunctionCall audit handle to the shim.

    The serialized request arrives on stdin (not an argv element) so a large
    brief cannot overflow the Windows 32,767-char command-line limit.
    """
    request_json = sys.stdin.read()
    # Honor the task's runtime contract: a >1h or uncapped (None) task must not
    # be killed at the 1h function default. Uncapped uses Modal's 24h ceiling; a
    # value above the cap is rejected as a capability block (see the helper).
    fn = evaluate_memo
    try:
        max_runtime = json.loads(request_json).get("max_runtime_seconds")
    except (TypeError, ValueError):
        max_runtime = None
    resolved = _resolve_function_timeout(max_runtime)
    if isinstance(resolved, dict):
        return json.dumps(resolved)
    timeout = resolved
    if timeout != _DEFAULT_TIMEOUT:
        fn = evaluate_memo.with_options(timeout=timeout)
    call = fn.spawn(request_json)
    result = json.loads(call.get())
    result["modal_call_id"] = call.object_id
    result["modal_log_url"] = call.get_dashboard_url()
    return json.dumps(result)

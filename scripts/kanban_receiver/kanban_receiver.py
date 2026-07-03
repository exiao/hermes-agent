#!/usr/bin/env python3
"""Standalone HTTP receiver that drops cards onto the local Kanban board.

This is the single inbound write surface connecting off-box producers (the
Render CPE website, the diligence inbox, Modal cron producers) to the Hermes
Kanban board living at ``$HERMES_HOME/kanban.db`` on this machine. Remote tiers
cannot write the local SQLite board directly, so they POST here instead.

Design decisions (see ~/.hermes/plans/cpe-chat-route-to-research-lead.md Card A
and ~/.hermes/plans/diligence-inbox-production.md section A+):

* **Standalone, not api_server.py.** The gateway's ``api_server`` is a large
  OpenAI-compatible platform adapter whose routes are baked into one hardcoded
  block with no auth'd custom-route registration hook. Widening its surface
  risks the multiplex gateway's always-on port (secondary profiles must not
  bind ports -> crash-loop risk). This service is a tiny, stdlib-only launchd
  daemon on its own port with a far smaller blast radius.

* **stdlib only.** No aiohttp / third-party imports, so it boots cleanly under a
  bare launchd context regardless of which venv the gateway uses. It shells out
  to the installed ``hermes kanban`` CLI (the same validated create/comment path
  a human or the dispatcher uses), inheriting ``HERMES_HOME``.

* **Fail closed.** ``CRON_SECRET`` (or ``KANBAN_RECEIVER_SECRET``) must be set;
  when it is unset the service refuses ALL writes with 403. Because the public
  hostname (kanban.getbloom.app) makes the secret gate the security boundary,
  an unset secret must never silently accept unauthenticated writes. This is the
  fail-CLOSED inversion of research-agent's ``_check_cron_auth`` (which
  fails-open for local testing); on a public surface fail-open is unacceptable.

Endpoints:
  GET  /health                -> {"ok": true, "ts": ...}  (no auth; liveness)
  POST /kanban/card-drop       -> create a card, returns {"id": ...}
  POST /kanban/comment         -> append a comment to an existing card

Auth: ``X-Cron-Secret`` header, compared with ``hmac.compare_digest``.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

LOG = logging.getLogger("kanban_receiver")

DEFAULT_PORT = 8646
MAX_BODY_BYTES = 64 * 1024  # generous for a card body, cheap DoS guard
CLI_TIMEOUT_SECONDS = 30

# Only these profiles may be targeted by an off-box drop. Keeps a leaked secret
# from spawning arbitrary lanes; extend deliberately. "none" creates an
# unassigned card (triaged by a human/orchestrator later).
ALLOWED_ASSIGNEES = {
    "none",
    "orchestrator",
    "equity-analyst",
    "memo-evaluator",
    "researcher",
    "dev",
    "pm",
    "designer",
    "qa",
    "infra-ops",
    "pr-babysitter",
    "code-reviewer",
    "content-creator",
    "ads-optimizer",
}


def _secret() -> Optional[str]:
    """The configured shared secret, or None when unset.

    Accepts KANBAN_RECEIVER_SECRET first (service-specific) then CRON_SECRET
    (the same secret the Modal / research-agent callers already send)."""
    for var in ("KANBAN_RECEIVER_SECRET", "CRON_SECRET"):
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    return None


def _hermes_bin() -> str:
    """Resolve the ``hermes`` CLI entrypoint.

    Prefer an explicit override, then PATH, then the interpreter running this
    service (``python -m hermes_cli.main``) as a last resort so the daemon works
    even when launchd's PATH is minimal."""
    override = os.environ.get("HERMES_BIN")
    if override:
        return override
    found = shutil.which("hermes")
    if found:
        return found
    return ""  # signals: fall back to python -m


_REDACT_FLAGS = {"--body"}


def _redact_args(args: list[str]) -> list[str]:
    """Return args safe for logging: drop user-supplied content (diligence /
    inbox text) so it never lands in the launchd stderr log.

    Sensitive content sources:
    - flag values after ``--body`` (card body on ``create``);
    - the trailing positional ``title`` on ``create`` (``-- <title>``), which is
      inbox/diligence-derived (same sensitivity class as the body);
    - the trailing positional ``text`` on ``comment`` (``comment <id> <text>``),
      which arrives after ``--`` alongside the safe card id.

    Rule: everything after ``--`` is a positional. For ``comment`` the FIRST
    such positional (the card id) is safe to log; every other trailing
    positional is redacted to ``<redacted>``.
    """
    out: list[str] = []
    skip = False
    is_comment = bool(args) and args[0] == "comment"
    seen_ddash = False
    positional_after_ddash = 0
    for a in args:
        if skip:
            skip = False
            continue
        if a in _REDACT_FLAGS:
            skip = True
            out.append(a)  # keep the flag name; its value is dropped
            out.append("<redacted>")
            continue
        if a == "--":
            seen_ddash = True
            out.append(a)
            continue
        if seen_ddash:
            positional_after_ddash += 1
            # Keep only the card id (first positional on `comment`); redact
            # every other trailing positional (comment text, create title).
            if is_comment and positional_after_ddash == 1:
                out.append(a)
            else:
                out.append("<redacted>")
            continue
        out.append(a)
    return out


def _run_hermes_kanban(args: list[str]) -> subprocess.CompletedProcess:
    """Invoke ``hermes kanban <args>`` inheriting the environment (HERMES_HOME)."""
    hermes = _hermes_bin()
    cmd = [hermes, "kanban", *args] if hermes else [sys.executable, "-m", "hermes_cli.main", "kanban", *args]
    LOG.info("exec: %s", " ".join(cmd[:2] + ["kanban"] + _redact_args(args)))
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=CLI_TIMEOUT_SECONDS,
        env=os.environ.copy(),
    )


def _opt_str(value: Any) -> Optional[str]:
    """Coerce a JSON field to a stripped str, or None if absent/wrong type.

    A client that sends ``title: 5`` or ``body: {...}`` must get a clean 400,
    not an uncaught AttributeError that resets the connection. Only real strings
    (and None/missing) are accepted; numbers/objects/lists are rejected."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value.strip()


def _is_goal(value: Any) -> bool:
    """Decide whether ``goal`` enables goal mode, accepting only real booleans.

    A producer that serializes ``goal`` as a string like ``"false"`` or ``"0"``
    is still truthy under a bare ``if payload.get("goal")`` check, which would
    dispatch an ordinary card as a multi-turn goal loop and burn the goal
    budget. Accept a JSON boolean directly; for string forms, treat the usual
    falsey tokens ("", "false", "0", "no", "off") as False and everything else
    True. Non-bool/non-str values (numbers, objects) fall back to truthiness.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "off")
    return bool(value)


def create_card(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Handle a /kanban/card-drop payload -> hermes kanban create.

    Contract: {assignee, title, body, dedupe_key?, priority?, goal?,
    goal_max_turns?}. Returns (status_code, json_body)."""
    # Reject non-string title/assignee/body up front so a malformed JSON value
    # (int, object, list) yields a clean 400 rather than an uncaught exception.
    for field in ("assignee", "title", "body"):
        if field in payload and payload[field] is not None and not isinstance(payload[field], str):
            return 400, {"error": f"{field} must be a string"}
    assignee = _opt_str(payload.get("assignee")) or ""
    title = _opt_str(payload.get("title")) or ""
    body = _opt_str(payload.get("body")) or ""

    if not assignee or not title:
        return 400, {"error": "assignee and title are required"}
    if assignee not in ALLOWED_ASSIGNEES:
        return 400, {"error": f"assignee '{assignee}' not permitted"}

    # `title` is a positional arg on `hermes kanban create`; `--body`/`--assignee`
    # are flags. Build all flags first, then append `-- <title>` LAST so a title
    # beginning with a dash can't be parsed as an option.
    args = ["create", "--body", body, "--json"]

    # The "none" sentinel means "unassigned / triage later". `hermes kanban
    # create` does NOT canonicalize it the way `assign`/`reassign` do, so
    # forwarding `--assignee none` would store the literal lane "none" and
    # strand the card in `ready` (no dispatcher serves it). Omit the flag
    # instead so the card lands genuinely unassigned.
    if assignee != "none":
        args += ["--assignee", assignee]

    dedupe_key = payload.get("dedupe_key")
    if dedupe_key and str(dedupe_key).strip():
        args += ["--idempotency-key", str(dedupe_key).strip()]

    priority = payload.get("priority")
    if priority is not None:
        try:
            args += ["--priority", str(int(priority))]
        except (TypeError, ValueError):
            return 400, {"error": "priority must be an integer"}

    if _is_goal(payload.get("goal")):
        args.append("--goal")
        gmt = payload.get("goal_max_turns")
        if gmt is not None:
            try:
                val = int(gmt)
            except (TypeError, ValueError):
                return 400, {"error": "goal_max_turns must be an integer"}
            # Guard against zero/negative-turn loops: a non-positive limit is
            # meaningless, so drop the flag and let the CLI apply its own
            # default rather than forwarding a value that stalls the worker.
            if val >= 1:
                args += ["--goal-max-turns", str(val)]

    # Author the card as the drop's assignee-agnostic origin so the audit trail
    # shows it arrived over the wire, not from a local human.
    args += ["--created-by", "card-drop"]

    # Positional title last, guarded by `--`.
    args += ["--", title]

    try:
        proc = _run_hermes_kanban(args)
    except subprocess.TimeoutExpired:
        return 504, {"error": "kanban create timed out"}

    if proc.returncode != 0:
        LOG.error("kanban create failed rc=%s stderr=%s", proc.returncode, proc.stderr[-500:])
        return 502, {"error": "kanban create failed", "detail": proc.stderr.strip()[-300:]}

    try:
        created = json.loads(proc.stdout)
    except json.JSONDecodeError:
        LOG.error("kanban create returned non-JSON: %s", proc.stdout[-300:])
        return 502, {"error": "kanban create returned unparseable output"}

    card_id = created.get("id")
    if not card_id:
        return 502, {"error": "kanban create returned no id"}
    return 200, {"id": card_id}


def comment_card(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Handle a /kanban/comment payload -> hermes kanban comment.

    Contract: {card_id, text}. Returns (status_code, json_body)."""
    for field in ("card_id", "text"):
        if field in payload and payload[field] is not None and not isinstance(payload[field], str):
            return 400, {"error": f"{field} must be a string"}
    card_id = _opt_str(payload.get("card_id")) or ""
    text = _opt_str(payload.get("text")) or ""
    if not card_id or not text:
        return 400, {"error": "card_id and text are required"}

    # `hermes kanban comment <task_id> <text...>` — both positional. Put the
    # optional `--author` first, then `--` and the two positionals, so a text
    # (or id) starting with a dash can't be misparsed as a flag.
    args = ["comment", "--author", "card-drop", "--", card_id, text]
    try:
        proc = _run_hermes_kanban(args)
    except subprocess.TimeoutExpired:
        return 504, {"error": "kanban comment timed out"}

    if proc.returncode != 0:
        LOG.error("kanban comment failed rc=%s stderr=%s", proc.returncode, proc.stderr[-500:])
        detail = proc.stderr.strip()[-300:]
        # Distinguish an unknown card (client error) from a real server fault.
        # `hermes kanban comment` on a missing id surfaces the CLI's
        # `kanban: unknown task <id>` (from add_comment's ValueError), so match
        # that alongside the generic phrasings.
        low = detail.lower()
        if "unknown task" in low or "no such task" in low or "not found" in low or "no such" in low:
            return 404, {"error": "card not found", "card_id": card_id}
        return 502, {"error": "kanban comment failed", "detail": detail}

    return 200, {"id": card_id, "commented": True}


class Handler(BaseHTTPRequestHandler):
    server_version = "kanban-receiver/1.0"

    # --- helpers -----------------------------------------------------------
    def _send_json(self, status: int, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        """Fail CLOSED: unset secret -> refuse; set secret -> constant-time match."""
        secret = _secret()
        if not secret:
            LOG.warning("write refused: no secret configured (fail-closed)")
            return False
        provided = self.headers.get("X-Cron-Secret", "")
        return hmac.compare_digest(provided, secret)

    def _read_json_body(self) -> Optional[dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002,A003
        LOG.info("%s - %s", self.address_string(), format % args)

    # --- routes ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/kanban/health"):
            self._send_json(200, {"ok": True, "ts": int(time.time())})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.rstrip("/")
        if route not in ("/kanban/card-drop", "/kanban/comment"):
            self._send_json(404, {"error": "not found"})
            return

        # Auth first; a bad/missing secret never reaches the board.
        if not self._authorized():
            self._send_json(403, {"error": "forbidden"})
            return

        payload = self._read_json_body()
        if payload is None:
            self._send_json(400, {"error": "invalid or oversized JSON body"})
            return

        if route == "/kanban/card-drop":
            status, obj = create_card(payload)
        else:
            status, obj = comment_card(payload)
        self._send_json(status, obj)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    port = int(os.environ.get("KANBAN_RECEIVER_PORT", DEFAULT_PORT))
    # Bind loopback only: the public path is the cloudflared tunnel, which
    # forwards to localhost. Never bind 0.0.0.0 (would expose the port on the
    # LAN, bypassing the tunnel's edge).
    host = os.environ.get("KANBAN_RECEIVER_HOST", "127.0.0.1")

    if not _secret():
        LOG.warning(
            "starting WITHOUT a secret configured -- ALL writes will be refused "
            "(403) until KANBAN_RECEIVER_SECRET or CRON_SECRET is set."
        )

    httpd = ThreadingHTTPServer((host, port), Handler)
    LOG.info("kanban receiver listening on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

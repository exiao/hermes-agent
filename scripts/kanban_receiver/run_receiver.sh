#!/bin/bash
# Launcher for the Kanban card-drop receiver launchd service.
#
# Sources the receiver secret from ~/.hermes/.env (never hardcode it in the
# plist, which lands in a world-readable LaunchAgents dir), then execs the
# stdlib-only receiver. Keeps HERMES_HOME so `hermes kanban` targets the real
# board.
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
export HERMES_HOME

REPO_ROOT="${KANBAN_RECEIVER_REPO:-$HOME/projects/hermes-agent}"

# Pull KANBAN_RECEIVER_SECRET / CRON_SECRET (and optional PORT) from .env
# without exporting the entire secret file into the process env. Only the keys
# the receiver reads are lifted.
ENV_FILE="$HERMES_HOME/.env"
if [[ -f "$ENV_FILE" ]]; then
  for key in KANBAN_RECEIVER_SECRET CRON_SECRET KANBAN_RECEIVER_PORT HERMES_BIN; do
    line="$(grep -E "^${key}=" "$ENV_FILE" | tail -1 || true)"
    if [[ -n "$line" ]]; then
      val="${line#*=}"
      # strip surrounding single/double quotes if present
      val="${val%\"}"; val="${val#\"}"
      val="${val%\'}"; val="${val#\'}"
      export "${key}=${val}"
    fi
  done
fi

# Prefer the repo venv (a source checkout may install `hermes` only there),
# then the installed `hermes` on PATH; fall back to the repo module.
# Prepending the venv bin lets _hermes_bin()'s `which hermes` resolve it and
# runs kanban_receiver.py under the venv interpreter so CLI imports succeed.
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
PY=""
for venv in "$REPO_ROOT/.venv" "$REPO_ROOT/venv"; do
  if [[ -x "$venv/bin/python3" ]]; then
    export PATH="$venv/bin:$PATH"
    PY="$venv/bin/python3"
    break
  fi
done

cd "$REPO_ROOT"
if [[ -n "$PY" ]]; then
  exec "$PY" scripts/kanban_receiver/kanban_receiver.py
fi
exec /usr/bin/env python3 scripts/kanban_receiver/kanban_receiver.py

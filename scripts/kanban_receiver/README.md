# Kanban card-drop receiver

The single inbound HTTP write surface connecting off-box producers (the Render
CPE website / diligence inbox, Modal cron producers) to this machine's local
Kanban board (`$HERMES_HOME/kanban.db`). Remote tiers can't write the local
SQLite board directly, so they POST here.

See the plans:
- `~/.hermes/plans/cpe-chat-route-to-research-lead.md` (Card A)
- `~/.hermes/plans/diligence-inbox-production.md` (section A+)

## Why standalone (not gateway/platforms/api_server.py)

`api_server.py` is a large OpenAI-compatible platform adapter whose routes are
baked into one hardcoded registration block with no auth'd custom-route hook.
Extending it risks the always-on multiplex gateway port (secondary profiles must
not bind ports → crash-loop risk). This receiver is a tiny **stdlib-only**
launchd daemon on its own loopback port (`8646`), far smaller blast radius, and
boots cleanly under a bare launchd context. It shells out to the installed
`hermes kanban` CLI — the same validated create/comment path a human uses.

## Endpoints

| Method | Path                  | Auth | Body |
|--------|-----------------------|------|------|
| GET    | `/health`             | none | — |
| POST   | `/kanban/card-drop`   | yes  | `{assignee, title, body, dedupe_key?, priority?, goal?, goal_max_turns?}` → `{id}` |
| POST   | `/kanban/comment`     | yes  | `{card_id, text}` → `{id, commented: true}` |

- `goal: true` (+ optional `goal_max_turns: N`) passes `--goal --goal-max-turns N`
  so the card runs as a Ralph-style goal loop.
- `dedupe_key` maps to `--idempotency-key`: a repeat drop with the same key
  collapses onto the existing card and returns its id (no duplicate).
- `assignee` must be one of the known lanes (`ALLOWED_ASSIGNEES` in the source).

## Auth: fail CLOSED

`X-Cron-Secret` header, compared with `hmac.compare_digest` against
`KANBAN_RECEIVER_SECRET` (preferred) or `CRON_SECRET`.

**When no secret is configured, the service refuses ALL writes with 403.** This
is the deliberate inversion of research-agent's `_check_cron_auth` (which
fails *open* for local testing). Because the receiver is reachable on the public
hostname `kanban.getbloom.app`, the secret gate IS the security boundary — a
fail-open default would let anyone who finds the URL spawn lanes. `/health` is
unauthenticated (liveness only).

## Config (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `KANBAN_RECEIVER_SECRET` / `CRON_SECRET` | *(unset → 403 all writes)* | shared secret |
| `KANBAN_RECEIVER_PORT` | `8646` | listen port |
| `KANBAN_RECEIVER_HOST` | `127.0.0.1` | bind addr (loopback only; never 0.0.0.0) |
| `HERMES_HOME` | `~/.hermes` | which board `hermes kanban` targets |
| `HERMES_BIN` | *(PATH lookup)* | explicit `hermes` binary override |

## Public reachability

Reuse the existing `hermes-webhooks` cloudflared tunnel (already serves
`webhooks.getbloom.app` + `proxy.getbloom.app`). Add ONE ingress rule to
`~/.cloudflared/config.yml`:

```yaml
  - hostname: kanban.getbloom.app
    service: http://localhost:8646
```

then `cloudflared tunnel route dns hermes-webhooks kanban.getbloom.app` and
restart the tunnel (`kill <cloudflared-pid>`; KeepAlive respawns and re-reads
config). Do NOT create a new tunnel. Render/Modal env:
`KANBAN_CARD_DROP_URL=https://kanban.getbloom.app/kanban/card-drop`.

## Install the launchd service

```bash
cp scripts/kanban_receiver/ai.hermes.kanban-receiver.plist ~/Library/LaunchAgents/
sed -i '' "s|__HOME__|$HOME|g" ~/Library/LaunchAgents/ai.hermes.kanban-receiver.plist
launchctl load ~/Library/LaunchAgents/ai.hermes.kanban-receiver.plist
# verify
curl -s http://127.0.0.1:8646/health
```

Logs: `/tmp/hermes/kanban-receiver-stdout.log` + `-stderr.log`.

## Test

```bash
python3 -m pytest scripts/kanban_receiver/test_kanban_receiver.py -q
```

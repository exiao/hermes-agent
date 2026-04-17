# Fork workflow

This is a personal fork of `NousResearch/hermes-agent`. Kept here so code changes
live as commits, not as MD patch notes scattered in `~/.hermes/plans/`.

## Layout

```
origin    → https://github.com/exiao/hermes-agent.git       (fetch + push)
upstream  → https://github.com/NousResearch/hermes-agent.git (fetch only;
                                                              push URL set to
                                                              DISABLED_NEVER_PUSH_UPSTREAM)
```

## Branches

| Branch | Role |
|---|---|
| `main` | Pristine mirror of `upstream/main`. Never commit here directly. |
| `live-config` | Long-lived integration branch. All local-only changes documented in `~/.hermes/plans/hermes-patches/*.md` converge here as real commits. This is what the running gateway checks out. |
| `feat/*`, `fix/*` | Short-lived feature/fix branches off `live-config`. Merge back via PR-to-self on GitHub (keeps history reviewable). |

### Commit ↔ patch-note mapping (as of 2026-04-17)

Every commit on `live-config` should reference its patch note in the body:

| Commit | Patch note |
|---|---|
| `feat(tools): add send_file tool …` | `send-file-attachment.md` |
| `feat(gateway): no-edit-mode streaming …` | `signal-streaming-patch.md`, `signal-streaming-split-patch.md` |
| `feat(gateway/signal): native formatting …` | `signal-reply-patch.md`, `signal-markdown-strip-patch.md`, `signal-italic-*.md` |
| `fix(cli): runtime provider inherits …` | `proxy-fix.md` |
| `fix(tools): strip temperature=0.1 …` | `tool-temperature-removed.md` |
| `fix(tools): isolate tool imports …` | `resilient-tool-discovery.md` |
| `feat(toolsets): register send_file …` | `send-file-toolset-registration.md` |

Notes that stay as MD (not code in this repo):
- `env-masked-display-writeback-bug.md` — postmortem, no code fix
- `proxy-watchdog-stale-pid-fix.md` — targets `~/.hermes/watchdog/`, outside repo
- `sentry-mcp-to-skill-migration.md` — config + skill files, outside repo
- `pre-upgrade-snapshot-*.md`, `upstream-pr-plan-*.md` — planning meta-docs

## Daily rules

1. **Never push to `upstream`.** The push URL is physically broken; if you see it
   work, something was misconfigured — stop and restore it.
2. **Never commit directly to `main`.** It exists only to track upstream.
3. **Work happens on feature branches**, merged into `live-config` via GitHub PR.
4. **`git stash push --include-untracked`** before branch ops. Never
   `git checkout -f` or `reset --hard` without stashing first (per project
   AGENTS.md rule).
5. **Config stays out.** `config.yaml`, `.env`, credentials — all live in
   `~/.hermes/` outside this checkout. The fork is pure code.
6. **Every commit on `live-config` references a patch note** in `~/.hermes/plans/hermes-patches/`.
   If you're about to commit something without one, write the note first — it
   doubles as the commit body and future-you's reapply guide.

## Syncing with upstream

```bash
git fetch upstream
git checkout main
git merge --ff-only upstream/main        # must fast-forward; if not, something
                                         # committed to main directly — bad
git push origin main

git checkout live-config
git rebase main                          # or merge, your call — rebase keeps
                                         # live-config's diff minimal
# resolve any conflicts, then:
git push --force-with-lease origin live-config
```

Do this weekly, or when an upstream fix is needed.

## Starting a feature

```bash
git checkout live-config
git pull
git checkout -b fix/some-descriptive-name
# ... work, commit, push ...
git push -u origin fix/some-descriptive-name
# open PR on GitHub against exiao/hermes-agent `live-config` branch
```

## Upstreaming something back

If a fix is general-purpose and worth contributing back to `NousResearch`:

```bash
git checkout -b upstream/fix-name main   # branch from upstream main, not live-config
git cherry-pick <commit-sha-from-live-config>
git push origin upstream/fix-name
# open PR on GitHub from exiao/hermes-agent:upstream/fix-name → NousResearch/hermes-agent:main
```

## Why this instead of MD patch notes alone

MD patch notes in `~/.hermes/plans/hermes-patches/` describe code behavior in
prose. On their own they:

- don't apply themselves
- rot silently when upstream changes
- can't be tested
- can't be reviewed as diffs
- can't be rolled back atomically

Code belongs in git. Patch notes stay — but now as **reapply guides paired
with a real commit on `live-config`**. The commit is the source of truth;
the MD is the human-readable "why" next to it.

"""Dangerous command approval -- the gate flow and per-session state.

Owns the session state (approvals, yolo, gateway queues, denial breaker), the three guard
entry points (``check_all_command_guards``, ``check_execute_code_guard``,
``request_tool_approval`` / ``_run_approval_gate``) and the shared human-decision engine
behind them. Leaves: ``approval_detection`` (hardline/dangerous patterns), ``approval_context``
(contextvars, config readers), ``approval_floors`` (pre-gate blocks, allowlist match),
``approval_prompt`` (CLI prompt, plugin transports, MCP elicitation), ``approval_gateway_wait``
(blocking gateway round-trip), ``approval_smart`` (guardian LLM), ``approval_human_wait``.
Leaves read facade-owned state (``_lock``, queues, denial breaker) back through ``tools.approval`` at
call time; sibling-defined names are imported from their defining module.
"""

from dataclasses import dataclass
import hashlib
import importlib
import logging
import os
import re
import shlex
import threading
from typing import Optional

from utils import env_var_enabled, is_truthy_value
from tools import approval_context
from tools import approval_detection as _approval_detection
from tools.approval_context import (
    _get_session_platform, _is_cron_approval_context,
    _is_gateway_approval_context, _is_interactive_cli, _is_single_query_approval_context,
    _is_unattended_platform_approval_context, _resolve_cli_approval_callback, _should_fall_through_to_cli_approval,
    _tirith_fail_open, get_current_session_key,
)
from tools.approval_detection import (
    _approval_key_aliases, _check_sudo_stdin_guard, detect_dangerous_command, detect_hardline_command,
)
from tools.approval_floors import (
    _command_matches_permanent_allowlist, _hardline_block_result, _match_user_deny_rule, _sudo_stdin_block_result,
    _user_deny_block_result,
)
from tools.approval_gateway_wait import _await_gateway_decision
from tools.approval_prompt import (
    _present_with_selected_transport, _transport_choice,
    prompt_dangerous_approval as _prompt_dangerous_approval)
from tools.approval_smart import _smart_verdict

logger = logging.getLogger(__name__)

_get_approval_config = approval_context._get_approval_config
_get_approval_mode = approval_context._get_approval_mode
_get_cron_approval_mode = approval_context._get_cron_approval_mode
_get_single_query_approval_mode = approval_context._get_single_query_approval_mode
_get_unattended_approval_mode = approval_context._get_unattended_approval_mode
_APPROVAL_SECRET_RE = re.compile(
    r"\b(?:sk|rk|pk|xox[baprs]|gh[pousr]|github_pat|hf|api)[-_][A-Za-z0-9_./+=-]{12,}\b")


def _redact_for_approval(text: str) -> str:
    from agent.redact import redact_sensitive_text
    return _APPROVAL_SECRET_RE.sub("***", redact_sensitive_text(text or ""))


def prompt_dangerous_approval(command: str, description: str, **kwargs):
    """Display-only redaction wrapper around the upstream prompt transport."""
    return _prompt_dangerous_approval(
        _redact_for_approval(command), _redact_for_approval(description), **kwargs)
_REDIRECT_TOKEN_RE = re.compile(
    r'\s*\d*(?:'
    # fd-dup (`2>&1`, `>&2`, `2>&-`, `2<&0`): operator + `&` + optional fd/`-`.
    # Stripped wholesale so the trailing `&` is not mistaken for a clause sep
    # (`kill 2>&1 $(pgrep -f hermes)` must still reach the pgrep target).
    r'[<>]&\d*-?'
    # redirect-to-target (`> /tmp/gateway.log`, `2>>x`, `&>x`): operator + path.
    r'|(?:&>>?|>>?|<)\s*[^\s;&|<>]+'
    r')'
)

# Single-character regex character classes used to obfuscate a pgrep target,
# e.g. `pgrep -f 'h[e]rmes-gatewa[y]'`. pgrep -f treats its argument as a regex,
# so `h[e]rmes` still matches the `hermes` process. Collapse `[x]` -> `x` (but
# not negated classes `[^x]`) on the self-termination scan view so obfuscated
# self-targets are still detected. Applied ONLY to the self-term scan.
_SINGLE_CHAR_CLASS_RE = re.compile(r'\[([^\]^])\]')

# Descriptions of the self-termination patterns that need redirect-stripping.
_SELF_TERM_DESCRIPTIONS = frozenset({
    "kill hermes/gateway process (self-termination)",
    "kill process via pgrep/pidof expansion (self-termination)",
    "kill process via backtick pgrep/pidof expansion (self-termination)",
})


def _build_self_term_scan_view(command_lower: str) -> str:
    """Build the command view used ONLY for self-termination pattern matching.

    Three transforms, in order:
      1. Strip redirect tokens (`> /tmp/gateway.log`, fd-dups like `2>&1`) so a
         redirect target with a self keyword is not a false positive, while a
         real kill target after a leading redirect is still reached.
      2. Neutralize clause separators (`;`, `&`, `|`) that appear INSIDE quotes:
         in `pkill -f "api-gateway|hermes-gateway"` the `|` is part of pkill's
         regex argument, not a shell pipeline, so it must not end the clause.
      3. Collapse single-char regex classes (`h[e]rmes` -> `hermes`): pgrep -f
         treats its argument as a regex, so `[e]` is an obfuscated `e`.
    """
    view = _REDIRECT_TOKEN_RE.sub(' ', command_lower)

    # Neutralize `;&|` inside single- or double-quoted spans (replace with space).
    def _scrub_quoted(m: re.Match) -> str:
        return m.group(0).translate({ord(';'): ' ', ord('&'): ' ', ord('|'): ' '})

    view = re.sub(r'"[^"]*"|\'[^\']*\'', _scrub_quoted, view)

    # Collapse `[x]` -> `x` (positive single-char classes only).
    view = _SINGLE_CHAR_CLASS_RE.sub(r'\1', view)
    return view


# =========================================================================
# Force-push carve-out (mirror the shell git-guard at ~/.local/bin/git)
# =========================================================================
# The shell PATH wrapper (~/.local/bin/git) already encodes the safe rule:
# force-pushing a FEATURE branch (rebasing your own PR) is allowed; it only
# blocks a force-push that could rewrite the default branch (main/master) —
# either via a main/master-targeting refspec or a bare force while standing on
# a main/master checkout. A force-WITH-LEASE to a feature branch sails through.
#
# The three force-push DANGEROUS_PATTERNS entries below do not make that
# distinction — they flag EVERY force push for an operator prompt, which stalls
# routine PR-rebase work headless. This carve-out narrows them to match the
# shell guard: a force-WITH-LEASE push whose target ref is NOT the default
# branch is auto-allowed; everything else (a bare force, any force whose refspec
# targets main/master) still prompts. The lease is the safety belt — it refuses
# if the remote moved since the last fetch, so even on a shared worktree branch
# the second pusher's lease bounces instead of clobbering. A bare force is NOT
# covered (no lease = no safety belt), so it keeps prompting.
#
# The main/master refspec forms here are the SAME ones the shell guard
# enumerates (lines ~49-62 of ~/.local/bin/git) so the two layers cannot
# disagree. The main/master push patterns in DANGEROUS_PATTERNS stay active as
# an independent backstop.
_FORCE_PUSH_DESCRIPTIONS = frozenset({
    "git force push (rewrites remote history)",
    "git force push short flag (rewrites remote history)",
    "git force-with-lease push (rewrites remote history)",
})

# A bare --force flag (NOT the --force-with-lease form). `--force-with-lease`
# contains the substring `--force`, so require a word boundary that the lease
# suffix does not satisfy: --force/--force=… followed by end or a non-hyphen.
_BARE_FORCE_FLAG_RE = re.compile(r'--force(?![\w-])')
# Short force flag: -f, or packed combos like -uf / -fv (a flag token that
# contains an `f`). Excludes long flags (already handled above). The other
# chars in the bundle may be letters OR digits — git push has numeric short
# flags `-4`/`-6` (IPv4/IPv6), so a packed `-4f` is IPv4 + a BARE force (which
# overrides the lease with no safety belt). Matching `[a-z0-9]` around `f`
# ensures `-4f`/`-6f` are caught as a bare force instead of slipping through.
_SHORT_FORCE_FLAG_RE = re.compile(r'(?<![\w-])-[a-z0-9]*f[a-z0-9]*(?![\w-])')
# Exclude a trailing hyphen too (`(?![\w-])`) so a different flag that merely
# starts with this string (e.g. `--force-with-lease-foo`) is not misread as the
# lease form — matches the boundary convention used by the bare-force regexes.
_FORCE_WITH_LEASE_RE = re.compile(r'--force-with-lease(?:=\S*)?(?![\w-])')
# Refspec/target forms that would rewrite the default branch — mirrors the
# shell guard's main/master enumeration (`+main`, `HEAD:main`, `*:main`,
# `main:main`, a bare `main`/`master` push target, the `+`/colon force forms),
# AND the fully-qualified `refs/heads/main` / `HEAD:refs/heads/main` forms.
# Anchor `main`/`master` on a preceding refspec-boundary char (`/`, `:`, `+`,
# or whitespace / start) so a `heads/`-prefixed or colon-RHS-qualified ref is
# caught, while a branch that merely CONTAINS the word (`mainline`, `my-main`,
# `main-event`) is not — the trailing `(?:\s|$)` keeps those un-gated.
_DEFAULT_BRANCH_PUSH_RE = re.compile(
    r'(?:^|[/:+\s])(?:main|master)(?:\s|$|[\'"`])'
)

# Long-lived integration branches the running gateway / deploy flow tracks
# directly. Auto-approving a headless leased force-push to one of these would
# bypass the PR-to-self review flow (AGENTS.md: `live-config` is the branch the
# live gateway checks out), so they are NEVER carved out — same posture as
# main/master. Lowercased; matched against the parsed destination branch name.
_PROTECTED_BRANCHES = frozenset({"main", "master", "live-config"})

# Push flags that consume the NEXT token as a value (space-separated form). If
# we don't drop the value too it survives token-splitting and gets miscounted
# as the remote or a refspec (`-o ci.skip origin` → `ci.skip` read as remote,
# `origin` read as a refspec). The `--flag=value` forms are already handled by
# the generic flag-strip, so only the space-separated short/long names matter.
# Only genuinely value-taking flags belong here — boolean flags like
# `--force-if-includes` must NOT be listed or they would wrongly eat the next
# positional (the remote/refspec).
_VALUE_FLAGS = frozenset({
    "-o", "--push-option", "--repo", "--receive-pack", "--exec",
    # `--recurse-submodules <check|on-demand|no>` consumes the next word in its
    # space form; if we don't drop the value it survives token-splitting and
    # gets miscounted as the remote/refspec (`--recurse-submodules on-demand
    # origin` → `origin` read as a refspec).
    "--recurse-submodules",
})

# A destination that names exactly one ordinary branch under refs/heads — the
# only shape we auto-approve. Accepts a bare `feature`, `refs/heads/feature`, or
# the `src:dst` form (we validate the RHS destination). Rejects wildcards (`*`),
# tag/other-namespace refs (`refs/tags/…`, `refs/notes/…`), the `HEAD`
# shorthand, the bare `:` matching-refspec, and empty destinations.
_VALID_BRANCH_NAME_RE = re.compile(r'^[\w./-]+$')


def _refspec_destination(refspec: str) -> Optional[str]:
    """Return the lowercased destination BRANCH NAME a push refspec writes to,
    or ``None`` if the refspec is not a single, ordinary ``refs/heads`` branch
    target this carve-out is willing to reason about.

    Handles `src:dst` (uses the RHS), bare `branch` / `refs/heads/branch`, and
    rejects: wildcards (`*`), `HEAD` (resolves to the current checkout — may be
    main), the bare `:` matching-refspec, an empty RHS (`src:` = delete), an
    empty SOURCE (`:dst` = the colon-prefix delete shorthand), and any ref
    outside `refs/heads/` (`refs/tags/…`, `refs/notes/…`, etc. — a leased tag
    rewrite must NOT slip through).
    """
    rs = refspec.strip()
    # Strip surrounding shell quotes the detection view may still carry (the
    # normalizer only removes EMPTY '' / "" literals). A quoted destination
    # like HEAD:'main' or 'feature' must validate on its bare branch name so a
    # quoted protected branch is still caught and a quoted feature branch still
    # carves out.
    rs = rs.strip('\'"`')
    if not rs or '*' in rs:
        return None
    # `src:dst` — the written ref is the RHS. A bare `:` (matching refspec) or a
    # `src:` (delete) leaves an empty dst → reject. An empty SOURCE (`:dst`) is
    # the colon-prefix DELETE shorthand (`git push origin :feature` removes the
    # remote ref) — just as destructive as `--delete`, so reject it too.
    #
    # In colon form, require the RHS destination to be fully qualified as
    # `refs/heads/<branch>`. Git resolves an unqualified RHS against the remote
    # namespace, so `HEAD:v1` can update an existing remote tag `refs/tags/v1`;
    # without remote ref resolution, the branch-only carve-out cannot safely
    # accept unqualified colon destinations. Bare source-only pushes like
    # `git push origin feature` are handled by the `else` branch and remain the
    # ordinary feature-branch convenience this carve-out exists to allow.
    colon_form = ':' in rs
    if colon_form:
        src, dst = rs.split(':', 1)
        if not src.strip('\'"`'):
            return None
    else:
        dst = rs
    dst = dst.strip('\'"`')
    if not dst:
        return None
    # `HEAD` resolves to whatever branch is checked out — could be main.
    if dst == 'head':
        return None
    # Qualified refs: only refs/heads/<branch> is a branch push; refs/tags/…,
    # refs/notes/…, refs/remotes/… and any other namespace are not carved out.
    # In colon form, an unqualified destination is also not carved out because
    # Git may expand it to an existing non-branch remote ref.
    if colon_form and not dst.startswith('refs/heads/'):
        return None
    if dst.startswith('refs/'):
        if not dst.startswith('refs/heads/'):
            return None
        dst = dst[len('refs/heads/'):]
    if not dst or not _VALID_BRANCH_NAME_RE.match(dst):
        return None
    return dst


def _is_safe_lease_push_to_feature_branch(command_lower: str) -> bool:
    """True iff *command_lower* is a force-WITH-LEASE git push to a single,
    ordinary, NON-protected branch — the exact case the shell git-guard
    auto-allows.

    *command_lower* is the already-normalized + lowercased command (same view
    the dangerous-pattern loop scans). Returns False (→ keep prompting) for:
      - anything that is not a `git push`,
      - a push with no `--force-with-lease` (a bare `--force`/`-f` has no lease
        safety belt, so it stays gated),
      - a push that ALSO carries a bare `--force`/`-f` (the bare force could
        still clobber regardless of the lease intent),
      - a `--all`/`--mirror` push (pushes EVERY local ref, incl. main/master —
        a single explicit destination ref cannot be reasoned about),
      - a push with NO explicit destination refspec (an omitted refspec follows
        push.default and may push the current branch — which could be main),
      - a refspec carrying a leading `+` (git documents `+<src>:<dst>` as the
        same forced update as `--force`, with NO lease safety belt),
      - a destination that is not a single `refs/heads/<branch>` target: the
        `HEAD` shorthand, the bare `:` matching-refspec, a wildcard
        (`refs/heads/*:refs/heads/*`), or a non-branch ref (`refs/tags/…`),
      - a push to a PROTECTED branch (main/master/live-config), in any form
        incl. `refs/heads/main` / `HEAD:refs/heads/main`.
    """
    if not re.search(r'\bgit\s+push\b', command_lower):
        return False
    # Require the lease form — the safety belt that makes auto-approve safe.
    if not _FORCE_WITH_LEASE_RE.search(command_lower):
        return False
    # Reject if a BARE force is also present (e.g. `--force --force-with-lease`
    # or `-f`). Strip the lease token first so its `--force` substring and any
    # `=<expected>` value don't read as a bare force.
    without_lease = _FORCE_WITH_LEASE_RE.sub(' ', command_lower)
    if _BARE_FORCE_FLAG_RE.search(without_lease) or _SHORT_FORCE_FLAG_RE.search(without_lease):
        return False
    # Reject a broadcast push: `--all`/`--mirror` push every local ref (incl.
    # main/master), so a "no main token in the args" check cannot vouch for them.
    # `--tags` (no short form) ALSO pushes tags alongside the named refspec — a
    # leased `--force-with-lease=refs/tags/v1:<old>` can rewrite a tag while the
    # visible refspec looks like a routine feature push, so the branch-only
    # carve-out must keep prompting for it.
    if re.search(r'(?<![\w-])--(?:all|mirror|tags)(?![\w-])', without_lease):
        return False
    # Reject a DELETE push (`--delete` / `-d`): it removes the remote ref
    # entirely (just as destructive as a force rewrite), and the flag-strip
    # below would erase the `--delete` token and leave the branch reading like a
    # routine rebase target. A deletion is never the carve-out's "rebase a
    # feature branch" intent, so keep prompting. The packed short form may carry
    # DIGITS too (`-4d` = IPv4 + delete), so match `[a-z0-9]` around `d` — same
    # reasoning as the packed numeric force bundle above.
    if re.search(r'(?<![\w-])(?:--delete|-[a-z0-9]*d[a-z0-9]*)(?![\w-])', without_lease):
        return False
    # Require an EXPLICIT destination refspec. An omitted refspec resolves via
    # push.default and may push the current branch (which could be main), so a
    # bare `git push --force-with-lease [origin]` must keep prompting.
    #
    # Strip EVERYTHING up to and including the `git push` verb so a shell prefix
    # (`cd repo && git push …`) does not survive as a phantom remote/refspec.
    args = re.sub(r'^.*?\bgit\s+push\b', ' ', without_lease)
    # Tokenize with shlex so a QUOTED value carrying whitespace (`--push-option
    # 'ci skip'`) collapses to a single token instead of leaking its tail
    # (`skip`) as a phantom remote/refspec. Fall back to a plain split if the
    # command has unbalanced quotes shlex can't parse (keep prompting on the
    # safe side rather than raising).
    try:
        tokens = shlex.split(args)
    except ValueError:
        return False
    # Drop value-flags AND the token they consume (`-o ci.skip`,
    # `--push-option 'ci skip'`, `--repo url`). The `--flag=value` form is one
    # token already, handled by the bare-flag strip below.
    pruned: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in _VALUE_FLAGS:
            skip_next = True  # consume this flag's value token
            continue
        pruned.append(tok)
    # Drop the remaining standalone `--flag` / `--flag=value` / `-f` flag tokens.
    # A short flag's leading char may be a letter (`--repo`, `-q`) OR a digit —
    # git push documents numeric short flags `-4`/`-6` (IPv4/IPv6); without the
    # digit they'd survive and be miscounted as a refspec. A remote/refspec
    # never starts with a dash, so this only drops flags. Also drop the bare `--`
    # option terminator: git does not treat it as a refspec (`git push origin --`
    # is an omitted-refspec push), so it must not count toward the explicit-
    # destination check below.
    tokens = [
        t for t in pruned if t != '--' and not re.match(r'--?[a-z0-9]', t)
    ]
    # First surviving token is the remote; a refspec must follow it.
    if len(tokens) < 2:
        return False
    refspecs = tokens[1:]
    # Reject the `git push <remote> ... tag <name>` shorthand: git documents
    # `tag <tag>` as sugar for `refs/tags/<tag>:refs/tags/<tag>`, and it may
    # appear at ANY refspec position (`origin tag v1`, `origin feature tag v1`),
    # so a leased `--force-with-lease=refs/tags/v1:<old> origin feature tag v1`
    # force-updates a TAG while the `tag`/`v1` tokens look like ordinary branch
    # refspecs. The shorthand is the literal `tag` keyword FOLLOWED by a name,
    # so reject whenever a `tag` token has another token after it (a lone
    # trailing `tag` is an ordinary branch named "tag", validated below).
    if any(rs == 'tag' and i + 1 < len(refspecs) for i, rs in enumerate(refspecs)):
        return False
    # Reject a leading-`+` refspec (forced update, no lease guarantee).
    if any(rs.startswith('+') or ':+' in rs for rs in refspecs):
        return False
    # Every refspec must name a single ordinary branch under refs/heads, and
    # NONE of them may target a protected branch. A refspec we can't parse to a
    # clean branch destination (tag ref, HEAD, `:`, wildcard) fails the push.
    for rs in refspecs:
        dst = _refspec_destination(rs)
        if dst is None:
            return False
        if dst in _PROTECTED_BRANCHES:
            return False
    # Final backstop: the main/master refspec patterns (catches forms the parse
    # above might normalize differently). Limit this to the parsed push refspecs
    # so an unrelated shell prefix/path like `cd main && git push ... feature`
    # does not defeat the safe feature-branch carve-out.
    if _DEFAULT_BRANCH_PUSH_RE.search(' '.join(refspecs)):
        return False
    return True



_SELF_PROC_TOKEN = (
    r'(?:(?<![\w-])hermes(?:-[\w.-]*)?(?![\w])|'
    r'(?<![\w-])hermes_cli(?:[\w.]*)?|(?<![\w-])gateway(?![\w-])|cli\.py)')
_STRICT_SELF_TERM_PATTERNS = (
    (re.compile(r'\b(?:pkill|killall)\b[^;&|]*' + _SELF_PROC_TOKEN, re.I | re.S),
     "kill hermes/gateway process (self-termination)"),
    (re.compile(r'\bkill\b[^;&|]*\$\(\s*(?:pgrep|pidof)\b[^)]*' + _SELF_PROC_TOKEN, re.I | re.S),
     "kill process via pgrep/pidof expansion (self-termination)"),
    (re.compile(r'\bkill\b[^;&|]*`\s*(?:pgrep|pidof)\b[^`]*' + _SELF_PROC_TOKEN, re.I | re.S),
     "kill process via backtick pgrep/pidof expansion (self-termination)"),
)
_FORK_EXTRA_PATTERNS = tuple((re.compile(pattern, re.I | re.S), description) for pattern, description in (
    (r'\bgit\s+push\b\s+\S+\s+\S*[/:+\s](?:main|master)(?:\s|$|[\'"`])',
     "git push to main/master (bypasses PR review)"),
    (r'\bgit\s+push\b\s+\S+\s+(?:main|master)(?:\s|$|[\'"`])',
     "git push to main/master (bypasses PR review)"),
    (r'\bgh\s+pr\s+merge\b', "gh pr merge (bypasses PR review — Eric merges)"),
    (r'\bgh\s+(?:release|repo|workflow)\s+delete\b', "gh delete destructive resource"),
    (r'\bgh\s+repo\s+archive\b', "gh repo archive (near-permanent change)"),
    (r'\bgh\s+api\b.*(?:-X|--method)\s+DELETE\b', "gh api raw DELETE call"),
))


def detect_dangerous_command(command: str) -> tuple:
    """Upstream detector plus fork self-process precision and protected-ref policy."""
    ad = _approval_detection
    if ad._command_parser_limit_exceeded(command):
        return True, ad._PARSER_LIMIT_DESCRIPTION, ad._PARSER_LIMIT_DESCRIPTION
    try:
        argv = shlex.split(command, posix=True)
        if len(argv) == 3 and argv[:2] == ["rm", "-f"]:
            import tempfile
            temp_dir = os.path.realpath(tempfile.gettempdir())
            basename = os.path.basename(argv[2])
            raw_temp = tempfile.gettempdir()
            expected = os.path.join(temp_dir, basename)
            raw_tmp_alias = os.path.join(raw_temp, basename) if raw_temp == "/tmp" else None
            target = os.path.realpath(argv[2])
            if (argv[2] in {expected, raw_tmp_alias}
                    and os.path.dirname(target) == temp_dir
                    and re.fullmatch(r"hermes-(?:verify|ad-hoc)-[A-Za-z0-9_.-]+", basename)):
                return False, None, None
    except ValueError:
        pass
    variants = list(ad._command_detection_variants(command))
    safe_lease = any(_is_safe_lease_push_to_feature_branch(value.lower()) for value in variants)
    for variant in variants:
        lowered = variant.lower()
        self_view = _build_self_term_scan_view(lowered)
        for pattern, description in ad.DANGEROUS_PATTERNS_COMPILED:
            if description in _SELF_TERM_DESCRIPTIONS:
                continue
            if safe_lease and description in _FORCE_PUSH_DESCRIPTIONS:
                continue
            if pattern.search(lowered):
                return True, description, description
        for pattern, description in _FORK_EXTRA_PATTERNS:
            if pattern.search(lowered):
                return True, description, description
        for pattern, description in _STRICT_SELF_TERM_PATTERNS:
            if pattern.search(self_view):
                return True, description, description
    normalized = ad._normalize_command_for_detection(command)
    for description, _ in ad._execution_flag_findings(normalized):
        return True, description, description
    if ad._is_shell_token_spliced_gateway_lifecycle(_build_self_term_scan_view(command)):
        desc = ad._GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION
        return True, desc, desc
    return False, None, None


# Frozen at import: reading os.environ per call would let any skill running in the process set
# this and bypass every approval check (prompt-injection escalation path).
_YOLO_MODE_FROZEN: bool = is_truthy_value(os.getenv("HERMES_YOLO_MODE", ""))


# --- Per-session approval state (thread-safe) -----------------------------------------------------------------------

_lock = threading.Lock()
_pending: dict[str, dict] = {}
_session_approved: dict[str, set] = {}
_session_yolo: set[str] = set()
_permanent_approved: set = set()

# --- Consecutive-denial circuit breaker for smart approvals ---------------------------------------------------------
# Each retry of a smart-denied command burns another guardian LLM call. After ``approvals.denial_breaker_threshold``
# consecutive guardian DENY verdicts in one session (default 3; 0 disables) the deny message escalates to a hard-stop
# instruction; any approval resets the tally. Only TOOL RESULT text changes — no history surgery, no interrupts — so
# it is prompt-cache-invariant. Capped so short-lived session keys cannot grow it without bound; oldest (least
# recently denied) entries are evicted.
_denial_tally: dict[str, int] = {}
_DENIAL_TALLY_MAX_SESSIONS = 256


def _get_denial_breaker_threshold() -> int:
    """``approvals.denial_breaker_threshold``: default 3; 0 or negative disables."""
    try:
        return int(approval_context._get_approval_config().get("denial_breaker_threshold", 3))
    except (ValueError, TypeError):
        return 3


def _record_denial(session_key: str) -> int:
    """Increment and return the session's consecutive guardian-denial count. Pop-and-reinsert
    keeps actively-denying sessions at the most-recent end so eviction drops idle keys."""
    with _lock:
        count = _denial_tally.pop(session_key, 0) + 1
        _denial_tally[session_key] = count
        while len(_denial_tally) > _DENIAL_TALLY_MAX_SESSIONS:
            _denial_tally.pop(next(iter(_denial_tally)))
        return count


def _reset_denials(session_key: str) -> None:
    """Clear the session's consecutive-denial tally (an approval happened)."""
    with _lock:
        _denial_tally.pop(session_key, None)


def _denial_breaker_addendum(session_key: str) -> str:
    """Escalated hard-stop text once the breaker has tripped, else ''. Read-only: callers
    increment via :func:`_record_denial`; the text is appended verbatim to the deny message."""
    with _lock:
        count = _denial_tally.get(session_key, 0)
    threshold = _get_denial_breaker_threshold()
    if threshold <= 0 or count < threshold:
        return ""
    # WARNING (was DEBUG): a failed/blocked guardian call is a real event the operator needs to see — the
    # whole point of #82846 is that the hang was invisible. Log the elapsed time and error class too.
    logger.warning(
        "Smart-approval circuit breaker tripped for session %s: %d consecutive denials (threshold %d)",
        session_key, count, threshold,
    )
    return (
        f" CIRCUIT BREAKER: {count} consecutive commands were blocked by "
        "the security reviewer. STOP attempting variations of this "
        "operation. Report the blocked operation to the user and either ask them to run it manually or use /approve."
    )

# --- Gateway approval queue (the blocking wait loop lives in approval_gateway_wait) ---------------------------------


# Optional free-text reason supplied with an explicit deny (``/deny <reason>``) so the agent can adapt
# instead of only hearing "denied". Ported from qwibitai/nanoclaw#2832.
_gateway_queues: dict[str, list] = {}        # session_key → [_ApprovalEntry, …]
_gateway_notify_cbs: dict[str, object] = {}  # session_key → callable(approval_data)


def register_gateway_notify(session_key: str, cb) -> None:
    """Register ``cb(approval_data: dict) -> None`` for sending approval requests. The callback
    bridges sync→async: it runs in the agent thread and must schedule the send on the loop."""
    with _lock:
        _gateway_notify_cbs[session_key] = cb


def unregister_gateway_notify(session_key: str) -> None:
    """Unregister the callback and wake ALL blocked threads for this session so
    they don't hang forever (agent run finished or interrupted)."""
    with _lock:
        _gateway_notify_cbs.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        entry.event.set()


def resolve_gateway_approval(session_key: str, choice: str,
                             resolve_all: bool = False,
                             reason: Optional[str] = None,
                             request_id: Optional[str] = None) -> int:
    """Unblock waiting agent thread(s) from the gateway's /approve or /deny handler.

    *resolve_all* resolves every pending approval (``/approve all``); otherwise the oldest
    (FIFO) or the one matching *request_id*. *reason* is the ``/deny <reason>`` free text,
    relayed to the agent in the BLOCKED message. Returns the number resolved.
    """
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return 0
        if request_id:
            targets = [entry for entry in queue if entry.data.get("request_id") == request_id]
            if not targets:
                return 0
            queue[:] = [entry for entry in queue if entry not in targets]
        elif resolve_all:
            targets = list(queue)
            queue.clear()
        else:
            targets = [queue.pop(0)]
        if not queue:
            _gateway_queues.pop(session_key, None)

    for entry in targets:
        entry.result = choice
        if reason:
            entry.reason = reason
        entry.event.set()
    return len(targets)


def list_gateway_approvals(session_key: str) -> list[dict]:
    """Return replay-safe snapshots of unresolved approvals for one session."""
    with _lock:
        return [dict(entry.data) for entry in _gateway_queues.get(session_key, [])]


def ack_gateway_approval(session_key: str, request_id: str) -> bool:
    """Record that a client received a particular pending approval request."""
    with _lock:
        for entry in _gateway_queues.get(session_key, []):
            if entry.data.get("request_id") == request_id:
                entry.acknowledged = True
                return True
    return False


def has_blocking_approval(session_key: str) -> bool:
    """Check if a session has one or more blocking gateway approvals waiting."""
    with _lock:
        return bool(_gateway_queues.get(session_key))


def get_pending_gateway_approval(session_key: str) -> dict | None:
    """Copy of the oldest unresolved gateway approval, for reconnecting clients
    to restore a prompt. Read-only snapshot — the queue stays authoritative."""
    if not session_key:
        return None
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return None
        return dict(queue[0].data)


def submit_pending(session_key: str, approval: dict):
    """Store a pending approval request for a session."""
    with _lock:
        _pending[session_key] = approval


def approve_session(session_key: str, pattern_key: str):
    """Approve a pattern for this session only."""
    with _lock:
        _session_approved.setdefault(session_key, set()).add(pattern_key)


def _release_permission_mode_dependents(session_key: str) -> None:
    """Drop resources whose immutable mode derives from Hermes YOLO. Lazy import so approval-only
    sessions never load computer-use; releasing on BOTH edges makes enabling YOLO replace a
    standard backend and disabling it revoke a private unrestricted daemon immediately."""
    try:
        from tools.computer_use.tool import release_computer_use_session

        release_computer_use_session(session_key)
    except Exception:
        logger.debug("Failed to release permission-mode dependent resources for %s", session_key, exc_info=True)


def _set_session_yolo(session_key: str, enabled: bool) -> None:
    if not session_key:
        return
    with _lock:
        (_session_yolo.add if enabled else _session_yolo.discard)(session_key)
    _release_permission_mode_dependents(session_key)


def enable_session_yolo(session_key: str) -> None:
    """Enable YOLO bypass for a single session key."""
    _set_session_yolo(session_key, True)


def disable_session_yolo(session_key: str) -> None:
    """Disable YOLO bypass for a single session key."""
    _set_session_yolo(session_key, False)


def clear_session(session_key: str) -> None:
    """Remove all approval and yolo state for a given session."""
    if not session_key:
        return
    with _lock:
        _session_approved.pop(session_key, None)
        _session_yolo.discard(session_key)
        _pending.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        # Cancel blocked waits now so the old run unwinds instead of idling until timeout.
        entry.result = "deny"
        entry.event.set()
    _release_permission_mode_dependents(session_key)
    # Session-persistent code kernels (local and remote) share this owner key and die at the same boundary so a
    # finished conversation cannot leak a live interpreter.
    for module, shutdown in (("tools.code_kernel", "shutdown_kernels_for_owner"),
                             ("tools.code_kernel_remote", "shutdown_remote_kernels_for_owner")):
        try:
            getattr(importlib.import_module(module), shutdown)(session_key)
        except Exception:
            pass


def is_session_yolo_enabled(session_key: str) -> bool:
    """Return True when YOLO bypass is enabled for a specific session."""
    if not session_key:
        return False
    with _lock:
        return session_key in _session_yolo


def is_current_session_yolo_enabled() -> bool:
    """Return True when the active approval session has YOLO bypass enabled."""
    return is_session_yolo_enabled(get_current_session_key(default=""))


def _yolo_active() -> bool:
    """CLI ``--yolo`` (process-scoped, frozen at import) or gateway ``/yolo``
    (session-scoped). Hardline / deny-rule floors run BEFORE this everywhere."""
    return _YOLO_MODE_FROZEN or is_current_session_yolo_enabled()


def is_approved(session_key: str, pattern_key: str) -> bool:
    """Session-scoped or permanent approval. Accepts the canonical key and the legacy
    regex-derived key so existing command_allowlist entries survive key migrations."""
    aliases = _approval_key_aliases(pattern_key)
    with _lock:
        approved = _permanent_approved | _session_approved.get(session_key, set())
    return any(alias in approved for alias in aliases)


def approve_permanent(pattern_key: str):
    """Add a pattern to the permanent allowlist."""
    with _lock:
        _permanent_approved.add(pattern_key)


def load_permanent(patterns: set):
    """Bulk-load permanent allowlist entries from config."""
    with _lock:
        _permanent_approved.update(patterns)


def _persist_choice(session_key: str, choice: str, warnings: list[tuple]) -> None:
    """Persist a human ``session``/``always`` choice for each ``(key, _, is_tirith)``. Tirith
    findings are session-max by design (no broad permanent allowlisting of content-level
    findings), so ``always`` downgrades them to session. ``once`` persists nothing."""
    for key, _, is_tirith in warnings:
        if choice not in ("session", "always"):
            continue
        approve_session(session_key, key)
        if choice == "always" and not is_tirith:
            approve_permanent(key)
            save_permanent_allowlist(_permanent_approved)


# --- Config persistence for permanent allowlist ---------------------------------------------------------------------

def load_permanent_allowlist() -> set:
    """Load ``command_allowlist`` from config and sync it into the approval state
    so is_approved() honors 'always' choices from previous sessions."""
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        patterns = set(config.get("command_allowlist", []) or [])
        if patterns:
            load_permanent(patterns)
        return patterns
    except Exception as e:
        logger.warning("Failed to load permanent allowlist: %s", e)
        return set()


def save_permanent_allowlist(patterns: set):
    """Save permanently allowed command patterns to config."""
    try:
        from hermes_cli.config import load_config, save_config
        config = load_config()
        config["command_allowlist"] = list(patterns)
        save_config(config)
    except Exception as e:
        logger.warning("Could not save allowlist: %s", e)


# --- Bypass check (yolo / mode=off) ---------------------------------------------------------------------------------

def is_approval_bypass_active_for_session(session_key: str) -> bool:
    """Canonical three-source bypass check: process ``--yolo`` (frozen at import), the
    session-scoped gateway ``/yolo`` toggle, ``approvals.mode: off``. Pure bypass
    sub-expression only — hardline blocklist / permanent allowlist are the caller's job."""
    return (_YOLO_MODE_FROZEN or is_session_yolo_enabled(session_key) or approval_context._get_approval_mode() == "off")


def is_approval_bypass_active() -> bool:
    """Return whether the current approval context has bypass enabled."""
    return is_approval_bypass_active_for_session(get_current_session_key(default=""))


# --- Result builders shared by the gates ----------------------------------------------------------------------------

def _approved() -> dict:
    return {"approved": True, "message": None}


def _denied(message: str, *, pattern_key: str, description: str, outcome: str, **extra) -> dict:
    """Standard non-consent result: the agent must not retry or rephrase."""
    return {"approved": False, "message": message, "pattern_key": pattern_key,
            "description": description, "outcome": outcome, "user_consent": False, **extra}


def _blocked(message: str, *, pattern_key: str, description: str) -> dict:
    """Non-interactive block (cron / -q / unattended / no-human): no consent keys."""
    return {"approved": False, "message": message, "pattern_key": pattern_key, "description": description}


def _user_approved(session_key: str, description: str) -> dict:
    """A human approval (incl. ESCALATE-then-approve or a smart-DENY owner
    override) resets the consecutive-denial tally."""
    _reset_denials(session_key)
    return {"approved": True, "message": None, "user_approved": True, "description": description}


def _gateway_notify_cb(session_key: str):
    with _lock:
        return _gateway_notify_cbs.get(session_key)


def _pending_result(spec, session_key: str, *, command: str, description: str,
                    pattern_key: str, pattern_keys: list[str], body: str | None,
                    smart_denied: bool) -> dict:
    """Queue an approval nobody can answer right now (no gateway notifier, no CLI panel) for
    ``/approve`` / ``/deny`` review. Command/code gates return the backward-compatible
    ``pending_approval`` shape (``pattern_keys`` + STOP text); the action gate ``approval_required``."""
    pending = {"command": command, "pattern_key": pattern_key}
    if spec.pending_keys:
        pending["pattern_keys"] = pattern_keys
    pending["description"] = description
    if smart_denied:
        pending.update(smart_denied=True, allow_permanent=False)
    submit_pending(session_key, pending)
    if not spec.pending_keys:
        return {
            "approved": False, "pattern_key": pattern_key, "status": "approval_required",
            "command": command, "description": description,
            "message": (f"⚠️ This action is potentially dangerous ({description}). "
                        f"Asking the user for approval.\n\n**Target:**\n```\n{command}\n```"),
        }
    body = body or f"**Command:**\n```\n{command}\n```"
    result = {
        "approved": False, "pattern_key": pattern_key, "status": "pending_approval",
        "approval_pending": True, "command": command, "description": description,
        "message": (
            f"⚠️ {description}. Asking the user for approval.\n\n{body}\n\n"
            f"STOP: do NOT re-run, rephrase, or re-issue this {spec.noun} — each "
            "variant sends the user ANOTHER approval card. Wait for the "
            "user's decision; if this turn must end, report that approval is pending."
        ),
    }
    if smart_denied:
        result.update(smart_denied=True, allow_permanent=False)
    return result


# --- Unattended contexts (nobody present to answer a prompt) --------------------------------------------------------

@dataclass(frozen=True)
class _Unattended:
    """One non-interactive context and the text every gate uses to explain it."""
    name: str       # "single_query" | "cron" | "unattended"
    cfg_key: str    # approvals.<cfg_key>: approve|deny
    clause: str     # "why nobody can approve" (lower-case sentence fragment)
    scope: str      # "in cron jobs" — completes "To allow ... {scope}"
    trust: str      # execute_code: "approve only if {trust}"

    def mode(self) -> str:
        name = f"_get_{self.name}_approval_mode"
        local_getter = globals()[name]
        original_getter = _ORIGINAL_UNATTENDED_GETTERS[name]
        return (getattr(approval_context, name) if local_getter is original_getter else local_getter)()

    def block_message(self, subject: str, *, noun: str, advice: str) -> str:
        return (f"BLOCKED: {subject} but {self.clause}. {advice} To allow {noun} {self.scope}, set "
                f"approvals.{self.cfg_key}: approve in config.yaml.")

    @property
    def exec_tail(self) -> str:
        return (f"{self.clause[0].upper()}{self.clause[1:]}. Use normal tools "
                f"instead, or set approvals.{self.cfg_key}: approve only if {self.trust}.")


_SINGLE_QUERY_CTX = _Unattended(
    "single_query", "single_query_mode",
    "single-query mode (-q) runs without a user present to approve it",
    "in single-query mode", "this single-query run is intentionally trusted",
)
_CRON_CTX = _Unattended(
    "cron", "cron_mode", "cron jobs run without a user present to approve it",
    "in cron jobs", "this cron profile is intentionally trusted",
)

_ORIGINAL_UNATTENDED_GETTERS = {
    "_get_single_query_approval_mode": _get_single_query_approval_mode,
    "_get_cron_approval_mode": _get_cron_approval_mode,
    "_get_unattended_approval_mode": _get_unattended_approval_mode,
}


def _unattended_contexts() -> list[_Unattended]:
    """Active unattended contexts in evaluation order: single-query first (``hermes chat -q``
    exports HERMES_INTERACTIVE=1 but nobody answers); cron beats a platform marker because
    cron binds the platform for delivery routing only."""
    contexts = []
    if _is_single_query_approval_context():
        contexts.append(_SINGLE_QUERY_CTX)
    if _is_cron_approval_context():
        contexts.append(_CRON_CTX)
    elif _is_unattended_platform_approval_context():
        contexts.append(_Unattended(
            "unattended", "unattended_mode",
            "this session runs on an unattended platform "
            f"({_get_session_platform()}) with no user present to approve it",
            "on unattended platforms", "sessions on this surface are intentionally trusted",
        ))
    return contexts


def _unattended_deny(command: str, ctx: _Unattended) -> dict | None:
    """Deny-mode handling for one unattended context (cron / -q / webhook); None = allow.

    Pattern detection first, then tirith so content-level threats (homograph URLs,
    pipe-to-interpreter, terminal injection) are caught even when the pattern detector misses.
    An un-importable tirith honours ``security.tirith_fail_open``: fail-closed means block,
    since nobody can approve.
    """
    if ctx.mode() != "deny":
        return None

    def block(subject: str) -> dict:
        return {"approved": False, "message": ctx.block_message(
            subject, noun="dangerous commands",
            advice="Find an alternative approach that avoids this command.")}

    is_dangerous, _pk, description = detect_dangerous_command(command)
    if is_dangerous:
        result = block(f"Command flagged as dangerous ({description})")
        if ctx.name == "single_query":
            result.update(pattern_key=_pk, description=description)
        return result
    try:
        from tools.tirith_security import check_command_security
        tirith = check_command_security(command)
    except ImportError:
        if _tirith_fail_open():
            return None
        return {"approved": False, "message": (
            "BLOCKED: the Tirith security scanner could not be imported and security.tirith_fail_open is false, "
            f"so this command cannot be silently allowed — and {ctx.clause}. "
            f"Find an alternative approach, install tirith, or set approvals.{ctx.cfg_key}: approve in config.yaml.")}
    if tirith.get("action") in ("block", "warn"):
        return block(_format_tirith_description(tirith))
    return None


# --- Human-decision engine shared by the three gates ----------------------------------------------------------------
# Every flagged action reaches a human the same way — selected plugin transport → gateway round-trip → pending
# fallback → CLI prompt → persist — so the consent contract (silence is not consent, deny is a hard halt, a smart-DENY
# override is one operation) cannot drift between gates. Only wording and a few policy knobs differ per flavor; they
# live in _GateSpec.

@dataclass(frozen=True)
class _GateSpec:
    noun: str                 # "command" | "code" — for the pending STOP text
    transport: bool           # offer the selected plugin transport first
    user_approved: bool       # human approval resets the denial tally
    redact_cli: bool          # CLI prompt + hooks see the redacted copy
    pending_keys: bool        # pending fallback: redacted ``pending_approval`` shape with
                              # pattern_keys (True) vs raw ``approval_required`` (False)
    # Message templates. ``{breaker}`` = the denial circuit-breaker addendum,
    # read only where a template shows it (reading it logs when tripped).
    notify_failed: str
    gateway_refused: str      # {reason}{reason_addendum}{timeout_addendum}{breaker}
    transport_denied: str     # {breaker}
    cli_timeout: str          # {breaker}
    cli_denied: str           # {description}{breaker}
    smart_log: str            # {command}{description}{session_key}


_STOP_COMMAND = (
    " The user has NOT consented to this action. Do NOT retry this command, do "
    "NOT rephrase it, and do NOT attempt the same outcome via a different "
    "command. Stop the current workflow and wait for the user to respond before "
    "taking any further destructive or irreversible action."
)
_STOP_ACTION = (
    " The user has NOT consented to this action. Do NOT retry it, do NOT "
    "rephrase it, and do NOT attempt the same outcome via a different path."
)

_COMMAND_GATE = _GateSpec(
    noun="command", transport=True, user_approved=True, redact_cli=False, pending_keys=True,
    notify_failed="BLOCKED: Failed to send approval request to user. Do NOT retry.",
    gateway_refused="BLOCKED: Command {reason}.{reason_addendum}" + _STOP_COMMAND
                    + "{timeout_addendum}{breaker}",
    transport_denied=(
        "BLOCKED: User denied this command through the selected approval "
        "transport. The user has NOT consented to this action. Do NOT retry or "
        "attempt the same outcome through another route.{breaker}"
    ),
    cli_timeout="BLOCKED: Command timed out without user response." + _STOP_COMMAND
                + " Silence is not consent.{breaker}",
    cli_denied="BLOCKED: User denied this command." + _STOP_COMMAND + "{breaker}",
    smart_log="Smart approval: auto-approved '{command}' ({description})",
)
_EXECUTE_CODE_GATE = _GateSpec(
    noun="code", transport=True, user_approved=True, redact_cli=True, pending_keys=True,
    notify_failed="BLOCKED: Failed to send execute_code approval request to user. Do NOT retry.",
    gateway_refused=(
        "BLOCKED: execute_code script {reason}.{reason_addendum} The user has "
        "NOT consented to running this code. Do NOT retry, do NOT rephrase the "
        "script, and do NOT attempt the same outcome via a different tool.{timeout_addendum}{breaker}"
    ),
    transport_denied=(
        "BLOCKED: User denied execute_code through the selected approval transport. The user has NOT consented."
    ),
    cli_timeout="BLOCKED: Action timed out without user response." + _STOP_ACTION
                + " Silence is not consent.{breaker}",
    cli_denied=(
        "BLOCKED: User denied execute_code script execution (matched "
        "'{description}'). Do NOT retry — the user has explicitly rejected it.{breaker}"
    ),
    smart_log="Smart approval: auto-approved execute_code for session {session_key}",
)
# Plugin-escalated tool calls / protected writes: no transport, no breaker,
# no user_approved marker (parity with the historical gate).
_ACTION_GATE = _GateSpec(
    noun="action", transport=False, user_approved=False, redact_cli=False, pending_keys=False,
    notify_failed="BLOCKED: Failed to send approval request to user. Do NOT retry.",
    gateway_refused="BLOCKED: Action {reason}.{reason_addendum}" + _STOP_ACTION
                    + "{timeout_addendum}",
    transport_denied="",
    cli_timeout="BLOCKED: Action timed out without user response." + _STOP_ACTION
                + " Silence is not consent.",
    cli_denied=(
        "BLOCKED: User denied this potentially dangerous action (matched "
        "'{description}'). Do NOT retry — the user has explicitly rejected it."
    ),
    smart_log="",
)


def _smart_gate(spec: _GateSpec, command: str, description: str, pattern_key: str,
                pattern_keys: list[str], session_key: str, *,
                human_present: bool) -> tuple[dict | None, bool]:
    """Guardian-LLM step -> ``(result, smart_denied_for_owner)``: a result ends the gate;
    ``smart_denied_for_owner`` means an interactive owner may still override the DENY for this
    one operation (once/deny only, nothing persists).

    APPROVE approves this command only — pattern-level persistence would let one benign
    command suppress review of later commands in the same broad detector category. A DENY
    counts toward the denial breaker even when an owner may override it. ESCALATE follows the
    normal, potentially persistent manual behavior.
    """
    verdict = _smart_verdict(command, description, pattern_key, pattern_keys, session_key)
    if verdict == "approve":
        _reset_denials(session_key)
        logger.debug(spec.smart_log.format(command=command[:60], description=description, session_key=session_key))
        return {"approved": True, "message": None, "smart_approved": True, "description": description}, False
    if verdict != "deny":
        return None, False
    _record_denial(session_key)
    if human_present:
        return None, True
    return {
        # Unattended programmatic platforms (webhook/msgraph_webhook/ api_server): respect unattended_mode
        # config. Resolves instantly — never a pending approval nobody can answer (#37284, #87509).
        "approved": False,
        "message": (f"BLOCKED by smart approval: {description}. The command was assessed as genuinely "
                    f"dangerous. Do NOT retry.{_denial_breaker_addendum(session_key)}"),
        "smart_denied": True,
    }, True


def _human_decision(spec: _GateSpec, *, command: str, description: str,
                    pattern_key: str, pattern_keys: list[str], warnings: list[tuple],
                    session_key: str, approval_callback, is_cli: bool, is_gateway: bool,
                    is_ask: bool, smart: bool = False,
                    permanent_capable: bool = True, pending_body=None) -> dict:
    """Ask a human (after the optional guardian-LLM step) and turn the answer into the gate result.

    ``warnings`` are the ``(key, _, is_tirith)`` tuples :func:`_persist_choice` stores on
    session/always. ``permanent_capable`` hides [a]lways when no key could be permanently
    allowlisted (pure-tirith prompts); a smart-DENY owner override reduces every surface to
    once/deny and persists nothing. ``pending_body`` is a thunk, built only once a human is
    actually asked, so a smart APPROVE never pays for redacting a large script.
    """
    from agent.redact import redact_sensitive_text

    smart_denied = False
    if smart:
        result, smart_denied = _smart_gate(spec, command, description, pattern_key, pattern_keys,
                                           session_key, human_present=is_cli or is_gateway or is_ask)
        if result is not None:
            return result
    pending_body = pending_body() if pending_body else None
    allow_permanent = permanent_capable and not smart_denied

    def deny(template: str, outcome: str, **fmt) -> dict:
        breaker = ""
        if "{breaker}" in template:
            breaker = _denial_breaker_addendum(session_key)
        deny_reason = fmt.pop("deny_reason", None)
        extra = {"deny_reason": deny_reason} if "reason" in fmt else {}
        return _denied(template.format(description=description, breaker=breaker, **fmt),
                       pattern_key=pattern_key, description=description,
                       outcome=outcome, **extra)

    def grant(choice: str) -> dict:
        # A smart-DENY owner override is always one operation, even if an older client returns "session" or "always".
        if not smart_denied:
            _persist_choice(session_key, choice, warnings)
        if spec.user_approved:
            return _user_approved(session_key, description)
        return _approved()

    if spec.transport:
        attempt = _present_with_selected_transport(
            command=command, description=description, pattern_key=pattern_key, pattern_keys=pattern_keys,
            session_key=session_key, surface="gateway" if (is_gateway or is_ask) else "cli",
            allow_session=not smart_denied, allow_permanent=allow_permanent,
        )
        choice, denied = _transport_choice(attempt, pattern_key=pattern_key, description=description)
        if denied is not None:
            return denied
        if choice is not None:
            if choice == "deny":
                _record_denial(session_key)
                return deny(spec.transport_denied, "denied")
            return grant(choice)

    # Gateway/async approval: block the agent thread until /approve or /deny, mirroring the CLI's synchronous input()
    # flow. The agent never sees "approval_required" here — it gets output or a definitive BLOCKED.
    if is_gateway or is_ask:
        # Redacted copies for user-visible rendering only (the gateway paints them into Discord/Slack); the raw
        # command still executes after approval and persistence keys off pattern_key.
        display_command = _redact_for_approval(command)
        display_description = _redact_for_approval(description)
        notify_cb = _gateway_notify_cb(session_key)
        if notify_cb is not None:
            # Smart DENY overrides are one-operation decisions, so the UI must not offer a
            # permanent scope. Session approval is safe for every non-Smart-DENY prompt —
            # including pure-tirith ones, where persistence already caps scope at session.
            data = {
                "command": display_command, "pattern_key": pattern_key,
                "pattern_keys": pattern_keys, "description": display_description,
                "allow_permanent": permanent_capable and not smart_denied,
                "allow_session": not smart_denied,
            }
            if smart_denied:
                data["smart_denied"] = True
            decision = _await_gateway_decision(session_key, notify_cb, data, surface="gateway")
            if decision.get("notify_failed"):
                return _denied(spec.notify_failed, pattern_key=pattern_key,
                               description=description, outcome="notify_failed")
            # Consent contract: silence is NOT consent, and an explicit deny is a hard
            # halt — both produce a BLOCKED outcome. ``/deny <reason>`` free text is
            # relayed verbatim so the agent can adapt rather than only hearing "denied".
            choice, deny_reason = decision["choice"], decision.get("reason")
            if not decision["resolved"]:
                return deny(spec.gateway_refused, "timeout", reason="timed out without user response",
                            reason_addendum="", timeout_addendum=" Silence is not consent.",
                            deny_reason=deny_reason)
            if choice is None or choice == "deny":
                return deny(spec.gateway_refused, "denied", reason="denied by user",
                            reason_addendum=(f' Reason given by the user: "{deny_reason}".' if deny_reason else ""),
                            timeout_addendum="", deny_reason=deny_reason)
            return grant(choice)

        # No gateway callback (cron, batch, or ask-mode leaked into an interactive CLI, historically via `import
        # gateway.run`): paint the local panel when possible instead of a pending_approval that makes the agent look
        # "auto-blocked".
        if not _should_fall_through_to_cli_approval(
            is_cli=is_cli, approval_callback=approval_callback, notify_cb=notify_cb,
        ):
            return _pending_result(
                spec, session_key, command=display_command, description=display_description, pattern_key=pattern_key,
                pattern_keys=pattern_keys, body=pending_body, smart_denied=smart_denied,
            )

    # CLI interactive: single combined prompt, wrapped in the pre/post plugin hooks.
    prompt_command = _redact_for_approval(command)
    prompt_description = _redact_for_approval(description)
    hook_kwargs = dict(command=prompt_command, description=prompt_description, pattern_key=pattern_key,
                       pattern_keys=list(pattern_keys), session_key=session_key, surface="cli")
    approval_context._fire_approval_hook("pre_approval_request", **hook_kwargs)
    choice = prompt_dangerous_approval(prompt_command, prompt_description, allow_permanent=allow_permanent,
                                       smart_denied=smart_denied, approval_callback=approval_callback)
    approval_context._fire_approval_hook("post_approval_response", **hook_kwargs, choice=choice)
    if choice == "timeout":
        return deny(spec.cli_timeout, "timeout")
    if choice == "deny":
        # No _record_denial(): the breaker counts consecutive guardian LLM
        # DENY verdicts, not deliberate human denials.
        return deny(spec.cli_denied, "denied")
    return grant(choice)


def _presence(approval_callback=None) -> tuple:
    """``(approval_callback, is_cli, is_gateway, is_ask)`` for the current context. Single-query
    (-q) exports HERMES_INTERACTIVE=1 but nobody answers prompts, and HERMES_EXEC_ASK has no
    human either — both are cleared so single_query_mode actually takes effect."""
    approval_callback = _resolve_cli_approval_callback(approval_callback)
    is_cli, is_gateway = _is_interactive_cli(), _is_gateway_approval_context()
    is_ask = env_var_enabled("HERMES_EXEC_ASK")
    if _is_single_query_approval_context():
        is_cli = is_gateway = is_ask = False
    return approval_callback, is_cli, is_gateway, is_ask


def _run_approval_gate(
    *, pattern_key: str, description: str, display_target: str, approval_callback=None,
    subject: str = "", noun: str = "flagged actions",
    advice: str = "Find an alternative approach that avoids this action.",
    cron_deny_message: str = "", single_query_deny_message: str = "", unattended_deny_message: str = "",
    autoapprove_log_prefix: str, fail_closed_when_no_human: bool = False, no_human_block_message: str = "",
) -> dict:
    """Shared human-approval gate for a flagged action (tool call or write): decision core for
    :func:`request_tool_approval` and the file-tool write gates.

    Order: yolo bypass → session-cache short-circuit → interactive/gateway/unattended branch →
    prompt → persistence. Input-shape checks (hardline, allowlist, pattern detection) are the
    caller's job. ``fail_closed_when_no_human``: a non-interactive, non-gateway, non-cron
    context BLOCKS instead of auto-approving, so a plugin-flagged action never runs ungated.
    Unattended deny text is ``ctx.block_message(subject, noun, advice)`` unless the caller passes
    an explicit ``*_deny_message`` (the file-tool write gates word their own).
    """
    # Hardline blocks are the caller's job BEFORE this gate, so yolo here only skips the recoverable approval layer.
    if _yolo_active():
        return _approved()
    session_key = get_current_session_key()
    if is_approved(session_key, pattern_key):
        return _approved()

    approval_callback, is_cli, is_gateway, is_ask = _presence(approval_callback)
    if not is_cli and not is_gateway:
        log_args = (autoapprove_log_prefix, pattern_key, description)
        # Every unattended context resolves instantly — never a pending approval nobody can answer.
        deny_messages = {
            "single_query": single_query_deny_message, "cron": cron_deny_message,
            "unattended": unattended_deny_message,
        }
        for ctx in _unattended_contexts():
            if ctx.mode() == "deny":
                message = deny_messages[ctx.name]
                if not message and ctx.name == "unattended":
                    # Platform contexts keep the generic wording (historical shape).
                    message = ctx.block_message(f"approval required ({description})", noun="flagged actions",
                                                advice="Find an alternative approach that avoids this action.")
                elif not message:
                    message = ctx.block_message(subject, noun=noun, advice=advice)
                return _blocked(message, pattern_key=pattern_key, description=description)
            if ctx.name == "single_query":
                # Return here rather than fall through: the fail-closed branch would
                # otherwise block what single_query_mode: approve just authorized.
                logger.warning("%s (pattern: %s): %s — single-query auto-approve "
                               "(approvals.single_query_mode: approve).", *log_args)
                return _approved()
            break  # cron/unattended approve-mode: auto-approve below
        else:
            if fail_closed_when_no_human:
                logger.warning("%s (pattern: %s): %s — no interactive user/gateway present; "
                               "BLOCKED (fail-closed). Set HERMES_INTERACTIVE or "
                               "HERMES_GATEWAY_SESSION to answer the prompt.", *log_args)
                return _blocked(no_human_block_message or (
                    f"BLOCKED: approval required ({description}) but no "
                    "interactive user or gateway is present to approve it."),
                    pattern_key=pattern_key, description=description)
        logger.warning("%s (pattern: %s): %s — set HERMES_INTERACTIVE or "
                       "HERMES_GATEWAY_SESSION to require approval.", *log_args)
        return _approved()

    return _human_decision(
        _ACTION_GATE, command=display_target, description=description, pattern_key=pattern_key,
        pattern_keys=[pattern_key], warnings=[(pattern_key, None, False)], session_key=session_key,
        approval_callback=approval_callback, is_cli=is_cli, is_gateway=is_gateway, is_ask=is_ask,
    )


def _should_skip_container_guards(env_type: str, has_host_access: bool = False) -> bool:
    """True when the backend is isolated enough to skip dangerous-command prompts. Docker is the
    exception once host paths are bind-mounted: ``rm -rf /workspace`` then reaches host files."""
    if env_type == "docker":
        return not has_host_access
    return env_type in ("singularity", "modal", "daytona", "vercel_sandbox")


def _user_deny_block(command: str) -> dict | None:
    """The operator's ``approvals.deny`` rules are documented as never bypassable — not by yolo,
    not by mode=off, and not by an isolated container either: they express intent about what the
    agent may DO, not what it can reach, so they are evaluated before the container fast path."""
    deny_pattern = _match_user_deny_rule(command)
    if deny_pattern is None:
        return None
    logger.warning("User deny rule %r blocked command: %s", deny_pattern, command[:200])
    return _user_deny_block_result(deny_pattern)


def _floor_block(command: str, *, sudo_guard: bool = False) -> dict | None:
    """Unconditional floors, BEFORE yolo / mode=off / cron approve-mode so no
    session-level setting can bypass them: hardline catastrophic commands,
    password-piping to ``sudo -S`` with no SUDO_PASSWORD configured (full guard
    only), and the user's own approvals.deny rules ("never, even under yolo")."""
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc, command)
    if sudo_guard:
        is_sudo_guess, sudo_guess_desc = _check_sudo_stdin_guard(command)
        if is_sudo_guess:
            logger.warning("Sudo stdin guard block: %s (command: %s)", sudo_guess_desc, command[:200])
            return _sudo_stdin_block_result(sudo_guess_desc)
    return _user_deny_block(command)


def check_dangerous_command(command: str, env_type: str,
                            approval_callback=None,
                            has_host_access: bool = False) -> dict:
    """Detect a dangerous command and handle approval (pattern layer only). ``has_host_access``:
    a Docker sandbox that bind-mounts host paths must not skip approval.
    Returns ``{"approved": True/False, "message": str or None, ...}``."""
    if _should_skip_container_guards(env_type, has_host_access=has_host_access):
        return _user_deny_block(command) or _approved()
    blocked = _floor_block(command)
    if blocked is not None:
        return blocked
    if _yolo_active():
        return _approved()
    if _command_matches_permanent_allowlist(command):
        return _approved()
    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    if not is_dangerous:
        return _approved()
    return _run_approval_gate(
        pattern_key=pattern_key, description=description, display_target=command, approval_callback=approval_callback,
        subject=f"Command flagged as dangerous ({description})", noun="dangerous commands",
        advice="Find an alternative approach that avoids this command.",
        autoapprove_log_prefix="AUTO-APPROVED dangerous command in non-interactive non-gateway context",
    )


def request_tool_approval(tool_name: str, reason: str, *, rule_key: str = "", approval_callback=None) -> dict:
    """Escalate an arbitrary tool call to the human-approval gate.

    Entry point for a plugin ``pre_tool_call`` hook returning ``{"action": "approve", ...}``:
    it asks the SAME human gate as Tier-2 dangerous shell patterns (session/permanent
    allowlist, CLI prompt, gateway pending, once/session/always/deny, timeout fail-closed), so
    the LLM cannot skip it. Cron honors ``approvals.cron_mode``; any OTHER non-interactive
    non-gateway context fails CLOSED. ``rule_key`` controls the ``[a]lways`` allowlist grain;
    when empty it is ``tool_name`` + a hash of ``reason`` so DISTINCT reasons on the same tool
    persist independently. Returns the ``check_dangerous_command`` result shape.
    """
    description = reason or f"Plugin requires approval for {tool_name}"
    if not rule_key:
        rule_key = f"{tool_name}:{hashlib.sha256(description.encode('utf-8')).hexdigest()[:12]}"
    subject = f"Tool '{tool_name}' requires approval ({description})"
    return _run_approval_gate(
        # Namespaced so plugin-rule approvals share the allowlist machinery without ever colliding with a real
        # command pattern key; the display target is a synthetic label for the display/allowlist layer.
        pattern_key=f"plugin_rule:{rule_key}", description=description,
        display_target=f"<{tool_name}> (plugin approval rule)", approval_callback=approval_callback,
        subject=subject, advice="Find an alternative approach.",
        autoapprove_log_prefix=f"plugin-escalated tool call '{tool_name}' in non-interactive non-gateway context",
        fail_closed_when_no_human=True,
        no_human_block_message=(f"BLOCKED: {subject} but no interactive user or gateway is present "
                                "to approve it. A plugin flagged this action for human confirmation."),
    )


# --- Combined pre-exec guard (tirith + dangerous command detection) -------------------------------------------------

def _format_tirith_description(tirith_result: dict) -> str:
    """Human-readable severity/title/description summary of tirith findings."""
    parts = []
    for f in tirith_result.get("findings") or []:
        severity, title, desc = f.get("severity", ""), f.get("title", ""), f.get("description", "")
        if title:
            text = f"{title}: {desc}" if desc else title
            parts.append(f"[{severity}] {text}" if severity else text)
    if not parts:
        summary = tirith_result.get("summary") or "security issue detected"
        return f"Security scan: {summary}"
    return "Security scan — " + "; ".join(parts)


def _tirith_scan(command: str) -> dict:
    """Tirith result for the interactive flow; an un-importable scanner allows
    (default) or, under fail-closed, synthesizes a HIGH warn finding that goes
    through the normal approval flow (#20733)."""
    try:
        from tools.tirith_security import check_command_security
        return check_command_security(command)
    except ImportError:
        if _tirith_fail_open():
            return {"action": "allow", "findings": [], "summary": ""}
        return {"action": "warn", "summary": "Tirith unavailable (fail-closed)", "findings": [{
            "rule_id": "tirith-import-error", "severity": "HIGH",
            "title": "Tirith security module unavailable",
            "description": ("The Tirith security scanner could not be imported. "
                            "Because security.tirith_fail_open is false, this "
                            "command cannot be silently allowed. Approve only if "
                            "you have verified the command is safe."),
        }]}


def check_all_command_guards(command: str, env_type: str,
                             approval_callback=None,
                             has_host_access: bool = False) -> dict:
    """Run all pre-exec security checks and return a single approval decision. Tirith and
    dangerous-command findings are presented as ONE combined approval request, so a gateway
    force=True replay cannot bypass one check when only the other was shown to the user.
    ``has_host_access``: a Docker sandbox with bind-mounted host paths takes the normal flow."""
    if _should_skip_container_guards(env_type, has_host_access=has_host_access):
        return _user_deny_block(command) or _approved()

    blocked = _floor_block(command, sudo_guard=True)
    if blocked is not None:
        return blocked

    approval_mode = _get_approval_mode()
    if _yolo_active() or approval_mode == "off":
        return _approved()
    if _command_matches_permanent_allowlist(command):
        return _approved()

    approval_callback, is_cli, is_gateway, is_ask = _presence(approval_callback)
    # Outside CLI/gateway/ask flows we never block on approvals: each
    # unattended context applies its configured deny/approve mode, else allow.
    if not is_cli and not is_gateway and not is_ask:
        for ctx in _unattended_contexts():
            result = _unattended_deny(command, ctx)
            if result is not None:
                return result
        return _approved()

    # Gather findings: warnings = [(pattern_key, description, is_tirith)]. Tirith block AND warn both go through the
    # approval flow (block used to be a hard stop) so users can inspect the findings and approve.
    tirith_result = _tirith_scan(command)
    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    warnings = []
    session_key = get_current_session_key()
    if tirith_result["action"] in {"block", "warn"}:
        findings = tirith_result.get("findings") or []
        rule_id = findings[0].get("rule_id", "unknown") if findings else "unknown"
        tirith_key = f"tirith:{rule_id}"
        if not is_approved(session_key, tirith_key):
            warnings.append((tirith_key, _format_tirith_description(tirith_result), True))
    if is_dangerous and not is_approved(session_key, pattern_key):
        warnings.append((pattern_key, description, False))
    if not warnings:
        return _approved()

    combined_desc = "; ".join(desc for _, desc, _ in warnings)
    primary_key = warnings[0][0]
    all_keys = [key for key, _, _ in warnings]

    # "Always" is offered when at least one warning is a dangerous-pattern key the persistence layer would actually
    # allowlist permanently. Pure-tirith findings are session-max by design, so a tirith-only prompt hides Always;
    # mixed prompts offer it (the pattern key persists, tirith downgrades to session — see _persist_choice).
    return _human_decision(
        _COMMAND_GATE, command=command, description=combined_desc,
        pattern_key=primary_key, pattern_keys=all_keys, warnings=warnings,
        session_key=session_key, approval_callback=approval_callback,
        is_cli=is_cli, is_gateway=is_gateway, is_ask=is_ask, smart=approval_mode == "smart",
        permanent_capable=any(not is_t for _, _, is_t in warnings),
    )


_EXECUTE_CODE_DESCRIPTION = (
    "execute_code script execution. The script can spawn subprocesses or "
    "mutate files without passing through terminal command approval; approval is one-shot for this run."
)


def _get_allow_execute_code() -> bool:
    """Whether the operator pre-trusts execute_code's one-shot script prompt."""
    value = _get_approval_config().get("allow_execute_code", False)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on", "allow"}
    return bool(value)


def check_execute_code_guard(code: str, env_type: str, has_host_access: bool = False) -> dict:
    """Approve an execute_code script before its child process is spawned.

    The script can call ``subprocess``/``os.system``/``ctypes`` directly, none of which pass
    through ``terminal()`` / ``DANGEROUS_PATTERNS``; in gateway/ask contexts we fail closed by
    approving the script as a whole. Same dict contract as ``check_all_command_guards``.
    Documented limitation: a purely local non-interactive non-gateway session returns approved
    (the terminal auto-approve contract); the hardline floor still blocks catastrophic
    ``terminal()`` commands the script issues.

    See #30882.
    The hardline floor still blocks catastrophic ``terminal()`` commands the script issues; running
    arbitrary code headlessly without any approval surface is trusted-by-config (set a gateway/ask surface
    or ``approvals.cron_mode`` to require approval). See #30882.
    """
    pattern_key = "execute_code"
    description = _EXECUTE_CODE_DESCRIPTION

    # Isolated backends already sandbox the child. vercel_sandbox has no host-bind concept so it stays always-skipped.
    if env_type == "vercel_sandbox":
        return _approved()
    if _should_skip_container_guards(env_type, has_host_access=has_host_access):
        return _approved()
    approval_mode = _get_approval_mode()
    if _yolo_active() or approval_mode == "off":
        return _approved()

    # (-q clears the presence flags, but its unattended context resolves first anyway.)
    approval_callback, is_cli, is_gateway, is_ask = _presence()
    # No user is present to approve arbitrary code in -q / cron / unattended
    # sessions: the first active context resolves instantly from its mode.
    for ctx in _unattended_contexts():
        if ctx.mode() == "deny":
            return _denied(
                "BLOCKED: execute_code runs arbitrary local Python (including "
                "subprocess calls that bypass shell-string approval checks). " + ctx.exec_tail,
                pattern_key=pattern_key, description=description, outcome="blocked",
            )
        return _approved()

    # Unattended/cron policy above stays authoritative; this opt-in only
    # suppresses the whole-script prompt on attended gateway/ask surfaces.
    if _get_allow_execute_code():
        return _approved()

    # Only gateway/ask contexts get the one-shot whole-script approval. In an interactive CLI the script's terminal()
    # calls are guarded per-call (context propagates into the RPC thread, #33057), so a whole-script prompt would fire
    # on every execute_code call. Ask-mode still takes this path even with INTERACTIVE set (how gateway/smart tests
    # and messaging ask-mode drive whole-script approval); when that leaks into a CLI with no notify callback, the
    # engine falls through to the CLI Dangerous Command panel instead of a silent pending_approval.
    if not is_gateway and not is_ask:
        return _approved()

    session_key = get_current_session_key()
    # Built only past the early-return gates so common paths don't copy a potentially-large script into this string.
    command = f"execute_code <<'PY'\n{code}\nPY"

    # Without this, "Approve session" / "Always" choices are stored but never
    # consulted, so every execute_code call re-prompts (#39275).
    if is_approved(session_key, pattern_key):
        return _approved()

    # Smart mode: an APPROVE only suppresses the redundant whole-script prompt; the per-call terminal() guards still
    # run independently. The gateway renders the pending payload to Discord/Slack, so the script body is redacted for
    # display; the raw code is what gets assessed and run.
    return _human_decision(
        _EXECUTE_CODE_GATE, command=command, description=description, pattern_key=pattern_key,
        pattern_keys=[pattern_key], warnings=[(pattern_key, None, False)], session_key=session_key,
        approval_callback=approval_callback, is_cli=is_cli, is_gateway=is_gateway, is_ask=is_ask,
        smart=approval_mode == "smart",
        pending_body=lambda: f"**Code:**\n```python\n{_redact_for_approval(code)}\n```",
    )


# Load permanent allowlist from config on module import
load_permanent_allowlist()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import contextlib  # noqa: F401,E402
import contextvars  # noqa: F401,E402
import fnmatch  # noqa: F401,E402
import functools  # noqa: F401,E402
import re  # noqa: F401,E402
import shlex  # noqa: F401,E402
import sys  # noqa: F401,E402
import tempfile  # noqa: F401,E402
import time  # noqa: F401,E402
import unicodedata  # noqa: F401,E402
import uuid  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'DANGEROUS_PATTERNS': ('tools.approval_detection', 'DANGEROUS_PATTERNS'),
    'DANGEROUS_PATTERNS_COMPILED': ('tools.approval_detection', 'DANGEROUS_PATTERNS_COMPILED'),
    'HARDLINE_PATTERNS': ('tools.approval_detection', 'HARDLINE_PATTERNS'),
    'HARDLINE_PATTERNS_COMPILED': ('tools.approval_detection', 'HARDLINE_PATTERNS_COMPILED'),
    'HUMAN_WAIT_MARGIN_S': ('tools.approval_human_wait', 'HUMAN_WAIT_MARGIN_S'),
    'cfg_get': ('hermes_cli.config', 'cfg_get'),
    'get_plugin_manager': ('tools.approval_prompt', 'get_plugin_manager'),
    'human_wait_ceiling': ('tools.approval_human_wait', 'human_wait_ceiling'),
    'human_wait_seconds': ('tools.approval_human_wait', 'human_wait_seconds'),
    'human_wait_window': ('tools.approval_human_wait', 'human_wait_window'),
    'is_interrupted': ('tools.interrupt', 'is_interrupted'),
    'request_elicitation_consent': ('tools.approval_prompt', 'request_elicitation_consent'),
    'reset_current_observability_context': ('tools.approval_context', 'reset_current_observability_context'),
    'reset_current_session_key': ('tools.approval_context', 'reset_current_session_key'),
    'reset_hermes_interactive_context': ('tools.approval_context', 'reset_hermes_interactive_context'),
    'set_current_observability_context': ('tools.approval_context', 'set_current_observability_context'),
    'set_current_session_key': ('tools.approval_context', 'set_current_session_key'),
    'set_hermes_interactive_context': ('tools.approval_context', 'set_hermes_interactive_context'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

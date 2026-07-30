"""Tests for scripts/ci/classify_changes.py.

Check some common patterns of file modifications and the CI lanes they should run.
We should always fail open. We may run a lane we didn't need, never skip one a
change could have broken.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "classify_changes.py"
_spec = importlib.util.spec_from_file_location("classify_changes", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load classify_changes.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
classify = _mod.classify
classify_for_event = _mod.classify_for_event
classify_live_config_push = _mod.classify_live_config_push
ci_review_files = _mod.ci_review_files

DEFAULT = {
    "python": True,
    "e2e": True,
    "frontend": True,
    "docker_meta": True,
    "site": True,
    "scan": True,
    "deps": True,
    "npm_lock": True,
    "mcp_catalog": False,
    "ci_review": True,
}


def _lanes(python=False, e2e=None, frontend=False, site=False, scan=False, deps=False, npm_lock=False, mcp_catalog=False, docker_meta=False, ci_review=False) -> dict[str, bool]:
    e2e = python if e2e is None else e2e
    return {
        "python": python,
        "e2e": e2e,
        "frontend": frontend,
        "docker_meta": docker_meta,
        "site": site,
        "scan": scan,
        "deps": deps,
        "npm_lock": npm_lock,
        "mcp_catalog": mcp_catalog,
        "ci_review": ci_review,
    }


CASES = {
    "docs-only → nothing heavy": (["README.md", "docs/guide.md"], _lanes()),
    "python source → python": (["run_agent.py"], _lanes(python=True, scan=True)),
    "dep manifest → python": (["pyproject.toml"], _lanes(python=True, scan=True, deps=True)),
    "uv.lock → python": (["uv.lock"], _lanes(python=True)),
    "ts package → frontend": (["apps/desktop/src/app.tsx"], _lanes(frontend=True)),
    "ui-tui → frontend": (["ui-tui/src/entry.ts"], _lanes(frontend=True)),
    # Lockfile bump shifts every TS package's tree, but not the Python suite.
    "root lockfile → frontend, not python": (["package-lock.json"], _lanes(frontend=True, npm_lock=True)),
    "nested lockfile → npm_lock": (["website/package-lock.json"], _lanes(site=True, npm_lock=True)),
    "website → site": (["website/docs/intro.md"], _lanes(site=True)),
    # SKILL.md reads like docs, but the skill-doc tests read skills/, so a
    # skill edit must still run Python.
    "skill md → python + site": (["skills/github/SKILL.md"], _lanes(python=True, site=True)),
    "dockerfile → docker meta": (["Dockerfile"], _lanes(docker_meta=True)),
    # Unknown top-level file keeps Python on rather than risk a silent skip.
    "unknown toplevel → python": (["Makefile"], _lanes(python=True)),
    "mixed docs+python → python": (["README.md", "agent/x.py"], _lanes(python=True, scan=True)),
    "mixed docs+frontend → frontend": (["README.md", "apps/x.tsx"], _lanes(frontend=True)),
    # Supply-chain lanes
    ".pth file → scan": (["evil.pth"], _lanes(python=True, scan=True)),
    "setup.py → scan": (["setup.py"], _lanes(python=True, scan=True)),
    "mcp catalog manifest → mcp_catalog": (
        ["optional-mcps/foo/manifest.yaml"],
        _lanes(python=True, mcp_catalog=True),
    ),
    "mcp_catalog.py → mcp_catalog": (
        ["hermes_cli/mcp_catalog.py"],
        _lanes(python=True, scan=True, mcp_catalog=True),
    ),
    # CI-sensitive files require explicit review label.
    "eslint config → ci_review": (
        ["apps/desktop/eslint.config.mjs"],
        _lanes(frontend=True, ci_review=True),
    ),
    "shared eslint config → ci_review": (
        ["eslint.config.shared.mjs"],
        _lanes(python=True, ci_review=True),
    ),
    "ui-tui eslint config → ci_review": (
        ["ui-tui/eslint.config.mjs"],
        _lanes(frontend=True, ci_review=True),
    ),
    "web eslint config → ci_review": (
        ["web/eslint.config.js"],
        _lanes(frontend=True, ci_review=True),
    ),
    "shared package eslint config → ci_review": (
        ["apps/shared/eslint.config.mjs"],
        _lanes(frontend=True, ci_review=True),
    ),
    "bootstrap-installer eslint config → ci_review": (
        ["apps/bootstrap-installer/eslint.config.mjs"],
        _lanes(frontend=True, ci_review=True),
    ),
    "prettier config → ci_review": (
        [".prettierrc"],
        _lanes(python=True, ci_review=True),
    ),
    "workflow yml → ci_review (also fail-open all)": (
        [".github/workflows/typecheck.yml"],
        DEFAULT,
    ),
    "composite action → ci_review (also fail-open all)": (
        [".github/actions/retry/action.yml"],
        DEFAULT,
    ),
    # Normal desktop source doesn't trigger ci_review.
    "desktop src → no ci_review": (
        ["apps/desktop/src/app.tsx"],
        _lanes(frontend=True),
    ),
    # Fail open: CI-config / empty / blank diffs run everything.
    ".github change → all": ([".github/workflows/tests.yml"], DEFAULT),
    "action change → all": ([".github/actions/detect-changes/action.yml"], DEFAULT),
    "empty diff → all": ([], DEFAULT),
    "blank lines → all": (["", "  "], DEFAULT),
}


@pytest.mark.parametrize("files,expected", CASES.values(), ids=CASES.keys())
def test_classify(files, expected):
    assert classify(files) == expected


LIVE_CONFIG_PUSH_CASES = {
    "live-config test-only push narrows without E2E": (
        "push", "refs/heads/live-config",
        ["tests/agent/test_tool_runtime.py"],
        _lanes(python=True, e2e=False),
    ),
    "live-config runtime push retains E2E": (
        "push", "refs/heads/live-config",
        ["gateway/runtime.py", "tests/gateway/test_runtime.py"],
        _lanes(python=True, e2e=True),
    ),
    "live-config E2E test push retains E2E": (
        "push", "refs/heads/live-config", ["tests/e2e/test_cli.py"],
        _lanes(python=True, e2e=True),
    ),
    "live-config shared test fixture push retains E2E": (
        "push", "refs/heads/live-config", ["tests/conftest.py"],
        _lanes(python=True, e2e=True),
    ),
    "live-config lockfile push retains E2E": (
        "push", "refs/heads/live-config", ["uv.lock"], _lanes(python=True, e2e=True),
    ),
    "live-config mixed code push fails open": (
        "push", "refs/heads/live-config",
        ["agent/tool_runtime.py", "apps/desktop/src/app.tsx"], DEFAULT,
    ),
    "main push keeps full CI": (
        "push", "refs/heads/main", ["agent/tool_runtime.py"], DEFAULT,
    ),
    "live-config unknown path push fails open": (
        "push", "refs/heads/live-config", ["new-runtime-surface.yaml"], DEFAULT,
    ),
}


@pytest.mark.parametrize(
    "event_name,ref,files,expected",
    LIVE_CONFIG_PUSH_CASES.values(),
    ids=LIVE_CONFIG_PUSH_CASES.keys(),
)
def test_classify_for_event(event_name, ref, files, expected):
    assert classify_for_event(event_name, ref, files) == expected


def test_classify_live_config_push_empty_diff_fails_open():
    assert classify_live_config_push([]) == DEFAULT


def test_ci_review_files_returns_only_sensitive_paths_sorted_and_unique():
    assert ci_review_files([
        "apps/desktop/src/app.tsx",
        ".github/workflows/ci.yml",
        "apps/desktop/eslint.config.mjs",
        ".github/workflows/ci.yml",
    ]) == [
        ".github/workflows/ci.yml",
        "apps/desktop/eslint.config.mjs",
    ]

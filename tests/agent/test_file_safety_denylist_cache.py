"""The read denylist is cached per Hermes home without changing verdicts.

``get_read_block_error`` runs once per file in the skills sync walk, so its
per-call cost sets the cost of enumerating ~10k files. These tests pin the two
properties that made caching safe: identical verdicts, and a cache keyed on the
Hermes home so a profile switch cannot inherit another profile's denylist.
"""

import os
from pathlib import Path

import pytest

from agent import file_safety
from agent.file_safety import get_read_block_error


@pytest.fixture(autouse=True)
def _clear_cache():
    file_safety._DENYLIST_CACHE.clear()
    yield
    file_safety._DENYLIST_CACHE.clear()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills" / ".hub" / "index-cache").mkdir(parents=True)
    (home / "mcp-tokens").mkdir(parents=True)
    (home / "auth").mkdir(parents=True)
    (home / "cache").mkdir(parents=True)
    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: home)
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: home)
    return home


class TestBlockedPathsStayBlocked:
    @pytest.mark.parametrize(
        "rel",
        [
            "auth.json",
            "auth.lock",
            ".anthropic_oauth.json",
            ".env",
            "webhook_subscriptions.json",
            os.path.join("auth", "google_oauth.json"),
            os.path.join("cache", "bws_cache.json"),
        ],
    )
    def test_credential_stores_are_denied(self, hermes_home, rel):
        target = hermes_home / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("secret")
        err = get_read_block_error(str(target))
        assert err is not None
        assert "credential store" in err

    def test_mcp_tokens_directory_is_denied(self, hermes_home):
        err = get_read_block_error(str(hermes_home / "mcp-tokens"))
        assert err is not None
        assert "MCP token directory" in err

    def test_file_inside_mcp_tokens_is_denied(self, hermes_home):
        target = hermes_home / "mcp-tokens" / "server.json"
        target.write_text("{}")
        err = get_read_block_error(str(target))
        assert err is not None
        assert "MCP token file" in err

    def test_nested_file_inside_mcp_tokens_is_denied(self, hermes_home):
        target = hermes_home / "mcp-tokens" / "a" / "b.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}")
        assert get_read_block_error(str(target)) is not None

    def test_skills_hub_cache_is_denied(self, hermes_home):
        target = hermes_home / "skills" / ".hub" / "index-cache" / "index.json"
        target.write_text("{}")
        err = get_read_block_error(str(target))
        assert err is not None
        assert "prompt injection" in err

    def test_project_env_file_is_still_denied(self, tmp_path, hermes_home):
        target = tmp_path / "project" / ".env"
        target.parent.mkdir(parents=True)
        target.write_text("KEY=1")
        assert get_read_block_error(str(target)) is not None


class TestAllowedPathsStayAllowed:
    @pytest.mark.parametrize(
        "rel",
        [
            os.path.join("skills", "coding", "simplify", "SKILL.md"),
            os.path.join("skills", "writer", "references", "voice.md"),
            os.path.join("plans", "a-plan.md"),
        ],
    )
    def test_real_content_is_not_blocked(self, hermes_home, rel):
        target = hermes_home / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("content")
        assert get_read_block_error(str(target)) is None

    def test_env_example_is_not_blocked(self, tmp_path, hermes_home):
        target = tmp_path / "project" / ".env.example"
        target.parent.mkdir(parents=True)
        target.write_text("KEY=")
        assert get_read_block_error(str(target)) is None

    def test_lookalike_outside_hermes_home_is_not_blocked(
        self, tmp_path, hermes_home
    ):
        # auth.json only matters under a Hermes dir; an unrelated project file
        # with the same name must stay readable.
        target = tmp_path / "someproject" / "auth.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}")
        assert get_read_block_error(str(target)) is None


class TestCacheKeying:
    def test_denylist_is_computed_once_per_home(self, hermes_home):
        target = hermes_home / "skills" / "a" / "SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_text("x")
        get_read_block_error(str(target))
        assert len(file_safety._DENYLIST_CACHE) == 1
        for _ in range(50):
            get_read_block_error(str(target))
        assert len(file_safety._DENYLIST_CACHE) == 1

    def test_switching_home_does_not_reuse_the_previous_denylist(
        self, tmp_path, monkeypatch
    ):
        """The bug a naive module-level cache would introduce."""
        home_a = tmp_path / "a" / ".hermes"
        home_b = tmp_path / "b" / ".hermes"
        for h in (home_a, home_b):
            (h / "mcp-tokens").mkdir(parents=True)
            (h / "auth.json").write_text("{}")

        monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: home_a)
        monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: home_a)
        assert get_read_block_error(str(home_a / "auth.json")) is not None
        # Profile A's store is not profile B's business, and vice versa.
        assert get_read_block_error(str(home_b / "auth.json")) is None

        monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: home_b)
        monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: home_b)
        assert get_read_block_error(str(home_b / "auth.json")) is not None, (
            "second home must build its own denylist, not inherit the first"
        )


class TestPerCallCost:
    def test_repeated_checks_do_not_rewalk_the_filesystem(
        self, hermes_home, monkeypatch
    ):
        """Pins the actual regression: resolution work per call, not wall time.

        A wall-clock threshold would be flaky on a loaded box; counting
        ``Path.resolve`` calls measures the thing that made the skills walk
        slow and stays stable everywhere.
        """
        target = hermes_home / "skills" / "a" / "SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_text("x")
        get_read_block_error(str(target))  # warm the cache

        calls = {"n": 0}
        real_resolve = Path.resolve

        def counting_resolve(self, *a, **kw):
            calls["n"] += 1
            return real_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", counting_resolve)
        for _ in range(10):
            get_read_block_error(str(target))

        # One resolve for the path under test, per call. The denylist itself
        # (24 resolutions before this change) is resolved once and reused.
        assert calls["n"] <= 10, (
            f"expected ~1 resolve per call, got {calls['n']} for 10 calls"
        )

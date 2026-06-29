import importlib
import os
import sys
from pathlib import Path

from hermes_cli.env_loader import load_hermes_dotenv


def test_user_env_overrides_stale_shell_values(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_BASE_URL=https://new.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("OPENAI_BASE_URL") == "https://new.example/v1"


def test_project_env_overrides_stale_shell_values_when_user_env_missing(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    project_env = tmp_path / ".env"
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://project.example/v1"


def test_project_env_is_sanitized_before_loading(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    project_env = tmp_path / ".env"
    project_env.write_text(
        "TELEGRAM_BOT_TOKEN=0123456789:test"
        "ANTHROPIC_API_KEY=sk-ant-test123\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "0123456789:test"
    assert os.getenv("ANTHROPIC_API_KEY") == "sk-ant-test123"


def test_user_env_takes_precedence_over_project_env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    user_env = home / ".env"
    project_env = tmp_path / ".env"
    user_env.write_text("OPENAI_BASE_URL=https://user.example/v1\n", encoding="utf-8")
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\nOPENAI_API_KEY=project-key\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [user_env, project_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://user.example/v1"
    assert os.getenv("OPENAI_API_KEY") == "project-key"


def test_null_bytes_in_user_env_are_stripped(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    # Null bytes can be introduced when copy-pasting API keys.
    env_file.write_text("GLM_API_KEY=abc\x00\x00\nOPENAI_API_KEY=sk-123\n", encoding="utf-8")

    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("GLM_API_KEY") == "abc"
    assert os.getenv("OPENAI_API_KEY") == "sk-123"


def test_main_import_applies_user_env_over_shell_values(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "OPENAI_BASE_URL=https://new.example/v1\nHERMES_INFERENCE_PROVIDER=custom\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")

    sys.modules.pop("hermes_cli.main", None)
    importlib.import_module("hermes_cli.main")

    assert os.getenv("OPENAI_BASE_URL") == "https://new.example/v1"
    assert os.getenv("HERMES_INFERENCE_PROVIDER") == "custom"


def test_profile_inherits_root_env_secret(tmp_path, monkeypatch):
    """A profile home pulls shared secrets from the root ~/.hermes/.env.

    The root .env is the single source of truth; the per-profile .env need not
    (and should not) duplicate shared secrets. Validates the durable
    "one .env, every profile references it" fix.
    """
    root = tmp_path / ".hermes"
    root.mkdir()
    (root / ".env").write_text("CPE_GITHUB_TOKEN=tok-from-root\n", encoding="utf-8")

    profile = root / "profiles" / "code-reviewer"
    profile.mkdir(parents=True)
    # Profile .env is intentionally secret-free.
    (profile / ".env").write_text("# no secrets here\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.delenv("CPE_GITHUB_TOKEN", raising=False)

    loaded = load_hermes_dotenv(hermes_home=profile)

    assert (root / ".env") in loaded
    assert (profile / ".env") in loaded
    # Root .env loads first as the base layer, profile .env on top.
    assert loaded.index(root / ".env") < loaded.index(profile / ".env")
    assert os.getenv("CPE_GITHUB_TOKEN") == "tok-from-root"


def test_profile_env_overrides_root_env(tmp_path, monkeypatch):
    """A profile may override a root value in its own .env (profile wins)."""
    root = tmp_path / ".hermes"
    root.mkdir()
    (root / ".env").write_text("OPENAI_BASE_URL=https://root.example/v1\n", encoding="utf-8")

    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    (profile / ".env").write_text("OPENAI_BASE_URL=https://profile.example/v1\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    load_hermes_dotenv(hermes_home=profile)

    assert os.getenv("OPENAI_BASE_URL") == "https://profile.example/v1"


def test_root_home_does_not_double_load_its_own_env(tmp_path, monkeypatch):
    """The default/root home loads its .env exactly once (no duplicate pass)."""
    root = tmp_path / ".hermes"
    root.mkdir()
    (root / ".env").write_text("OPENAI_API_KEY=sk-root\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=root)

    assert loaded == [root / ".env"]
    assert os.getenv("OPENAI_API_KEY") == "sk-root"


def test_isolated_home_does_not_inherit_real_root_env(tmp_path, monkeypatch):
    """An explicit isolated hermes_home must NOT inherit the real root .env.

    Regression for the secret-isolation bug: when a caller passes an explicit
    hermes_home WITHOUT exporting a matching HERMES_HOME (tests, embeddings),
    the root layer used to be resolved from the HERMES_HOME env var, so the
    user's real ~/.hermes/.env leaked its secrets into the unrelated home.
    The root must be derived from the passed home_path instead — an isolated
    home that is not a <root>/profiles/<name> path has no shared root and
    loads ONLY its own .env.
    """
    # A "real" root .env exists at the platform default location.
    real_root = tmp_path / ".hermes"
    real_root.mkdir()
    (real_root / ".env").write_text("CPE_GITHUB_TOKEN=secret-from-real-root\n", encoding="utf-8")

    # An isolated home elsewhere on disk (NOT under ~/.hermes, NOT a profile).
    isolated = tmp_path / "isolated_home"
    isolated.mkdir()
    (isolated / ".env").write_text("OPENAI_API_KEY=sk-isolated\n", encoding="utf-8")

    # Point Path.home() at tmp_path so the platform default root is the fake
    # real_root above — but do NOT export HERMES_HOME (the buggy code read it).
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("CPE_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=isolated)

    # Only the isolated home's own .env is loaded; the real root .env is NOT.
    assert loaded == [isolated / ".env"]
    assert (real_root / ".env") not in loaded
    assert os.getenv("OPENAI_API_KEY") == "sk-isolated"
    assert os.getenv("CPE_GITHUB_TOKEN") is None


def test_get_default_hermes_root_derives_from_passed_home(tmp_path, monkeypatch):
    """get_default_hermes_root(home) uses the arg, not the HERMES_HOME env var."""
    from hermes_constants import get_default_hermes_root

    native = tmp_path / ".hermes"
    native.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # Env var points somewhere unrelated; the explicit arg must win.
    monkeypatch.setenv("HERMES_HOME", str(native))

    # A profile path resolves to its <root>.
    profile = native / "profiles" / "worker"
    assert get_default_hermes_root(profile) == native

    # An isolated custom home (not under ~/.hermes, not a profile) is its own root.
    isolated = tmp_path / "isolated_home"
    assert get_default_hermes_root(isolated) == isolated

    # No arg → falls back to the env var (historical behavior preserved).
    assert get_default_hermes_root() == native


def test_get_default_hermes_root_resolves_relative_home(tmp_path, monkeypatch):
    """A relative ``home`` must yield an ABSOLUTE root, independent of cwd.

    The returned root is later used to locate ``.env``; a relative path would
    be re-anchored to whatever cwd is active at that point. Resolving upfront
    keeps the root stable.
    """
    from hermes_constants import get_default_hermes_root

    native = tmp_path / ".hermes"
    native.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)

    # Run from inside tmp_path and pass a relative isolated home.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "isolated_home").mkdir()

    root = get_default_hermes_root("isolated_home")
    assert root.is_absolute()
    assert root == (tmp_path / "isolated_home").resolve()

    # A relative profile path still resolves to its absolute <root>.
    (native / "profiles" / "worker").mkdir(parents=True)
    monkeypatch.chdir(native / "profiles")
    rel_profile_root = get_default_hermes_root("worker")
    assert rel_profile_root.is_absolute()
    assert rel_profile_root == native.resolve()


def test_get_default_hermes_root_symlinked_profile_keeps_real_root(tmp_path, monkeypatch):
    """A profile dir that is a symlink outside the root still derives the real root.

    Regression for the codex P2: ``<root>/profiles/<name>`` may be a symlink to
    storage outside the Hermes root. Resolving the home with ``.resolve()`` first
    would follow the link and erase the textual ``profiles/<name>`` segment, so the
    profile-shape check would miss and the function would return the link TARGET as
    its own root -- breaking shared-root ``.env`` secret isolation. The root must be
    derived from the logical profile path BEFORE following symlinks.
    """
    from hermes_constants import get_default_hermes_root

    native = tmp_path / ".hermes"
    (native / "profiles").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)

    # The profile's storage lives OUTSIDE the Hermes root; the logical profile
    # dir <root>/profiles/worker is a symlink to it.
    target = tmp_path / "external_storage" / "worker_data"
    target.mkdir(parents=True)
    logical_profile = native / "profiles" / "worker"
    logical_profile.symlink_to(target, target_is_directory=True)

    # Despite the symlink, the root is the real Hermes root (grandparent of the
    # logical profile path), NOT the link target's parent.
    assert get_default_hermes_root(logical_profile) == native.resolve()


def test_symlinked_profile_still_loads_shared_root_env(tmp_path, monkeypatch):
    """End-to-end: a symlinked profile dir still inherits the shared root .env.

    Ties the get_default_hermes_root symlink fix to the secret-isolation
    guarantee this PR adds: load_hermes_dotenv must load the shared
    <root>/.env BEFORE the profile's own .env even when the profile dir is a
    symlink to storage outside the root.
    """
    root = tmp_path / ".hermes"
    (root / "profiles").mkdir(parents=True)
    (root / ".env").write_text("CPE_GITHUB_TOKEN=shared-root-secret\n", encoding="utf-8")

    target = tmp_path / "external_storage" / "worker_data"
    target.mkdir(parents=True)
    (target / ".env").write_text("OPENAI_API_KEY=sk-profile\n", encoding="utf-8")
    profile = root / "profiles" / "worker"
    profile.symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("CPE_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=profile)

    # Shared root .env loaded as the base layer, profile .env on top.
    assert (root / ".env") in loaded
    assert (profile / ".env") in loaded
    assert os.getenv("CPE_GITHUB_TOKEN") == "shared-root-secret"
    assert os.getenv("OPENAI_API_KEY") == "sk-profile"


def test_get_default_hermes_root_alias_to_native_profile(tmp_path, monkeypatch):
    """A home symlink OUTSIDE ~/.hermes whose target is a native profile is native.

    Regression for codex P2 "Preserve native profile symlink aliases": when
    HERMES_HOME is a symlink outside ~/.hermes that resolves to
    ~/.hermes/profiles/<name>, the root must still be the native root so the
    shared ~/.hermes/.env is loaded (alias/wrapper deployments).
    """
    from hermes_constants import get_default_hermes_root

    native = tmp_path / ".hermes"
    profile = native / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)

    # An alias OUTSIDE the root that points AT the native profile dir.
    alias = tmp_path / "coder_alias"
    alias.symlink_to(profile, target_is_directory=True)

    assert get_default_hermes_root(alias) == native.resolve()


def test_get_default_hermes_root_symlink_under_root_to_outside_is_custom(tmp_path, monkeypatch):
    """A symlink UNDER ~/.hermes pointing to outside storage is a custom home.

    Regression for codex P2 "Do not inherit root secrets for symlinked custom
    homes": ~/.hermes/work -> /mnt/work must resolve to the OUTSIDE target as
    its own root (no shared ~/.hermes/.env leak), not be treated as native
    merely because the entry point is textually under ~/.hermes.
    """
    from hermes_constants import get_default_hermes_root

    native = tmp_path / ".hermes"
    native.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)

    outside = tmp_path / "mnt" / "work-hermes"
    outside.mkdir(parents=True)
    work = native / "work"
    work.symlink_to(outside, target_is_directory=True)

    # The custom home is its own root (the resolved outside target), NOT native.
    assert get_default_hermes_root(work) == outside.resolve()
    assert get_default_hermes_root(work) != native.resolve()


def test_symlinked_custom_home_under_root_does_not_inherit_root_env(tmp_path, monkeypatch):
    """End-to-end: a symlinked custom home under ~/.hermes does NOT load root .env."""
    native = tmp_path / ".hermes"
    native.mkdir()
    (native / ".env").write_text("CPE_GITHUB_TOKEN=root-secret\n", encoding="utf-8")

    outside = tmp_path / "mnt" / "work-hermes"
    outside.mkdir(parents=True)
    (outside / ".env").write_text("OPENAI_API_KEY=sk-work\n", encoding="utf-8")
    work = native / "work"
    work.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("CPE_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=work)

    # Only the custom home's own .env loads; the native root .env is NOT inherited.
    assert (native / ".env") not in loaded
    assert os.getenv("OPENAI_API_KEY") == "sk-work"
    assert os.getenv("CPE_GITHUB_TOKEN") is None


def test_empty_hermes_home_env_falls_back_to_default_root(tmp_path, monkeypatch):
    """A bare HERMES_HOME= export must fall back to ~/.hermes, not cwd.

    Regression for codex P2 "Treat empty HERMES_HOME as unset": an empty
    HERMES_HOME used to make home_path == Path("") (cwd), so the root was the
    working directory and the real ~/.hermes/.env was skipped.
    """
    native = tmp_path / ".hermes"
    native.mkdir()
    (native / ".env").write_text("CPE_GITHUB_TOKEN=root-secret\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HERMES_HOME", "")  # exported empty, NOT unset
    monkeypatch.delenv("CPE_GITHUB_TOKEN", raising=False)
    # Run from an unrelated cwd to prove the root is not derived from it.
    work_cwd = tmp_path / "some" / "cwd"
    work_cwd.mkdir(parents=True)
    monkeypatch.chdir(work_cwd)

    loaded = load_hermes_dotenv()  # no explicit hermes_home

    assert (native / ".env") in loaded
    assert os.getenv("CPE_GITHUB_TOKEN") == "root-secret"

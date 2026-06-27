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

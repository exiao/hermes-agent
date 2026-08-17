import codecs
import importlib
import os
import sys
from pathlib import Path

from hermes_cli.env_loader import load_hermes_dotenv


def test_recovered_update_retry_skips_external_secret_sources(tmp_path, monkeypatch):
    """The post-recovery updater must not remap native vault dependencies."""
    import hermes_cli.env_loader as env_loader
    from hermes_cli import _early_recovery

    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("UPDATE_RETRY_DOTENV=loaded\n", encoding="utf-8")
    monkeypatch.delenv("UPDATE_RETRY_DOTENV", raising=False)
    monkeypatch.setattr(_early_recovery, "_UPDATE_RETRY_RECOVERED", True)
    external_calls = []
    monkeypatch.setattr(
        env_loader,
        "_apply_external_secret_sources",
        lambda path: external_calls.append(path),
    )

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.environ["UPDATE_RETRY_DOTENV"] == "loaded"
    assert external_calls == []


def test_utf8_bom_does_not_mangle_first_key(tmp_path, monkeypatch):
    """A leading UTF-8 BOM must not prefix the first key name in os.environ.

    PowerShell 5.1 ``Set-Content -Encoding UTF8`` and Windows Notepad write
    a BOM (EF BB BF). With encoding=utf-8, python-dotenv keeps U+FEFF on the
    first key so the canonical name is absent and callers see "not configured".
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_bytes(
        b"\xef\xbb\xbfFIRST_KEY=first-value\nSECOND_KEY=second-value\n"
    )

    monkeypatch.delenv("FIRST_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)
    monkeypatch.delenv("\ufeffFIRST_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("FIRST_KEY") == "first-value"
    assert os.getenv("SECOND_KEY") == "second-value"
    assert os.environ.get("\ufeffFIRST_KEY") is None


def test_bomless_utf8_env_still_loads(tmp_path, monkeypatch):
    """BOM-less UTF-8 .env files must keep loading after utf-8-sig."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-plain\nSECOND_KEY=ok\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("OPENAI_API_KEY") == "sk-plain"
    assert os.getenv("SECOND_KEY") == "ok"


def test_latin1_env_falls_back(tmp_path, monkeypatch):
    """Invalid UTF-8 bytes must still load via the latin-1 fallback."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    # 0xE9 is "é" in latin-1 and not a valid UTF-8 lead sequence alone.
    env_file.write_bytes(b"LATIN1_VALUE=caf\xe9\n")

    monkeypatch.delenv("LATIN1_VALUE", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("LATIN1_VALUE") == "café"


def test_utf8_bom_preserves_first_api_key_name(tmp_path, monkeypatch):
    """Real-world case: BOM + first line is a provider API key name."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_bytes(
        b"\xef\xbb\xbfANTHROPIC_API_KEY=sk-test-123\nSECOND_KEY=ok\n"
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)
    monkeypatch.delenv("\ufeffANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("ANTHROPIC_API_KEY") == "sk-test-123"
    assert os.getenv("SECOND_KEY") == "ok"
    assert os.environ.get("\ufeffANTHROPIC_API_KEY") is None


def test_utf8_bom_plus_invalid_utf8_preserves_first_key(tmp_path, monkeypatch):
    """BOM + non-UTF-8 body must load via latin-1 without mangling the first key.

    utf-8-sig only applies on the primary path. When invalid UTF-8 forces the
    latin-1 fallback, a leading EF BB BF would otherwise become part of the
    first key name under latin-1 and drop the canonical name.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    # BOM + valid first key + latin-1 é (0xE9) in a later value.
    env_file.write_bytes(
        b"\xef\xbb\xbfANTHROPIC_API_KEY=sk-test-123\nBAD=caf\xe9\n"
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("BAD", raising=False)
    monkeypatch.delenv("\ufeffANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("ANTHROPIC_API_KEY") == "sk-test-123"
    assert os.getenv("BAD") == "café"
    assert os.environ.get("\ufeffANTHROPIC_API_KEY") is None

def test_bomless_latin1_env_still_loads(tmp_path, monkeypatch):
    """BOM-less cp1252/latin-1 .env files must keep loading after the BOM strip."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_bytes(b"LATIN1_VALUE=caf\xe9\nOTHER=ok\n")

    monkeypatch.delenv("LATIN1_VALUE", raising=False)
    monkeypatch.delenv("OTHER", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("LATIN1_VALUE") == "café"
    assert os.getenv("OTHER") == "ok"

def test_latin1_fallback_stream_honors_override(tmp_path, monkeypatch):
    """Stream-based latin-1 fallback must honor override= identically to dotenv_path."""
    from hermes_cli.env_loader import _load_dotenv_with_fallback

    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    # Invalid UTF-8 forces the stream/latin-1 path.
    env_file.write_bytes(b"OVERRIDE_PROBE=from-file\nLATIN1_VALUE=caf\xe9\n")

    monkeypatch.setenv("OVERRIDE_PROBE", "from-shell")
    monkeypatch.delenv("LATIN1_VALUE", raising=False)

    # override=False: shell value must win (same as dotenv_path form).
    _load_dotenv_with_fallback(env_file, override=False)
    assert os.getenv("OVERRIDE_PROBE") == "from-shell"
    assert os.getenv("LATIN1_VALUE") == "café"

    # override=True: file value must win (user-env path).
    _load_dotenv_with_fallback(env_file, override=True)
    assert os.getenv("OVERRIDE_PROBE") == "from-file"
    assert os.getenv("LATIN1_VALUE") == "café"

def test_latin1_fallback_stream_preserves_interpolation(tmp_path, monkeypatch):
    """Stream/latin-1 path must still expand ${VAR} like the dotenv_path form."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    # 0xE9 forces latin-1 fallback; ${FOO} must still expand.
    env_file.write_bytes(b"FOO=bar\nBAR=${FOO}\nLATIN1_VALUE=caf\xe9\n")

    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAR", raising=False)
    monkeypatch.delenv("LATIN1_VALUE", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("FOO") == "bar"
    assert os.getenv("BAR") == "bar"
    assert os.getenv("LATIN1_VALUE") == "café"

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
# ---------------------------------------------------------------------------
# UTF-16 / UTF-32 .env sanitizer coverage
#
# UTF-8 BOM handling for _load_dotenv_with_fallback is covered above (#65124).
# This section covers the sanitizer rewrite path for UTF-16/32 (and UTF-8 /
# cp1252 regression guards for that path).
# ---------------------------------------------------------------------------


def _assert_clean_utf8_env_on_disk(env_file, *, first_key: str) -> None:
    """On-disk file must be clean UTF-8: no BOM, no U+FFFD, canonical key."""
    after = env_file.read_bytes()
    assert not after.startswith(codecs.BOM_UTF8)
    assert not after.startswith(codecs.BOM_UTF16_LE)
    assert not after.startswith(codecs.BOM_UTF16_BE)
    text = after.decode("utf-8")  # strict — raises if not clean UTF-8
    assert "\ufffd" not in text
    assert text.startswith(f"{first_key}=") or f"\n{first_key}=" in text
    assert first_key.encode("ascii") in after




def test_utf16_le_bom_preserves_non_ascii_values(tmp_path, monkeypatch):
    """UTF-16-LE+BOM rewrite must preserve non-ASCII values (not just ASCII keys).

    Uses non-credential var names so _sanitize_loaded_credentials does not
    strip non-ASCII from values (that path only targets *_KEY/*_TOKEN/etc.).
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    content = "GREETING=café\nCJK_LABEL=日本語\n"
    env_file.write_bytes(codecs.BOM_UTF16_LE + content.encode("utf-16-le"))

    monkeypatch.delenv("GREETING", raising=False)
    monkeypatch.delenv("CJK_LABEL", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("GREETING") == "café"
    assert os.getenv("CJK_LABEL") == "日本語"
    after = env_file.read_bytes()
    assert after.decode("utf-8")  # strict
    assert "café".encode("utf-8") in after
    assert "日本語".encode("utf-8") in after
    assert b"\xef\xbf\xbd" not in after


def test_utf32_le_bom_leaves_file_untouched(tmp_path, caplog):
    """UTF-32-LE BOM: refuse-to-mangle (leave bytes untouched + warning).

    UTF-32-LE's BOM starts with UTF-16-LE's FF FE; sniff order must check
    UTF-32 first so we never misdetect and corrupt.

    Exercises ``_sanitize_env_file_if_needed`` only: the dotenv load path
    is out of scope here (#65124's surface) and still cannot ingest UTF-32.
    """
    import logging

    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    env_file = tmp_path / ".env"
    content = "HERMES_TEST_KEY=hello_utf32\nSECOND_KEY=world\n"
    raw = codecs.BOM_UTF32_LE + content.encode("utf-32-le")
    env_file.write_bytes(raw)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        _sanitize_env_file_if_needed(env_file)

    assert env_file.read_bytes() == raw  # untouched
    assert any("UTF-32" in r.message for r in caplog.records)




def test_utf32_warning_fires_once_per_path(tmp_path, caplog, monkeypatch):
    """Three sanitize calls on the same UTF-32 file → exactly one warning.

    Matches house style for warn-once (module-level seen-set, same class as
    ``_WARNED_KEYS``): hot-reload / multi-entry load must not spam logs.
    """
    import logging

    import hermes_cli.env_loader as env_loader
    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    # Isolate process-level seen-set so other tests' paths don't leak in.
    monkeypatch.setattr(env_loader, "_WARNED_UTF32_PATHS", set())

    env_file = tmp_path / ".env"
    content = "HERMES_TEST_KEY=hello_utf32\nSECOND_KEY=world\n"
    raw = codecs.BOM_UTF32_LE + content.encode("utf-32-le")
    env_file.write_bytes(raw)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        _sanitize_env_file_if_needed(env_file)
        _sanitize_env_file_if_needed(env_file)
        _sanitize_env_file_if_needed(env_file)

    utf32_warnings = [r for r in caplog.records if "UTF-32" in r.message]
    assert len(utf32_warnings) == 1
    assert env_file.read_bytes() == raw




def test_plain_utf8_env_regression(tmp_path, monkeypatch):
    """Plain UTF-8 .env must keep loading after the UTF-16 sanitize changes."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    before = b"OPENAI_API_KEY=sk-plain\nSECOND_KEY=ok\n"
    env_file.write_bytes(before)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("OPENAI_API_KEY") == "sk-plain"
    assert os.getenv("SECOND_KEY") == "ok"
    # No spurious rewrite of an already-clean file.
    assert env_file.read_bytes() == before


def test_cp1252_env_regression_does_not_crash(tmp_path, monkeypatch):
    """cp1252/latin-1 body must not crash sanitize; ASCII keys still usable.

    0xE9 is 'é' in cp1252 and incomplete as UTF-8. First line does not begin
    with U+FFFD, so the FFFD guard must not refuse the whole file.

    Sanitize leaves the file bytes alone when the only "change" is
    errors=replace on values (original already replace-decoded equals
    sanitized), so _load_dotenv_with_fallback's latin-1 path recovers café.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    before = b"ASCII_KEY=ok\nLATIN1_VALUE=caf\xe9\n"
    env_file.write_bytes(before)

    monkeypatch.delenv("ASCII_KEY", raising=False)
    monkeypatch.delenv("LATIN1_VALUE", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("ASCII_KEY") == "ok"
    assert os.getenv("LATIN1_VALUE") == "café"
    # Sanitize must not have rewritten (would have persisted U+FFFD).
    assert env_file.read_bytes() == before


# ---------------------------------------------------------------------------
# Profile .env isolation: inherited known-key cleanup
# ---------------------------------------------------------------------------


def test_known_keys_absent_from_user_env_are_cleared(tmp_path, monkeypatch):
    """Known Hermes keys inherited from parent process are removed when absent
    from the profile's .env.

    This is the startup equivalent of ``reload_env()``'s known-key cleanup and
    fixes the isolation gap where one profile's ACP/provider settings silently
    leak into another profile's runtime via ``os.environ`` inheritance.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "OPENAI_BASE_URL=https://profile.example/v1\n", encoding="utf-8"
    )

    # Inherited known keys from parent process / other profile
    monkeypatch.setenv("OPENAI_BASE_URL", "https://stale.example/v1")
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")
    monkeypatch.setenv("COPILOT_CLI_PATH", "/usr/bin/claude-code")
    # Unrelated shell var must NOT be touched
    monkeypatch.setenv("MY_SHELL_ONLY_VAR", "keep-me")

    load_hermes_dotenv(hermes_home=home)

    # OPENAI_BASE_URL is defined in the profile .env → overridden to the new value
    assert os.getenv("OPENAI_BASE_URL") == "https://profile.example/v1"
    # HERMES_ACP_AUTH_METHOD and COPILOT_CLI_PATH are NOT in the profile .env → cleared
    assert "HERMES_ACP_AUTH_METHOD" not in os.environ
    assert "COPILOT_CLI_PATH" not in os.environ
    # Unrelated shell vars must survive
    assert os.getenv("MY_SHELL_ONLY_VAR") == "keep-me"


def test_empty_assignment_in_user_env_is_preserved(tmp_path, monkeypatch):
    """An explicit ``KEY=`` (empty value) in the profile .env keeps the key
    in ``os.environ`` — distinct from a key absent from .env entirely.

    Empty ``HERMES_ACP_AUTH_METHOD=`` tells the ACP adapter to skip
    ``authenticate`` (the key exists, its value is just empty).  This is the
    documented workaround for the leak and must still work after the cleanup.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("HERMES_ACP_AUTH_METHOD=\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")
    monkeypatch.setenv("COPILOT_CLI_PATH", "/usr/bin/sneaky")  # NOT in .env → cleared

    load_hermes_dotenv(hermes_home=home)

    # KEY= in .env keeps the key (now empty string)
    assert "HERMES_ACP_AUTH_METHOD" in os.environ
    assert os.environ["HERMES_ACP_AUTH_METHOD"] == ""
    # COPILOT_CLI_PATH is absent from .env → cleared
    assert "COPILOT_CLI_PATH" not in os.environ


def test_no_user_env_does_not_clear_anything(tmp_path, monkeypatch):
    """When no profile .env exists (bare profile), load_hermes_dotenv must not
    wipe inherited known keys — the bare-profile case follows #66930 / #67027
    semantics and the user's shell environment should not be mutilated.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    # No .env in home — bare profile

    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_ACP_AUTH_METHOD") == "cursor_login"
    assert os.getenv("PATH") == "/usr/bin:/bin"


def test_known_key_explicitly_set_in_user_env_is_kept(tmp_path, monkeypatch):
    """A known Hermes key that IS explicitly set in the profile .env survives
    the cleanup (overrides the inherited value).
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "HERMES_ACP_AUTH_METHOD=claude_code_cli\n", encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_ACP_AUTH_METHOD") == "claude_code_cli"


def test_export_prefixed_known_key_in_user_env_is_kept(tmp_path, monkeypatch):
    """A known Hermes key defined with the bash-compatible ``export KEY=value``
    form in the profile .env must be recognized as defined and survive the
    cleanup - mirrors the ``export `` stripping in config.py's load_env()
    (#6659).
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "export HERMES_ACP_AUTH_METHOD=claude_code_cli\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")
    load_hermes_dotenv(hermes_home=home)
    assert os.getenv("HERMES_ACP_AUTH_METHOD") == "claude_code_cli"


def test_shell_exported_credentials_survive_cleanup(tmp_path, monkeypatch):
    """User-shell-exported provider credentials must NOT be scrubbed.

    ``export OPENAI_API_KEY=…`` in the shell with a ``.env`` that doesn't
    contain the key is a documented, legitimate flow (see
    test_dump_env_visibility.py). The startup cleanup is scoped to
    _PROFILE_MANAGED_ENV_KEYS (ACP routing keys) precisely so it can never
    delete shell-supplied credentials — a process cannot distinguish a
    shell export from parent-process leakage, so credential isolation is
    owned by read-time secret scoping instead.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("SOME_OTHER_KEY=x\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:token-from-shell")
    # A profile-managed routing key inherited alongside them IS cleared.
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("OPENAI_API_KEY") == "sk-from-shell"
    assert os.getenv("ANTHROPIC_API_KEY") == "sk-ant-from-shell"
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "12345:token-from-shell"
    assert "HERMES_ACP_AUTH_METHOD" not in os.environ


def test_cleanup_scope_is_the_profile_managed_set():
    """Lock the invariant: the startup scrub set contains only behavioral
    ACP/routing keys — never credential-shaped keys. If this fails, someone
    widened _PROFILE_MANAGED_ENV_KEYS toward the full known-key set, which
    re-introduces the shell-export deletion bug.
    """
    from hermes_cli.env_loader import _PROFILE_MANAGED_ENV_KEYS

    for key in _PROFILE_MANAGED_ENV_KEYS:
        assert not key.endswith(("_API_KEY", "_TOKEN", "_SECRET")), (
            f"{key} looks credential-shaped; startup scrub must not "
            "cover credentials — read-time secret scoping owns those"
        )


# ---------------------------------------------------------------------------
# config.yaml terminal.* re-apply after dotenv loads (#29186 / #67323)
#
# load_hermes_dotenv loads .env with override=True, so a stale
# TERMINAL_ENV=docker in .env used to silently beat config.yaml's
# terminal.backend on every reload (gateway per-turn reload, cron standalone
# runs). The bridge re-applies config.yaml's EXPLICIT terminal keys last via
# the shared hermes_cli.config.apply_terminal_config_to_env helper.
# ---------------------------------------------------------------------------


def _seed_terminal_home(tmp_path, monkeypatch, *, config_yaml=None, env_text=None):
    home = tmp_path / "hermes"
    home.mkdir()
    if config_yaml is not None:
        (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    if env_text is not None:
        (home / ".env").write_text(env_text, encoding="utf-8")
    # The bridge is scoped to the process HERMES_HOME (a different profile's
    # load must not bridge this process's config), so point the process at
    # the seeded home like a real gateway/cron process would be.
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_config_yaml_terminal_backend_overrides_stale_env(tmp_path, monkeypatch):
    """Regression for #29186: a leftover TERMINAL_ENV=docker in ~/.hermes/.env
    must not silently override the user's choice in config.yaml. config.yaml
    is the documented source of truth, so its value must win after load."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="terminal:\n  backend: local\n",
        env_text="TERMINAL_ENV=docker\n",
    )

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "local"


def test_config_yaml_terminal_backend_overrides_stale_shell(tmp_path, monkeypatch):
    """config.yaml must also beat a stale TERMINAL_ENV exported in the shell
    (e.g. set in ~/.zshrc when the user was experimenting with docker)."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="terminal:\n  backend: local\n",
    )

    monkeypatch.setenv("TERMINAL_ENV", "docker")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "local"


def test_no_terminal_section_leaves_env_value_alone(tmp_path, monkeypatch):
    """When config.yaml has no terminal section, the .env value is still the
    user's active setting — the bridge must NOT clobber it with merged
    defaults."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="display:\n  streaming: true\n",
        env_text="TERMINAL_ENV=docker\n",
    )

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "docker"


def test_config_yaml_terminal_omitted_key_does_not_clear_env(tmp_path, monkeypatch):
    """If config.yaml has a terminal block but no `backend`, the .env value
    must survive (only explicit config keys override env)."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="terminal:\n  timeout: 600\n",
        env_text="TERMINAL_ENV=docker\n",
    )

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "docker"
    assert os.getenv("TERMINAL_TIMEOUT") == "600"


def test_other_profile_home_does_not_bridge_process_config(tmp_path, monkeypatch):
    """Loading a DIFFERENT profile's .env must not re-bridge this process's
    config.yaml — the shared bridge reads the process-global config, so
    applying it for another home would stamp the wrong profile's terminal
    settings into the env."""
    process_home = tmp_path / "process-home"
    process_home.mkdir()
    (process_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(process_home))

    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    (other_home / ".env").write_text("TERMINAL_ENV=docker\n", encoding="utf-8")

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=other_home)

    # The other profile's .env value stands; the process config was not applied.
    assert os.getenv("TERMINAL_ENV") == "docker"

"""Regression: os.environ mutation race in load_dotenv -> KeyError.

python-dotenv's ``resolve_variables`` does ``env.update(os.environ)`` once
per interpolated ``.env`` line. ``dict.update(os._Environ)`` enumerates the
mapping's keys, then fetches each by key. ``os.environ`` is a live, shared
mutable mapping; Hermes pins/unpins ``HERMES_KANBAN_BOARD`` on it on the fly
(hermes_cli/main.py, gateway/kanban_watchers.py). When a reload's enumerate
races a concurrent unpin, the key is listed but gone by the time it's
fetched -> ``KeyError`` (crashed a chat turn, 2026-06-29).

These tests install a racing ``os.environ`` stand-in that drops a key after
the first full enumeration -- exactly what a concurrent unpin does between
two of dotenv's per-line ``env.update(os.environ)`` passes. On the pre-fix
code the second pass raises ``KeyError``; after the fix
``_load_dotenv_with_fallback`` snapshots the environment once and feeds
dotenv the frozen copy, so no later pass can observe the deletion.
"""

from collections.abc import MutableMapping
import os
import subprocess
import sys
import threading

import pytest

from dotenv import load_dotenv as dotenv_load_dotenv

import hermes_cli.env_loader as env_loader


class _RacingEnviron:
    """os.environ-like mapping that drops a key mid-life, like a concurrent unpin.

    Mimics ``os._Environ`` enough for ``dict.update(self)`` and ``in``:
    ``keys()`` + ``__getitem__``. The volatile key stays *enumerated* (so a
    reader that calls ``keys()`` then fetches each still tries to fetch it),
    but its value disappears after the first full enumeration -- so the very
    next ``__getitem__`` for it raises ``KeyError``. That is exactly the
    enumerate-then-get TOCTOU: a key listed by ``keys()`` is gone by the time
    it's fetched.

    The loader's one-time snapshot reads the value during the first
    enumeration (while it's still present) and never re-reads the live
    mapping, so it is immune. An *unfixed* reader that re-reads ``os.environ``
    once per ``.env`` line hits the second enumeration and raises.
    """

    def __init__(self, base: dict, volatile_key: str, volatile_value: str):
        self._base = dict(base)
        self._volatile_key = volatile_key
        self._volatile_value = volatile_value
        self._enumerations = 0
        # Once True, fetching the volatile key raises (the unpin happened).
        self._dropped = False
        # Write-through record so dotenv's set_as_environment_variables
        # (os.environ[k] = v) doesn't explode; we only care about the read
        # race here.
        self.writes: dict[str, str] = {}

    def _all_keys(self):
        keys = list(self._base.keys())
        keys.append(self._volatile_key)
        keys.extend(k for k in self.writes if k not in keys)
        return keys

    def keys(self):
        self._enumerations += 1
        # After the first full enumeration completed, the concurrent unpin
        # has landed: the value is gone even though the key was just listed.
        if self._enumerations >= 2:
            self._dropped = True
        return self._all_keys()

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key):
        if key == self._volatile_key:
            if self._dropped and key not in self.writes:
                raise KeyError(key)
            if key in self.writes:
                return self.writes[key]
            return self._volatile_value
        if key in self.writes:
            return self.writes[key]
        return self._base[key]

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __setitem__(self, key, value):
        self.writes[key] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        # Used by _sanitize_loaded_credentials (list(os.environ.items())).
        out = []
        for k in self._all_keys():
            try:
                out.append((k, self[k]))
            except KeyError:
                pass
        return out


class _InitialSnapshotRaceEnviron(_RacingEnviron):
    """Drops the volatile key during the initial snapshot's keys/get window."""

    def keys(self):
        self._enumerations += 1
        self._dropped = True
        return self._all_keys()


class _CaseNormalizingEnviron(dict):
    """Windows-like env mapping that normalizes keys case-insensitively."""

    def encodekey(self, key):
        return key.upper()

    def __getitem__(self, key):
        return super().__getitem__(self.encodekey(key))

    def __setitem__(self, key, value):
        return super().__setitem__(self.encodekey(key), value)

    def __delitem__(self, key):
        return super().__delitem__(self.encodekey(key))


def _write_interpolating_env(tmp_path):
    """A .env with >=2 interpolated lines.

    Interpolation (the ``${...}`` / value-bearing lines) is what forces
    python-dotenv into the ``env.update(os.environ)`` branch, once per line,
    giving multiple read windows.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FOO=alpha-${HERMES_KANBAN_BOARD}\n"
        "BAR=beta-${HERMES_KANBAN_BOARD}\n"
        "BAZ=gamma\n",
        encoding="utf-8",
    )
    return env_file


def test_pre_fix_environ_race_raises_keyerror(tmp_path, monkeypatch):
    """Without the snapshot, dotenv's per-line re-read trips the TOCTOU."""
    env_file = _write_interpolating_env(tmp_path)
    racing = _RacingEnviron(
        {"PATH": "/usr/bin"}, "HERMES_KANBAN_BOARD", "main"
    )
    # dotenv reads the module-global ``os.environ``; point it at the racer.
    monkeypatch.setattr(os, "environ", racing)

    # Call python-dotenv directly (the UNFIXED reader path) and prove it
    # raises -- this is the bug, and it guards against a future change that
    # silently makes the snapshot a no-op.
    raised = False
    try:
        dotenv_load_dotenv(dotenv_path=env_file, override=True, encoding="utf-8")
    except KeyError as exc:
        raised = "HERMES_KANBAN_BOARD" in str(exc)
    assert raised, "expected the os.environ mutation race to raise KeyError pre-fix"


def test_loader_snapshots_environ_and_survives_race(tmp_path, monkeypatch):
    """The fixed loader snapshots os.environ once -> no KeyError under the race."""
    env_file = _write_interpolating_env(tmp_path)
    racing = _RacingEnviron(
        {"PATH": "/usr/bin"}, "HERMES_KANBAN_BOARD", "main"
    )
    # Both dotenv's module and the loader resolve ``os.environ`` through the
    # ``os`` module; patch it there so the loader snapshots the racer.
    monkeypatch.setattr(os, "environ", racing)

    # Must NOT raise. The loader snapshots os.environ before handing it to
    # python-dotenv, so the volatile key is stable for every per-line pass.
    env_loader._load_dotenv_with_fallback(env_file, override=True)

    # And the interpolation actually resolved against the snapshot value
    # (write-through landed on the real/raced mapping).
    assert racing.get("FOO") == "alpha-main"
    assert racing.get("BAR") == "beta-main"
    assert racing.get("BAZ") == "gamma"


def test_stable_environ_is_mutable_mapping_with_snapshot_copy():
    """Mapping helpers should stay coherent through MutableMapping methods."""
    real = {"EXISTING": "old"}
    stable = env_loader._StableEnviron(real)

    assert isinstance(stable, MutableMapping)
    assert stable.copy() == {"EXISTING": "old"}

    stable.update({"NEW": "value"})
    assert real["NEW"] == "value"
    assert stable["NEW"] == "value"

    copied = stable.copy()
    real["EXISTING"] = "mutated-outside-snapshot"
    assert copied["EXISTING"] == "old"
    assert stable["EXISTING"] == "old"


def test_stable_environ_preserves_case_normalized_key_reads():
    """Windows normalizes env keys, so snapshot lookups must too."""
    real = _CaseNormalizingEnviron()
    real["PATH"] = "old"
    stable = env_loader._StableEnviron(real)

    assert "Path" in stable
    assert stable["Path"] == "old"
    assert stable.get("Path") == "old"
    assert stable.setdefault("Path", "new") == "old"
    assert real["PATH"] == "old"


def test_stable_environ_delete_does_not_mutate_read_snapshot():
    """Concurrent unpin deletes must not invalidate an active keys/get pass."""
    real = {"HERMES_KANBAN_BOARD": "main"}
    stable = env_loader._StableEnviron(real)

    keys = stable.keys()
    assert stable.pop("HERMES_KANBAN_BOARD", None) == "main"

    assert "HERMES_KANBAN_BOARD" in keys
    assert stable["HERMES_KANBAN_BOARD"] == "main"
    assert stable.get("HERMES_KANBAN_BOARD") is None
    assert "HERMES_KANBAN_BOARD" not in stable
    assert "HERMES_KANBAN_BOARD" not in stable.copy()
    assert stable.pop("HERMES_KANBAN_BOARD", None) is None
    assert "HERMES_KANBAN_BOARD" not in real


def test_stable_environ_post_snapshot_writes_stay_out_of_read_keys():
    real = {}
    stable = env_loader._StableEnviron(real)

    stable["HERMES_KANBAN_BOARD"] = "main"
    keys = stable.keys()
    del stable["HERMES_KANBAN_BOARD"]

    assert "HERMES_KANBAN_BOARD" not in keys
    assert stable.get("HERMES_KANBAN_BOARD") is None
    assert "HERMES_KANBAN_BOARD" not in stable
    assert "HERMES_KANBAN_BOARD" not in real


def test_loader_tolerates_key_deleted_during_initial_snapshot(tmp_path, monkeypatch):
    """The first stable snapshot must not reintroduce the same TOCTOU crash."""
    env_file = _write_interpolating_env(tmp_path)
    racing = _InitialSnapshotRaceEnviron(
        {"PATH": "/usr/bin"}, "HERMES_KANBAN_BOARD", "main"
    )
    monkeypatch.setattr(os, "environ", racing)

    env_loader._load_dotenv_with_fallback(env_file, override=True)

    assert racing.get("FOO") == "alpha-"
    assert racing.get("BAR") == "beta-"
    assert racing.get("BAZ") == "gamma"


def test_load_dotenv_swap_is_serialized_by_process_lock(tmp_path, monkeypatch):
    """Concurrent loads should wait for the process-wide os.environ swap lock."""
    env_file = _write_interpolating_env(tmp_path)
    entered = threading.Event()

    def fake_load_dotenv(**_kwargs):
        entered.set()
        return True

    monkeypatch.setattr(env_loader, "load_dotenv", fake_load_dotenv)

    env_loader._ENV_LOAD_LOCK.acquire()
    worker = threading.Thread(
        target=env_loader._load_dotenv_with_fallback,
        kwargs={"path": env_file, "override": True},
    )
    worker.start()
    try:
        assert not entered.wait(0.05)
    finally:
        env_loader._ENV_LOAD_LOCK.release()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert entered.is_set()


def test_stable_environ_rejects_bytes_keys_like_real_environ():
    """Regression: bytes keys must not alias str entries.

    ``_encode_key`` used to swallow the ``TypeError`` from
    ``os._Environ.encodekey`` and fall back to the raw key, so
    ``stable[b'PATH']`` returned the value of the str key ``'PATH'``.
    ``os.get_exec_path`` reads both spellings, saw two, and raised
    ``ValueError: env cannot contain 'PATH' and b'PATH' keys`` -- every
    subprocess spawned while the wrapper was installed died (observed
    2026-08-28: kanban dispatcher ticks + the earnings-card cron).
    """
    stable = env_loader._StableEnviron(os.environ)

    for probe in (
        lambda m: m[b"PATH"],
        lambda m: m.get(b"PATH"),
        lambda m: b"PATH" in m,
    ):
        with pytest.raises(TypeError):
            probe(os.environ)
        with pytest.raises(TypeError):
            probe(stable)

    assert stable["PATH"] == os.environ["PATH"]


def test_subprocess_works_while_stable_environ_is_installed(monkeypatch):
    """The wrapper must survive os.get_exec_path / subprocess.run."""
    stable = env_loader._StableEnviron(os.environ)
    monkeypatch.setattr(os, "environ", stable)

    assert os.get_exec_path()
    proc = subprocess.run(
        [sys.executable, "-c", "print('ok')"],
        stdout=subprocess.PIPE, text=True, timeout=60, check=True,
    )
    assert proc.stdout.strip() == "ok"

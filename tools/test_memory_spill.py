#!/usr/bin/env python3
"""Tests for memory_tool overflow → spill-to-pending behavior.

Run: python3 tools/test_memory_spill.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fresh_home():
    d = tempfile.mkdtemp(prefix="memspill-")
    os.environ["HERMES_HOME"] = d
    (Path(d) / "memories").mkdir(parents=True, exist_ok=True)
    (Path(d) / "episodes").mkdir(parents=True, exist_ok=True)
    return Path(d)


def _store(limit):
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import tools.memory_tool as mt
    importlib.reload(mt)
    s = mt.MemoryStore(memory_char_limit=limit, user_char_limit=limit)
    s.load_from_disk()
    return s, mt


def check(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond:
        raise SystemExit(1)


def test_under_limit_writes_to_file():
    home = _fresh_home()
    s, _ = _store(2000)
    r = s.add("memory", "[2026-06-17][fact] small entry one")
    check("under-limit add succeeds", r.get("success"))
    txt = (home / "memories" / "MEMORY.md").read_text()
    check("under-limit add lands in MEMORY.md", "small entry one" in txt)
    pend = home / "episodes" / ".pending.md"
    check("under-limit add does NOT spill", not pend.exists() or pend.read_text().strip() == "")


def test_over_limit_spills():
    home = _fresh_home()
    s, _ = _store(120)
    s.add("memory", "[2026-06-17][rule] " + "x" * 90)  # fills near limit
    big = "[2026-06-17][fact] this entry pushes past the configured limit boundary"
    r = s.add("memory", big)
    check("over-limit add returns success (no drop)", r.get("success"))
    check("over-limit add reports spilled_to_pending", r.get("spilled_to_pending") is True)
    pend = (home / "episodes" / ".pending.md").read_text()
    check("over-limit entry is in pending", "pushes past the configured limit" in pend)
    check("pending line is TARGET-tab prefixed", pend.startswith("memory\t"))


def test_over_limit_dedups():
    home = _fresh_home()
    s, _ = _store(120)
    s.add("memory", "[2026-06-17][rule] " + "x" * 90)
    big = "[2026-06-17][fact] duplicate overflow fact that should only land once"
    s.add("memory", big)
    r2 = s.add("memory", big)
    check("repeated overflow still returns success", r2.get("success"))
    check("repeated overflow is deduped (not spilled twice)", r2.get("spilled_to_pending") is False)
    pend = (home / "episodes" / ".pending.md").read_text()
    check("pending contains the fact exactly once", pend.count("duplicate overflow fact") == 1)


if __name__ == "__main__":
    print("[1] under-limit add writes to file, no spill")
    test_under_limit_writes_to_file()
    print("[2] over-limit add spills to pending, returns success")
    test_over_limit_spills()
    print("[3] repeated over-limit add dedups")
    test_over_limit_dedups()
    print("\nALL PASS")

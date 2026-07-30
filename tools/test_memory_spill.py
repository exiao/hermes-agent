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
    txt = (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    check("under-limit add lands in MEMORY.md", "small entry one" in txt)
    pend = home / "episodes" / ".pending.md"
    check("under-limit add does NOT spill", not pend.exists() or pend.read_text(encoding="utf-8").strip() == "")


def test_over_limit_spills():
    home = _fresh_home()
    s, _ = _store(120)
    s.add("memory", "[2026-06-17][rule] " + "x" * 90)  # fills near limit
    big = "[2026-06-17][fact] this entry pushes past the configured limit boundary"
    r = s.add("memory", big)
    check("over-limit add returns success (no drop)", r.get("success"))
    check("over-limit add reports spilled_to_pending", r.get("spilled_to_pending") is True)
    pend = (home / "episodes" / ".pending.md").read_text(encoding="utf-8")
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
    pend = (home / "episodes" / ".pending.md").read_text(encoding="utf-8")
    check("pending contains the fact exactly once", pend.count("duplicate overflow fact") == 1)


def test_cross_target_does_not_false_dedup():
    """A user add must not be suppressed by a same-text memory line already pending."""
    home = _fresh_home()
    s, mt = _store(120)
    # Seed pending with a memory-target line, then spill the same text as user.
    mt._spill_to_pending("memory", "[2026-06-17][fact] shared sentence about pipelines")
    status = mt._spill_to_pending("user", "[2026-06-17][fact] shared sentence about pipelines")
    check("user spill not suppressed by memory line", status == "written")
    pend = (home / "episodes" / ".pending.md").read_text(encoding="utf-8")
    check("pending has both target rows", pend.count("shared sentence about pipelines") == 2)


def test_prefix_substring_does_not_false_dedup():
    """content_key containing a target word (e.g. 'memory') must not match the prefix column."""
    home = _fresh_home()
    s, mt = _store(120)
    status = mt._spill_to_pending("memory", "[2026-06-17][fact] memory usage stayed flat all week")
    check("entry whose text contains 'memory' still writes", status == "written")


def test_same_day_distinct_facts_not_collapsed():
    """Two short same-day facts must not cross the Jaccard threshold via shared metadata."""
    home = _fresh_home()
    s, mt = _store(120)
    a = mt._spill_to_pending("user", "[2026-06-17][pref] User prefers Python")
    b = mt._spill_to_pending("user", "[2026-06-17][pref] User prefers Rust")
    check("first distinct fact written", a == "written")
    check("second distinct fact NOT deduped", b == "written")


def test_multiline_entry_flattened():
    """A multiline overflow entry must land as a single TARGET\\tLINE record."""
    home = _fresh_home()
    s, mt = _store(120)
    status = mt._spill_to_pending("memory", "[2026-06-17][fact] line one\nline two\nline three")
    check("multiline entry written", status == "written")
    pend = (home / "episodes" / ".pending.md").read_text(encoding="utf-8")
    body_lines = [ln for ln in pend.splitlines() if ln.strip()]
    check("multiline collapsed to one physical record", len(body_lines) == 1)
    check("single record is target-prefixed", body_lines[0].startswith("memory\t"))
    check("no orphan continuation lines", body_lines[0].endswith("line one line two line three"))


def test_spill_write_failure_surfaces_error():
    """If the spill write fails, add() must report success=False, not a false success."""
    home = _fresh_home()
    s, mt = _store(120)
    # Make .pending.md unwritable by putting a directory where the file should go.
    (home / "episodes" / ".pending.md").mkdir(parents=True, exist_ok=True)
    s.add("memory", "[2026-06-17][rule] " + "x" * 90)
    r = s.add("memory", "[2026-06-17][fact] this overflow cannot be spilled because pending is a dir")
    check("write-failure add reports success=False", r.get("success") is False)
    check("write-failure add did not claim spilled", r.get("spilled_to_pending") is False)
    check("write-failure add returns an error message", bool(r.get("error")))


if __name__ == "__main__":
    print("[1] under-limit add writes to file, no spill")
    test_under_limit_writes_to_file()
    print("[2] over-limit add spills to pending, returns success")
    test_over_limit_spills()
    print("[3] repeated over-limit add dedups")
    test_over_limit_dedups()
    print("[4] cross-target add not false-deduped")
    test_cross_target_does_not_false_dedup()
    print("[5] prefix-substring text not false-deduped")
    test_prefix_substring_does_not_false_dedup()
    print("[6] same-day distinct facts not collapsed")
    test_same_day_distinct_facts_not_collapsed()
    print("[7] multiline entry flattened to one record")
    test_multiline_entry_flattened()
    print("[8] spill write failure surfaces an error")
    test_spill_write_failure_surfaces_error()
    print("\nALL PASS")

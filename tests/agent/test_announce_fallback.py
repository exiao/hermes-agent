"""Tests for the ``agent.announce_fallback`` config flag.

When ``announce_fallback`` is True the agent emits fallback status lines live
(via ``_emit_status`` → CLI ``_vprint`` + gateway ``status_callback``) instead
of buffering them.  When False (default) the lines are buffered and dropped on
successful recovery, preserving the quiet behavior.

The helper under test, ``AIAgent._announce_or_buffer_fallback``, only depends on
three attributes (``_announce_fallback``, ``_emit_status``, ``_buffer_status``),
so we bind the real, unbound method to a lightweight stub rather than importing
the full ~4800-line module with its heavy optional deps.
"""

from __future__ import annotations

import re

import pytest


def _load_helper():
    """Extract the real ``_announce_or_buffer_fallback`` method source from
    run_agent.py and bind it onto a stub class, so the test exercises the
    shipped implementation without importing the whole module."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2] / "run_agent.py").read_text()
    m = re.search(
        r"(    def _announce_or_buffer_fallback\(self.*?)(?=\n    (?:async )?def )",
        src,
        re.S,
    )
    assert m, "could not locate _announce_or_buffer_fallback in run_agent.py"
    ns: dict = {}
    exec("class _Stub:\n" + m.group(1), ns)
    return ns["_Stub"]


@pytest.fixture()
def stub():
    Stub = _load_helper()
    s = Stub()
    s.emitted = []
    s.buffered = []
    s._emit_status = lambda msg: s.emitted.append(msg)
    s._buffer_status = lambda msg: s.buffered.append(msg)
    return s


def test_flag_on_emits_live(stub):
    stub._announce_fallback = True
    stub._announce_or_buffer_fallback("⚠️ trying fallback...")
    assert stub.emitted == ["⚠️ trying fallback..."]
    assert stub.buffered == []


def test_flag_off_buffers(stub):
    stub._announce_fallback = False
    stub._announce_or_buffer_fallback("⚠️ trying fallback...")
    assert stub.buffered == ["⚠️ trying fallback..."]
    assert stub.emitted == []


def test_missing_attr_defaults_to_buffer(stub):
    # No _announce_fallback set at all → safe default is buffer (quiet).
    assert not hasattr(stub, "_announce_fallback")
    stub._announce_or_buffer_fallback("⚠️ trying fallback...")
    assert stub.buffered == ["⚠️ trying fallback..."]
    assert stub.emitted == []

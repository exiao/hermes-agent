"""The generated sandbox stub docstrings must describe the real return contract.

Bug (Aug 2026): the `terminal` stub's docstring said only

    Returns dict with "output" and "exit_code".

The dict actually carries up to five keys. When a command is rejected BEFORE
it runs (shell guard, lifecycle guard, unavailable backend), `output` is an
empty string, `exit_code` is -1, and the reason lives in `error`/`traceback`.
A caller who trusted the docstring and read only `output` saw "" and concluded
the command had succeeded silently, misdiagnosing a hard failure as a
no-op — which is exactly what happened during the lifecycle_guard NUL-path
investigation.

These tests assert the contract is documented AND that the generated module is
still valid, executable Python on both transports.
"""
import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.code_execution_tool import (
    SANDBOX_ALLOWED_TOOLS,
    generate_hermes_tools_module,
)

TRANSPORTS = ("uds", "file")


def _exec_module(transport, tools=("terminal", "read_file")):
    src = generate_hermes_tools_module(list(tools), transport=transport)
    ast.parse(src)  # syntactically valid
    namespace = {}
    exec(compile(src, "<generated>", "exec"), namespace)  # and executable
    return namespace


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_generated_module_is_valid_python(transport):
    ns = _exec_module(transport)
    assert callable(ns["terminal"])
    assert callable(ns["read_file"])


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_terminal_docstring_documents_failure_contract(transport):
    """The discriminator: a caller must be told to check exit_code."""
    doc = _exec_module(transport)["terminal"].__doc__
    assert doc, "terminal stub has no docstring"
    # The keys that actually appear on a pre-execution failure.
    for key in ("output", "exit_code", "error", "traceback"):
        assert key in doc, f"docstring never mentions {key!r}"
    # The behavioral warning, not just a key list. Without this the docstring
    # can list the keys and still let a caller read `output` alone.
    assert "exit_code" in doc and "empty" in doc.lower(), (
        "docstring must warn that output is EMPTY on a pre-execution failure"
    )


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_every_generated_stub_has_a_docstring(transport):
    """Guard the whole family, not just the one tool that had the bug."""
    ns = _exec_module(transport, tools=sorted(SANDBOX_ALLOWED_TOOLS))
    for name, obj in ns.items():
        if name.startswith("_") or not callable(obj):
            continue
        if name in {"json_parse", "shell_quote", "retry"}:
            continue  # built-in helpers, documented separately
        assert obj.__doc__, f"generated stub {name!r} has no docstring"


def test_docstring_change_does_not_break_arg_passing():
    """A multi-line docstring must not disturb the generated call expression.

    The stubs are built by string concatenation, so an unescaped newline or
    quote in the docstring could silently corrupt the `_call(...)` line below
    it. Assert the signature and body survived.
    """
    src = generate_hermes_tools_module(["terminal"], transport="uds")
    tree = ast.parse(src)
    func = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "terminal"
    )
    assert [a.arg for a in func.args.args] == ["command", "timeout", "workdir"]
    ret = func.body[-1]
    assert isinstance(ret, ast.Return)
    assert isinstance(ret.value, ast.Call)
    assert ret.value.func.id == "_call"

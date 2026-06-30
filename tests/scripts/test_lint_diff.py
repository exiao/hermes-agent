from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_lint_diff_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "lint_diff.py"
    spec = importlib.util.spec_from_file_location("lint_diff_under_test", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ty_omitted_union_count_is_not_part_of_diff_identity():
    lint_diff = _load_lint_diff_module()
    base = [
        {
            "path": "hermes_cli/config.py",
            "rule": "unresolved-attribute",
            "line": 5253,
            "message": "Attribute `items` is not defined on `int`, `str`, `list[Unknown]`, `float`, `None` in union `Unknown | int | str | ... omitted 18 union elements`",
        }
    ]
    head = [
        {
            "path": "hermes_cli/config.py",
            "rule": "unresolved-attribute",
            "line": 5231,
            "message": "Attribute `items` is not defined on `int`, `str`, `list[Unknown]`, `float`, `None` in union `Unknown | int | str | ... omitted 17 union elements`",
        }
    ]

    new, fixed, unchanged = lint_diff._diff(base, head)

    assert new == []
    assert fixed == []
    assert unchanged == head
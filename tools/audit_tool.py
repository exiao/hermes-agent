"""Evidence-only tool surface for delegated audit children."""

from __future__ import annotations

import json
from pathlib import Path

from tools.registry import no_cache_check_fn, registry, tool_error

_MAX_READ_LINES = 2_000
_MAX_READ_BYTES = 1_000_000


AUDIT_READ_FILE_SCHEMA = {
    "name": "audit_read_file",
    "description": (
        "Read a parent-supplied audit evidence file. Paths outside the supplied evidence "
        "scope, symlink escapes, directories, and special files are refused."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Evidence file path supplied by the parent."},
            "offset": {"type": "integer", "minimum": 1, "default": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_READ_LINES, "default": 2000},
        },
        "required": ["path"],
    },
}


def _scope_paths() -> tuple[Path, ...]:
    from agent.delegation_context import audit_evidence_paths

    return tuple(Path(path) for path in audit_evidence_paths())


def _resolve_evidence_file(raw_path: str) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_file():
            return None
        if any(root.is_file() and path == root for root in _scope_paths()):
            return path
        if any(root.is_dir() and root in path.parents for root in _scope_paths()):
            return path
    except OSError:
        return None
    return None


def audit_read_file(path: str, offset: int = 1, limit: int = 2000, **_kwargs) -> str:
    """Read only a canonical file inside the parent-provided evidence scope."""
    from agent.delegation_context import is_audit_child_context

    if not is_audit_child_context():
        return tool_error("audit_read_file is available only to audit children")
    try:
        offset, limit = int(offset), int(limit)
    except (TypeError, ValueError):
        return tool_error("offset and limit must be integers")
    if offset < 1 or limit < 1 or limit > _MAX_READ_LINES:
        return tool_error(f"offset must be >= 1 and limit must be 1-{_MAX_READ_LINES}")
    resolved = _resolve_evidence_file(path)
    if resolved is None:
        return tool_error("path is outside the parent-supplied audit evidence scope")
    try:
        if resolved.stat().st_size > _MAX_READ_BYTES:
            return tool_error(f"evidence file exceeds the {_MAX_READ_BYTES} byte limit")
        lines: list[str] = []
        total_lines = 0
        with resolved.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                total_lines = line_number
                if offset <= line_number < offset + limit:
                    lines.append(f"{line_number}|{line.rstrip(chr(10))}\n")
    except OSError as exc:
        return tool_error(f"could not read evidence file: {exc}")
    return json.dumps({
        "content": "".join(lines),
        "total_lines": total_lines,
        "path": str(resolved),
        "next_offset": offset + limit if total_lines >= offset + limit else None,
    }, ensure_ascii=False)


def _check_audit_surface() -> bool:
    try:
        from agent.delegation_context import is_audit_child_context

        return is_audit_child_context()
    except Exception:
        return False


registry.register(
    name="audit_read_file",
    toolset="audit",
    schema=AUDIT_READ_FILE_SCHEMA,
    handler=lambda args, **kwargs: audit_read_file(**args, **kwargs),
    check_fn=no_cache_check_fn(_check_audit_surface),
)

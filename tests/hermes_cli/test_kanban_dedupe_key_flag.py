"""`--dedupe-key` is the current name; `--idempotency-key` still works.

Eric renamed the flag (2026-08-13) because "idempotency" is jargon. The old
name stays as a permanent alias: the babysit-pr-detector cron and a pile of
skills/scripts already pass it, and a silent break there means duplicate cards,
which is the exact bug the flag exists to prevent.
"""

import argparse

import pytest

from hermes_cli import kanban as kanban_cli


def _parser() -> argparse.ArgumentParser:
    """Build the real kanban subcommand tree, as `hermes kanban ...` does."""
    root = argparse.ArgumentParser(prog="hermes")
    subs = root.add_subparsers(dest="command")
    kanban_cli.build_parser(subs)
    return root


def _parse(argv):
    return _parser().parse_args(["kanban"] + argv)


@pytest.mark.parametrize("flag", ["--dedupe-key", "--idempotency-key"])
def test_create_accepts_both_flag_names(flag):
    args = _parse(["create", "some title", flag, "localreview:o/r#1-abc123def456"])
    assert args.dedupe_key == "localreview:o/r#1-abc123def456"


@pytest.mark.parametrize("flag", ["--dedupe-key", "--idempotency-key"])
def test_swarm_accepts_both_flag_names(flag):
    args = _parse([
        "swarm", "root title",
        "--worker", "dev",
        "--verifier", "qa",
        "--synthesizer", "writer",
        flag, "swarm:key-1",
    ])
    assert args.dedupe_key == "swarm:key-1"


def test_absent_flag_defaults_to_none():
    args = _parse(["create", "some title"])
    assert getattr(args, "dedupe_key", "MISSING") is None


def test_new_name_appears_in_create_help():
    """The name Eric types must be the one the CLI advertises."""
    for action in _parser()._subparsers._group_actions[0].choices["kanban"]._subparsers._group_actions[0].choices.items():
        name, sub = action
        if name == "create":
            assert "--dedupe-key" in sub.format_help()
            return
    pytest.fail("create subparser not found")


def test_tool_schema_exposes_dedupe_key():
    """The model-facing schema drives whether an agent passes the key at all."""
    from tools import kanban_tools

    props = kanban_tools.KANBAN_CREATE_SCHEMA["parameters"]["properties"]
    assert "dedupe_key" in props, "renamed field missing from the tool schema"

"""Heredoc stdin must feed the WHOLE command, not just its last word.

Backends without a stdin channel (Modal, Daytona) receive stdin by having it
appended as a shell heredoc. The redirect was attached to the bare command
string, but a redirect binds only to the single command it follows. For a
multi-command sequence the body therefore went to the LAST command.

That is exactly what ``FileOperations._atomic_write`` emits:

    set -e; ...; trap '...' EXIT; cat > "$tmp"; mv -f "$tmp" "$t"; trap - EXIT

so the file body was fed to ``trap - EXIT`` and ``cat`` read the sandbox's
real stdin instead. Observed live on Modal-backed kanban lanes as writes that
either produced an empty file or blocked until the command timeout (300.6s,
303.7s, 300.8s on a single card).
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.modal_utils import wrap_modal_stdin_heredoc


# The real shape emitted by FileOperations._atomic_write: a multi-command
# sequence whose stdin consumer (`cat`) is NOT the final command.
def _atomic_write_script(target: str) -> str:
    parent = str(Path(target).parent)
    return (
        "set -e; "
        f"d='{parent}'; t='{target}'; "
        "tmp=\"$(mktemp -p \"$d\" '.hermes-tmp.XXXXXX' 2>/dev/null)\"; "
        '[ -n "$tmp" ] || { echo "no temp" >&2; exit 1; }; '
        "trap 'rm -f \"$tmp\"' EXIT; "
        'cat > "$tmp"; '
        'mv -f "$tmp" "$t"; '
        "trap - EXIT"
    )


def _run(wrapped: str) -> subprocess.CompletedProcess:
    # stdin closed, mirroring a sandbox exec with no stdin channel.
    return subprocess.run(
        ["bash", "-c", wrapped], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize(
    "embed",
    [BaseEnvironment._embed_stdin_heredoc, wrap_modal_stdin_heredoc],
    ids=["base", "modal_utils"],
)
class TestHeredocFeedsWholeCommand:
    def test_atomic_write_body_reaches_the_file(self, embed, tmp_path):
        """The regression: body must land in the file, not go to `trap`."""
        target = str(tmp_path / "out.md")
        body = "first line\nsecond line\n"

        result = _run(embed(_atomic_write_script(target), body))

        assert result.returncode == 0, result.stderr
        assert Path(target).exists(), "atomic write produced no file"
        # The body must be present — an empty file is the silent-corruption
        # failure mode this guards against.
        assert Path(target).read_text().startswith(body)

    def test_single_command_still_works(self, embed, tmp_path):
        """The brace group must not regress the simple one-command case."""
        target = tmp_path / "single.txt"
        result = _run(embed(f"cat > {target}", "single body\n"))

        assert result.returncode == 0, result.stderr
        assert target.read_text().startswith("single body\n")

    def test_trailing_comment_does_not_swallow_the_brace(self, embed, tmp_path):
        """A command ending in a comment must not eat the closing `}`.

        This is why the closing brace goes on its own line.
        """
        target = tmp_path / "commented.txt"
        result = _run(embed(f"cat > {target}  # trailing comment", "body\n"))

        assert result.returncode == 0, result.stderr
        assert target.read_text().startswith("body\n")

    def test_exit_code_propagates(self, embed):
        """The brace group must not mask the command's exit status."""
        result = _run(embed("cat > /dev/null; exit 3", "body\n"))
        assert result.returncode == 3

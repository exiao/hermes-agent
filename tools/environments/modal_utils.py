"""PLUGIN-COMPAT stub (revert-scheduled; see COMPAT_MANIFEST.md).

``tools.environments.modal_utils`` was deleted in the Sep 2026 decomposition (its callers were folded into the
execution backends). Importing it no longer provides anything; this stub exists only so an external
plugin's ``import tools.environments.modal_utils`` does not raise at import time.
"""

import uuid


def wrap_modal_stdin_heredoc(command: str, stdin_data: str) -> str:
    """Compatibility helper: attach stdin to a group containing the whole command."""
    delimiter = f"HERMES_STDIN_{uuid.uuid4().hex[:12]}"
    return f"{{ {command}\n}} << '{delimiter}'\n{stdin_data}\n{delimiter}"

"""Send File Tool — delivers arbitrary files as platform attachments.

The agent calls this tool when a user asks to "send", "share", or "attach"
a file.  The handler validates the file (exists, not empty, size limit)
and returns a ``MEDIA:<path>`` tag.  The gateway's ``extract_media()``
picks up that tag and routes it through ``send_document()`` (or the
appropriate image/video/audio sender based on extension).

This closes the gap where ``send_document()`` existed on every platform
adapter but nothing in the agent toolset could reach it for non-media
file types.
"""

import json
import logging
from pathlib import Path

from tools.registry import registry

logger = logging.getLogger(__name__)

# 100 MB hard limit — matches Signal's attachment ceiling
MAX_FILE_SIZE = 100 * 1024 * 1024


SEND_FILE_SCHEMA = {
    "name": "send_file",
    "description": (
        "Attach a local file to the current chat (Signal, Telegram, Discord, "
        "Slack, etc.). Any file type: docs, config, code, archives, images, "
        "audio, video. Returns a MEDIA:<path> string.\n\n"
        "You MUST echo the returned MEDIA:<path> verbatim in your final reply "
        "text (own line, blank line above). The gateway scans reply text, not "
        "tool results. Skipping this step silently sends zero attachments. "
        "For multiple files, one MEDIA line per file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or ~/relative path to the file",
            },
            "caption": {
                "type": "string",
                "description": "Optional caption prepended to the returned MEDIA line",
            },
        },
        "required": ["file_path"],
    },
}


def send_file_tool(file_path: str, caption: str = "") -> str:
    """Validate *file_path* and return a ``MEDIA:`` tag for gateway delivery."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})
    if not path.is_file():
        return json.dumps({"error": f"Not a file (maybe a directory?): {file_path}"})

    size = path.stat().st_size
    if size == 0:
        return json.dumps({"error": f"File is empty: {file_path}"})
    if size > MAX_FILE_SIZE:
        size_mb = size / (1024 * 1024)
        return json.dumps({"error": f"File too large: {size_mb:.1f} MB (max 100 MB)"})

    logger.info("send_file: queuing %s (%d bytes) for delivery", path, size)

    # The gateway's extract_media() regex picks up MEDIA:<path> tags from
    # the agent response and routes them to send_image_file / send_voice /
    # send_video / send_document depending on extension.
    result = f"MEDIA:{path}"
    if caption:
        result = f"{caption}\n{result}"

    return result


def check_send_file() -> bool:
    """Always available — file sending works on all platforms."""
    return True


registry.register(
    name="send_file",
    toolset="files",
    schema=SEND_FILE_SCHEMA,
    handler=lambda args, **kw: send_file_tool(
        file_path=args.get("file_path", ""),
        caption=args.get("caption", ""),
    ),
    check_fn=check_send_file,
    emoji="📎",
)

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
    "type": "function",
    "function": {
        "name": "send_file",
        "description": (
            "Send a local file as a native attachment to the current chat. "
            "Works for any file type: documents (.pdf, .md, .txt, .docx), "
            "config files (.yaml, .json, .toml), code (.py, .js, .ts), "
            "archives (.zip, .tar.gz), images, audio, video, and more. "
            "The file is delivered as a platform-native attachment "
            "(e.g., Signal file, Telegram document, Slack file upload). "
            "Use this instead of pasting file contents as text when the "
            "user asks to 'send', 'share', or 'attach' a file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or ~/relative path to the file to send",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional message to accompany the file",
                },
            },
            "required": ["file_path"],
        },
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

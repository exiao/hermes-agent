"""Direct HTTP fetch — the free first attempt before a paid extract backend.

Most pages worth reading are plain server-rendered HTML. Sending those to a
metered extract API (Firecrawl, Tavily, Exa, Parallel) costs a credit and a
round-trip to do what a single GET plus a tag strip already does. This module
is that first attempt: fetch the URL directly, and only hand it to the paid
backend when the direct read is not good enough.

It is deliberately conservative. It gives up — rather than returning thin or
wrong content — whenever it cannot be confident, because a bad "success" here
silently replaces a good paid extraction:

- non-HTML content types (PDF and friends) are left to the backend, which
  parses formats this module cannot
- JS-shell pages (little text, many script tags) are left to the backend,
  which renders them
- anything under ``MIN_USEFUL_CHARS`` of extracted text is treated as a miss
- any HTTP error, timeout, redirect loop, or oversized body is a miss

No new dependency: the extraction is stdlib ``html.parser``. ``markdownify``,
``bs4``, ``trafilatura`` and friends are NOT installed in the runtime venv
(checked on both the dev box and the BloomBot VPS), so adding one would mean
shipping a dependency to every install for a best-effort fast path.
"""

from __future__ import annotations

import asyncio
import logging
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# A page that yields less than this much text is treated as a miss and handed
# to the paid backend. Cookie walls, JS shells, and redirect stubs all land
# well under it; a genuine article clears it easily.
MIN_USEFUL_CHARS = 600

# Per-URL wall clock. The point of this path is to be FASTER than the paid
# backend, so a slow direct fetch is a miss: fall back rather than stack this
# timeout on top of the backend's own.
DEFAULT_TIMEOUT_S = 8.0

# Refuse to buffer more than this. Protects against a multi-hundred-MB body
# being pulled into memory before we discover it is not useful.
MAX_BYTES = 5_000_000

# Content we can actually handle with a tag strip. Everything else (PDF,
# octet-stream, images) is exactly what the paid backends are good at.
_HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain")

_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "canvas", "iframe"}
)
# Tags whose content is chrome, not article text.
_CHROME_TAGS = frozenset({"nav", "header", "footer", "aside", "form"})

_BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "article", "br", "li", "tr", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6", "pre",
    }
)

_WS_RUN = re.compile(r"[ \t\r\f\v]+")
_BLANK_RUN = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    """Collect visible text, dropping script/style and page chrome."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth = 0
        self._chrome_depth = 0
        self._script_tags = 0
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            if tag == "script":
                self._script_tags += 1
        elif tag in _CHROME_TAGS:
            self._chrome_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _CHROME_TAGS:
            self._chrome_depth = max(0, self._chrome_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()
            return
        if self._skip_depth or self._chrome_depth:
            return
        if data.strip():
            self._chunks.append(data)

    @property
    def script_tag_count(self) -> int:
        return self._script_tags

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = unescape(raw)
        raw = _WS_RUN.sub(" ", raw)
        raw = "\n".join(line.strip() for line in raw.split("\n"))
        return _BLANK_RUN.sub("\n\n", raw).strip()


def _extract_text(html: str) -> tuple[str, str, int]:
    """Return ``(text, title, script_tag_count)`` for an HTML document."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 — malformed markup is a miss, not a crash
        logger.debug("direct-fetch: HTML parse failed: %s", exc)
        return "", "", 0
    return parser.text(), parser.title, parser.script_tag_count


def _looks_like_js_shell(text: str, script_tags: int) -> bool:
    """True when the page is probably rendered client-side.

    A JS app ships many script tags and little server-rendered prose. Those
    are precisely the pages a headless-browser backend handles and this one
    cannot, so they are a miss.
    """
    return script_tags >= 5 and len(text) < 2000


async def fetch_direct(
    url: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    min_chars: int = MIN_USEFUL_CHARS,
) -> Optional[Dict[str, Any]]:
    """Try to read ``url`` with a plain GET.

    Returns a result dict shaped like an extract provider's
    (``url``/``title``/``content``/``error``) on success, or ``None`` when the
    caller should fall back to the configured extract backend. Never raises:
    every failure path is a ``None`` so the fallback always gets its turn.
    """
    try:
        import httpx  # noqa: F401
    except ImportError:  # pragma: no cover - httpx is a hard runtime dep
        return None

    from tools.url_safety import (
        async_is_safe_url,
        create_ssrf_safe_async_client,
        redirect_target_from_response,
    )
    from tools.website_policy import check_website_access

    if check_website_access(url):
        logger.debug("direct-fetch miss (blocked by website policy): %s", url)
        return None

    async def _ssrf_redirect_guard(response: Any) -> None:
        """Re-validate every redirect target: a public URL can 302 to 127.0.0.1."""
        redirect_url = redirect_target_from_response(response)
        if not redirect_url:
            return
        if not await async_is_safe_url(redirect_url):
            raise ValueError(f"Blocked redirect to private address: {redirect_url}")
        if check_website_access(redirect_url):
            raise ValueError(f"Blocked redirect by website policy: {redirect_url}")

    try:
        async with create_ssrf_safe_async_client(
            follow_redirects=True,
            event_hooks={"response": [_ssrf_redirect_guard]},
            timeout=timeout_s,
            headers={
                # Some sites 403 an unknown agent. Identify honestly but in a
                # shape servers expect from a reader.
                "User-Agent": (
                    "Mozilla/5.0 (compatible; HermesAgent/1.0; +https://github.com/"
                    "NousResearch/hermes-agent)"
                ),
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
                "Accept-Language": "en-US,en;q=0.9",
            },
        ) as client:
            response = await client.get(url)

        if response.status_code >= 400:
            logger.debug("direct-fetch miss (HTTP %s): %s", response.status_code, url)
            return None

        content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type and not any(content_type.startswith(t) for t in _HTML_TYPES):
            logger.debug("direct-fetch miss (content-type %s): %s", content_type, url)
            return None

        body = response.content
        if len(body) > MAX_BYTES:
            logger.debug("direct-fetch miss (body %d bytes): %s", len(body), url)
            return None

        html = response.text
    except Exception as exc:  # noqa: BLE001 — any transport failure is just a miss
        logger.debug("direct-fetch miss (%s): %s", type(exc).__name__, url)
        return None

    text, title, script_tags = _extract_text(html)

    if len(text) < min_chars:
        logger.debug("direct-fetch miss (%d chars < %d): %s", len(text), min_chars, url)
        return None
    if _looks_like_js_shell(text, script_tags):
        logger.debug("direct-fetch miss (JS shell, %d scripts): %s", script_tags, url)
        return None

    logger.info("direct-fetch hit (%d chars, no API credit): %s", len(text), url)
    return {"url": url, "title": title, "content": text, "error": None}


async def fetch_many_direct(
    urls: List[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    min_chars: int = MIN_USEFUL_CHARS,
) -> Dict[str, Dict[str, Any]]:
    """Fetch ``urls`` concurrently, returning only the hits keyed by URL.

    Misses are simply absent from the mapping, so the caller can send exactly
    the remaining URLs to the paid backend in one batch.
    """
    if not urls:
        return {}

    results = await asyncio.gather(
        *(fetch_direct(u, timeout_s=timeout_s, min_chars=min_chars) for u in urls),
        return_exceptions=True,
    )

    hits: Dict[str, Dict[str, Any]] = {}
    for url, result in zip(urls, results):
        if isinstance(result, dict):
            hits[url] = result
        elif isinstance(result, BaseException):
            logger.debug("direct-fetch miss (gather %s): %s", type(result).__name__, url)
    return hits

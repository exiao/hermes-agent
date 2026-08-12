"""Direct-fetch-first path for ``web_extract``.

The contract: try a plain GET before spending an extract-backend credit, and
fall back to the configured backend for anything the direct read cannot
handle confidently. These assert both halves — that a good page skips the
paid backend entirely, and that every miss still reaches it with results in
the caller's original order.

No network: httpx is stubbed. The ordering tests are the important ones —
a direct hit changes which URLs the backend sees, so a reassembly bug would
silently drop pages or return them against the wrong URL.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from tools import web_direct_fetch as dfetch


ARTICLE = (
    "<html><head><title>Real Article</title></head><body>"
    "<nav>home about contact</nav>"
    "<article><p>" + ("Substantive reporting about the subject. " * 40) + "</p></article>"
    "<script>var tracking = 1;</script>"
    "<footer>copyright</footer>"
    "</body></html>"
)


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        content_type: str = "text/html; charset=utf-8",
        content: Optional[bytes] = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = {"content-type": content_type}
        self.content = content if content is not None else text.encode()


class _FakeClient:
    """Stands in for httpx.AsyncClient; serves a canned response per URL."""

    def __init__(self, responses: Dict[str, Any], calls: List[str]) -> None:
        self._responses = responses
        self._calls = calls

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str) -> _FakeResponse:
        self._calls.append(url)
        result = self._responses[url]
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def fake_http(monkeypatch):
    """Patch httpx.AsyncClient; returns (responses, calls) for the test to fill."""
    import httpx

    responses: Dict[str, Any] = {}
    calls: List[str] = []

    def _factory(*args: object, **kwargs: object) -> _FakeClient:
        return _FakeClient(responses, calls)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    return responses, calls


# ── the helper itself ────────────────────────────────────────────────────────

def test_good_html_page_is_a_hit(fake_http):
    responses, _ = fake_http
    responses["https://x.test/a"] = _FakeResponse(text=ARTICLE)

    result = asyncio.run(dfetch.fetch_direct("https://x.test/a"))

    assert result is not None
    assert result["title"] == "Real Article"
    assert "Substantive reporting" in result["content"]
    assert result["error"] is None
    # Chrome and scripts are stripped, not returned as content.
    assert "var tracking" not in result["content"]
    assert "copyright" not in result["content"]


@pytest.mark.parametrize(
    "response,reason",
    [
        (_FakeResponse(status_code=404, text=ARTICLE), "http error"),
        (_FakeResponse(text="%PDF-1.4 binary", content_type="application/pdf"), "pdf"),
        (_FakeResponse(text="<html><body><p>too short</p></body></html>"), "thin page"),
        (
            _FakeResponse(
                text="<html><body><div id=root></div>"
                + "<script>a</script>" * 8
                + "</body></html>"
            ),
            "js shell",
        ),
    ],
)
def test_unhandleable_pages_are_misses(fake_http, response, reason):
    """A miss must return None so the caller falls back — never thin content."""
    responses, _ = fake_http
    responses["https://x.test/a"] = response

    assert asyncio.run(dfetch.fetch_direct("https://x.test/a")) is None, reason


def test_transport_error_is_a_miss_not_a_crash(fake_http):
    responses, _ = fake_http
    responses["https://x.test/a"] = RuntimeError("connection reset")

    assert asyncio.run(dfetch.fetch_direct("https://x.test/a")) is None


def test_fetch_many_returns_only_hits(fake_http):
    responses, _ = fake_http
    responses["https://x.test/good"] = _FakeResponse(text=ARTICLE)
    responses["https://x.test/bad"] = _FakeResponse(status_code=500, text="")

    hits = asyncio.run(
        dfetch.fetch_many_direct(["https://x.test/good", "https://x.test/bad"])
    )

    assert list(hits) == ["https://x.test/good"]


# ── integration with web_extract_tool ────────────────────────────────────────

class _RecordingProvider:
    """Extract provider that records exactly which URLs it was asked for."""

    name = "fake-backend"
    display_name = "Fake Backend"

    def __init__(self) -> None:
        self.seen: List[List[str]] = []

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], format: str = None) -> List[Dict[str, Any]]:
        self.seen.append(list(urls))
        return [
            {"url": u, "title": "from-backend", "content": f"backend:{u}", "error": None}
            for u in urls
        ]


@pytest.fixture
def wired(monkeypatch, fake_http):
    """web_extract_tool with SSRF allowed, plugins stubbed, fake backend."""
    from tools import web_tools

    provider = _RecordingProvider()

    async def _safe(url: str) -> bool:
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", _safe)
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fake-backend")

    import agent.web_search_registry as registry

    monkeypatch.setattr(registry, "get_provider", lambda name: provider)
    monkeypatch.setattr(registry, "get_active_extract_provider", lambda: provider)
    monkeypatch.setattr(registry, "_disabled_web_plugin_for", lambda capability: None)

    return web_tools, provider, fake_http


def _run_extract(web_tools, urls):
    raw = asyncio.run(web_tools.web_extract_tool(urls=urls))
    return json.loads(raw)["results"]


def test_direct_hit_never_reaches_the_paid_backend(wired, monkeypatch):
    web_tools, provider, (responses, calls) = wired
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
    responses["https://x.test/a"] = _FakeResponse(text=ARTICLE)

    results = _run_extract(web_tools, ["https://x.test/a"])

    assert provider.seen == [], "backend was called for a page direct fetch handled"
    assert calls == ["https://x.test/a"]
    assert "Substantive reporting" in results[0]["content"]


def test_miss_falls_through_to_the_backend(wired, monkeypatch):
    web_tools, provider, (responses, _) = wired
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
    responses["https://x.test/pdf"] = _FakeResponse(
        text="%PDF-1.4", content_type="application/pdf"
    )

    results = _run_extract(web_tools, ["https://x.test/pdf"])

    assert provider.seen == [["https://x.test/pdf"]]
    assert results[0]["content"] == "backend:https://x.test/pdf"


def test_mixed_batch_keeps_caller_order_and_splits_correctly(wired, monkeypatch):
    """The ordering guarantee: direct hits and backend results interleave correctly."""
    web_tools, provider, (responses, _) = wired
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

    urls = [
        "https://x.test/pdf1",   # miss  -> backend
        "https://x.test/good1",  # hit   -> direct
        "https://x.test/pdf2",   # miss  -> backend
        "https://x.test/good2",  # hit   -> direct
    ]
    responses["https://x.test/pdf1"] = _FakeResponse(
        text="%PDF", content_type="application/pdf"
    )
    responses["https://x.test/good1"] = _FakeResponse(text=ARTICLE)
    responses["https://x.test/pdf2"] = _FakeResponse(
        text="%PDF", content_type="application/pdf"
    )
    responses["https://x.test/good2"] = _FakeResponse(text=ARTICLE)

    results = _run_extract(web_tools, urls)

    # Only the two misses were paid for.
    assert provider.seen == [["https://x.test/pdf1", "https://x.test/pdf2"]]
    # Every input URL comes back, in the order the caller asked for.
    assert [r["url"] for r in results] == urls
    assert results[0]["content"] == "backend:https://x.test/pdf1"
    assert "Substantive reporting" in results[1]["content"]
    assert results[2]["content"] == "backend:https://x.test/pdf2"
    assert "Substantive reporting" in results[3]["content"]


def test_disabling_restores_backend_only_behavior(wired, monkeypatch):
    web_tools, provider, (responses, calls) = wired
    monkeypatch.setattr(
        web_tools, "_load_web_config", lambda: {"direct_fetch_first": False}
    )
    responses["https://x.test/a"] = _FakeResponse(text=ARTICLE)

    results = _run_extract(web_tools, ["https://x.test/a"])

    assert calls == [], "direct fetch ran while disabled"
    assert provider.seen == [["https://x.test/a"]]
    assert results[0]["content"] == "backend:https://x.test/a"


# ── security gates: a redirect and the blocklist must both produce a miss ────

class _RedirectingClient(_FakeClient):
    """Fake client that runs the response event hooks, like httpx does."""

    def __init__(self, responses, calls, hooks):
        super().__init__(responses, calls)
        self._hooks = hooks

    async def get(self, url: str) -> Any:
        response = await super().get(url)
        for hook in self._hooks:
            await hook(response)
        return response


def test_redirect_to_private_address_is_a_miss(monkeypatch):
    """A public URL that 302s to loopback must never return a body."""
    calls: List[str] = []
    redirect = _FakeResponse(status_code=302, text="")
    redirect.is_redirect = True
    redirect.headers = {"location": "http://127.0.0.1/internal"}
    redirect.url = "https://x.test/redir"
    responses = {"https://x.test/redir": redirect}

    def _factory(*args: object, **kwargs: object) -> _RedirectingClient:
        hooks = (kwargs.get("event_hooks") or {}).get("response", [])
        return _RedirectingClient(responses, calls, hooks)

    monkeypatch.setattr(
        "tools.url_safety.create_ssrf_safe_async_client", _factory
    )

    assert asyncio.run(dfetch.fetch_direct("https://x.test/redir")) is None
    assert calls == ["https://x.test/redir"]


def test_blocklisted_host_is_a_miss(fake_http, monkeypatch):
    """A policy-blocked host must not be fetched at all."""
    responses, calls = fake_http
    responses["https://blocked.test/a"] = _FakeResponse(text=ARTICLE)
    monkeypatch.setattr(
        "tools.website_policy.check_website_access",
        lambda url, *a, **k: {"host": "blocked.test", "message": "blocked"},
    )

    assert asyncio.run(dfetch.fetch_direct("https://blocked.test/a")) is None
    assert calls == [], "blocked URL was fetched anyway"

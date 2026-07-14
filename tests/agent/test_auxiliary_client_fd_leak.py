"""Repro + fix proof for the auxiliary-client FD leak (kanban.db corruption L1).

``agent/auxiliary_client.py`` used to evict/replace cached httpx clients by only
marking the inner async httpx client ``ClientState.CLOSED`` (``_force_close_async_httpx``)
— which leaves the underlying OS socket OPEN. In a short-lived CLI that is fine
(the OS reaps the fd at process exit), but the gateway (up for days) and kanban
workers (up for hours) evict aux clients continuously and stranded CLOSED/CLOSE_WAIT
sockets accumulated until the host hit its fd ceiling and a SQLite WAL/SHM mmap
tore (``kanban.db`` "database disk image is malformed").

The fix (``_release_cached_client_fds``) actually closes the pool's raw sockets at
every eviction path. These tests exercise the REAL object graph with real httpx
against a local stdlib HTTP server (no network, no mocks) on a now-dead event loop
— the exact stranded-fd condition — and assert:

  * the old state-mark path leaves the fd OPEN (documents the leak), and
  * ``_release_cached_client_fds`` / ``cleanup_stale_async_clients`` free it.
"""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

httpx = pytest.importorskip("httpx")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def local_http_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        srv.shutdown()


def _make_dead_loop_client(url: str):
    """Build a wrapper mimicking AsyncOpenAI (``._client`` is an httpx.AsyncClient),
    perform one request so a keep-alive connection is pooled, then close the loop.

    Returns a SimpleNamespace whose ``_client`` is the (now loop-dead) httpx client.
    """
    from types import SimpleNamespace

    loop = asyncio.new_event_loop()

    async def _go():
        hc = httpx.AsyncClient()
        await hc.get(url)
        return hc

    inner = loop.run_until_complete(_go())
    loop.close()  # loop now dead -> pooled sockets stranded, aclose() cannot run
    return SimpleNamespace(_client=inner), loop


def _pool_socket_fds(inner_httpx):
    """Return the fileno() of every raw OS socket still held by the httpx pool."""
    fds = []
    pool = inner_httpx._transport._pool
    for conn in list(pool._connections):
        stream = conn._connection._network_stream._stream
        holder = getattr(stream, "transport_stream", None) or stream
        sock = getattr(getattr(holder, "_transport", None), "_sock", None)
        if sock is not None:
            fds.append(sock.fileno())
    return fds


class TestAuxClientFdRelease:
    def test_state_mark_leaks_fd(self, local_http_server):
        """Documents the leak: state-marking CLOSED leaves the OS socket OPEN."""
        from agent.auxiliary_client import _force_close_async_httpx

        client, _loop = _make_dead_loop_client(local_http_server)
        before = _pool_socket_fds(client._client)
        assert before and all(fd >= 0 for fd in before), "expected a pooled socket"

        _force_close_async_httpx(client)  # old eviction behavior

        after = _pool_socket_fds(client._client)
        # fd is still open — this is the leak the task fixes.
        assert all(fd >= 0 for fd in after), (
            "state-mark unexpectedly closed the fd; test no longer reproduces the leak"
        )

    def test_release_closes_fd(self, local_http_server):
        """The fix frees the raw socket fd even with the owning loop dead."""
        from agent.auxiliary_client import _release_cached_client_fds

        client, _loop = _make_dead_loop_client(local_http_server)
        before = _pool_socket_fds(client._client)
        assert before and all(fd >= 0 for fd in before)

        _release_cached_client_fds(client)  # new eviction behavior

        after = _pool_socket_fds(client._client)
        assert all(fd == -1 for fd in after), (
            f"expected all pool sockets closed (fileno -1), got {after}"
        )

    def test_release_closes_real_client_wrapper(self, local_http_server):
        """Wrappers expose the SDK client via ``_real_client``; that path must free fds too."""
        from types import SimpleNamespace
        from agent.auxiliary_client import _release_cached_client_fds

        inner_wrapper, _loop = _make_dead_loop_client(local_http_server)
        wrapper = SimpleNamespace(_real_client=inner_wrapper)

        _release_cached_client_fds(wrapper)

        after = _pool_socket_fds(inner_wrapper._client)
        assert all(fd == -1 for fd in after), after

    def test_cleanup_stale_async_clients_releases_fd(self, local_http_server):
        """The post-turn stale-loop sweep must release fds, not just state-mark."""
        import agent.auxiliary_client as ac

        client, dead_loop = _make_dead_loop_client(local_http_server)
        cache_key = ("test-provider", True, local_http_server, "k", "", (), False, "", "", "m")
        with ac._client_cache_lock:
            ac._client_cache[cache_key] = (client, "m", dead_loop)
        try:
            before = _pool_socket_fds(client._client)
            assert before and all(fd >= 0 for fd in before)

            ac.cleanup_stale_async_clients()

            after = _pool_socket_fds(client._client)
            assert all(fd == -1 for fd in after), after
            assert cache_key not in ac._client_cache
        finally:
            with ac._client_cache_lock:
                ac._client_cache.pop(cache_key, None)

    def test_release_none_is_safe(self):
        from agent.auxiliary_client import _release_cached_client_fds

        _release_cached_client_fds(None)  # must not raise

    def test_release_sync_client_closes_pool(self, local_http_server):
        """A cached SYNC client is closed via its own close() (pool released)."""
        from agent.auxiliary_client import _release_cached_client_fds

        sync = httpx.Client()
        sync.get(local_http_server)
        assert not sync.is_closed

        _release_cached_client_fds(type("W", (), {"_client": sync})())

        # The sync httpx pool close() marks the client closed and reaps sockets.
        assert sync.is_closed

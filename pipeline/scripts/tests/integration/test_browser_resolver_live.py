"""Live-smoke: реальный резолв через Puppeteer-core+Lightpanda (не мок, spec
headless-browser-resolver). Требует Node в PATH и бинарь pipeline/browser/lightpanda
(оба gitignored -> в CI и на свежем клоне отсутствуют -> skipif). Резолвит
ЛОКАЛЬНУЮ фикстуру (http.server на loopback) — не зависит от интернета/внешних
сайтов, только от реального Node+Lightpanda-тулчейна."""
from __future__ import annotations

import http.server
import socket
import threading
from collections.abc import Iterator

import pytest

from core.browser_resolver import is_available, resolve

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(not is_available(), reason="Node и/или pipeline/browser/lightpanda не установлены"),
]

_FIXTURE_HTML = (
    "<!doctype html><html><head><title>fixture</title></head>"
    "<body><div id='marker'>BROWSER-RESOLVER-LIVE-OK</div></body></html>"
)


@pytest.fixture(scope="module")
def fixture_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("browser_resolver_fixture")
    (root / "page.html").write_text(_FIXTURE_HTML, encoding="utf-8")

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **kw)  # noqa: E731
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()


def test_resolve_renders_local_fixture_page(fixture_server: str) -> None:
    result = resolve(f"{fixture_server}/page.html", wait_ms=1500, timeout=30)
    assert result.ok is True
    assert "BROWSER-RESOLVER-LIVE-OK" in result.html


def test_resolve_unreachable_host_returns_engine_error_page_not_exception() -> None:
    """Живой факт (не предположение): на CouldntConnect Lightpanda НЕ бросает —
    рендерит СВОЮ внутреннюю страницу «Navigation failed» и отдаёт её как обычный
    ok=True результат (resolve.mjs ловит отказ goto() и всё равно читает
    page.content(), см. комментарий в resolve.mjs). Это не инструментальный
    отказ (BrowserResolverUnavailable не поднимается) — контракт resolve()
    честно возвращает то, что фактически отрендерил движок; отличать «страница
    реальна» от «страница — чужой рендер ошибки» — забота классификатора на
    стороне acquisition.py (WAF-маркеры/размерный порог), не этого модуля."""
    result = resolve("http://127.0.0.1:1", wait_ms=500, timeout=30)
    assert result.ok is True
    assert "BROWSER-RESOLVER-LIVE-OK" not in result.html

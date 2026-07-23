"""Тесты discovery/connectors/eurlex.py (spec discovery-eurlex).

Fetch/parse РАЗДЕЛЕНЫ (чартер, тест-принципы): эти тесты — чистый parse/config/retry
на синтетических фикстурах, БЕЗ реальной сети (``urllib.request.urlopen`` монкипатчится,
зеркало ``test_openrouter.py``). Живой смок (``discover.py discover --only eurlex
--dry-run`` на боевом эндпоинте) — вне CI, спек §Тестовое покрытие.
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
import yaml

from discovery.connectors import eurlex

# --- load_config ---


def test_load_config_reads_real_tracked_config() -> None:
    """pipeline/config/discovery_eurlex.yaml — настоящий трекаемый файл, не фикстура."""
    config = eurlex.load_config()
    assert config.enabled is True
    assert config.sparql_endpoint == "http://publications.europa.eu/webapi/rdf/sparql"
    assert "http://eurovoc.europa.eu/3030" in config.eurovoc_concepts
    assert "http://eurovoc.europa.eu/3740" not in config.eurovoc_concepts  # robotics исключён (v4)
    assert config.expression_language == "ENG"
    assert config.result_limit == 1000
    assert config.timeout_seconds == 60.0


def test_load_config_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "discovery_eurlex.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": False,
                "sparql_endpoint": "http://example.org/sparql",
                "eurovoc_concepts": ["http://eurovoc.europa.eu/3030"],
                "expression_language": "ENG",
                "result_limit": 10,
                "timeout_seconds": 30,
            }
        ),
        encoding="utf-8",
    )
    config = eurlex.load_config(path)
    assert config.enabled is False
    assert config.eurovoc_concepts == ("http://eurovoc.europa.eu/3030",)
    assert config.result_limit == 10
    assert config.timeout_seconds == 30.0


# --- fetch_sparql: retry/backoff (зеркало test_openrouter.py) ---


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _sparql_payload() -> dict[str, Any]:
    return {"head": {"vars": ["celex"]}, "results": {"bindings": []}}


def _http_error(code: int, body: bytes = b"{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://example.org/sparql", code, "err", Message(), io.BytesIO(body)
    )


def test_fetch_sparql_succeeds_on_first_try(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout=60.0: _FakeResponse(_sparql_payload())
    )
    out = eurlex.fetch_sparql("SELECT * WHERE {}", endpoint="http://example.org/sparql", timeout=60.0)
    assert out["head"]["vars"] == ["celex"]


def test_fetch_sparql_retries_after_two_429s(monkeypatch: Any) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(url: str, timeout: float = 60.0) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _http_error(429)
        return _FakeResponse(_sparql_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = eurlex.fetch_sparql("SELECT * WHERE {}", endpoint="http://example.org/sparql", timeout=60.0)
    assert out["head"]["vars"] == ["celex"]
    assert calls["n"] == 3
    assert sleeps == [eurlex.RETRY_SCHEDULE[0], eurlex.RETRY_SCHEDULE[1]]


def test_fetch_sparql_non_retryable_4xx_raises_immediately_without_sleep(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout=60.0: (_ for _ in ()).throw(_http_error(400, b"malformed query")),
    )
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        eurlex.fetch_sparql("bad query", endpoint="http://example.org/sparql", timeout=60.0)
    assert sleeps == []


def test_fetch_sparql_5xx_is_retried_then_exhausts(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout=60.0: (_ for _ in ()).throw(_http_error(503))
    )
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(RuntimeError, match="исчерпаны попытки"):
        eurlex.fetch_sparql("SELECT * WHERE {}", endpoint="http://example.org/sparql", timeout=60.0)
    assert len(sleeps) == len(eurlex.RETRY_SCHEDULE)


def test_fetch_sparql_network_errors_are_retried(monkeypatch: Any) -> None:
    calls = {"n": 0}

    def fake_urlopen(url: str, timeout: float = 60.0) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection refused")
        return _FakeResponse(_sparql_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    out = eurlex.fetch_sparql("SELECT * WHERE {}", endpoint="http://example.org/sparql", timeout=60.0)
    assert out["head"]["vars"] == ["celex"]
    assert calls["n"] == 2


# --- diff_cursor ---


def test_diff_cursor_first_run_all_fresh_all_seen() -> None:
    fresh, cursor = eurlex.diff_cursor(["32024R1689", "32025R2653"], None)
    assert fresh == {"32024R1689", "32025R2653"}
    assert cursor == {"seen_celex": ["32024R1689", "32025R2653"]}


def test_diff_cursor_repeat_run_same_ids_no_new_fresh() -> None:
    cursor: dict[str, Any] = {"seen_celex": ["32024R1689", "32025R2653"]}
    fresh, new_cursor = eurlex.diff_cursor(["32024R1689", "32025R2653"], cursor)
    assert fresh == set()
    assert new_cursor == cursor


def test_diff_cursor_new_id_added_only_new_one_fresh() -> None:
    cursor: dict[str, Any] = {"seen_celex": ["32024R1689"]}
    fresh, new_cursor = eurlex.diff_cursor(["32024R1689", "32026R0150"], cursor)
    assert fresh == {"32026R0150"}
    assert new_cursor == {"seen_celex": ["32024R1689", "32026R0150"]}


def test_diff_cursor_monotonic_never_shrinks_when_upstream_result_shrinks() -> None:
    cursor: dict[str, Any] = {"seen_celex": ["32024R1689", "32025R2653"]}
    # апстрим "потерял" 32025R2653 в текущем прогоне — seen не должен его выбросить (§Вне скоупа)
    fresh, new_cursor = eurlex.diff_cursor(["32024R1689"], cursor)
    assert fresh == set()
    assert new_cursor["seen_celex"] == ["32024R1689", "32025R2653"]

"""Тесты discovery/connectors/oecd.py (spec discovery-oecd).

Fetch/parse РАЗДЕЛЕНЫ (принцип aiforgood/eurlex): эти тесты — чистый parse/config/retry
на синтетических фикстурах, БЕЗ реальной сети (``urllib.request.urlopen`` монкипатчится,
зеркало ``test_aiforgood.py``). Живой смок (``discover.py discover --only oecd
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

from discovery.connectors import oecd

_BASE_CONFIG = oecd.OecdConfig(
    enabled=True,
    endpoint="https://example.org/policy-initiatives",
    user_agent="test-agent/1.0",
    crawl_delay_seconds=0.0,
    timeout_seconds=30.0,
    include_categories=("Cat A",),
    probe_category="Projects",
    probe_initiative_types=("Type A",),
)

# --- load_config ---


def test_load_config_reads_real_tracked_config() -> None:
    """pipeline/config/discovery_oecd.yaml — настоящий трекаемый файл, не фикстура."""
    config = oecd.load_config()
    assert config.enabled is True
    assert config.endpoint == "https://api.oecdai.org/policy-initiatives"
    assert "claude" not in config.user_agent.lower()
    assert config.crawl_delay_seconds == 2.0
    assert config.timeout_seconds == 30.0
    assert "National – Strategy" in config.include_categories
    assert "Regulations, guidelines and standards" in config.include_categories
    assert config.probe_category == "AI policy initiatives, programmes and projects"
    assert config.probe_initiative_types == ("AI use cases/projects in the public sector",)


def test_load_config_custom_path(tmp_path: Path) -> None:
    """Анти-orphan: КАЖДОЕ поле конфига доходит до ``OecdConfig``, ни одно не потеряно
    (урок orphan-бага ``non_us_include_all`` agora, PR #36 — спек §4)."""
    path = tmp_path / "discovery_oecd.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": False,
                "endpoint": "https://example.org/policy-initiatives",
                "user_agent": "test-agent/1.0",
                "crawl_delay_seconds": 1.5,
                "timeout_seconds": 5,
                "include_categories": ["Cat A", "Cat B"],
                "probe_category": "Projects",
                "probe_initiative_types": ["Type A"],
            }
        ),
        encoding="utf-8",
    )
    config = oecd.load_config(path)
    assert config.enabled is False
    assert config.endpoint == "https://example.org/policy-initiatives"
    assert config.user_agent == "test-agent/1.0"
    assert config.crawl_delay_seconds == 1.5
    assert config.timeout_seconds == 5.0
    assert config.include_categories == ("Cat A", "Cat B")
    assert config.probe_category == "Projects"
    assert config.probe_initiative_types == ("Type A",)


# --- fetch_json: retry/backoff (зеркало test_aiforgood.py) ---


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _http_error(code: int, body: bytes = b"{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://example.org/policy-initiatives", code, "err", Message(), io.BytesIO(body)
    )


def test_fetch_json_succeeds_on_first_try(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=30.0: _FakeResponse({"data": []})
    )
    out = oecd.fetch_json(
        1, endpoint="http://example.org/policy-initiatives", user_agent="test-agent/1.0",
        timeout=30.0,
    )
    assert out == {"data": []}


def test_fetch_json_builds_page_query_param(monkeypatch: Any) -> None:
    seen_urls: list[str] = []

    def fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        seen_urls.append(req.full_url)
        return _FakeResponse({"data": []})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    oecd.fetch_json(
        7, endpoint="http://example.org/policy-initiatives", user_agent="test-agent/1.0",
        timeout=30.0,
    )
    assert seen_urls == ["http://example.org/policy-initiatives?page=7"]


def test_fetch_json_retries_after_two_429s(monkeypatch: Any) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _http_error(429)
        return _FakeResponse({"data": []})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = oecd.fetch_json(
        1, endpoint="http://example.org/policy-initiatives", user_agent="test-agent/1.0",
        timeout=30.0,
    )
    assert out == {"data": []}
    assert calls["n"] == 3
    assert sleeps == [oecd.RETRY_SCHEDULE[0], oecd.RETRY_SCHEDULE[1]]


def test_fetch_json_non_retryable_4xx_raises_immediately_without_sleep(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=30.0: (_ for _ in ()).throw(_http_error(404, b"not found")),
    )
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(RuntimeError, match="HTTP 404"):
        oecd.fetch_json(
            1, endpoint="http://example.org/policy-initiatives", user_agent="test-agent/1.0",
            timeout=30.0,
        )
    assert sleeps == []


def test_fetch_json_5xx_is_retried_then_exhausts(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=30.0: (_ for _ in ()).throw(_http_error(503)),
    )
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(RuntimeError, match="исчерпаны попытки"):
        oecd.fetch_json(
            1, endpoint="http://example.org/policy-initiatives", user_agent="test-agent/1.0",
            timeout=30.0,
        )
    assert len(sleeps) == len(oecd.RETRY_SCHEDULE)


def test_fetch_json_network_errors_are_retried(monkeypatch: Any) -> None:
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection refused")
        return _FakeResponse({"data": []})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    out = oecd.fetch_json(
        1, endpoint="http://example.org/policy-initiatives", user_agent="test-agent/1.0",
        timeout=30.0,
    )
    assert out == {"data": []}
    assert calls["n"] == 2


def test_fetch_json_sends_configured_user_agent(monkeypatch: Any) -> None:
    seen_headers: dict[str, str] = {}

    def fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        seen_headers.update(req.headers)
        return _FakeResponse({"data": []})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    oecd.fetch_json(
        1, endpoint="http://example.org/policy-initiatives", user_agent="G2AI-corpus-research/1.0",
        timeout=30.0,
    )
    assert seen_headers.get("User-agent") == "G2AI-corpus-research/1.0"


# --- fetch_all_pages: full-scan + shape-гейт (спек §3) ---


def _page(records: list[dict[str, Any]], *, current: int, last: int, total: int) -> dict[str, Any]:
    return {"data": records, "currentPage": current, "lastPage": last, "total": total}


def test_fetch_all_pages_walks_until_last_page() -> None:
    pages = {
        1: _page([{"id": 1}, {"id": 2}], current=1, last=3, total=6),
        2: _page([{"id": 3}, {"id": 4}], current=2, last=3, total=6),
        3: _page([{"id": 5}, {"id": 6}], current=3, last=3, total=6),
    }

    def fake_fetch(page: int, **_: Any) -> dict[str, Any]:
        return pages[page]

    out = oecd.fetch_all_pages(_BASE_CONFIG, fetch=fake_fetch, sleep=lambda s: None)
    assert [r["id"] for r in out] == [1, 2, 3, 4, 5, 6]


def test_fetch_all_pages_sleeps_between_pages_not_before_first() -> None:
    pages = {
        1: _page([{"id": 1}], current=1, last=2, total=2),
        2: _page([{"id": 2}], current=2, last=2, total=2),
    }
    sleeps: list[float] = []

    def fake_fetch(page: int, **_: Any) -> dict[str, Any]:
        return pages[page]

    oecd.fetch_all_pages(_BASE_CONFIG, fetch=fake_fetch, sleep=lambda s: sleeps.append(s))
    assert sleeps == [_BASE_CONFIG.crawl_delay_seconds]


def test_fetch_all_pages_single_page_no_sleep() -> None:
    page = _page([{"id": 1}], current=1, last=1, total=1)
    sleeps: list[float] = []
    out = oecd.fetch_all_pages(
        _BASE_CONFIG, fetch=lambda page_num, **_: page, sleep=lambda s: sleeps.append(s)
    )
    assert [r["id"] for r in out] == [1]
    assert sleeps == []


def test_fetch_all_pages_empty_data_raises() -> None:
    page = {"data": [], "currentPage": 1, "lastPage": 1, "total": 0}
    with pytest.raises(RuntimeError, match="backend изменил форму"):
        oecd.fetch_all_pages(_BASE_CONFIG, fetch=lambda page_num, **_: page, sleep=lambda s: None)


def test_fetch_all_pages_missing_data_key_raises() -> None:
    page = {"currentPage": 1, "lastPage": 1, "total": 5}
    with pytest.raises(RuntimeError, match="backend изменил форму"):
        oecd.fetch_all_pages(_BASE_CONFIG, fetch=lambda page_num, **_: page, sleep=lambda s: None)


def test_fetch_all_pages_zero_total_raises() -> None:
    page = _page([{"id": 1}], current=1, last=1, total=0)
    with pytest.raises(RuntimeError, match="backend изменил форму"):
        oecd.fetch_all_pages(_BASE_CONFIG, fetch=lambda page_num, **_: page, sleep=lambda s: None)


def test_fetch_all_pages_missing_last_page_raises() -> None:
    page = {"data": [{"id": 1}], "currentPage": 1, "total": 1}
    with pytest.raises(RuntimeError, match="backend изменил форму"):
        oecd.fetch_all_pages(_BASE_CONFIG, fetch=lambda page_num, **_: page, sleep=lambda s: None)


def test_fetch_all_pages_later_page_non_list_data_raises() -> None:
    pages = {
        1: _page([{"id": 1}], current=1, last=2, total=2),
        2: {"data": "not-a-list", "currentPage": 2, "lastPage": 2, "total": 2},
    }
    with pytest.raises(RuntimeError, match="изменил форму на странице 2"):
        oecd.fetch_all_pages(
            _BASE_CONFIG, fetch=lambda page_num, **_: pages[page_num], sleep=lambda s: None
        )


def test_fetch_all_pages_total_mismatch_after_full_scan_raises() -> None:
    pages = {
        1: _page([{"id": 1}], current=1, last=2, total=99),
        2: _page([{"id": 2}], current=2, last=2, total=99),
    }
    with pytest.raises(RuntimeError, match="заявлен total=99, реально получено 2"):
        oecd.fetch_all_pages(
            _BASE_CONFIG, fetch=lambda page_num, **_: pages[page_num], sleep=lambda s: None
        )


# --- save_snapshot: атомарная страховочная запись сырья (спек §3) ---


def test_save_snapshot_writes_raw_records_as_json(tmp_path: Path) -> None:
    path = tmp_path / "oecd" / "policy-initiatives-latest.json"
    records = [{"id": 1, "englishName": "X"}, {"id": 2, "englishName": "Y"}]
    oecd.save_snapshot(records, path=path)
    assert json.loads(path.read_text(encoding="utf-8")) == records


def test_save_snapshot_overwrites_existing(tmp_path: Path) -> None:
    path = tmp_path / "oecd" / "policy-initiatives-latest.json"
    oecd.save_snapshot([{"id": 1}], path=path)
    oecd.save_snapshot([{"id": 2}], path=path)
    assert json.loads(path.read_text(encoding="utf-8")) == [{"id": 2}]


def test_save_snapshot_no_staging_leftover(tmp_path: Path) -> None:
    path = tmp_path / "oecd" / "policy-initiatives-latest.json"
    oecd.save_snapshot([{"id": 1}], path=path)
    assert list(path.parent.glob(".*.part")) == []


# --- diff_cursor: seen-id, монотонный (зеркало test_aiforgood.py) ---


def test_diff_cursor_first_run_all_fresh_all_seen() -> None:
    fresh, cursor = oecd.diff_cursor(["a", "b"], None)
    assert fresh == {"a", "b"}
    assert cursor == {"seen_ids": ["a", "b"]}


def test_diff_cursor_repeat_run_same_ids_no_new_fresh() -> None:
    fresh, cursor = oecd.diff_cursor(["a", "b"], {"seen_ids": ["a", "b"]})
    assert fresh == set()
    assert cursor == {"seen_ids": ["a", "b"]}


def test_diff_cursor_new_id_added_only_new_one_fresh() -> None:
    fresh, cursor = oecd.diff_cursor(["a", "b", "c"], {"seen_ids": ["a", "b"]})
    assert fresh == {"c"}
    assert cursor == {"seen_ids": ["a", "b", "c"]}


def test_diff_cursor_monotonic_never_shrinks_when_upstream_result_shrinks() -> None:
    fresh, cursor = oecd.diff_cursor(["a"], {"seen_ids": ["a", "b"]})
    assert fresh == set()
    assert cursor == {"seen_ids": ["a", "b"]}

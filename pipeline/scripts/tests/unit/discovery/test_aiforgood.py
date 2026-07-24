"""Тесты discovery/connectors/aiforgood.py (spec aiforgood-standards).

Fetch/parse РАЗДЕЛЕНЫ (принцип eurlex/agora): эти тесты — чистый parse/config/retry/
пагинация на синтетических фикстурах, БЕЗ реальной сети (``urllib.request.urlopen``
монкипатчится, зеркало ``test_eurlex.py``/``test_openrouter.py``). Живой смок
(``discover.py discover --only aiforgood --dry-run`` на боевом эндпоинте) — вне CI,
спек §Тестовое покрытие.
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import replace
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
import yaml

from discovery.connectors import aiforgood

# --- load_config ---


def test_load_config_reads_real_tracked_config() -> None:
    """pipeline/config/discovery_aiforgood.yaml — настоящий трекаемый файл, не фикстура."""
    config = aiforgood.load_config()
    assert config.enabled is True
    assert config.ajax_endpoint == "https://aiforgood.itu.int/wp-admin/admin-ajax.php"
    assert config.topic == "tx518"
    assert set(config.exclude_groups) == {"gx1401", "gx1178"}
    assert "draft" in config.exclude_status_substrings
    assert "under development" in config.exclude_status_substrings
    assert "ClaudeBot" not in config.user_agent and "claude" not in config.user_agent.lower()
    assert config.crawl_delay_seconds == 10.0
    assert config.page_size == 10
    assert config.timeout_seconds == 30.0


def test_load_config_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "discovery_aiforgood.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": False,
                "ajax_endpoint": "https://example.org/admin-ajax.php",
                "topic": "tx1",
                "exclude_groups": ["gxA"],
                "exclude_status_substrings": ["draft"],
                "user_agent": "test-agent/1.0",
                "crawl_delay_seconds": 1.0,
                "page_size": 10,
                "timeout_seconds": 5,
            }
        ),
        encoding="utf-8",
    )
    config = aiforgood.load_config(path)
    assert config.enabled is False
    assert config.exclude_groups == ("gxA",)
    assert config.timeout_seconds == 5.0


# --- fetch_json: retry/backoff (зеркало test_eurlex.py) ---


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
        "http://example.org/admin-ajax.php", code, "err", Message(), io.BytesIO(body)
    )


def test_fetch_json_succeeds_on_first_try(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=30.0: _FakeResponse({"ok": True})
    )
    out = aiforgood.fetch_json(
        {"action": "get_groups"}, endpoint="http://example.org/admin-ajax.php",
        user_agent="test-agent/1.0", timeout=30.0,
    )
    assert out == {"ok": True}


def test_fetch_json_retries_after_two_429s(monkeypatch: Any) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _http_error(429)
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = aiforgood.fetch_json(
        {"action": "get_groups"}, endpoint="http://example.org/admin-ajax.php",
        user_agent="test-agent/1.0", timeout=30.0,
    )
    assert out == {"ok": True}
    assert calls["n"] == 3
    assert sleeps == [aiforgood.RETRY_SCHEDULE[0], aiforgood.RETRY_SCHEDULE[1]]


def test_fetch_json_non_retryable_4xx_raises_immediately_without_sleep(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=30.0: (_ for _ in ()).throw(_http_error(400, b"bad request")),
    )
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        aiforgood.fetch_json(
            {"action": "get_groups"}, endpoint="http://example.org/admin-ajax.php",
            user_agent="test-agent/1.0", timeout=30.0,
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
        aiforgood.fetch_json(
            {"action": "get_groups"}, endpoint="http://example.org/admin-ajax.php",
            user_agent="test-agent/1.0", timeout=30.0,
        )
    assert len(sleeps) == len(aiforgood.RETRY_SCHEDULE)


def test_fetch_json_network_errors_are_retried(monkeypatch: Any) -> None:
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection refused")
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    out = aiforgood.fetch_json(
        {"action": "get_groups"}, endpoint="http://example.org/admin-ajax.php",
        user_agent="test-agent/1.0", timeout=30.0,
    )
    assert out == {"ok": True}
    assert calls["n"] == 2


def test_fetch_json_sends_configured_user_agent(monkeypatch: Any) -> None:
    """robots.txt Disallow: ClaudeBot — коннектор обязан слать нейтральный UA (§4/OQ1)."""
    seen_headers: dict[str, str] = {}

    def fake_urlopen(req: Any, timeout: float = 30.0) -> Any:
        seen_headers.update(req.headers)
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    aiforgood.fetch_json(
        {"action": "get_groups"}, endpoint="http://example.org/admin-ajax.php",
        user_agent="G2AI-corpus-research/1.0", timeout=30.0,
    )
    assert seen_headers.get("User-agent") == "G2AI-corpus-research/1.0"


# --- фикстуры конфига ---


_BASE_CONFIG = aiforgood.AiforgoodConfig(
    enabled=True,
    ajax_endpoint="https://example.org/admin-ajax.php",
    topic="tx518",
    exclude_groups=("gx1401", "gx1178"),
    exclude_status_substrings=("draft", "under development"),
    user_agent="test-agent/1.0",
    crawl_delay_seconds=0.0,
    page_size=10,
    timeout_seconds=30.0,
)


def _config(**overrides: Any) -> aiforgood.AiforgoodConfig:
    return replace(_BASE_CONFIG, **overrides)


# --- get_groups / group_total ---


def test_get_groups_parses_live_response_shape() -> None:
    """Форма живьём подтверждена 2026-07-24: {"success": true, "data": [...]}."""
    payload = {
        "success": True,
        "data": [
            {"id": "gx0", "text": "ITU-T <strong>(654)</strong>", "children": True, "data": {"total": 654}},
            {"id": "gx1401", "text": "ISO/IEC <strong>(81)</strong>", "children": True, "data": {"total": 81}},
        ],
    }
    groups = aiforgood.get_groups(_config(), fetch=lambda *a, **kw: payload)
    assert [g["id"] for g in groups] == ["gx0", "gx1401"]
    group_totals = {g["id"]: aiforgood.group_total(g) for g in groups}
    assert group_totals == {"gx0": 654, "gx1401": 81}


def test_get_groups_passes_topic_and_action(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        captured.update(params)
        return {"success": True, "data": []}

    aiforgood.get_groups(_config(topic="tx518"), fetch=fake_fetch)
    assert captured == {"action": "get_groups", "topic": "tx518"}


def test_group_total_missing_data_defaults_zero() -> None:
    assert aiforgood.group_total({"id": "gx0"}) == 0


# --- get_standards_page / paginate_group ---


def _standards_page(records: list[dict[str, Any]], *, total: int) -> dict[str, Any]:
    return {"standards": records, "totalCount": total, "facets": []}


def _standard(id_value: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "id_value": id_value,
        "standard_name": f"ITU-T {id_value}",
        "standard_title": "Example standard",
        "standard_status": "In force",
        "standard_url": f"https://www.itu.int/rec/{id_value}",
        "standard_summary": "A summary.",
        "standard_type": "Recommendation",
    }
    base.update(overrides)
    return base


def test_paginate_group_walks_all_pages_until_total_exhausted() -> None:
    """1 группа, 25 записей, page_size=10 -> 3 страницы (10+10+5), не только первая."""
    all_records = [_standard(f"r{i}") for i in range(25)]
    pages_requested: list[int] = []

    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        idx = int(params["index"])
        pages_requested.append(idx)
        batch = all_records[idx : idx + 10]
        return _standards_page(batch, total=25)

    out = aiforgood.paginate_group(_config(), group_id="gx0", fetch=fake_fetch, sleep=lambda s: None)
    assert pages_requested == [0, 10, 20]
    assert [r["id_value"] for r in out] == [f"r{i}" for i in range(25)]


def test_paginate_group_sleeps_crawl_delay_between_pages_not_before_first() -> None:
    all_records = [_standard(f"r{i}") for i in range(15)]
    sleeps: list[float] = []

    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        idx = int(params["index"])
        return _standards_page(all_records[idx : idx + 10], total=15)

    aiforgood.paginate_group(
        _config(crawl_delay_seconds=7.0), group_id="gx0", fetch=fake_fetch,
        sleep=lambda s: sleeps.append(s),
    )
    assert sleeps == [7.0]  # 2 страницы -> 1 пауза между ними, не перед первым запросом


def test_paginate_group_single_page_no_sleep() -> None:
    all_records = [_standard(f"r{i}") for i in range(3)]
    sleeps: list[float] = []

    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        return _standards_page(all_records, total=3)

    aiforgood.paginate_group(_config(), group_id="gx0", fetch=fake_fetch, sleep=lambda s: sleeps.append(s))
    assert sleeps == []


def test_paginate_group_empty_group_returns_empty_list() -> None:
    out = aiforgood.paginate_group(
        _config(), group_id="gx1141", fetch=lambda *a, **kw: _standards_page([], total=0),
        sleep=lambda s: None,
    )
    assert out == []


def test_get_standards_page_passes_group_and_index() -> None:
    captured: dict[str, Any] = {}

    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        captured.update(params)
        return _standards_page([], total=0)

    aiforgood.get_standards_page(_config(topic="tx518"), group_id="gx0", index=10, fetch=fake_fetch)
    assert captured == {"action": "get_standards", "topic": "tx518", "group": "gx0", "index": "10"}


# --- diff_cursor (зеркало test_eurlex.py) ---


def test_diff_cursor_first_run_all_fresh_all_seen() -> None:
    fresh, cursor = aiforgood.diff_cursor(["a", "b"], None)
    assert fresh == {"a", "b"}
    assert cursor == {"seen_ids": ["a", "b"]}


def test_diff_cursor_repeat_run_same_ids_no_new_fresh() -> None:
    fresh, cursor = aiforgood.diff_cursor(["a", "b"], {"seen_ids": ["a", "b"]})
    assert fresh == set()
    assert cursor == {"seen_ids": ["a", "b"]}


def test_diff_cursor_new_id_added_only_new_one_fresh() -> None:
    fresh, cursor = aiforgood.diff_cursor(["a", "b", "c"], {"seen_ids": ["a", "b"]})
    assert fresh == {"c"}
    assert cursor == {"seen_ids": ["a", "b", "c"]}


def test_diff_cursor_monotonic_never_shrinks_when_upstream_result_shrinks() -> None:
    fresh, cursor = aiforgood.diff_cursor(["a"], {"seen_ids": ["a", "b"]})
    assert fresh == set()
    assert cursor == {"seen_ids": ["a", "b"]}

"""Тесты discovery/connectors/aiforgood.py (spec aiforgood-standards).

Fetch/parse РАЗДЕЛЕНЫ (принцип eurlex/agora): эти тесты — чистый parse/config/retry/
пагинация на синтетических фикстурах, БЕЗ реальной сети (``urllib.request.urlopen``
монкипатчится, зеркало ``test_eurlex.py``/``test_openrouter.py``). Живой смок
(``discover.py discover --only aiforgood --dry-run`` на боевом эндпоинте) — вне CI,
спек §Тестовое покрытие.
"""
from __future__ import annotations

import datetime as dt
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

from core import schema
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


# --- load_standards_bodies (справочник §3) — регресс-гвард против рассинхрона ---


def test_load_standards_bodies_reads_real_tracked_vocab() -> None:
    bodies = aiforgood.load_standards_bodies()
    assert set(bodies) == {"itu-t", "itu-r", "ietf", "u4ssc", "etsi", "tta"}
    assert bodies["itu-t"]["kind"] == "international"
    assert "ITU Telecommunication" in bodies["itu-t"]["full_name"]
    assert bodies["etsi"]["kind"] == "sectoral"
    assert bodies["tta"]["kind"] == "national"


def test_group_id_to_entity_all_targets_present_in_vocab() -> None:
    """Регресс-гвард: каждый entity_id, который коннектор способен породить
    (GROUP_ID_TO_ENTITY), обязан существовать в vocab_standards_bodies.yaml."""
    bodies = aiforgood.load_standards_bodies()
    for entity_id in aiforgood.GROUP_ID_TO_ENTITY.values():
        assert entity_id in bodies, entity_id


# --- is_excluded_status / _valid_url ---


@pytest.mark.parametrize(
    "status,expected",
    [
        ("In force", False),
        ("Published", False),
        ("Under development/Draft", True),
        ("DRAFT", True),
        ("under development", True),
        (None, False),
        ("", False),
    ],
)
def test_is_excluded_status(status: str | None, expected: bool) -> None:
    assert aiforgood.is_excluded_status(status, ("draft", "under development")) is expected


def test_valid_url_accepts_http_and_https() -> None:
    assert aiforgood._valid_url("https://www.itu.int/rec/T-REC-E.475") is True
    assert aiforgood._valid_url("http://example.org") is True


@pytest.mark.parametrize("url", [None, "", "ftp://example.org", "not-a-url"])
def test_valid_url_rejects_missing_or_non_http(url: str | None) -> None:
    assert aiforgood._valid_url(url) is False


# --- _map_record (§4) ---


def test_map_record_field_mapping() -> None:
    record = _standard(
        "14148",
        standard_name="ITU-T E.475 (01/2020)",
        standard_title="Interoperability testing",
        standard_status="In force",
        standard_type="Recommendation",
        standard_summary="A summary of the standard.",
        standard_url="https://www.itu.int/myworkspace/t-rec/item?id=14148",
    )
    cand = aiforgood._map_record(record, entity_id="itu-t", issuer_full_name="ITU Telecommunication Standardization Sector (ITU-T)")
    assert cand is not None
    assert cand.title == "ITU-T E.475 (01/2020): Interoperability testing"
    assert cand.issuer == "ITU Telecommunication Standardization Sector (ITU-T)"
    assert cand.source_url == "https://www.itu.int/myworkspace/t-rec/item?id=14148"
    assert cand.native_id == "14148"
    assert cand.native_summary == "A summary of the standard."
    assert cand.native_tags == ["ITU AI Standards Exchange: In force", "type: Recommendation"]
    assert cand.language is None
    assert cand.doc_date is None
    assert cand.jurisdiction is None
    assert cand.connector_id == "aiforgood"


def test_map_record_title_falls_back_to_single_part_when_other_empty() -> None:
    record = _standard("1", standard_name="ITU-T E.475", standard_title="")
    cand = aiforgood._map_record(record, entity_id="itu-t", issuer_full_name="ITU-T")
    assert cand is not None
    assert cand.title == "ITU-T E.475"


def test_map_record_no_title_at_all_is_skipped() -> None:
    record = _standard("1", standard_name="", standard_title="")
    assert aiforgood._map_record(record, entity_id="itu-t", issuer_full_name="ITU-T") is None


def test_map_record_missing_url_is_skipped() -> None:
    record = _standard("1", standard_url="")
    assert aiforgood._map_record(record, entity_id="itu-t", issuer_full_name="ITU-T") is None


def test_map_record_dash_summary_becomes_none_not_fabricated() -> None:
    record = _standard("1", standard_summary="-")
    cand = aiforgood._map_record(record, entity_id="itu-t", issuer_full_name="ITU-T")
    assert cand is not None
    assert cand.native_summary is None


def test_map_record_raw_hash_deterministic_and_changes_with_content() -> None:
    a = aiforgood._map_record(_standard("1"), entity_id="itu-t", issuer_full_name="ITU-T")
    b = aiforgood._map_record(_standard("1"), entity_id="itu-t", issuer_full_name="ITU-T")
    c = aiforgood._map_record(_standard("1", standard_status="Withdrawn"), entity_id="itu-t", issuer_full_name="ITU-T")
    assert a is not None and b is not None and c is not None
    assert a.raw_hash == b.raw_hash
    assert a.raw_hash != c.raw_hash


def test_candidate_requires_language_override_like_agora() -> None:
    """aiforgood-кандидат несёт language=None (организация — не страна) — промоушен
    требует явный override триажа (тот же механизм, что AGORA, PR #36)."""
    cand = aiforgood._map_record(_standard("1"), entity_id="itu-t", issuer_full_name="ITU-T")
    assert cand is not None
    record = schema.promote_candidate(
        cand,
        id="itu-t-e475-test",
        entity_id="itu-t",
        track=schema.Track.tech_standards,
        issuer_type=schema.IssuerType.standards_body,
        geo_scope=schema.GeoScope.international,
        doc_type="technical_standard",
        authority="voluntary_standard",
        relevance=schema.Relevance(
            target_fit=schema.TargetFit.primary,
            axis="digital_sovereignty",
            assessed_stage=schema.AssessedStage.triage,
            rationale="test",
            assessed_date=dt.date(2026, 7, 24),
        ),
        language="en",
    )
    assert record.language == "en"


# --- discover_aiforgood (§4) ---


def _bodies() -> dict[str, dict[str, str]]:
    return {
        entity_id: {"kind": "international", "full_name": entity_id.upper()}
        for entity_id in aiforgood.GROUP_ID_TO_ENTITY.values()
    }


def _groups_payload(*groups: tuple[str, int]) -> dict[str, Any]:
    return {
        "success": True,
        "data": [{"id": gid, "text": f"{gid} ({total})", "data": {"total": total}} for gid, total in groups],
    }


def test_discover_aiforgood_excludes_paid_catalog_groups_never_fetched() -> None:
    """gx1401 (ISO/IEC) в exclude_groups — get_standards для неё не должен вызываться вовсе."""
    fetch_calls: list[dict[str, str]] = []

    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        fetch_calls.append(dict(params))
        if params["action"] == "get_groups":
            return _groups_payload(("gx0", 1), ("gx1401", 81))
        assert params["group"] != "gx1401", "excluded group must not be paginated"
        return {"standards": [_standard("1")], "totalCount": 1, "facets": []}

    result = aiforgood.discover_aiforgood(
        None, config=aiforgood.load_config(), fetch=fake_fetch, sleep=lambda s: None, bodies=_bodies()
    )
    assert result.diagnostics["excluded_groups"] == 1
    assert all(c.native_tags for c in result.candidates)
    group_params = [c["group"] for c in fetch_calls if c["action"] == "get_standards"]
    assert "gx1401" not in group_params


def test_discover_aiforgood_unknown_group_skipped_with_diagnostic() -> None:
    """Группа не в exclude_groups и не в GROUP_ID_TO_ENTITY — новая организация,
    пропускается с диагностикой, не угадывается."""
    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        if params["action"] == "get_groups":
            return _groups_payload(("gx9999", 5))
        raise AssertionError("unknown group must not be paginated")

    result = aiforgood.discover_aiforgood(
        None, config=aiforgood.load_config(), fetch=fake_fetch, sleep=lambda s: None, bodies=_bodies()
    )
    assert result.diagnostics["skipped_unknown_group"] == 1
    assert result.candidates == []


def test_discover_aiforgood_draft_status_skipped() -> None:
    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        if params["action"] == "get_groups":
            return _groups_payload(("gx0", 2))
        return {
            "standards": [
                _standard("1", standard_status="Under development/Draft"),
                _standard("2", standard_status="In force"),
            ],
            "totalCount": 2,
            "facets": [],
        }

    result = aiforgood.discover_aiforgood(
        None, config=aiforgood.load_config(), fetch=fake_fetch, sleep=lambda s: None, bodies=_bodies()
    )
    assert result.diagnostics["skipped_draft"] == 1
    assert [c.native_id for c in result.candidates] == ["2"]


def test_discover_aiforgood_first_run_all_fresh() -> None:
    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        if params["action"] == "get_groups":
            return _groups_payload(("gx0", 1))
        return {"standards": [_standard("1")], "totalCount": 1, "facets": []}

    result = aiforgood.discover_aiforgood(
        None, config=aiforgood.load_config(), fetch=fake_fetch, sleep=lambda s: None, bodies=_bodies()
    )
    assert result.diagnostics["status"] == "fetched"
    assert len(result.candidates) == 1
    assert result.cursor == {"seen_ids": ["1"]}


def test_discover_aiforgood_repeat_run_same_result_is_no_new() -> None:
    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        if params["action"] == "get_groups":
            return _groups_payload(("gx0", 1))
        return {"standards": [_standard("1")], "totalCount": 1, "facets": []}

    result = aiforgood.discover_aiforgood(
        {"seen_ids": ["1"]}, config=aiforgood.load_config(), fetch=fake_fetch, sleep=lambda s: None,
        bodies=_bodies(),
    )
    assert result.diagnostics["status"] == "no_new"
    assert result.candidates == []


def test_discover_aiforgood_new_id_appears_only_it_is_fresh() -> None:
    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        if params["action"] == "get_groups":
            return _groups_payload(("gx0", 2))
        return {
            "standards": [_standard("1"), _standard("2")],
            "totalCount": 2,
            "facets": [],
        }

    result = aiforgood.discover_aiforgood(
        {"seen_ids": ["1"]}, config=aiforgood.load_config(), fetch=fake_fetch, sleep=lambda s: None,
        bodies=_bodies(),
    )
    assert [c.native_id for c in result.candidates] == ["2"]
    assert result.cursor == {"seen_ids": ["1", "2"]}


def test_discover_aiforgood_invalid_url_skipped_not_crashing_batch() -> None:
    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        if params["action"] == "get_groups":
            return _groups_payload(("gx0", 2))
        return {
            "standards": [_standard("1", standard_url=""), _standard("2")],
            "totalCount": 2,
            "facets": [],
        }

    result = aiforgood.discover_aiforgood(
        None, config=aiforgood.load_config(), fetch=fake_fetch, sleep=lambda s: None, bodies=_bodies()
    )
    assert result.diagnostics["skipped_no_title_or_url"] == 1
    assert [c.native_id for c in result.candidates] == ["2"]


def test_aiforgood_connector_uses_configured_topic_and_multiple_orgs() -> None:
    """Смок нескольких организаций в одном прогоне — маппинг issuer идёт по entity_id, не
    по сырому тексту группы."""
    def fake_fetch(params: dict[str, str], **kw: Any) -> dict[str, Any]:
        if params["action"] == "get_groups":
            return _groups_payload(("gx0", 1), ("gx1043", 1))
        group = params["group"]
        return {"standards": [_standard(f"{group}-1")], "totalCount": 1, "facets": []}

    result = aiforgood.discover_aiforgood(
        None, config=aiforgood.load_config(), fetch=fake_fetch, sleep=lambda s: None, bodies=_bodies()
    )
    issuers = {c.native_id: c.issuer for c in result.candidates}
    assert issuers == {"gx0-1": "ITU-T", "gx1043-1": "ETSI"}

"""Тесты discovery/connectors/eurlex.py (spec discovery-eurlex).

Fetch/parse РАЗДЕЛЕНЫ (чартер, тест-принципы): эти тесты — чистый parse/config/retry
на синтетических фикстурах, БЕЗ реальной сети (``urllib.request.urlopen`` монкипатчится,
зеркало ``test_openrouter.py``). Живой смок (``discover.py discover --only eurlex
--dry-run`` на боевом эндпоинте) — вне CI, спек §Тестовое покрытие.
"""
from __future__ import annotations

import datetime as dt
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

from core import schema
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


# --- синтетические SPARQL-JSON фикстуры (форма — как реально отдаёт CELLAR) ---


def _binding(value: str) -> dict[str, str]:
    return {"type": "literal", "value": value}


def _uri_binding(value: str) -> dict[str, str]:
    return {"type": "uri", "value": value}


def _row(
    celex: str,
    *,
    date: str | None = None,
    title: str | None = None,
    author: str | None = None,
    concept: str = "http://eurovoc.europa.eu/3030",
) -> dict[str, dict[str, str]]:
    row: dict[str, dict[str, str]] = {"celex": _binding(celex), "concept": _uri_binding(concept)}
    if date is not None:
        row["date"] = _binding(date)
    if title is not None:
        row["title"] = _binding(title)
    if author is not None:
        row["authorLabel"] = _binding(author)
    return row


def _sparql_json(rows: list[dict[str, dict[str, str]]]) -> dict[str, Any]:
    return {
        "head": {"vars": ["celex", "date", "title", "authorLabel", "concept"]},
        "results": {"bindings": rows},
    }


def _fake_config(**overrides: object) -> eurlex.EurlexConfig:
    base: dict[str, object] = dict(
        enabled=True,
        sparql_endpoint="http://example.org/sparql",
        eurovoc_concepts=("http://eurovoc.europa.eu/3030",),
        expression_language="ENG",
        result_limit=500,
        timeout_seconds=60.0,
    )
    base.update(overrides)
    return eurlex.EurlexConfig(**base)  # type: ignore[arg-type]


# --- build_query (§3) ---


def test_build_query_includes_values_for_each_configured_concept() -> None:
    config = _fake_config(
        eurovoc_concepts=("http://eurovoc.europa.eu/3030", "http://eurovoc.europa.eu/3293")
    )
    query = eurlex.build_query(config)
    assert "<http://eurovoc.europa.eu/3030>" in query
    assert "<http://eurovoc.europa.eu/3293>" in query
    assert "LIMIT 500" in query


def test_build_query_no_celex_type_filter() -> None:
    """Регресс-гвард против ре-интродукции v1-фильтра (§0/OQ1) — запрос не сужает
    по CELEX-сектору/типу, только по EuroVoc-концепту (тип решает триаж, не коннектор)."""
    query = eurlex.build_query(_fake_config())
    assert "resource_legal_id_celex" in query  # celex как идентификатор — есть
    assert "REGEX" not in query  # но не как форма-фильтр


def test_build_query_no_untyped_string_equality_on_celex() -> None:
    """Регресс-гвард против ловушки CELLAR: нетипизированный строковый литерал в
    FILTER по CELEX -> тихий пустой результат ([[reference-ai-policy-databases]])."""
    query = eurlex.build_query(_fake_config())
    assert '?celex = "' not in query


def test_build_query_empty_concepts_raises() -> None:
    with pytest.raises(ValueError, match="eurovoc_concepts"):
        eurlex.build_query(_fake_config(eurovoc_concepts=()))


def test_build_query_uses_expression_language_from_config() -> None:
    query = eurlex.build_query(_fake_config(expression_language="EST"))
    assert "authority/language/EST" in query


# --- parse_bindings ---


def test_parse_bindings_extracts_values() -> None:
    sparql_json = _sparql_json(
        [_row("32024R1689", date="2024-06-13", title="AI Act", author="European Parliament")]
    )
    rows = eurlex.parse_bindings(sparql_json)
    assert rows == [
        {
            "celex": "32024R1689",
            "date": "2024-06-13",
            "title": "AI Act",
            "authorLabel": "European Parliament",
            "concept": "http://eurovoc.europa.eu/3030",
        }
    ]


def test_parse_bindings_missing_optional_var_is_none_not_keyerror() -> None:
    sparql_json = _sparql_json([_row("32024R1689")])  # без date/title/author
    rows = eurlex.parse_bindings(sparql_json)
    assert rows[0]["date"] is None
    assert rows[0]["title"] is None
    assert rows[0]["authorLabel"] is None


def test_parse_bindings_is_pure_no_side_effects() -> None:
    """Fetch/parse РАЗДЕЛЕНЫ (чартер) — parse_bindings чистая функция, без сети/файлов."""
    sparql_json = _sparql_json(
        [_row("32024R1689", date="2024-06-13", title="AI Act", author="EP")]
    )
    assert eurlex.parse_bindings(sparql_json) == eurlex.parse_bindings(sparql_json)


# --- group_by_celex ---


def test_group_by_celex_merges_co_authors() -> None:
    rows = eurlex.parse_bindings(
        _sparql_json(
            [
                _row("32024R1689", date="2024-06-13", title="AI Act", author="European Parliament"),
                _row(
                    "32024R1689",
                    date="2024-06-13",
                    title="AI Act",
                    author="Council of the European Union",
                ),
            ]
        )
    )
    grouped = eurlex.group_by_celex(rows)
    assert set(grouped.keys()) == {"32024R1689"}
    assert grouped["32024R1689"]["authors"] == [
        "European Parliament",
        "Council of the European Union",
    ]
    assert grouped["32024R1689"]["title"] == "AI Act"


def test_group_by_celex_multi_concept_match_deduplicates_into_one_entry() -> None:
    rows = eurlex.parse_bindings(
        _sparql_json(
            [
                _row(
                    "32024R1689", date="2024-06-13", title="AI Act", author="EP",
                    concept="http://eurovoc.europa.eu/3030",
                ),
                _row(
                    "32024R1689", date="2024-06-13", title="AI Act", author="EP",
                    concept="http://eurovoc.europa.eu/c_5a195ffd",
                ),
            ]
        )
    )
    grouped = eurlex.group_by_celex(rows)
    assert len(grouped) == 1
    assert grouped["32024R1689"]["concepts"] == {
        "http://eurovoc.europa.eu/3030",
        "http://eurovoc.europa.eu/c_5a195ffd",
    }
    assert grouped["32024R1689"]["authors"] == ["EP"]  # дубль автора не повторился


def test_group_by_celex_row_without_celex_is_ignored() -> None:
    rows: list[dict[str, str | None]] = [
        {"celex": None, "date": None, "title": None, "authorLabel": None, "concept": None}
    ]
    assert eurlex.group_by_celex(rows) == {}


# --- decode_celex_type (§5.1) ---


def test_decode_celex_type_legal_act_regulation() -> None:
    assert eurlex.decode_celex_type("32024R1689") == "EUR-Lex: legal act, Regulation"


def test_decode_celex_type_preparatory_act_proposal() -> None:
    assert eurlex.decode_celex_type("52026PC0502") == "EUR-Lex: preparatory act, proposal"


def test_decode_celex_type_merger_notification() -> None:
    assert "merger notification" in eurlex.decode_celex_type("52026M12417")


def test_decode_celex_type_unknown_type_code_passes_through_raw_not_crash() -> None:
    result = eurlex.decode_celex_type("92026Q0001")
    assert result == "EUR-Lex: other, Q"


def test_decode_celex_type_malformed_string_does_not_crash() -> None:
    assert eurlex.decode_celex_type("not-a-celex") == "EUR-Lex: other, unknown"
    assert eurlex.decode_celex_type("") == "EUR-Lex: other, unknown"


# --- concept_label ---


def test_concept_label_known_concepts() -> None:
    assert eurlex.concept_label("http://eurovoc.europa.eu/3030") == "artificial intelligence"
    assert eurlex.concept_label("http://eurovoc.europa.eu/3293") == "cybernetics"


def test_concept_label_unknown_concept_falls_back_to_uri_suffix() -> None:
    assert eurlex.concept_label("http://eurovoc.europa.eu/9999") == "9999"


# --- map_rows_to_candidates / _map_group (§5) ---


def test_map_rows_wide_set_no_type_filter_all_pass() -> None:
    """Регресс-гвард: merger/preparatory/binding — ВСЕ проходят в кандидаты, различие
    только в native_tags (тип — pre-signal триажу, не гейт коннектора, §0/OQ1)."""
    rows = eurlex.parse_bindings(
        _sparql_json(
            [
                _row(
                    "52026M12417", date="2026-07-07",
                    title="Prior notification of a concentration", author="European Commission",
                ),
                _row(
                    "52026PC0502", date="2026-06-03",
                    title="Proposal for a Regulation", author="European Commission",
                ),
                _row("32024R1689", date="2024-06-13", title="AI Act", author="European Parliament"),
            ]
        )
    )
    candidates, skipped = eurlex.map_rows_to_candidates(rows)
    assert skipped == 0
    assert {c.native_id for c in candidates} == {"52026M12417", "52026PC0502", "32024R1689"}


def test_map_group_empty_title_corrigendum_skipped_not_crashed() -> None:
    rows = eurlex.parse_bindings(
        _sparql_json(
            [
                _row("32024R1689R(04)", date="2026-05-04", title=None, author="European Commission"),
                _row("32024R1689", date="2024-06-13", title="AI Act", author="European Parliament"),
            ]
        )
    )
    candidates, skipped = eurlex.map_rows_to_candidates(rows)
    assert skipped == 1
    assert [c.native_id for c in candidates] == ["32024R1689"]


def test_map_group_field_mapping() -> None:
    rows = eurlex.parse_bindings(
        _sparql_json(
            [
                _row(
                    "32024R1689", date="2024-06-13", title="AI Act", author="European Parliament",
                    concept="http://eurovoc.europa.eu/3030",
                )
            ]
        )
    )
    candidates, _ = eurlex.map_rows_to_candidates(rows)
    cand = candidates[0]
    assert cand.source_url == (
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689"
    )
    assert cand.language == "en"
    assert cand.jurisdiction == "European Union"
    assert cand.issuer == "European Parliament"
    assert cand.doc_date == dt.date(2024, 6, 13)
    assert cand.rights == schema.Rights.cc_by
    assert cand.connector_id == "eurlex"
    assert any(t.startswith("EuroVoc:") for t in cand.native_tags or [])
    assert cand.normalized_url is not None
    assert len(cand.raw_hash) == 64


def test_map_group_missing_date_gives_none() -> None:
    rows = eurlex.parse_bindings(_sparql_json([_row("32024R1689", title="AI Act", author="EP")]))
    candidates, _ = eurlex.map_rows_to_candidates(rows)
    assert candidates[0].doc_date is None


def test_map_group_no_author_gives_none_issuer_not_fabricated() -> None:
    """EUR-Lex не даёт автора для этой (гипотетической) строки — issuer=None, не
    фабрикуем дефолт ('European Union' и т.п.); строгий промоушен упадёт явно (§6)."""
    rows = eurlex.parse_bindings(_sparql_json([_row("32024R1689", title="AI Act")]))
    candidates, _ = eurlex.map_rows_to_candidates(rows)
    assert candidates[0].issuer is None


def test_raw_hash_deterministic_and_changes_with_content() -> None:
    rows_a = eurlex.parse_bindings(
        _sparql_json([_row("32024R1689", date="2024-06-13", title="AI Act", author="EP")])
    )
    rows_b = eurlex.parse_bindings(
        _sparql_json([_row("32024R1689", date="2024-06-13", title="AI Act", author="EP")])
    )
    cand_a, _ = eurlex.map_rows_to_candidates(rows_a)
    cand_b, _ = eurlex.map_rows_to_candidates(rows_b)
    assert cand_a[0].raw_hash == cand_b[0].raw_hash

    rows_c = eurlex.parse_bindings(
        _sparql_json(
            [_row("32024R1689", date="2024-06-13", title="AI Act (amended)", author="EP")]
        )
    )
    cand_c, _ = eurlex.map_rows_to_candidates(rows_c)
    assert cand_c[0].raw_hash != cand_a[0].raw_hash


def test_candidate_promotable_without_language_override() -> None:
    """EUR-Lex-кандидат несёт language='en' сам — в отличие от AGORA, override не нужен (§6)."""
    rows = eurlex.parse_bindings(
        _sparql_json(
            [_row("32024R1689", date="2024-06-13", title="AI Act", author="European Parliament")]
        )
    )
    candidates, _ = eurlex.map_rows_to_candidates(rows)
    record = schema.promote_candidate(
        candidates[0],
        id="eu-ai-act-test",
        entity_id="eu",
        track=schema.Track.intl_xperience,
        issuer_type=schema.IssuerType.igo,
        geo_scope=schema.GeoScope.regional,
        doc_type="legislation",
        authority="binding_law",
        relevance=schema.Relevance(
            target_fit=schema.TargetFit.primary,
            axis="digital_sovereignty",
            assessed_stage=schema.AssessedStage.triage,
            rationale="test",
            assessed_date=dt.date(2026, 7, 23),
        ),
        source_format=schema.SourceFormat.html,
    )
    assert record.language == "en"


# --- discover_eurlex (§3/§4) ---


def test_discover_eurlex_first_run_all_fresh() -> None:
    sparql_json = _sparql_json(
        [
            _row("32024R1689", date="2024-06-13", title="AI Act", author="European Parliament"),
            _row(
                "32025R2653", date="2025-12-19", title="Some Regulation",
                author="Council of the European Union",
            ),
        ]
    )

    def fake_fetch(query: str, *, endpoint: str, timeout: float) -> dict[str, Any]:
        return sparql_json

    result = eurlex.discover_eurlex(None, config=_fake_config(), fetch=fake_fetch)
    assert {c.native_id for c in result.candidates} == {"32024R1689", "32025R2653"}
    assert result.cursor == {"seen_celex": ["32024R1689", "32025R2653"]}
    assert result.diagnostics["status"] == "fetched"


def test_discover_eurlex_repeat_run_same_result_is_no_new() -> None:
    sparql_json = _sparql_json(
        [_row("32024R1689", date="2024-06-13", title="AI Act", author="European Parliament")]
    )

    def fake_fetch(query: str, *, endpoint: str, timeout: float) -> dict[str, Any]:
        return sparql_json

    first = eurlex.discover_eurlex(None, config=_fake_config(), fetch=fake_fetch)
    second = eurlex.discover_eurlex(first.cursor, config=_fake_config(), fetch=fake_fetch)
    assert second.candidates == []
    assert second.diagnostics["status"] == "no_new"
    assert second.cursor == first.cursor


def test_discover_eurlex_new_celex_appears_only_it_is_fresh() -> None:
    first_json = _sparql_json(
        [_row("32024R1689", date="2024-06-13", title="AI Act", author="EP")]
    )
    second_json = _sparql_json(
        [
            _row("32024R1689", date="2024-06-13", title="AI Act", author="EP"),
            _row("32026R0150", date="2026-01-16", title="New Regulation", author="Council"),
        ]
    )
    calls = {"n": 0}

    def fake_fetch(query: str, *, endpoint: str, timeout: float) -> dict[str, Any]:
        calls["n"] += 1
        return first_json if calls["n"] == 1 else second_json

    first = eurlex.discover_eurlex(None, config=_fake_config(), fetch=fake_fetch)
    second = eurlex.discover_eurlex(first.cursor, config=_fake_config(), fetch=fake_fetch)
    assert [c.native_id for c in second.candidates] == ["32026R0150"]
    assert second.diagnostics["status"] == "fetched"


def test_eurlex_connector_implements_protocol() -> None:
    conn = eurlex.EurlexConnector(enabled=True)
    assert conn.id == "eurlex"
    assert conn.kind == schema.ConnectorKind.registry
    assert conn.enabled is True

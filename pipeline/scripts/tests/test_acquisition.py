"""Тесты лестницы добычи: детекция блока и маршрутизация (без реальной сети —
файлы/subprocess/сетевые вызовы подделываются)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

import schema
from acquisition import (
    AcquisitionBlocked,
    AcquisitionDead,
    AcquisitionOutcome,
    ClassifiedResponse,
    classify_response,
    fetch_and_classify,
    next_rung,
    persist_acquisition_state,
    run_ladder,
)
from test_schema import valid_record

REAL_PDF_BODY = b"%PDF-1.6\n" + b"x" * 4000  # больше MIN_EXPECTED_PDF_SIZE

CLOUDFLARE_BLOCK_HEADERS = (
    "HTTP/2 403 \n"
    "date: Wed, 15 Jul 2026 15:43:50 GMT\n"
    "content-type: text/html; charset=UTF-8\n"
    "cf-ray: a1b9e294cfd4e293-BEG\n"
    "server: cloudflare\n"
)
CLOUDFLARE_BLOCK_BODY = b"<html><body>Sorry, you have been blocked</body></html>"

CF_COOKIE_ONLY_HEADERS = (
    "HTTP/2 200 \n"
    "content-type: text/html; charset=UTF-8\n"
    "set-cookie: __cf_bm=0Cmh0PjncfxIOzhyqGyzrl7s; HttpOnly; SameSite=None; Secure\n"
)

DEAD_404_HEADERS = "HTTP/1.1 404 Not Found\ncontent-type: text/html\n"
DEAD_410_HEADERS = "HTTP/1.1 410 Gone\ncontent-type: text/html\n"

OK_HEADERS = "HTTP/1.1 200 OK\ncontent-type: application/pdf\n"

REDIRECT_THEN_BLOCK_HEADERS = (
    "HTTP/1.1 301 Moved Permanently\n"
    "location: https://ai.gov.ae/final.pdf\n"
    "\n"
    "HTTP/2 403 \n"
    "cf-ray: a1b9e294cfd4e293-BEG\n"
    "content-type: text/html\n"
)


def test_classify_valid_pdf_is_ok() -> None:
    result = classify_response(REAL_PDF_BODY, OK_HEADERS)
    assert result.outcome == AcquisitionOutcome.ok
    assert result.http_status == 200


def test_classify_cloudflare_block_via_cf_ray_header() -> None:
    result = classify_response(CLOUDFLARE_BLOCK_BODY, CLOUDFLARE_BLOCK_HEADERS)
    assert result.outcome == AcquisitionOutcome.blocked
    assert result.http_status == 403


def test_classify_cloudflare_block_via_cookie_only() -> None:
    # 200 + __cf_bm cookie + не-PDF тело -> тоже блок (сама Cloudflare-кука уже отпечаток,
    # даже без cf-ray в этом конкретном ответе).
    result = classify_response(b"<html>checking your browser</html>", CF_COOKIE_ONLY_HEADERS)
    assert result.outcome == AcquisitionOutcome.blocked


def test_classify_challenge_body_marker() -> None:
    body = b"<html><h1>Attention Required! | Cloudflare</h1></html>"
    result = classify_response(body, CLOUDFLARE_BLOCK_HEADERS)
    assert result.outcome == AcquisitionOutcome.blocked
    assert "WAF" in result.reason or "challenge" in result.reason


def test_classify_dead_404() -> None:
    result = classify_response(b"<html>not found</html>", DEAD_404_HEADERS)
    assert result.outcome == AcquisitionOutcome.dead
    assert result.http_status == 404


def test_classify_dead_410() -> None:
    result = classify_response(b"", DEAD_410_HEADERS)
    assert result.outcome == AcquisitionOutcome.dead
    assert result.http_status == 410


def test_classify_small_unexpected_body_is_blocked() -> None:
    # 200, не PDF, без явных challenge-маркеров, но подозрительно маленький -> блок, не "ok".
    result = classify_response(b"tiny", OK_HEADERS)
    assert result.outcome == AcquisitionOutcome.blocked


def test_classify_redirect_keeps_only_final_hop_status_and_headers() -> None:
    # Финальный статус (403) должен победить промежуточный 301; cf-ray виден только
    # в финальном блоке заголовков -> детектор не должен потеряться в редиректе.
    result = classify_response(CLOUDFLARE_BLOCK_BODY, REDIRECT_THEN_BLOCK_HEADERS)
    assert result.http_status == 403
    assert result.outcome == AcquisitionOutcome.blocked


def test_fetch_and_classify_omits_dash_f_and_captures_headers(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """curl вызывается БЕЗ -f (иначе тело/заголовки блока/dead-ответа теряются —
    см. решение №1 спека) и с -D для заголовков.
    """
    captured_cmd: list[str] = []

    def fake_run(cmd: list[str], check: bool) -> None:  # noqa: FBT001 — сигнатура subprocess.run
        captured_cmd.extend(cmd)
        d_index = cmd.index("-D")
        o_index = cmd.index("-o")
        Path(cmd[d_index + 1]).write_text(CLOUDFLARE_BLOCK_HEADERS, encoding="utf-8")
        Path(cmd[o_index + 1]).write_bytes(CLOUDFLARE_BLOCK_BODY)

    monkeypatch.setattr("acquisition.subprocess.run", fake_run)
    dest = tmp_path / "doc.pdf"
    result = fetch_and_classify("https://ai.gov.ae/doc.pdf", dest, user_agent="test-ua")

    assert "-f" not in captured_cmd
    assert "-D" in captured_cmd
    assert result.outcome == AcquisitionOutcome.blocked
    assert result.http_status == 403


# --- маршрутизация лестницы (§2/§4/§9 спека) ---

NORMAL = schema.Sensitivity.normal
CONFIDENTIAL = schema.Sensitivity.confidential
DIRECT = schema.AcquisitionMethod.direct
OFFICIAL_ALT = schema.AcquisitionMethod.official_alt
MANUAL = schema.AcquisitionMethod.manual
ARCHIVE = schema.AcquisitionMethod.archive


@pytest.mark.parametrize(
    "outcome,rung,has_alt,sensitivity,expected",
    [
        (AcquisitionOutcome.ok, DIRECT, True, NORMAL, None),
        (AcquisitionOutcome.blocked, DIRECT, True, NORMAL, OFFICIAL_ALT),
        (AcquisitionOutcome.blocked, DIRECT, False, NORMAL, MANUAL),
        (AcquisitionOutcome.blocked, OFFICIAL_ALT, True, NORMAL, MANUAL),
        (AcquisitionOutcome.dead, DIRECT, False, NORMAL, ARCHIVE),
        (AcquisitionOutcome.dead, OFFICIAL_ALT, True, NORMAL, ARCHIVE),
        (AcquisitionOutcome.dead, DIRECT, False, CONFIDENTIAL, MANUAL),
    ],
)
def test_next_rung_transition_table(
    outcome: AcquisitionOutcome,
    rung: schema.AcquisitionMethod,
    has_alt: bool,
    sensitivity: schema.Sensitivity,
    expected: schema.AcquisitionMethod | None,
) -> None:
    assert next_rung(outcome, rung, has_official_alt=has_alt, sensitivity=sensitivity) == expected


def _scripted_fetch(responses: list[ClassifiedResponse]) -> Any:
    """Подмена ``fetch_and_classify``: выдаёт по одному ответу из списка на каждый вызов."""
    calls = {"n": 0}

    def fake(url: str, dest: Path, *, user_agent: str, timeout: int = 30) -> ClassifiedResponse:
        response = responses[calls["n"]]
        calls["n"] += 1
        return response

    return fake


def _rec(**over: Any) -> schema.SourceRecord:
    data = valid_record()
    data.update(over)
    return schema.SourceRecord.model_validate(data)


def test_run_ladder_direct_ok(tmp_path: Path, monkeypatch: Any) -> None:
    ok = ClassifiedResponse(AcquisitionOutcome.ok, 200, "valid PDF")
    monkeypatch.setattr("acquisition.fetch_and_classify", _scripted_fetch([ok]))
    result = run_ladder(_rec(), tmp_path / "doc.pdf", user_agent="test-ua")
    assert result.method == DIRECT
    assert result.fidelity == schema.Fidelity.live


def test_run_ladder_blocked_no_alt_raises_blocked(tmp_path: Path, monkeypatch: Any) -> None:
    blocked = ClassifiedResponse(AcquisitionOutcome.blocked, 403, "WAF challenge signature detected")
    monkeypatch.setattr("acquisition.fetch_and_classify", _scripted_fetch([blocked]))
    with pytest.raises(AcquisitionBlocked):
        run_ladder(_rec(), tmp_path / "doc.pdf", user_agent="test-ua")  # official_alt_url не задан


def test_run_ladder_blocked_falls_through_to_official_alt(tmp_path: Path, monkeypatch: Any) -> None:
    blocked = ClassifiedResponse(AcquisitionOutcome.blocked, 403, "WAF challenge signature detected")
    ok = ClassifiedResponse(AcquisitionOutcome.ok, 200, "valid PDF")
    monkeypatch.setattr("acquisition.fetch_and_classify", _scripted_fetch([blocked, ok]))
    rec = _rec(official_alt_url="https://example.org/alt.pdf")
    result = run_ladder(rec, tmp_path / "doc.pdf", user_agent="test-ua")
    assert result.method == OFFICIAL_ALT
    assert result.fidelity == schema.Fidelity.rehost


def test_run_ladder_blocked_on_official_alt_raises_blocked(tmp_path: Path, monkeypatch: Any) -> None:
    blocked = ClassifiedResponse(AcquisitionOutcome.blocked, 403, "WAF challenge signature detected")
    monkeypatch.setattr("acquisition.fetch_and_classify", _scripted_fetch([blocked, blocked]))
    rec = _rec(official_alt_url="https://example.org/alt.pdf")
    with pytest.raises(AcquisitionBlocked):
        run_ladder(rec, tmp_path / "doc.pdf", user_agent="test-ua")


def test_run_ladder_dead_normal_raises_dead(tmp_path: Path, monkeypatch: Any) -> None:
    dead = ClassifiedResponse(AcquisitionOutcome.dead, 404, "HTTP 404")
    monkeypatch.setattr("acquisition.fetch_and_classify", _scripted_fetch([dead]))
    with pytest.raises(AcquisitionDead):
        run_ladder(_rec(), tmp_path / "doc.pdf", user_agent="test-ua")  # sensitivity=normal (дефолт)


def test_run_ladder_dead_confidential_raises_blocked_not_dead(tmp_path: Path, monkeypatch: Any) -> None:
    dead = ClassifiedResponse(AcquisitionOutcome.dead, 404, "HTTP 404")
    monkeypatch.setattr("acquisition.fetch_and_classify", _scripted_fetch([dead]))
    rec = _rec(sensitivity="confidential")
    with pytest.raises(AcquisitionBlocked) as exc_info:
        run_ladder(rec, tmp_path / "doc.pdf", user_agent="test-ua")
    assert "confidential" in str(exc_info.value)


# --- round-trip-safe запись в sources.yaml ---

REALISTIC_SOURCES_YAML = """\
# Реестр первоисточников G2AI-корпуса — единый источник истины.
# Схема и валидация:      pipeline/scripts/schema.py + validate_sources.py

- id: sg-imda-mgf-agentic-2026
  title: "Model AI Governance Framework for Agentic AI"
  dates:
    published: 2026-05-20
  summary: >-
    Модельный фреймворк управления агентным ИИ: оценка и ограничение рисков на
    входе, значимая человеческая подотчётность.
  status: verified
"""


def test_persist_acquisition_state_preserves_comments_and_formatting(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(REALISTIC_SOURCES_YAML, encoding="utf-8")

    changed = persist_acquisition_state(
        sources_path, "sg-imda-mgf-agentic-2026",
        acquisition_method=schema.AcquisitionMethod.direct,
        fidelity=schema.Fidelity.live,
        checked=dt.date(2026, 7, 15),
    )

    result = sources_path.read_text(encoding="utf-8")
    assert "# Реестр первоисточников" in result  # комментарий шапки не потерян
    assert "summary: >-" in result  # folded scalar не переформатирован
    assert "acquisition_method: direct" in result
    assert "acquisition_checked: 2026-07-15" in result
    assert changed == {
        "acquisition_method": (None, "direct"),
        "acquisition_checked": (None, dt.date(2026, 7, 15)),
        "fidelity": (None, "live"),
    }


def test_persist_acquisition_state_noop_when_unchanged(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(REALISTIC_SOURCES_YAML, encoding="utf-8")
    kwargs = dict(
        acquisition_method=schema.AcquisitionMethod.direct,
        fidelity=schema.Fidelity.live,
        checked=dt.date(2026, 7, 15),
    )
    persist_acquisition_state(sources_path, "sg-imda-mgf-agentic-2026", **kwargs)
    after_first = sources_path.read_text(encoding="utf-8")

    changed = persist_acquisition_state(sources_path, "sg-imda-mgf-agentic-2026", **kwargs)
    assert changed == {}
    assert sources_path.read_text(encoding="utf-8") == after_first  # второй прогон не трогает файл


def test_persist_acquisition_state_unknown_id_raises(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(REALISTIC_SOURCES_YAML, encoding="utf-8")
    with pytest.raises(ValueError):
        persist_acquisition_state(
            sources_path, "nonexistent-id",
            acquisition_method=schema.AcquisitionMethod.direct,
            fidelity=schema.Fidelity.live,
            checked=dt.date(2026, 7, 15),
        )


def test_persist_acquisition_state_atomic_on_dump_failure(tmp_path: Path, monkeypatch: Any) -> None:
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(REALISTIC_SOURCES_YAML, encoding="utf-8")
    original = sources_path.read_text(encoding="utf-8")

    import acquisition as acq

    def raising_dump(self: Any, data: Any, stream: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(acq.YAML, "dump", raising_dump)

    with pytest.raises(RuntimeError):
        persist_acquisition_state(
            sources_path, "sg-imda-mgf-agentic-2026",
            acquisition_method=schema.AcquisitionMethod.direct,
            fidelity=schema.Fidelity.live,
            checked=dt.date(2026, 7, 15),
        )
    # Главная гарантия атомарности: исходный файл не тронут при сбое сериализации
    # (возможный осиротевший .tmp — тот же паттерн, что у _do_convert/_do_frontmatter).
    assert sources_path.read_text(encoding="utf-8") == original

"""Тесты convert/cloud_ocr.py (spec convert-cloud-tier §2): батчирование,
чекпоинт, outline-контекст, кэш-заголовок. Сеть/pdfplumber-рендер мокаются —
чистая логика планирования и I/O сайдкаров тестируется без реального PDF."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from convert.cloud_ocr import (
    OCR_BATCH_MAX_MB,
    OCR_BATCH_PAGES,
    _lang_name,
    _load_parts,
    _ordered_texts,
    _parts_path,
    _plan_batches,
    _save_parts,
    convert_scan,
)
from core.fsio import sha256_file


# --- _lang_name ---


def test_lang_name_known_code() -> None:
    assert _lang_name("cnr") == "Montenegrin (Latin script)"


def test_lang_name_unknown_code_warns_and_falls_back(caplog: Any) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        assert _lang_name("xx") == "the source language"
    assert "xx" in caplog.text


def test_lang_name_none_defaults_to_english() -> None:
    assert _lang_name(None) == "English"


# --- _plan_batches: потолок по страницам И по байтам ---


def test_plan_batches_splits_on_page_count() -> None:
    rendered = [(f"uri{i}", 1000) for i in range(OCR_BATCH_PAGES + 5)]
    batches = _plan_batches(rendered)
    assert len(batches) == 2
    assert batches[0][:2] == (1, OCR_BATCH_PAGES)
    assert batches[1][:2] == (OCR_BATCH_PAGES + 1, OCR_BATCH_PAGES + 5)


def test_plan_batches_splits_on_byte_ceiling() -> None:
    max_bytes = OCR_BATCH_MAX_MB * 1024 * 1024
    heavy = max_bytes // 2  # 2 страницы влезают ровно, 3-я уже не влезает
    rendered = [("u1", heavy), ("u2", heavy), ("u3", heavy)]
    batches = _plan_batches(rendered)
    assert len(batches) == 2
    assert batches[0][:2] == (1, 2)
    assert batches[1][:2] == (3, 3)


def test_plan_batches_single_oversized_page_becomes_its_own_batch() -> None:
    max_bytes = OCR_BATCH_MAX_MB * 1024 * 1024
    rendered = [("u1", max_bytes * 2)]  # патологически огромная одна страница
    batches = _plan_batches(rendered)
    assert len(batches) == 1
    assert batches[0][:2] == (1, 1)


def test_plan_batches_small_doc_single_batch() -> None:
    rendered = [(f"uri{i}", 1000) for i in range(5)]
    batches = _plan_batches(rendered)
    assert len(batches) == 1
    assert batches[0][:2] == (1, 5)
    assert batches[0][2] == [f"uri{i}" for i in range(5)]


# --- _ordered_texts: числовая, не лексикографическая сортировка ---


def test_ordered_texts_numeric_sort_beyond_nine_batches() -> None:
    parts = {"100-118": "c", "2-19": "b", "1-1": "a"}
    assert _ordered_texts(parts) == ["a", "b", "c"]


# --- сайдкар .cloudocr.parts.yaml: заголовок/инвалидация ---


def test_load_parts_missing_file_returns_empty(tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    assert _load_parts(raw, model="m", raw_sha256="a" * 64) == {}


def test_save_then_load_parts_roundtrip(tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    _save_parts(raw, model="m", raw_sha256="a" * 64, parts={"1-5": "text"})
    assert _load_parts(raw, model="m", raw_sha256="a" * 64) == {"1-5": "text"}


def test_load_parts_header_mismatch_discards_everything(tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    _save_parts(raw, model="model-a", raw_sha256="a" * 64, parts={"1-5": "text"})
    assert _load_parts(raw, model="model-b", raw_sha256="a" * 64) == {}  # другая модель
    assert _load_parts(raw, model="model-a", raw_sha256="b" * 64) == {}  # другой sha


def test_load_parts_batch_ceiling_recalibration_discards_checkpoint(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Потолки батчей — часть header (находка приёмки чекпоинта 1): чекпоинт, снятый
    при иной нарезке (рекалибровка §2.1 после обрыва), несовместим с новым планом —
    без инвалидации финальная склейка _ordered_texts тихо задублировала бы страницы."""
    raw = tmp_path / "raw.pdf"
    _save_parts(raw, model="m", raw_sha256="a" * 64, parts={"1-20": "text"})
    monkeypatch.setattr("convert.cloud_ocr.OCR_BATCH_PAGES", 18)
    assert _load_parts(raw, model="m", raw_sha256="a" * 64) == {}


def test_parts_path_is_dot_prefixed(tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    assert _parts_path(raw).name == ".cloudocr.parts.yaml"


# --- convert_scan: сквозной сценарий с мокнутым рендером/сетью ---


class _FakePage:
    pass


class _FakePdf:
    def __init__(self, n_pages: int) -> None:
        self.pages = [_FakePage() for _ in range(n_pages)]

    def __enter__(self) -> "_FakePdf":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _patch_pdf(monkeypatch: Any, n_pages: int) -> None:
    monkeypatch.setattr("convert.cloud_ocr.pdfplumber.open", lambda path: _FakePdf(n_pages))


def _patch_render(monkeypatch: Any, sizes: list[int]) -> None:
    it = iter(sizes)

    def fake_render(page: Any) -> tuple[str, int]:
        size = next(it)
        return f"uri-{size}", size

    monkeypatch.setattr("convert.cloud_ocr._render_page", fake_render)


def _patch_key(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_convert_scan_missing_key_raises(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"pdf")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        convert_scan(raw, "en", model="m")


def test_convert_scan_single_batch_calls_once_and_finalizes(monkeypatch: Any, tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"pdf bytes for sha")
    _patch_key(monkeypatch)
    _patch_pdf(monkeypatch, 3)
    _patch_render(monkeypatch, [1000, 1000, 1000])

    calls: list[dict[str, Any]] = []

    def fake_chat(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        calls.append(payload)
        return {"choices": [{"message": {"content": "# Title\n\nBody text."}}]}

    monkeypatch.setattr("convert.cloud_ocr.openrouter.chat_request", fake_chat)

    text = convert_scan(raw, "en", model="test-model")
    assert text == "# Title\n\nBody text."
    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    images = [p for p in calls[0]["messages"][0]["content"] if p["type"] == "image_url"]
    assert len(images) == 3
    assert not _parts_path(raw).exists()  # финализация удалила чекпоинт


def test_convert_scan_batch_two_gets_outline_context(monkeypatch: Any, tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"pdf bytes")
    _patch_key(monkeypatch)
    n_pages = OCR_BATCH_PAGES + 2
    _patch_pdf(monkeypatch, n_pages)
    _patch_render(monkeypatch, [1000] * n_pages)

    prompts: list[str] = []

    def fake_chat(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        prompt_text = payload["messages"][0]["content"][0]["text"]
        prompts.append(prompt_text)
        if len(prompts) == 1:
            return {"choices": [{"message": {"content": "# Chapter One\n\nBody."}}]}
        return {"choices": [{"message": {"content": "## Chapter Two\n\nMore body."}}]}

    monkeypatch.setattr("convert.cloud_ocr.openrouter.chat_request", fake_chat)

    convert_scan(raw, "en", model="m")
    assert len(prompts) == 2
    assert "Continuation of the same document" not in prompts[0]
    assert "Continuation of the same document" in prompts[1]
    assert "# Chapter One" in prompts[1]


def test_convert_scan_checkpoints_after_each_batch_and_resumes(monkeypatch: Any, tmp_path: Path) -> None:
    """kill-9-паттерн PR #16: отказ батча 2 после успеха батча 1 оставляет
    чекпоинт с батчем 1; повторный вызов добирает ТОЛЬКО батч 2 (сеть не
    трогает уже добытое)."""
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"pdf bytes")
    _patch_key(monkeypatch)
    n_pages = OCR_BATCH_PAGES + 3
    _patch_pdf(monkeypatch, n_pages)
    _patch_render(monkeypatch, [1000] * n_pages)

    calls = {"n": 0}

    def failing_on_second(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("сеть оборвалась")
        return {"choices": [{"message": {"content": f"# Batch {calls['n']}"}}]}

    monkeypatch.setattr("convert.cloud_ocr.openrouter.chat_request", failing_on_second)

    with pytest.raises(RuntimeError, match="сеть оборвалась"):
        convert_scan(raw, "en", model="m")

    assert _parts_path(raw).exists()
    saved = yaml.safe_load(_parts_path(raw).read_text(encoding="utf-8"))
    assert list(saved["parts"].keys()) == [f"1-{OCR_BATCH_PAGES}"]

    # рестарт: batch 1 из чекпоинта, сеть зовётся ТОЛЬКО для batch 2
    def succeed_now(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        calls["n"] += 1
        return {"choices": [{"message": {"content": f"# Batch {calls['n']}"}}]}

    monkeypatch.setattr("convert.cloud_ocr.openrouter.chat_request", succeed_now)
    _patch_render(monkeypatch, [1000] * n_pages)  # рестарт снова рендерит все страницы
    text = convert_scan(raw, "en", model="m")
    assert calls["n"] == 3  # 2 (до обрыва) + 1 (добор) — не 4
    assert not _parts_path(raw).exists()
    assert "# Batch 1" in text and "# Batch 3" in text


def test_convert_scan_header_change_discards_partial_checkpoint(monkeypatch: Any, tmp_path: Path) -> None:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"pdf bytes")
    _patch_key(monkeypatch)
    _save_parts(raw, model="old-model", raw_sha256=sha256_file(raw), parts={"1-1": "stale"})
    _patch_pdf(monkeypatch, 1)
    _patch_render(monkeypatch, [1000])

    calls: list[dict[str, Any]] = []

    def fake_chat(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        calls.append(payload)
        return {"choices": [{"message": {"content": "# Fresh"}}]}

    monkeypatch.setattr("convert.cloud_ocr.openrouter.chat_request", fake_chat)

    text = convert_scan(raw, "en", model="new-model")
    assert text == "# Fresh"
    assert len(calls) == 1  # старый чекпоинт (другая модель) не переиспользован

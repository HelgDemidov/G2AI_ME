"""Тесты оркестратора: реконсиляция стадий, синк frontmatter, dry-run, изоляция отказов.

Всё CI-safe — без сети (download), pdfplumber (convert) и модели (index).
Раскладка — папка-документ (corpus-layout-v2): пути выводятся из <root>/<track>/<entity>/<id>/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from acquire import acquisition
from convert import cloud_ocr, converters, figures_vlm
from index import corpus_index
from core import schema
from acquire.acquisition import AcquisitionOutcome, ClassifiedResponse
from core import fsio
from run_pipeline import (
    Stage,
    _adopt_untracked_raw,
    _compose_md,
    _do_convert,
    _do_download,
    _do_figures,
    _do_frontmatter,
    _needs_index_rebuild,
    _read_index_fingerprint,
    _sha256,
    needed_stages,
    process_docs,
    rebuild_index,
)
from core.schema import SourceRecord, render_frontmatter
from tests.support import valid_record, write_doc


def make(**over: Any) -> SourceRecord:
    data = valid_record()
    data.update(over)
    return SourceRecord.model_validate(data)


def _place(
    rec: SourceRecord,
    root: Path,
    *,
    raw: bytes | None = None,
    md: str | None = None,
    state: dict[str, Any] | None = None,
) -> None:
    """Разложить raw.pdf/doc.md/.state.yaml в выведенную папку-документ."""
    import yaml as _yaml

    d = schema.doc_dir(rec, root)
    d.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (d / "raw.pdf").write_bytes(raw)
    if md is not None:
        (d / "doc.md").write_text(md, encoding="utf-8")
    if state is not None:
        (d / ".state.yaml").write_text(_yaml.safe_dump(state, allow_unicode=True), encoding="utf-8")


def test_needs_all_when_nothing_exists(tmp_path: Path) -> None:
    assert needed_stages(make(), tmp_path) == [Stage.download, Stage.convert, Stage.frontmatter]


def test_needs_convert_when_md_missing(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")  # raw есть, sha неизвестен -> download не нужен
    assert needed_stages(rec, tmp_path) == [Stage.convert, Stage.frontmatter]


def test_sha256_mismatch_triggers_download(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", state={"sha256": "0" * 64})  # sha не совпадёт с содержимым
    assert Stage.download in needed_stages(rec, tmp_path)


def _stamp_converter_state(rec: SourceRecord, root: Path) -> None:
    """Дописать converter_name/version (текущий реестр) в УЖЕ существующее .state.yaml,
    сохранив прочие поля (sha256/raw_size/…) — для тестов, где состояние уже подготовлено
    _adopt_untracked_raw и фокус не на версионировании конвертера."""
    state_path = schema.state_file(rec, root)
    state = schema.load_state(state_path)
    conv = converters._CONVERTERS["pdf"]
    state.converter_name, state.converter_version = conv.name, conv.version
    schema.save_state(state_path, state)


def _current_converter_state() -> dict[str, str]:
    """Состояние с converter_name/version, ТЕКУЩИМИ у реального реестра pdf — изолирует
    тесты, не посвящённые версионированию конвертера, от converter_changed в needed_stages
    (иначе легаси-состояние без этих полей само по себе требовало бы Stage.convert, см.
    отдельные test_needed_stages_*_converter_* ниже, которые как раз это и проверяют)."""
    conv = converters._CONVERTERS["pdf"]
    return {"converter_name": conv.name, "converter_version": conv.version}


def test_up_to_date_no_stages(tmp_path: Path) -> None:
    rec = make()
    _place(  # синхронный frontmatter, sha неизвестен, конвертер актуален
        rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""), state=_current_converter_state()
    )
    assert needed_stages(rec, tmp_path) == []


def test_force_redoes_all(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""), state=_current_converter_state())
    assert needed_stages(rec, tmp_path, force=True) == [Stage.download, Stage.convert, Stage.frontmatter]


def test_frontmatter_drift_detected(tmp_path: Path) -> None:
    rec = make()
    _place(
        rec, tmp_path, raw=b"pdf", md="---\nid: stale-old\n---\n\nBody.\n",
        state=_current_converter_state(),
    )
    assert needed_stages(rec, tmp_path) == [Stage.frontmatter]


def test_do_frontmatter_syncs_and_idempotent(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, md="Body only, no frontmatter.\n")
    assert _do_frontmatter(rec, tmp_path) is True
    content = schema.md_file(rec, tmp_path).read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "Body only" in content
    assert _do_frontmatter(rec, tmp_path) is False  # второй раз — уже синхронно


def test_do_frontmatter_uses_fsio_atomic_write(tmp_path: Path, monkeypatch: Any) -> None:
    """Мигрировано на fsio.atomic_write_text — единая staging-политика (dot-файл,
    не отдельный .tmp-суффикс)."""
    rec = make()
    _place(rec, tmp_path, md="Body only, no frontmatter.\n")
    calls: list[Path] = []
    real = fsio.atomic_write_text

    def spy(target: Path, text: str) -> None:
        calls.append(target)
        real(target, text)

    monkeypatch.setattr("run_pipeline.fsio.atomic_write_text", spy)
    _do_frontmatter(rec, tmp_path)
    assert calls == [schema.md_file(rec, tmp_path)]


def _fake_converter(fn: Any, *, name: str = "pdf", version: str = "test") -> Any:
    """Подменить реестр реестра одним fake-конвертером: fn(raw, out, language) -> None.
    Адаптер поглощает ``record=`` (ConvertFn Protocol, spec convert-cloud-tier) —
    большинство тестов не о record, писать его в каждый fake было бы шумом."""
    def adapter(raw: Path, out: Path, language: str | None, *, record: Any = None) -> None:
        fn(raw, out, language)
    return converters.Converter(name, version, adapter)


def test_do_convert_staging_uses_dot_prefix_on_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """_do_convert мигрирован на fsio.staging_path — при отказе конвертации огрызок
    остаётся под dot-префиксным именем (совместимым с fsio.cleanup_staging), не
    старым '.tmp'-суффиксом."""
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        dst.write_bytes(b"")  # пустой вывод -> _do_convert бросит RuntimeError

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))

    with pytest.raises(RuntimeError, match="пустой файл"):
        _do_convert(rec, tmp_path)

    staging_files = list(schema.doc_dir(rec, tmp_path).glob(".*.part"))
    assert [p.name for p in staging_files] == [".doc.md.part"]


def test_do_convert_success_leaves_no_staging(tmp_path: Path, monkeypatch: Any) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        dst.write_text("converted body", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))
    _do_convert(rec, tmp_path)

    assert schema.md_file(rec, tmp_path).read_text(encoding="utf-8") == "converted body"
    assert list(schema.doc_dir(rec, tmp_path).glob(".*.part")) == []


def test_do_convert_writes_converter_name_and_version_to_state(tmp_path: Path, monkeypatch: Any) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        dst.write_text("body", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert, version="7"))
    _do_convert(rec, tmp_path)

    state = schema.load_state(schema.state_file(rec, tmp_path))
    assert (state.converter_name, state.converter_version) == ("pdf", "7")


def test_do_convert_refreshes_sha256_when_converter_mutates_raw_in_place(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """convert-ocr (2026-07-17): OCR-нормализация мутирует raw.pdf IN-PLACE (один файл,
    без сайдкара .ocr.pdf) — _do_convert обязан пересчитать sha256/размер/mtime ПОСЛЕ
    конвертации, иначе следующий stat-guard (needed_stages) увидит расхождение со
    старой записью и решит, что raw «повреждён», затребовав передобычу."""
    rec = make()
    _place(rec, tmp_path, raw=b"original scanned bytes")

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        raw.write_bytes(b"ocr-normalized bytes, different content")  # имитирует _ocr_normalize
        dst.write_text("body", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))
    _do_convert(rec, tmp_path)

    raw = schema.raw_file(rec, tmp_path)
    assert raw is not None
    state = schema.load_state(schema.state_file(rec, tmp_path))
    assert state.sha256 == _sha256(raw)
    assert state.raw_size == raw.stat().st_size
    assert state.raw_mtime_ns == raw.stat().st_mtime_ns
    # реконсиляция после мутации не должна ложно требовать передобычу
    assert Stage.download not in needed_stages(rec, tmp_path)


def test_do_convert_passes_record_language_to_converter(tmp_path: Path, monkeypatch: Any) -> None:
    rec = make(language="et")
    _place(rec, tmp_path, raw=b"pdf")
    seen: list[str | None] = []

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        seen.append(language)
        dst.write_text("body", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))
    _do_convert(rec, tmp_path)

    assert seen == ["et"]


def test_do_convert_writes_lint_defects_to_state(tmp_path: Path, monkeypatch: Any) -> None:
    """C1 (spec convert-hardening): дефектный вывод конвертера (без единого
    заголовка) фиксируется в .state.yaml — машиночитаемо для worksheet'а
    батч-триажа (spec discovery-manual), НЕ роняет конвертацию."""
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        dst.write_text("Just plain prose, no headings at all.", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))
    _do_convert(rec, tmp_path)

    state = schema.load_state(schema.state_file(rec, tmp_path))
    assert "no-headings" in state.lint_defects


def test_do_convert_clean_output_has_no_lint_defects(tmp_path: Path, monkeypatch: Any) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        dst.write_text("# Title\n\nClean body text.", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))
    _do_convert(rec, tmp_path)

    state = schema.load_state(schema.state_file(rec, tmp_path))
    assert state.lint_defects == []


def test_do_convert_unsupported_format_raises(tmp_path: Path) -> None:
    rec = make()
    d = schema.doc_dir(rec, tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw.xyz").write_bytes(b"data")
    with pytest.raises(converters.UnsupportedFormat):
        _do_convert(rec, tmp_path)


# --- needed_stages: converter_changed (реконсиляция реконверсии) ---


def test_needed_stages_replans_convert_when_converter_version_changed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    rec = make()
    _place(
        rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""),
        state={"converter_name": "pdf", "converter_version": "0"},
    )
    monkeypatch.setitem(
        converters._CONVERTERS, "pdf", _fake_converter(lambda raw, out, lang: None, version="1")
    )
    assert needed_stages(rec, tmp_path) == [Stage.convert, Stage.frontmatter]


def test_needed_stages_no_convert_when_converter_version_matches(
    tmp_path: Path, monkeypatch: Any
) -> None:
    rec = make()
    _place(
        rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""),
        state={"converter_name": "pdf", "converter_version": "1"},
    )
    monkeypatch.setitem(
        converters._CONVERTERS, "pdf", _fake_converter(lambda raw, out, lang: None, version="1")
    )
    assert needed_stages(rec, tmp_path) == []


def test_needed_stages_legacy_state_without_converter_fields_replans_convert(tmp_path: Path) -> None:
    """.state.yaml без converter_name/version (легаси) -> mismatch с None -> одноразовая
    реконверсия на первом прогоне после мерджа спека (см. spec §3)."""
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))  # без state вовсе
    assert needed_stages(rec, tmp_path) == [Stage.convert, Stage.frontmatter]


def test_dry_run_no_side_effects(tmp_path: Path) -> None:
    rec = make()
    results = process_docs(
        [rec], tmp_path,
        force=False, dry_run=True, no_download=False, pause=0,
    )
    assert results[0].done == [Stage.download, Stage.convert, Stage.frontmatter]
    assert not schema.doc_dir(rec, tmp_path).exists()  # ничего не создано


def test_dry_run_does_not_adopt_untracked_raw(tmp_path: Path) -> None:
    """--dry-run обязан быть no-op: усыновление (запись .state.yaml с посчитанным
    sha) не должно происходить, даже если есть неотслеженный raw."""
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))  # актуален, sha не отслежен
    process_docs([rec], tmp_path, force=False, dry_run=True, no_download=False, pause=0)
    assert not schema.state_file(rec, tmp_path).exists()


def test_failure_isolation(tmp_path: Path) -> None:
    a = make(id="a-doc-2026", entity_id="aa")  # нужен download
    b = make(id="b-doc-2026", entity_id="bb")
    _place(  # b актуален (вкл. версию конвертера — иначе понадобился бы реальный _do_convert)
        b, tmp_path, raw=b"pdf", md=_compose_md(b, ""), state=_current_converter_state()
    )

    results = process_docs(
        [a, b], tmp_path,
        force=False, dry_run=False, no_download=True, pause=0,  # a падает на download
    )
    ra = next(r for r in results if r.doc_id == "a-doc-2026")
    rb = next(r for r in results if r.doc_id == "b-doc-2026")
    assert ra.error is not None and "download" in ra.error  # a упал, но не оборвал батч
    assert rb.up_to_date is True


def test_do_download_writes_state(tmp_path: Path, monkeypatch: Any) -> None:
    """Сквозная проводка: _do_download -> run_ladder -> запись .state.yaml (не meta.yaml)."""
    rec = make()
    ok = ClassifiedResponse(AcquisitionOutcome.ok, 200, "valid PDF")

    def fake_fetch(url: str, dest: Path, *, user_agent: str, timeout: int = 30, **kw: Any) -> ClassifiedResponse:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4 fake content")
        return ok

    monkeypatch.setattr("acquire.acquisition.fetch_and_classify", fake_fetch)

    _do_download(rec, tmp_path, pause=0)

    assert (schema.doc_dir(rec, tmp_path) / "raw.pdf").read_bytes() == b"%PDF-1.4 fake content"
    st = schema.load_state(schema.state_file(rec, tmp_path))
    assert st.acquisition_method is not None and st.acquisition_method.value == "direct"
    assert st.fidelity is not None and st.fidelity.value == "live"
    assert st.sha256 is not None


def test_do_download_html_record_targets_raw_html(tmp_path: Path, monkeypatch: Any) -> None:
    """source_format=html -> цель скачивания raw.html, не raw.pdf (ext-маршрутизация)."""
    rec = make(source_format="html")
    ok = ClassifiedResponse(AcquisitionOutcome.ok, 200, "valid HTML")

    def fake_fetch(url: str, dest: Path, *, user_agent: str, timeout: int = 30, **kw: Any) -> ClassifiedResponse:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"<html>fake content</html>")
        return ok

    monkeypatch.setattr("acquire.acquisition.fetch_and_classify", fake_fetch)

    _do_download(rec, tmp_path, pause=0)

    assert (schema.doc_dir(rec, tmp_path) / "raw.html").read_bytes() == b"<html>fake content</html>"
    assert schema.raw_file(rec, tmp_path) == schema.doc_dir(rec, tmp_path) / "raw.html"


def test_do_download_cleans_staging_on_batch_block(tmp_path: Path, monkeypatch: Any) -> None:
    """При AcquisitionBlocked в батче (interactive=False) staging убирается —
    challenge-тело (реалистично: curl -o пишет его в part ДО классификации) не
    остаётся под именем, которое schema.raw_file мог бы усыновить как оригинал."""
    rec = make()

    def fake_run_ladder(rec_: SourceRecord, dest: Path, *, user_agent: str) -> Any:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"<html>Attention Required</html>")  # как реальный curl -o на challenge
        raise acquisition.AcquisitionBlocked(rec_.source_url, "direct blocked (WAF challenge)")

    monkeypatch.setattr("acquire.acquisition.run_ladder", fake_run_ladder)

    with pytest.raises(acquisition.AcquisitionBlocked):
        _do_download(rec, tmp_path, pause=0, interactive=False)

    doc_dir = schema.doc_dir(rec, tmp_path)
    assert list(doc_dir.glob(".*.part")) == []  # staging убран в finally
    assert schema.raw_file(rec, tmp_path) is None  # challenge не усыновлён как оригинал


def test_do_download_replaces_prior_raw_of_different_ext(tmp_path: Path, monkeypatch: Any) -> None:
    """Смена канала/формата: новый raw.pdf публикуется, прежний raw.html удаляется (single-raw)."""
    rec = make()
    doc_dir = schema.doc_dir(rec, tmp_path)
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "raw.html").write_bytes(b"<html>old html original</html>")

    ok = ClassifiedResponse(AcquisitionOutcome.ok, 200, "valid PDF")

    def fake_fetch(url: str, dest: Path, *, user_agent: str, timeout: int = 30, **kw: Any) -> ClassifiedResponse:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4 fake content")
        return ok

    monkeypatch.setattr("acquire.acquisition.fetch_and_classify", fake_fetch)

    _do_download(rec, tmp_path, pause=0)

    assert schema.raw_file(rec, tmp_path) == doc_dir / "raw.pdf"  # ровно один raw.* — новый
    assert not (doc_dir / "raw.html").exists()


def test_process_docs_cleans_stale_staging_before_planning(tmp_path: Path) -> None:
    """Реконсиляционная чистка (§1d): останки упавшего прогона убираются сами."""
    rec = make()
    _place(  # актуален (вкл. версию конвертера)
        rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""), state=_current_converter_state()
    )
    doc_dir = schema.doc_dir(rec, tmp_path)
    (doc_dir / ".raw.pdf.part").write_bytes(b"stale leftover")

    results = process_docs(
        [rec], tmp_path, force=False, dry_run=False, no_download=False, pause=0
    )

    assert not (doc_dir / ".raw.pdf.part").exists()
    assert results[0].up_to_date is True


def test_planning_failure_isolated_from_batch(tmp_path: Path) -> None:
    """Папка с двумя raw.* ломает планирование ОДНОГО документа (ValueError у
    raw_file), но не рвёт батч — вопреки прежнему поведению, где needed_stages()
    вызывался вне per-doc try и такая папка убивала весь прогон."""
    broken = make(id="broken-doc-2026", entity_id="bb")
    broken_dir = schema.doc_dir(broken, tmp_path)
    broken_dir.mkdir(parents=True, exist_ok=True)
    (broken_dir / "raw.pdf").write_bytes(b"pdf")
    (broken_dir / "raw.html").write_bytes(b"html")  # два raw.* -> ValueError у raw_file

    ok = make(id="ok-doc-2026", entity_id="oo")
    _place(  # актуален (вкл. версию конвертера)
        ok, tmp_path, raw=b"pdf", md=_compose_md(ok, ""), state=_current_converter_state()
    )

    results = process_docs(
        [broken, ok], tmp_path, force=False, dry_run=False, no_download=False, pause=0
    )

    r_broken = next(r for r in results if r.doc_id == "broken-doc-2026")
    r_ok = next(r for r in results if r.doc_id == "ok-doc-2026")
    assert r_broken.error is not None and r_broken.error.startswith("planning:")
    assert r_ok.up_to_date is True  # батч не оборван


def test_render_frontmatter_used_in_compose(tmp_path: Path) -> None:
    rec = make()
    composed = _compose_md(rec, "old body")
    assert composed.startswith(render_frontmatter(rec))
    assert composed.rstrip().endswith("old body")


# --- реконсиляция пересборки индекса по fingerprint (не по in-run флагу) ---


def _corpus_doc(root: Path, **over: Any) -> Path:
    rec = valid_record()
    rec.update(over)
    return write_doc(root, rec, raw=b"pdf", md="some body text")


def test_needs_index_rebuild_true_when_no_db(tmp_path: Path) -> None:
    _corpus_doc(tmp_path)
    needs, fp = _needs_index_rebuild(tmp_path, tmp_path / "nope.db", force=False)
    assert needs is True
    assert fp  # непустой отпечаток посчитан


def test_needs_index_rebuild_false_when_fingerprint_matches(tmp_path: Path) -> None:
    _corpus_doc(tmp_path)
    db = tmp_path / "c.db"
    conn = corpus_index.create_db(db)
    corpus_index.write_meta(conn, "corpus_fingerprint", corpus_index.corpus_fingerprint(tmp_path))
    conn.commit()
    conn.close()

    needs, _ = _needs_index_rebuild(tmp_path, db, force=False)
    assert needs is False


def test_needs_index_rebuild_true_when_corpus_changed(tmp_path: Path) -> None:
    _corpus_doc(tmp_path)
    db = tmp_path / "c.db"
    conn = corpus_index.create_db(db)
    corpus_index.write_meta(conn, "corpus_fingerprint", "stale-fingerprint-from-before")
    conn.commit()
    conn.close()

    needs, _ = _needs_index_rebuild(tmp_path, db, force=False)
    assert needs is True


def test_needs_index_rebuild_true_when_force(tmp_path: Path) -> None:
    _corpus_doc(tmp_path)
    db = tmp_path / "c.db"
    conn = corpus_index.create_db(db)
    corpus_index.write_meta(conn, "corpus_fingerprint", corpus_index.corpus_fingerprint(tmp_path))
    conn.commit()
    conn.close()

    needs, _ = _needs_index_rebuild(tmp_path, db, force=True)  # совпадает, но --force
    assert needs is True


def test_rebuild_index_writes_fingerprint_on_success(tmp_path: Path, monkeypatch: Any) -> None:
    _corpus_doc(tmp_path)
    monkeypatch.setattr("index.bge_tokenizer.token_counter", lambda: (lambda text: len(text.split())))

    db = tmp_path / "c.db"
    status = rebuild_index(tmp_path, db, embed=False)

    assert "чанков" in status
    # отпечаток, записанный пересборкой, обязан совпасть со свежепосчитанным —
    # иначе следующий прогон не смог бы честно no-op'нуть по гейту.
    assert _read_index_fingerprint(db) == corpus_index.corpus_fingerprint(tmp_path)


def test_rebuild_index_without_model_does_not_touch_existing_fingerprint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Модель временно недоступна: rebuild_index не трогает уже существующий индекс/
    отпечаток — следующий прогон (когда модель появится) честно доиндексирует.
    Мокаем token_counter напрямую — реальное наличие/отсутствие bge-m3 на машине,
    где гоняются тесты, не должно влиять на детерминизм этого теста."""

    def fake_token_counter() -> Any:
        raise FileNotFoundError("bge-m3 не найден (тест)")

    monkeypatch.setattr("index.bge_tokenizer.token_counter", fake_token_counter)

    db = tmp_path / "c.db"
    conn = corpus_index.create_db(db)
    corpus_index.write_meta(conn, "corpus_fingerprint", "old-fingerprint")
    conn.commit()
    conn.close()

    status = rebuild_index(tmp_path, db, embed=False)

    assert "пропущен" in status
    assert _read_index_fingerprint(db) == "old-fingerprint"


# --- усыновление sha ручных raw + stat-guard пересчёта ---


def test_adopt_untracked_raw_no_state_backfills_sha_and_stat(tmp_path: Path) -> None:
    rec = make()
    content = b"manually placed pdf content"
    _place(rec, tmp_path, raw=content)

    _adopt_untracked_raw(rec, tmp_path)

    state = schema.load_state(schema.state_file(rec, tmp_path))
    assert state.sha256 == hashlib.sha256(content).hexdigest()
    assert state.raw_size == len(content)
    assert state.raw_mtime_ns is not None


def test_adopt_untracked_raw_no_raw_is_noop(tmp_path: Path) -> None:
    rec = make()  # папка документа не создана вовсе
    _adopt_untracked_raw(rec, tmp_path)  # не должно бросать
    assert schema.load_state(schema.state_file(rec, tmp_path)) == schema.OperationalState()


def test_adopt_untracked_raw_idempotent(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")
    _adopt_untracked_raw(rec, tmp_path)
    state1 = schema.load_state(schema.state_file(rec, tmp_path))
    _adopt_untracked_raw(rec, tmp_path)  # повторно — уже отслежен, ничего не меняет
    state2 = schema.load_state(schema.state_file(rec, tmp_path))
    assert state1 == state2


def test_adopt_untracked_raw_backfills_old_state_when_sha_still_matches(tmp_path: Path) -> None:
    """.state.yaml старого формата (sha есть, guard-полей нет) — бэкфилл, если
    содержимое подтверждённо совпадает с уже записанным sha."""
    rec = make()
    content = b"stable content since before this feature"
    _place(rec, tmp_path, raw=content, state={"sha256": hashlib.sha256(content).hexdigest()})

    _adopt_untracked_raw(rec, tmp_path)

    state = schema.load_state(schema.state_file(rec, tmp_path))
    assert state.raw_size == len(content)
    assert state.raw_mtime_ns is not None


def test_adopt_untracked_raw_does_not_backfill_when_sha_mismatches(tmp_path: Path) -> None:
    """Старый .state.yaml с sha, НЕ совпадающим с текущим содержимым (файл разошёлся
    ДО того, как эта фича появилась) — guard-поля не бэкфиллятся вслепую; needed_stages
    сама поймает расхождение и запланирует download."""
    rec = make()
    _place(rec, tmp_path, raw=b"actual content", state={"sha256": "0" * 64})

    _adopt_untracked_raw(rec, tmp_path)

    state = schema.load_state(schema.state_file(rec, tmp_path))
    assert state.sha256 == "0" * 64  # не тронут — не наше дело исправлять чужой sha
    assert state.raw_size is None
    assert state.raw_mtime_ns is None
    assert Stage.download in needed_stages(rec, tmp_path)  # расхождение всё равно поймано


def test_needed_stages_stat_guard_skips_sha_recompute_when_unchanged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))  # актуален
    _adopt_untracked_raw(rec, tmp_path)  # заполняет guard-поля
    _stamp_converter_state(rec, tmp_path)  # + converter-поля, сохраняя guard-поля

    calls = {"n": 0}
    real_sha256 = _sha256

    def counting_sha256(path: Path) -> str:
        calls["n"] += 1
        return real_sha256(path)

    monkeypatch.setattr("run_pipeline._sha256", counting_sha256)

    assert needed_stages(rec, tmp_path) == []
    assert calls["n"] == 0  # stat совпал — полное чтение файла не потребовалось


def test_needed_stages_stat_guard_recomputes_when_mtime_changed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))
    _adopt_untracked_raw(rec, tmp_path)
    raw = schema.raw_file(rec, tmp_path)
    assert raw is not None
    raw.touch()  # тот же контент, новый mtime -> guard не совпадёт

    calls = {"n": 0}
    real_sha256 = _sha256

    def counting_sha256(path: Path) -> str:
        calls["n"] += 1
        return real_sha256(path)

    monkeypatch.setattr("run_pipeline._sha256", counting_sha256)

    needed_stages(rec, tmp_path)
    assert calls["n"] == 1  # stat разошёлся -> честно перечитан


def test_corrupted_raw_after_adoption_triggers_download(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"original content", md=_compose_md(rec, ""))
    _adopt_untracked_raw(rec, tmp_path)
    _stamp_converter_state(rec, tmp_path)
    assert needed_stages(rec, tmp_path) == []  # усыновлён, актуален

    raw = schema.raw_file(rec, tmp_path)
    assert raw is not None
    raw.write_bytes(b"corrupted, different content, different size")

    assert Stage.download in needed_stages(rec, tmp_path)


# --- дефолты API-first: --embed-backend / изоляция отказа индексной стадии
# (spec embed-api-first §4) ---


def test_main_embed_backend_flag_propagates_to_rebuild_index(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    captured: dict[str, Any] = {}

    def fake_rebuild(*args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("run_pipeline.rebuild_index", fake_rebuild)
    from run_pipeline import main

    assert main([str(sources), "--db", str(tmp_path / "c.db"), "--embed", "--embed-backend", "bge"]) == 0
    assert captured["embed_backend"] == "bge"
    assert captured["embed"] is True


def test_main_embed_backend_defaults_to_openrouter(tmp_path: Path, monkeypatch: Any) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    captured: dict[str, Any] = {}

    def fake_rebuild(*args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("run_pipeline.rebuild_index", fake_rebuild)
    from run_pipeline import main

    assert main([str(sources), "--db", str(tmp_path / "c.db"), "--embed"]) == 0
    assert captured["embed_backend"] == "openrouter"


def test_main_index_stage_failure_reported_nonzero_not_raised(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Отказ векторной стадии (облако после ретраев/нет ключа) не роняет прогон
    исключением: репорт + ненулевой exit (FTS-часть закоммичена до эмбеддинга)."""
    sources = tmp_path / "sources"
    sources.mkdir()

    def failing_rebuild(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("OpenRouter: исчерпаны попытки")

    monkeypatch.setattr("run_pipeline.rebuild_index", failing_rebuild)
    from run_pipeline import main

    assert main([str(sources), "--db", str(tmp_path / "c.db"), "--embed"]) == 1


def test_rebuild_index_openrouter_skips_confidential_only_chunks(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """sensitivity-гейт в rebuild_index (spec embed-api-first §3.3): облачный бэкенд
    не эмбеддит чанки confidential-документов; пропуск репортится в статусе."""
    sources = tmp_path / "sources"
    conf = {**valid_record(), "id": "sg-conf-doc-2026", "sensitivity": "confidential"}
    pub = {**valid_record(), "id": "sg-pub-doc-2026"}
    write_doc(sources, conf, md="конфиденциальный текст закрытого документа")
    write_doc(sources, pub, md="публичный текст открытого документа")

    # чанковка без реальной модели: токены = слова (лениво импортируемый token_counter)
    monkeypatch.setattr("index.bge_tokenizer.token_counter", lambda: (lambda s: len(s.split())))
    monkeypatch.setattr("run_pipeline.load_dotenv", lambda: None)

    seen: list[str] = []

    class _CapturingCloud:
        name = "cloud"
        dim = 2
        max_tokens: Any = None

        def embed(self, texts: list[str], *, kind: str = "doc") -> Any:
            import numpy as np

            seen.extend(texts)
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr("run_pipeline.get_embedder", lambda backend, **kw: _CapturingCloud())

    status = rebuild_index(sources, tmp_path / "c.db", embed=True, embed_backend="openrouter")
    assert "только-confidential" in status
    assert any("публичный" in t for t in seen)
    assert not any("конфиденциальный" in t for t in seen)


# --- witness-линт: гейт в _do_convert (spec convert-cloud-tier §3) ---


def test_do_convert_witness_skipped_when_no_cloud_ocr_model(tmp_path: Path, monkeypatch: Any) -> None:
    """Локальный (не облачный) конвертированный документ — cloud_ocr_model не
    выставлен, witness_checks НЕ должен вызываться вовсе."""
    rec = make()
    _place(rec, tmp_path, raw=b"raw bytes")

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        dst.write_text("body text", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))
    monkeypatch.setattr("run_pipeline._raw_text", lambda raw, fmt: "witness text totally different")
    monkeypatch.setattr(
        "run_pipeline.lint.witness_checks",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("witness не должен был вызываться")),
    )
    _do_convert(rec, tmp_path)  # не должно упасть

    state = schema.load_state(schema.state_file(rec, tmp_path))
    assert not any(d.startswith("cloud-ocr-") for d in state.lint_defects)


def test_do_convert_witness_skipped_when_raw_sha256_stale(tmp_path: Path, monkeypatch: Any) -> None:
    """cloud_ocr_model выставлен, НО от старого raw (текущий фолбэк на локальный
    путь после провала облака на изменившемся скане) — witness неприменим."""
    rec = make()
    _place(
        rec, tmp_path, raw=b"raw bytes",
        state={"cloud_ocr_model": "m", "cloud_ocr_raw_sha256": "0" * 64},
    )

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        dst.write_text("body text", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))
    monkeypatch.setattr("run_pipeline._raw_text", lambda raw, fmt: "witness text totally different")
    monkeypatch.setattr(
        "run_pipeline.lint.witness_checks",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("witness не должен был вызываться")),
    )
    _do_convert(rec, tmp_path)  # не должно упасть


def test_do_convert_witness_runs_when_cloud_ocr_matches_current_raw(tmp_path: Path, monkeypatch: Any) -> None:
    """cloud_ocr_model выставлен И sha256 совпадает с ТЕКУЩИМ raw — doc.md этого
    прогона подтверждённо облачный, witness обязан отработать и добавить defect."""
    rec = make()
    raw_bytes = b"raw bytes matching cloud state"
    matching_sha256 = _sha256_bytes(raw_bytes)
    _place(
        rec, tmp_path, raw=raw_bytes,
        state={"cloud_ocr_model": "m", "cloud_ocr_raw_sha256": matching_sha256},
    )

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        dst.write_text("cloud output body", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))
    monkeypatch.setattr(
        "run_pipeline._raw_text", lambda raw, fmt: "witness text unrelated to cloud output entirely"
    )
    _do_convert(rec, tmp_path)

    state = schema.load_state(schema.state_file(rec, tmp_path))
    assert any(d.startswith("cloud-ocr-text-loss") for d in state.lint_defects)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- стадия figures: планирование/диспетчеризация/CLI (spec convert-cloud-tier §5/§6) ---


def test_needed_stages_schedules_figures_after_fresh_convert_when_cloud_allowed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rec = make()
    assert needed_stages(rec, tmp_path) == [Stage.download, Stage.convert, Stage.figures, Stage.frontmatter]


def test_needed_stages_convert_without_key_skips_figures(tmp_path: Path) -> None:
    """Дефолт conftest — ключа нет: свежий convert НЕ тянет figures (гейт закрыт)."""
    rec = make()
    assert needed_stages(rec, tmp_path) == [Stage.download, Stage.convert, Stage.frontmatter]


def test_needed_stages_no_cloud_flag_skips_figures_even_with_key(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(converters, "_CLOUD_DISABLED", True)
    rec = make()
    assert needed_stages(rec, tmp_path) == [Stage.download, Stage.convert, Stage.frontmatter]


def test_needed_stages_self_heals_bare_marker_without_forcing_convert(tmp_path: Path, monkeypatch: Any) -> None:
    """Документ уже сконвертирован (converter версия совпадает, raw не менялся), но
    doc.md несёт необработанный маркер (напр. первый прогон после апгрейда на
    convert-cloud-tier) — desired-state самовосстановление: figures планируется
    БЕЗ форсированной реконверсии."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rec = make()
    md_body = (
        "prose\n\n"
        "> [Figure, p. 1, region aaaaaaaaaaaa — structure not reconstructed]\n"
        "> Labels (reading order not guaranteed): X\n"
    )
    _place(
        rec, tmp_path, raw=b"raw bytes", md=_compose_md(rec, md_body),
        state=_current_converter_state(),
    )
    assert needed_stages(rec, tmp_path) == [Stage.figures, Stage.frontmatter]


def test_needed_stages_no_figures_when_all_markers_already_injected(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rec = make()
    md_body = (
        "prose\n\n"
        "> [Figure, p. 1, region aaaaaaaaaaaa — VLM interpretation (m); "
        "reconstruction, verify against original]\n\nAlready injected prose.\n"
    )
    _place(
        rec, tmp_path, raw=b"raw bytes", md=_compose_md(rec, md_body),
        state=_current_converter_state(),
    )
    assert needed_stages(rec, tmp_path) == []


def test_needed_stages_cloudocr_cache_missing_triggers_convert(tmp_path: Path, monkeypatch: Any) -> None:
    """ФС-реконсиляция §6.4: сайдкар .cloudocr.md удалён вручную (единственный способ
    инвалидации кэша — отдельного флага нет) -> следующий прогон обязан переиграть
    convert (тот заново позвонит в облако через cache-мисс _cached_or_call_cloud)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rec = make()
    raw_bytes = b"raw bytes for cloudocr cache test"
    _place(
        rec, tmp_path, raw=raw_bytes, md=_compose_md(rec, "body"),
        state={
            **_current_converter_state(),
            "cloud_ocr_model": "m", "cloud_ocr_raw_sha256": _sha256_bytes(raw_bytes),
        },
    )
    # .cloudocr.md сознательно НЕ создан — это и есть условие теста
    assert Stage.convert in needed_stages(rec, tmp_path)


def test_needed_stages_cloudocr_cache_present_no_forced_convert(tmp_path: Path, monkeypatch: Any) -> None:
    """Контрольный случай: тот же state, но сайдкар НА МЕСТЕ — реконсиляция не
    должна ложно требовать конвертацию (иначе каждый прогон облачного скана бы
    заново конвертировался вхолостую)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rec = make()
    raw_bytes = b"raw bytes for cloudocr cache present test"
    _place(
        rec, tmp_path, raw=raw_bytes, md=_compose_md(rec, "body"),
        state={
            **_current_converter_state(),
            "cloud_ocr_model": "m", "cloud_ocr_raw_sha256": _sha256_bytes(raw_bytes),
        },
    )
    (schema.doc_dir(rec, tmp_path) / ".cloudocr.md").write_text("cached", encoding="utf-8")
    assert Stage.convert not in needed_stages(rec, tmp_path)


def test_do_figures_calls_apply_figures_pass_with_active_model(tmp_path: Path, monkeypatch: Any) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"raw bytes", md=_compose_md(rec, "body"))
    calls: list[dict[str, Any]] = []

    def fake_apply(md: Path, raw: Path, *, model: str) -> bool:
        calls.append({"md": md, "raw": raw, "model": model})
        return True

    monkeypatch.setattr(figures_vlm, "apply_figures_pass", fake_apply)
    monkeypatch.setattr(cloud_ocr, "ACTIVE_MODEL", "test-active-model")
    _do_figures(rec, tmp_path)
    assert len(calls) == 1
    assert calls[0]["model"] == "test-active-model"


def test_process_docs_dispatches_figures_stage(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    rec = make()

    def fake_convert(raw: Path, dst: Path, language: str | None) -> None:
        dst.write_text("prose\n\n> [Figure, p. 1, region aaaaaaaaaaaa — structure not reconstructed]\n"
                        "> Labels (reading order not guaranteed): X\n", encoding="utf-8")

    monkeypatch.setitem(converters._CONVERTERS, "pdf", _fake_converter(fake_convert))
    _place(rec, tmp_path, raw=b"pdf")

    figures_calls: list[str] = []
    monkeypatch.setattr(
        "run_pipeline._do_figures", lambda rec_, root: figures_calls.append(rec_.id)
    )

    results = process_docs([rec], tmp_path, force=False, dry_run=False, no_download=True, pause=0)
    assert results[0].error is None
    assert Stage.figures in results[0].done
    assert figures_calls == [rec.id]


def test_main_no_cloud_flag_disables_cloud_path(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    from run_pipeline import main

    assert main([str(sources), "--no-cloud"]) == 0
    assert converters._CLOUD_DISABLED is True


def test_main_vlm_model_flag_overrides_active_model(tmp_path: Path, monkeypatch: Any) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    monkeypatch.setattr(cloud_ocr, "ACTIVE_MODEL", cloud_ocr.ACTIVE_MODEL)  # регистрируем авто-восстановление
    from run_pipeline import main

    assert main([str(sources), "--vlm-model", "google/gemini-3-pro-preview"]) == 0
    assert cloud_ocr.ACTIVE_MODEL == "google/gemini-3-pro-preview"

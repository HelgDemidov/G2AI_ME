"""Тесты оркестратора: реконсиляция стадий, синк frontmatter, dry-run, изоляция отказов.

Всё CI-safe — без сети (download), pdfplumber (convert) и модели (index).
Раскладка — папка-документ (corpus-layout-v2): пути выводятся из <root>/<track>/<entity>/<id>/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

import acquisition
import corpus_index
import schema
from acquisition import AcquisitionOutcome, ClassifiedResponse
import fsio
from run_pipeline import (
    Stage,
    _adopt_untracked_raw,
    _compose_md,
    _do_convert,
    _do_download,
    _do_frontmatter,
    _needs_index_rebuild,
    _read_index_fingerprint,
    _sha256,
    needed_stages,
    process_docs,
    rebuild_index,
)
from schema import SourceRecord, render_frontmatter
from test_schema import valid_record, write_doc


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


def test_up_to_date_no_stages(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))  # синхронный frontmatter, sha неизвестен
    assert needed_stages(rec, tmp_path) == []


def test_force_redoes_all(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))
    assert needed_stages(rec, tmp_path, force=True) == [Stage.download, Stage.convert, Stage.frontmatter]


def test_frontmatter_drift_detected(tmp_path: Path) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md="---\nid: stale-old\n---\n\nBody.\n")
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


def test_do_convert_staging_uses_dot_prefix_on_failure(tmp_path: Path, monkeypatch: Any) -> None:
    """_do_convert мигрирован на fsio.staging_path — при отказе конвертации огрызок
    остаётся под dot-префиксным именем (совместимым с fsio.cleanup_staging), не
    старым '.tmp'-суффиксом."""
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")

    def fake_convert(src: str, dst: str) -> None:
        Path(dst).write_bytes(b"")  # пустой вывод -> _do_convert бросит RuntimeError

    monkeypatch.setattr("run_pipeline.pdf_convert", fake_convert)

    with pytest.raises(RuntimeError, match="пустой файл"):
        _do_convert(rec, tmp_path)

    staging_files = list(schema.doc_dir(rec, tmp_path).glob(".*.part"))
    assert [p.name for p in staging_files] == [".doc.md.part"]


def test_do_convert_success_leaves_no_staging(tmp_path: Path, monkeypatch: Any) -> None:
    rec = make()
    _place(rec, tmp_path, raw=b"pdf")

    def fake_convert(src: str, dst: str) -> None:
        Path(dst).write_text("converted body", encoding="utf-8")

    monkeypatch.setattr("run_pipeline.pdf_convert", fake_convert)
    _do_convert(rec, tmp_path)

    assert schema.md_file(rec, tmp_path).read_text(encoding="utf-8") == "converted body"
    assert list(schema.doc_dir(rec, tmp_path).glob(".*.part")) == []


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
    _place(b, tmp_path, raw=b"pdf", md=_compose_md(b, ""))  # b актуален

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

    def fake_fetch(url: str, dest: Path, *, user_agent: str, timeout: int = 30) -> ClassifiedResponse:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4 fake content")
        return ok

    monkeypatch.setattr("acquisition.fetch_and_classify", fake_fetch)

    _do_download(rec, tmp_path, pause=0)

    assert (schema.doc_dir(rec, tmp_path) / "raw.pdf").read_bytes() == b"%PDF-1.4 fake content"
    st = schema.load_state(schema.state_file(rec, tmp_path))
    assert st.acquisition_method is not None and st.acquisition_method.value == "direct"
    assert st.fidelity is not None and st.fidelity.value == "live"
    assert st.sha256 is not None


def test_do_download_cleans_staging_on_batch_block(tmp_path: Path, monkeypatch: Any) -> None:
    """При AcquisitionBlocked в батче (interactive=False) staging убирается —
    challenge-тело (реалистично: curl -o пишет его в part ДО классификации) не
    остаётся под именем, которое schema.raw_file мог бы усыновить как оригинал."""
    rec = make()

    def fake_run_ladder(rec_: SourceRecord, dest: Path, *, user_agent: str) -> Any:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"<html>Attention Required</html>")  # как реальный curl -o на challenge
        raise acquisition.AcquisitionBlocked(rec_.source_url, "direct blocked (WAF challenge)")

    monkeypatch.setattr("acquisition.run_ladder", fake_run_ladder)

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

    def fake_fetch(url: str, dest: Path, *, user_agent: str, timeout: int = 30) -> ClassifiedResponse:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4 fake content")
        return ok

    monkeypatch.setattr("acquisition.fetch_and_classify", fake_fetch)

    _do_download(rec, tmp_path, pause=0)

    assert schema.raw_file(rec, tmp_path) == doc_dir / "raw.pdf"  # ровно один raw.* — новый
    assert not (doc_dir / "raw.html").exists()


def test_process_docs_cleans_stale_staging_before_planning(tmp_path: Path) -> None:
    """Реконсиляционная чистка (§1d): останки упавшего прогона убираются сами."""
    rec = make()
    _place(rec, tmp_path, raw=b"pdf", md=_compose_md(rec, ""))  # актуален
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
    _place(ok, tmp_path, raw=b"pdf", md=_compose_md(ok, ""))  # актуален

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
    monkeypatch.setattr("bge_tokenizer.token_counter", lambda: (lambda text: len(text.split())))

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

    monkeypatch.setattr("bge_tokenizer.token_counter", fake_token_counter)

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
    assert needed_stages(rec, tmp_path) == []  # усыновлён, актуален

    raw = schema.raw_file(rec, tmp_path)
    assert raw is not None
    raw.write_bytes(b"corrupted, different content, different size")

    assert Stage.download in needed_stages(rec, tmp_path)

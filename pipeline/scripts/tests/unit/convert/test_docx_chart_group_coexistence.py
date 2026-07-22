"""Характеризационные тесты (spec chart-data-extraction, hardening §2 —
"characterization-тесты ПЕРВЫМ коммитом, до любого удаления"): docx-документ
с ОБЕИМИ разновидностями вырезаемых объектов — composite-группой (kind="group",
остаётся на VLM) и нативным c:chart (kind="chart", станет data-driven) —
одновременно. Существующие тесты (``test_docx_groups.py``) покрывают каждый
kind ПООДИНОЧКЕ; этот файл — регресс-guard на то, что переключение резолюции
chart-kind на data-driven (следующие коммиты) НЕ задевает group-путь (детект/
вырезка/сентинел/резолюция), который делит с ним общий код
(``docx_groups._iter_objects``/``extract_and_strip_groups``).

Сборка ``inject_group_markers`` для kind="chart" в этом файле пинует ТЕКУЩЕЕ
(до-рефакторинга) поведение — маркер, а не data-driven рендер; эта ОДНА
ассерция намеренно изменится, когда резолюция chart-kind станет data-driven
(остальные — group-путь и детект/вырезка — обязаны остаться зелёными без
изменений на протяжении всего рефакторинга)."""
from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO
from pathlib import Path

from convert.docx_groups import (
    DocxGroup,
    all_media_ids,
    extract_and_strip_groups,
    extract_group_docx,
    inject_group_markers,
)
from tests.support import build_docx_with_shape_group_and_inline_chart


def test_group_and_chart_coexist_both_detected_with_distinct_kinds(tmp_path: Path) -> None:
    raw = tmp_path / "raw.docx"
    img = b"x" * 100
    raw.write_bytes(
        build_docx_with_shape_group_and_inline_chart(
            ["Group caption"], {"a.png": img}, ["Chart Title"]
        )
    )
    _rewritten, groups = extract_and_strip_groups(raw)
    assert len(groups) == 2
    kinds = {g.kind for g in groups}
    assert kinds == {"group", "chart"}
    group = next(g for g in groups if g.kind == "group")
    chart = next(g for g in groups if g.kind == "chart")
    assert group.media_ids == frozenset({hashlib.sha256(img).hexdigest()[:12]})
    assert group.captions == ("Group caption",)
    assert chart.media_ids == frozenset()
    assert chart.captions == ("Chart Title",)


def test_group_and_chart_coexist_both_stripped_to_sentinels(tmp_path: Path) -> None:
    raw = tmp_path / "raw.docx"
    raw.write_bytes(
        build_docx_with_shape_group_and_inline_chart(["Cap"], {"a.png": b"y" * 50}, ["Title"])
    )
    rewritten, groups = extract_and_strip_groups(raw)
    with zipfile.ZipFile(BytesIO(rewritten)) as z:
        doc_xml = z.read("word/document.xml").decode("utf-8")
    assert "mc:AlternateContent" not in doc_xml
    assert "c:chart" not in doc_xml
    for group in groups:
        assert group.id12 in doc_xml


def test_group_and_chart_coexist_chart_part_survives_in_rewritten_zip(tmp_path: Path) -> None:
    """Сентинел заменяет ТОЛЬКО anchor в document.xml — chart-парт
    (``word/charts/chart1.xml``) остаётся в архиве нетронутым (нужен резолюции,
    сейчас VLM-рендеру через ``extract_group_docx``, впоследствии
    data-driven-парсеру ``chart_data.parse_chart``)."""
    raw = tmp_path / "raw.docx"
    raw.write_bytes(
        build_docx_with_shape_group_and_inline_chart(["Cap"], {}, ["Title"])
    )
    rewritten, _groups = extract_and_strip_groups(raw)
    with zipfile.ZipFile(BytesIO(rewritten)) as z:
        assert "word/charts/chart1.xml" in z.namelist()


def test_group_and_chart_coexist_injection_gives_each_its_own_marker_kind(tmp_path: Path) -> None:
    """ТЕКУЩЕЕ (до-рефакторинга) поведение ``inject_group_markers``: ОБА kind
    сегодня дают одинаковую грамматику маркера (только noun различается) —
    эта ассерция для kind="chart" ожидаемо изменится, когда резолюция
    станет data-driven (см. докстроку модуля); group-часть — нет."""
    raw = tmp_path / "raw.docx"
    raw.write_bytes(
        build_docx_with_shape_group_and_inline_chart(["Group cap"], {}, ["Chart cap"])
    )
    rewritten, groups = extract_and_strip_groups(raw)
    with zipfile.ZipFile(BytesIO(rewritten)) as z:
        doc_xml = z.read("word/document.xml").decode("utf-8")
    # Сентинелы не проходят через mammoth в этом тесте (нет реального
    # текстового потока) — инъецируем маркер напрямую в сырые sentinel-строки,
    # как это делает converters._convert_docx после markdownify.
    group = next(g for g in groups if g.kind == "group")
    chart = next(g for g in groups if g.kind == "chart")
    text = f"before {doc_xml.count('DOCXGROUPSENTINEL')} " + "".join(
        f"DOCXGROUPSENTINEL{g.id12}" for g in groups
    )
    result = inject_group_markers(text, groups)
    assert f"> [Figure, docx group {group.id12} — composite content not analyzed]" in result
    assert "> captions: Group cap" in result
    assert f"> [Figure, docx chart {chart.id12} — chart content not analyzed]" in result
    assert "> captions: Chart cap" in result


def test_group_path_render_extraction_unaffected_by_coexisting_chart(tmp_path: Path) -> None:
    """``extract_group_docx`` для kind="group" (питает ``_render_docx_group``
    -> soffice -> VLM, спек §2 «group-путь БЕЗ ИЗМЕНЕНИЙ») продолжает находить
    и изолировать ИМЕННО группу, а не подхватывает соседний чарт, когда оба
    присутствуют в одном документе."""
    raw = tmp_path / "raw.docx"
    raw.write_bytes(
        build_docx_with_shape_group_and_inline_chart(["Cap"], {"a.png": b"z" * 60}, ["Title"])
    )
    _rewritten, groups = extract_and_strip_groups(raw)
    group = next(g for g in groups if g.kind == "group")
    mini = extract_group_docx(raw, group.id12)
    assert mini is not None
    with zipfile.ZipFile(BytesIO(mini)) as z:
        doc_xml = z.read("word/document.xml").decode("utf-8")
    assert "wpg:wgp" in doc_xml
    assert "c:chart" not in doc_xml


def test_all_media_ids_includes_both_kinds_media() -> None:
    g1 = DocxGroup(id12="a" * 12, media_ids=frozenset({"111111111111"}), captions=(), kind="group")
    g2 = DocxGroup(id12="b" * 12, media_ids=frozenset(), captions=(), kind="chart")
    assert all_media_ids([g1, g2]) == frozenset({"111111111111"})

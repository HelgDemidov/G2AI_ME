"""Тесты docx_groups.py (spec convert-docx §2-ter): детект composite-групп
(mc:AlternateContent/wpg:wgp), сентинел-замена, извлечение media_ids/captions,
пост-инъекция маркера. Ни сети, ни soffice — чистый XML in-memory."""
from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO
from pathlib import Path

from convert.docx_groups import (
    SENTINEL_PREFIX,
    all_media_ids,
    extract_and_strip_groups,
    inject_group_markers,
)
from tests.support import build_docx_with_shape_group, build_minimal_docx


def test_no_groups_returns_original_bytes_unchanged(tmp_path: Path) -> None:
    raw = tmp_path / "raw.docx"
    orig = build_minimal_docx(["Just plain text, no groups."])
    raw.write_bytes(orig)
    rewritten, groups = extract_and_strip_groups(raw)
    assert groups == []
    assert rewritten == orig


def test_detects_single_group_with_media_and_captions(tmp_path: Path) -> None:
    raw = tmp_path / "raw.docx"
    img = b"x" * 100
    raw.write_bytes(
        build_docx_with_shape_group(["Before."], ["Caption A", "Caption B"], {"a.png": img}, ["After."])
    )
    _rewritten, groups = extract_and_strip_groups(raw)
    assert len(groups) == 1
    group = groups[0]
    assert group.media_ids == frozenset({hashlib.sha256(img).hexdigest()[:12]})
    assert group.captions == ("Caption A", "Caption B")


def test_numeric_only_captions_filtered_as_position_junk(tmp_path: Path) -> None:
    raw = tmp_path / "raw.docx"
    raw.write_bytes(
        build_docx_with_shape_group(["Before."], ["-635", "1231900", "Real caption"], {}, ["After."])
    )
    _rewritten, groups = extract_and_strip_groups(raw)
    assert groups[0].captions == ("Real caption",)


def test_duplicate_captions_deduplicated_in_order(tmp_path: Path) -> None:
    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_docx_with_shape_group(["Before."], ["Repeat", "Other", "Repeat"], {}, ["After."]))
    _rewritten, groups = extract_and_strip_groups(raw)
    assert groups[0].captions == ("Repeat", "Other")


def test_group_with_no_images_has_empty_media_ids(tmp_path: Path) -> None:
    """Живой кейс §2-ter.1: EU Data Act flow — все иконки мельче порога/вектор,
    в group_media_ids не входит НИЧЕГО, но группа как таковая детектится и
    получает маркер (captions её не теряют)."""
    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_docx_with_shape_group(["Before."], ["Only text, no raster"], {}, ["After."]))
    _rewritten, groups = extract_and_strip_groups(raw)
    assert len(groups) == 1
    assert groups[0].media_ids == frozenset()
    assert groups[0].captions == ("Only text, no raster",)


def test_sentinel_replaces_group_in_rewritten_document_xml(tmp_path: Path) -> None:
    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_docx_with_shape_group(["Before."], ["Cap"], {}, ["After."]))
    rewritten, groups = extract_and_strip_groups(raw)
    with zipfile.ZipFile(BytesIO(rewritten)) as z:
        doc_xml = z.read("word/document.xml").decode("utf-8")
    assert "mc:AlternateContent" not in doc_xml
    assert f"{SENTINEL_PREFIX}{groups[0].id12}" in doc_xml


def test_id12_deterministic_across_repeated_calls(tmp_path: Path) -> None:
    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_docx_with_shape_group(["Before."], ["Cap"], {"a.png": b"y" * 50}, ["After."]))
    _r1, groups1 = extract_and_strip_groups(raw)
    _r2, groups2 = extract_and_strip_groups(raw)
    assert groups1[0].id12 == groups2[0].id12


def test_multiple_groups_get_distinct_ids_and_own_media(tmp_path: Path) -> None:
    """Билдер рассчитан на одну группу на документ — сверяем на ДВУХ отдельных
    однокгрупповых документах, что id12 действительно зависит от содержимого
    (разные картинки/подписи -> разные id, разные media_ids), что и требуется
    для корректной работы на документе с несколькими группами (живой кейс —
    вырезка отчёта, 3 группы)."""
    img_a, img_b = b"a" * 100, b"b" * 100
    raw_a = tmp_path / "a.docx"
    raw_a.write_bytes(build_docx_with_shape_group([], ["Cap A"], {"a.png": img_a}, []))
    raw_b = tmp_path / "b.docx"
    raw_b.write_bytes(build_docx_with_shape_group([], ["Cap B"], {"b.png": img_b}, []))
    _rewritten_a, groups_a = extract_and_strip_groups(raw_a)
    _rewritten_b, groups_b = extract_and_strip_groups(raw_b)
    assert groups_a[0].id12 != groups_b[0].id12
    assert groups_a[0].media_ids != groups_b[0].media_ids


def test_inject_group_markers_replaces_bare_sentinel() -> None:
    from convert.docx_groups import DocxGroup

    group = DocxGroup(id12="abc123def456", media_ids=frozenset(), captions=("Foo", "Bar"))
    text = f"Before.\n\n{SENTINEL_PREFIX}abc123def456\n\nAfter."
    result = inject_group_markers(text, [group])
    assert "> [Figure, docx group abc123def456 — composite content not analyzed]" in result
    assert "> captions: Foo; Bar" in result
    assert SENTINEL_PREFIX not in result
    assert result.index("Before.") < result.index("abc123def456") < result.index("After.")


def test_inject_group_markers_consumes_bold_wrapping() -> None:
    """Markdownify иногда оборачивает сентинел в ** (унаследованный rPr run'а,
    который сентинел заменил, живой кейс — блок 23 фикстуры) — регекс должен
    поглотить обрамление целиком, не оставляя висячих звёздочек."""
    from convert.docx_groups import DocxGroup

    group = DocxGroup(id12="abc123def456", media_ids=frozenset(), captions=())
    text = f"Before. **{SENTINEL_PREFIX}abc123def456** After."
    result = inject_group_markers(text, [group])
    assert "**" not in result
    assert "> [Figure, docx group abc123def456" in result


def test_inject_group_markers_empty_captions_says_no_text() -> None:
    from convert.docx_groups import DocxGroup

    group = DocxGroup(id12="abc123def456", media_ids=frozenset(), captions=())
    result = inject_group_markers(f"{SENTINEL_PREFIX}abc123def456", [group])
    assert "> captions: (нет текста)" in result


def test_inject_group_markers_no_groups_is_noop() -> None:
    text = "Nothing to replace here."
    assert inject_group_markers(text, []) == text


def test_all_media_ids_unions_across_groups() -> None:
    from convert.docx_groups import DocxGroup

    g1 = DocxGroup(id12="a" * 12, media_ids=frozenset({"111111111111", "222222222222"}), captions=())
    g2 = DocxGroup(id12="b" * 12, media_ids=frozenset({"222222222222", "333333333333"}), captions=())
    assert all_media_ids([g1, g2]) == frozenset({"111111111111", "222222222222", "333333333333"})


def test_all_media_ids_empty_for_no_groups() -> None:
    assert all_media_ids([]) == frozenset()

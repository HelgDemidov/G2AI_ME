"""Round-trip грамматики маркеров: производитель ↔ потребитель (spec
convert-knowledge-seam-hardening §5).

Аудит шва (Б5) нашёл грамматику маркеров в ТРЁХ несшитых копиях: рендереры
(``pdf_graphics``/``docx_groups``/``converters``), распознающие регексы
(``figures_vlm``) и рукописные литералы в двух тестовых файлах. Каждая сторона была
запинена ОТДЕЛЬНО, а связи между ними не утверждал никто: смена формы маркера прошла
бы оба теста порознь, а ``has_bare_markers`` тихо перестала бы находить маркеры —
стадия figures просто не планировалась бы (no-op без ошибки).

Здесь НЕТ ни одного литерала маркера: каждый вход строится вызовом настоящего
рендерера, каждый вердикт — вызовом настоящего потребителя. Намеренно НЕ ловимые
формы зафиксированы так же явно — иначе «перестало ловиться» и «так задумано»
неразличимы.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from convert import docx_groups, pdf_graphics, xlsx_charts
from convert.converters import _docx_image_markers
from convert.figures_vlm import (
    _render_injected_docx_group,
    _render_injected_docx_image,
    _render_injected_figure,
    _render_injected_image,
    has_bare_markers,
)
from tests.support import build_docx_with_choice_only_images

_ID12 = "abc123def456"


class _Word:
    def __init__(self, text: str) -> None:
        self.text, self.x0, self.x1, self.top, self.bottom = text, 0.0, 10.0, 0.0, 5.0


def _region(kind: str, **extra: object) -> pdf_graphics.Region:
    return pdf_graphics.Region(
        bbox=(0.0, 0.0, 100.0, 100.0), elements=[], words=[_Word("Label A")], kind=kind, id=_ID12,
        **extra,  # type: ignore[arg-type]
    )


# --- маркеры, которые стадия figures ОБЯЗАНА подхватывать ---


def test_opaque_region_marker_is_recognized() -> None:
    marker = pdf_graphics.render_region_block(_region("opaque"), page=6)
    assert has_bare_markers(marker) is True


def test_raster_image_marker_is_recognized() -> None:
    image = pdf_graphics.Element("image", 0.0, 0.0, 10.0, 10.0, content_hash="a" * 64)
    assert has_bare_markers(pdf_graphics.render_raster_marker(2, image)) is True


def test_docx_group_marker_is_recognized() -> None:
    marker = docx_groups._render_group_marker(_ID12, ("caption",), "group")
    assert has_bare_markers(marker) is True


def test_docx_image_marker_is_recognized(tmp_path: Path) -> None:
    """Маркер строится НАСТОЯЩИМ фолбэк-проходом по реальному docx (не литералом),
    поэтому тест ломается и от правки формы, и от правки orphan-фильтра."""
    data = b"\x89PNG" + b"x" * 8000
    raw = tmp_path / "d.docx"
    raw.write_bytes(build_docx_with_choice_only_images(["para"], {"pic.png": data}))
    block = _docx_image_markers(raw)
    assert hashlib.sha256(data).hexdigest()[:12] in block
    assert has_bare_markers(block) is True


# --- формы, НЕ подхватываемые СОЗНАТЕЛЬНО (иначе «сломалось» неотличимо от «так надо») ---


@pytest.mark.parametrize("kind,extra", [
    ("grid", {"cells": [["a", "b"], ["c", "d"]]}),
    ("sequence", {"items": ["one", "two"]}),
])
def test_reconstructed_region_markers_are_not_escalated(kind: str, extra: dict[str, object]) -> None:
    """Грид/sequence уже РЕКОНСТРУИРОВАНЫ детерминированно — эскалировать в VLM нечего
    (spec convert-graphics §2)."""
    marker = pdf_graphics.render_region_block(_region(kind, **extra), page=3)
    assert has_bare_markers(marker) is False


def test_docx_chart_caption_fallback_is_not_escalated() -> None:
    """Нативный чарт без numCache остаётся честным статичным маркером навсегда
    (spec chart-data-extraction §4.3), а не уходит в soffice+VLM."""
    assert has_bare_markers(docx_groups._render_group_marker(_ID12, ("cap",), "chart")) is False


def test_xlsx_chart_marker_is_not_escalated() -> None:
    chart = xlsx_charts.XlsxChart(id12=_ID12, sheet="S1", anchor_cell="A1", captions=("cap",))
    assert has_bare_markers(xlsx_charts.render_chart_marker(chart)) is False


# --- инъецированные формы: идемпотентность стадии по построению ---


@pytest.mark.parametrize("injected", [
    _render_injected_figure(6, _ID12, "model/x", "prose"),
    _render_injected_image(20, _ID12, "model/x", "prose"),
    _render_injected_docx_image(_ID12, "model/x", "prose"),
    _render_injected_docx_group(_ID12, "model/x", "prose"),
])
def test_injected_forms_are_not_bare_markers(injected: str) -> None:
    """Повторный прогон стадии не должен видеть уже обработанный блок — на этом стоит
    идемпотентность без отдельного флага «уже сделано»."""
    assert has_bare_markers(injected) is False

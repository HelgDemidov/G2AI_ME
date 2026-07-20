"""Live-smoke: реальный рендер xlsx-чарта через soffice (не мок, spec
convert-xlsx §3). Требует установленный soffice (в CI отсутствует ->
skipif) и реальную тестовую фикстуру (tests/fixtures/local/, gitignored ->
skipif при свежем клоне). НЕ вызывает VLM/сеть — только проверяет, что
extract_chart_workbook + рендерер производят валидный data-URI на реальной
книге стороннего автора (не нашей openpyxl-синтетике)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from convert.figures_vlm import _render_xlsx_chart

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "local" / "govtech-2025-stats-excerpt.xlsx"

pytestmark = [
    pytest.mark.libreoffice,
    pytest.mark.skipif(shutil.which("soffice") is None, reason="soffice не установлен (см. spec §3)"),
    pytest.mark.skipif(not _FIXTURE.exists(), reason="тестовая фикстура fixtures/local отсутствует (gitignored)"),
]

# Известные id24 чартов листа Stats фикстуры (spec §3, живой чекпоинт
# 2026-07-20): подмножество разных типов (doughnut/radar/bar + два bar-
# производных без title-конфликтов) — не все 24, довольно для проверки
# самой механики extract_chart_workbook на реальной чужой книге.
_KNOWN_CHART_IDS = [
    "81e3f64eb12d",  # doughnut, "Institutional Responsibility for GovTech"
    "96a22b948568",  # radar, "GovTech Maturity Index Components"
    "257ff3eeb85d",  # bar, "Income Level Distribution"
    "68977664ff7f",  # bar, "Regional Distribution"
    "cfc18ed7ef06",  # bar, "Distribution by Groups"
]


@pytest.mark.parametrize("chart_id", _KNOWN_CHART_IDS)
def test_render_xlsx_chart_real_fixture_produces_valid_data_uri(chart_id: str) -> None:
    data_uri = _render_xlsx_chart(_FIXTURE, chart_id)
    assert data_uri is not None
    assert data_uri.startswith("data:image/jpeg;base64,")
    # грубая проверка, что кроп реально захватил содержимое чарта, а не
    # пустой/крошечный угол страницы
    assert len(data_uri) > 5000


def test_render_xlsx_chart_unknown_id_returns_none() -> None:
    assert _render_xlsx_chart(_FIXTURE, "0" * 12) is None

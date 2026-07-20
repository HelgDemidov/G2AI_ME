"""Live-smoke: реальный рендер docx composite-группы через soffice (не мок,
spec convert-docx §2-ter). Требует установленный soffice (в CI отсутствует ->
skipif) и реальную тестовую фикстуру (tests/fixtures/local/, gitignored ->
skipif при свежем клоне). НЕ вызывает VLM/сеть — только проверяет, что
рендерер производит валидный data-URI на реальных группах документа."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from convert.figures_vlm import _render_docx_group

_FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "local"
    / "iot-report-2022-national-strategies-excerpt.docx"
)

pytestmark = [
    pytest.mark.libreoffice,
    pytest.mark.skipif(shutil.which("soffice") is None, reason="soffice не установлен (см. spec §2-ter.4)"),
    pytest.mark.skipif(not _FIXTURE.exists(), reason="тестовая фикстура fixtures/local отсутствует (gitignored)"),
]

# Известные id renderable-объектов фикстуры (spec §2-ter.2 + ultimate-тест
# 2026-07-20, после трансплантации 5 объектов из полного отчёта):
# группы — «Программа European Large-Scale IoT Pilots», EU Data Act flow,
# «Матрица приоритетных секторов IoT в Бразилии», «5G/IMT-2020 сценарии»,
# «Карта Wi-Fi 6E»; chart — bar-график CAPEX/OPEX частных сетей LTE/5G.
_KNOWN_GROUP_IDS = [
    "31cb26ede622",
    "863b94a50ac0",
    "5fef7b6067d0",
    "cf269e703022",
    "33be0a31a485",
    "34d4b5014cb4",
]


@pytest.mark.parametrize("group_id", _KNOWN_GROUP_IDS)
def test_render_docx_group_real_fixture_produces_valid_data_uri(group_id: str) -> None:
    data_uri = _render_docx_group(_FIXTURE, group_id)
    assert data_uri is not None
    assert data_uri.startswith("data:image/jpeg;base64,")
    # грубая проверка, что кроп реально захватил содержимое диаграммы, а не
    # пустой/крошечный угол страницы
    assert len(data_uri) > 5000


def test_render_docx_group_unknown_id_returns_none() -> None:
    assert _render_docx_group(_FIXTURE, "0" * 12) is None

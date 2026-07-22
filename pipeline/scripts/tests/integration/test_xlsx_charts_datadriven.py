"""Локальный детерминированный чек (spec chart-data-extraction §Тестовое
покрытие + §Финальная приёмка): data-driven chart-путь на РЕАЛЬНОЙ, не
синтетической книге стороннего автора (``tests/fixtures/local/
govtech-2025-charts.xlsx``, World Bank GovTech Maturity Index Dataset,
CC BY 4.0, gitignored -> skipif при свежем клоне). ВСЕ 55 встроенных
чартов сохранены byte-identical (``xl/charts/*``/``xl/drawings/*`` не
тронуты) — обрезаны только избыточные строки/колонки листов и раздутые
per-cell ``<hyperlinks>`` (см. ``make_govtech_charts_fixture.py`` рядом,
регенерирует фикстуру из полной книги детерминированно). Заменила
изначальный ПОЛНЫЙ оригинал (5.4 МБ, 2 листа с 700k+ живых ячеек) 2026-07-22
по решению пользователя: полная книга оказалась слишком тяжёлой для
локального full-gate (``openpyxl.load_workbook`` сам по себе — 16.5 с из-за
объёма живых данных, никак не связанных с чартами — ``parse_chart`` читает
ТОЛЬКО закэшированные ``<c:numCache>``/``<c:strCache>`` внутри chart-парта,
не ячейки листа). После обрезки то же покрытие (55/55 чартов достижимы,
0 крашей, 33/55 mermaid) при ``_convert_xlsx`` ~5 с/прогон (было ~22 с).
Отличается от прежней урезанной Stats-only фикстуры (24/55 чартов
достижимы, остальные — осколки хирургического удаления 8 ЛИСТОВ целиком)
методом: здесь обрезаны только строки/колонки ВНУТРИ каждого из 9 листов,
ни один лист/drawing/chart-парт не удалён — orphan-чартов в принципе не
может возникнуть. В отличие от удалённого ``test_xlsx_charts_live.py``
(spec convert-xlsx §3, требовал системный soffice для рендера картинки) —
этот чек НЕ имеет внешних системных зависимостей: parse_chart/render_chart
чистые функции, ``_convert_xlsx`` — openpyxl+lxml, никакого soffice/сети.
Живёт в ``tests/integration/`` (не ``unit/``) исключительно из-за
зависимости на негерметичный внешний файл, не системный ресурс."""
from __future__ import annotations

from pathlib import Path

import pytest

from convert import chart_data, chart_render, xlsx_charts
from convert.converters import _convert_xlsx

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "local" / "govtech-2025-charts.xlsx"

pytestmark = pytest.mark.skipif(
    not _FIXTURE.exists(), reason="тестовая фикстура fixtures/local отсутствует (gitignored)"
)

# Все 55 физических xl/charts/*.xml части достижимы (3 листа — Stats/Regions/
# Trends несут drawing-анкеры) — обрезка листов не трогает ни один chart/drawing
# парт, поэтому осколков-сирот нет (extract_charts == голый zip-глоб по счёту,
# в кои-то веки совпадают, но код всё равно должен идти через первый — не глоб).
_EXPECTED_REACHABLE_CHARTS = 55
_KNOWN_DOUGHNUT_ID = "81e3f64eb12d"  # "Institutional Responsibility for GovTech" (Stats!BO37)
_KNOWN_RADAR_ID = "96a22b948568"  # "GovTech Maturity Index Components" (Stats!AP25)


def test_all_reachable_charts_parse_and_render_without_crash() -> None:
    charts = xlsx_charts.extract_charts(_FIXTURE)
    roots = xlsx_charts.extract_chart_roots(_FIXTURE)
    assert len(charts) == _EXPECTED_REACHABLE_CHARTS

    crashes: list[tuple[str, str]] = []
    mermaid_count = 0
    for chart in charts:
        try:
            data = chart_data.parse_chart(roots[chart.id12])
            rendered = chart_render.render_chart(data)
        except Exception as exc:  # noqa: BLE001 — сеть безопасности: ловим ЛЮБОЙ крах на реальных данных
            crashes.append((chart.id12, repr(exc)))
            continue
        assert rendered is not None, f"{chart.id12}: пустое извлечение на реальном чарте с numCache"
        if "```mermaid" in rendered:
            mermaid_count += 1
    assert crashes == []
    # Нижний порог, не точное число — ловит грубую регрессию маппинга типов,
    # не переобучен на текущий точный подсчёт (живой замер финальной приёмки
    # 2026-07-22: 33/55 — column/bar/combo/radar/doughnut получают mermaid,
    # stacked-бары/scatter честно только таблица).
    assert mermaid_count >= 25


def test_known_doughnut_chart_type_and_categories() -> None:
    roots = xlsx_charts.extract_chart_roots(_FIXTURE)
    data = chart_data.parse_chart(roots[_KNOWN_DOUGHNUT_ID])
    assert data.chart_type == "doughnut"
    assert data.title == "Institutional Responsibility for GovTech"
    assert "Ministry of ICT" in data.categories
    assert len(data.series) == 1
    assert all(v is not None for v in data.series[0].values)


def test_known_radar_chart_type_and_series_count() -> None:
    roots = xlsx_charts.extract_chart_roots(_FIXTURE)
    data = chart_data.parse_chart(roots[_KNOWN_RADAR_ID])
    assert data.chart_type == "radar"
    assert data.title == "GovTech Maturity Index Components"
    assert len(data.categories) == 4  # CGSI/PSDI/DCEI/GTEI
    assert len(data.series) == 3  # Regional Avg / Global Avg / Mozambique (живой факт фикстуры)


def test_convert_xlsx_full_fixture_produces_stable_output_with_provenance(tmp_path: Path) -> None:
    out = tmp_path / "out.md"
    _convert_xlsx(_FIXTURE, out, "en")
    text = out.read_text(encoding="utf-8")
    assert "## Stats" in text
    assert "> лист Stats, якорь" in text
    assert "Institutional Responsibility for GovTech" in text
    assert "GovTech Maturity Index Components" in text
    assert "```mermaid" in text
    # Все 55 достижимых чартов извлекаются непусто (test_all_reachable_charts_
    # parse_and_render_without_crash) -> ни один не должен упасть на честный
    # caption-фолбэк маркер в полном прогоне конвертера.
    assert "chart content not analyzed" not in text


def test_convert_xlsx_full_fixture_deterministic_across_runs(tmp_path: Path) -> None:
    """Golden-safety (спек §5): реконверсия -> идентичный вывод, что и было
    невозможно гарантировать для VLM-пути. ``raw`` НЕ модифицируется
    (``xlsx_charts`` module docstring) — прогон дважды по одному и тому же
    файлу безопасен."""
    out1, out2 = tmp_path / "out1.md", tmp_path / "out2.md"
    _convert_xlsx(_FIXTURE, out1, "en")
    _convert_xlsx(_FIXTURE, out2, "en")
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_all_mermaid_blocks_accepted_by_real_mermaid_js() -> None:
    """Постоянный аналог финальной приёмки 2026-07-22 (спек §Финальная
    приёмка): не только наши эвристики (``test_chart_render_visual.py``,
    синтетические ChartData), а ВСЕ реальные mermaid-блоки, которые
    ``render_chart`` производит из этой конкретной книги — через настоящий
    mermaid.js (``mermaidx``, теперь runtime-зависимость, requirements.txt —
    ``render_chart`` уже гейтит каждый блок через неё сам, этот тест дублирует
    проверку явно, как постоянный регресс-чек). Живой замер: 33/55 чартов
    дают mermaid, ~116 мс/диаграмма на прогретом движке."""
    import mermaidx

    charts = xlsx_charts.extract_charts(_FIXTURE)
    roots = xlsx_charts.extract_chart_roots(_FIXTURE)
    failures: list[tuple[str, str]] = []
    mermaid_count = 0
    for chart in charts:
        data = chart_data.parse_chart(roots[chart.id12])
        rendered = chart_render.render_chart(data)
        if rendered is None or "```mermaid" not in rendered:
            continue
        mermaid_count += 1
        fence = rendered.split("```mermaid\n", 1)[1].split("```", 1)[0]
        try:
            mermaidx.render(fence).svg()
        except Exception as exc:  # noqa: BLE001 — любой отказ реального рендера — находка теста
            failures.append((chart.id12, repr(exc)[:200]))
    assert mermaid_count >= 25
    assert failures == []

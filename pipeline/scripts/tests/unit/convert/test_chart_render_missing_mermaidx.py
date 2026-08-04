"""Regression coverage for the mermaidx-absent path in chart_render.py.

Root cause (found during a refigure-extraction code review, 2026-08-04):
``mermaid_renders()``'s own docstring promises "any failure, including the
library's own absence, -> False, fail-safe" — but the ``import mermaidx``
statement sat OUTSIDE the function's try/except, so a genuinely missing
mermaidx raised an uncaught ``ModuleNotFoundError`` instead of degrading.
Dormant in this repo today only because ``requirements.txt`` pins mermaidx
as a hard dependency of the whole pipeline; becomes a real crash the moment
mermaidx is an optional per-format extra (refigure's ``[docx]``/``[xlsx]``
packaging plan).

Simulates absence by monkeypatching ``chart_render.mermaidx`` to ``None``
(what the module-level optional import sets it to) rather than actually
uninstalling the package — mermaidx stays a real, exercised dependency for
every other test in this suite.
"""

from __future__ import annotations

import logging

import pytest

from convert import chart_render
from convert.chart_data import ChartData, ChartSeries
from convert.chart_render import mermaid_renders, render_chart


@pytest.fixture(autouse=True)
def _reset_warn_once_cache() -> None:
    """``_warn_missing_mermaidx`` is ``lru_cache(maxsize=1)`` — module-level
    state that would otherwise leak across tests/test order."""
    chart_render._warn_missing_mermaidx.cache_clear()


def _data(*, chart_type: str = "pie", title: str | None = "T") -> ChartData:
    return ChartData(
        chart_type=chart_type,
        title=title,
        value_axis_title=None,
        value_format=None,
        stacked=False,
        categories=("A", "B"),
        series=(ChartSeries(name="S", values=(1.0, 2.0), kind=chart_type),),
    )


def test_mermaid_renders_does_not_raise_when_mermaidx_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug itself: this used to raise ModuleNotFoundError uncaught."""
    monkeypatch.setattr(chart_render, "mermaidx", None)
    result = mermaid_renders('pie title T\n"a" : 10')
    assert result is False


def test_render_chart_falls_back_to_table_only_when_mermaidx_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full real call path (what docx_groups.py / converters.py actually call) —
    must degrade to table-only, not crash the whole document conversion."""
    monkeypatch.setattr(chart_render, "mermaidx", None)
    out = render_chart(_data())
    assert out is not None
    assert "```mermaid" not in out
    assert "| Category | S |" in out


def test_warning_logged_when_mermaidx_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(chart_render, "mermaidx", None)
    with caplog.at_level(logging.WARNING, logger="convert.chart_render"):
        mermaid_renders('pie title T\n"a" : 10')
    assert len(caplog.records) == 1
    assert "mermaidx" in caplog.records[0].message
    assert "install" in caplog.records[0].message.lower()


def test_warning_logged_only_once_across_multiple_calls(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The whole point of lru_cache(maxsize=1): a document with 55 charts
    (the real govtech fixture) must not spam 55 identical warnings."""
    monkeypatch.setattr(chart_render, "mermaidx", None)
    with caplog.at_level(logging.WARNING, logger="convert.chart_render"):
        for _ in range(5):
            mermaid_renders('pie title T\n"a" : 10')
        render_chart(_data())
        render_chart(_data(chart_type="column"))
    assert len(caplog.records) == 1


def test_no_warning_when_mermaidx_present_and_render_succeeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression guard: the fix must not change the happy path at all —
    mermaidx IS installed in this test environment, real render must still work."""
    with caplog.at_level(logging.WARNING, logger="convert.chart_render"):
        result = mermaid_renders('pie title T\n"a" : 10')
    assert result is True
    assert len(caplog.records) == 0


def test_real_render_failure_returns_false_without_missing_dependency_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A genuine bad-render (mermaidx present, syntax mermaid.js rejects) must
    keep degrading via the existing `except Exception -> False` path, and must
    NOT be conflated with (or trigger) the missing-dependency warning."""
    with caplog.at_level(logging.WARNING, logger="convert.chart_render"):
        result = mermaid_renders("this is not valid mermaid syntax at all {{{")
    assert result is False
    assert len(caplog.records) == 0

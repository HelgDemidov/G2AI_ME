"""``ChartData`` -> markdown (spec chart-data-extraction §3): GFM-таблица
ВСЕГДА (лосслесс), mermaid ДОПОЛНИТЕЛЬНО — только для типов, у которых
объективно ЕСТЬ mermaid-конструкт (предел вокабуляра самого mermaid, не
осторожность): ``pie``/``doughnut``->``pie``, ``column``/``bar``/``line``/
``area``/``combo``->``xychart-beta`` (bar+line combo = overlay двух серий
разного kind), ``radar``->``radar-beta``. Всё прочее (``scatter``, stacked-бар,
bubble/waterfall/treemap/sunburst/boxplot/3D/лог-шкалы) -> mermaid-конструкта
НЕТ вовсе, только таблица. ``verify+fallback``: форма рискованна (несовпадение
длин серии/категорий, пропуски, >1 серия у pie) -> mermaid снимается, таблица
остаётся всегда (см. Design rationale спека).

Двухуровневая верификация mermaid перед выдачей (решение пользователя
2026-07-22, эволюция того же verify+fallback): структурные эвристики выше
ловят рискованную ФОРМУ, но не гарантируют, что mermaid.js реально примет
СИНТАКСИС/СЕМАНТИКУ результата — живой пример: кавычки в ``pie title``
синтаксически валидны, но рендерятся буквально. Финальный гейт —
``_mermaid_renders``: настоящий рендер через ``mermaidx`` (embedded QuickJS,
runtime-зависимость с 2026-07-22), любой отказ -> откат к таблице-only, НЕ
крах конвертации."""
from __future__ import annotations

import re
from typing import cast

from convert.chart_data import ChartData

_STRIP_RE = re.compile(r"[\[\]{}()]")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")

_PIE_LIKE = frozenset({"pie", "doughnut"})
_XYCHART_LIKE = frozenset({"column", "bar", "line", "area", "combo"})


def _sanitize_label(text: str) -> str:
    """Живой дефект convert-xlsx (VLM-эра): скобки/кавычки в лейбле ломают
    mermaid-парсер — вырезаем ``[]{}()``, двойные кавычки заменяем на
    одинарные (лейблы всегда в двойных кавычках в выводе)."""
    cleaned = _STRIP_RE.sub("", text).replace('"', "'").strip()
    return cleaned or "?"


def _slug(text: str, fallback_idx: int) -> str:
    """Голое слово без пробелов/пунктуации — id узла radar-beta (``curve
    id[...]``): mermaid требует именно такой id ПЕРЕД лейблом."""
    s = _SLUG_RE.sub("_", text).strip("_").lower()
    if not s or not s[0].isalpha():
        s = f"s{fallback_idx}{s}"
    return s


def _fmt_num(v: float) -> str:
    return f"{v:.4g}"


def _count_decimal_digits(fmt: str) -> int:
    if "." not in fmt:
        return 0
    frac = re.split(r"[;\s]", fmt.split(".", 1)[1])[0]
    return sum(1 for ch in frac if ch in "0#")


def _format_value(value: float | None, value_format: str | None) -> str:
    """Прагматичное форматирование (§3): ``0.0%``->×100+«%», число десятичных
    из кода; неизвестный/отсутствующий формат -> округление ~4 знач. цифры
    (НЕ сырой float — живой дефект govtech: ``0.58909698401216537`` без
    формата нечитаем)."""
    if value is None:
        return ""
    if value_format:
        if "%" in value_format:
            decimals = _count_decimal_digits(value_format.split("%", 1)[0])
            return f"{value * 100:.{decimals}f}%"
        if re.search(r"[0#]", value_format):
            decimals = _count_decimal_digits(value_format)
            return f"{value:,.{decimals}f}"
    return _fmt_num(value)


def _caption(data: ChartData) -> str | None:
    parts = [p for p in (data.title, data.value_axis_title) if p]
    return " — ".join(parts) if parts else None


def _row_labels(data: ChartData) -> tuple[str, ...]:
    if data.categories:
        return data.categories
    max_len = max((len(s.values) for s in data.series), default=0)
    return tuple(str(i + 1) for i in range(max_len))


def _table(data: ChartData) -> str:
    header = ["Category"] + [s.name or f"Series {i + 1}" for i, s in enumerate(data.series)]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for i, label in enumerate(_row_labels(data)):
        row = [label.replace("\n", " ")] + [
            _format_value(s.values[i] if i < len(s.values) else None, data.value_format)
            for s in data.series
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _series_shape_ok(data: ChartData) -> bool:
    """verify-часть правила verify+fallback-to-table: mermaid не умеет
    представить пропуск/несовпадение длины — таблица переживает это честно
    (пустая ячейка), mermaid просто снимается целиком."""
    if not data.categories:
        return False
    return all(len(s.values) == len(data.categories) and all(v is not None for v in s.values) for s in data.series)


def _dense(values: tuple[float | None, ...]) -> tuple[float, ...]:
    """``values`` только ПОСЛЕ ``_series_shape_ok`` подтвердил отсутствие
    ``None`` — узкий cast вместо повторной рантайм-проверки/фильтрации
    (фильтрация читалась бы как «отбросить None», а инвариант уже — «их нет»)."""
    return cast(tuple[float, ...], values)


def _mermaid_pie(data: ChartData) -> str | None:
    if len(data.series) != 1 or not _series_shape_ok(data):
        return None
    values = _dense(data.series[0].values)
    if any(v < 0 for v in values):
        return None  # доля не может быть отрицательной — форма рискованна
    # "pie title <text>" — ПЛОСКАЯ строка, БЕЗ кавычек (в отличие от data-
    # лейблов ниже, которые mermaid требует в кавычках): найдено на реальном
    # рендере govtech-фикстуры (пользователь попросил визуально проверить
    # результат) — кавычки, обёрнутые вокруг title, mermaid НЕ интерпретирует
    # как delimiter, а рендерит буквально (хвостовая `"` была видна в SVG).
    # mermaid-parser-bundle/mermaid.parse() эту форму пропускали как валидную
    # (кавычки внутри плоской строки — не грамматическая ошибка), поэтому
    # синтакс-валидация её не поймала — только визуальный рендер.
    lines = [f"pie title {_sanitize_label(data.title)}"] if data.title else ["pie"]
    for cat, v in zip(data.categories, values, strict=True):
        lines.append(f'    "{_sanitize_label(cat)}" : {_fmt_num(v)}')
    return "```mermaid\n" + "\n".join(lines) + "\n```"


def _mermaid_xychart(data: ChartData) -> str | None:
    if data.stacked or not _series_shape_ok(data):
        return None
    lines = ["xychart-beta"]
    x_axis = ", ".join(f'"{_sanitize_label(c)}"' for c in data.categories)
    lines.append(f"x-axis [{x_axis}]")
    y_label = _sanitize_label(data.value_axis_title) if data.value_axis_title else "Value"
    all_values = [v for s in data.series for v in _dense(s.values)]
    y_min, y_max = min(0.0, min(all_values)), max(all_values)
    if y_min == y_max:
        y_max = y_min + 1  # xychart-beta требует различимый диапазон оси
    lines.append(f'y-axis "{y_label}" {_fmt_num(y_min)} --> {_fmt_num(y_max)}')
    for s in data.series:
        kind = "bar" if s.kind in ("bar", "column") else "line"
        values_str = ", ".join(_fmt_num(v) for v in _dense(s.values))
        lines.append(f"{kind} [{values_str}]")
    return "```mermaid\n" + "\n".join(lines) + "\n```"


def _mermaid_radar(data: ChartData) -> str | None:
    if not _series_shape_ok(data):
        return None
    axis = ", ".join(f'{_slug(cat, i)}["{_sanitize_label(cat)}"]' for i, cat in enumerate(data.categories))
    lines = [f"axis {axis}"]
    for i, s in enumerate(data.series):
        label = _sanitize_label(s.name) if s.name else f"Series {i + 1}"
        sid = _slug(s.name or f"series{i}", i)
        values_str = ", ".join(_fmt_num(v) for v in _dense(s.values))
        lines.append(f'curve {sid}["{label}"]{{{values_str}}}')
    return "```mermaid\nradar-beta\n" + "\n".join(lines) + "\n```"


def _mermaid_renders(code: str) -> bool:
    """Настоящая render-проверка через ``mermaidx`` (spec chart-data-extraction,
    решение пользователя 2026-07-22 — сначала dev-тест, затем штатный
    runtime-гейт): структурные эвристики выше (``_series_shape_ok`` и т.п.)
    ловят РИСКОВАННУЮ форму, но не гарантируют, что mermaid.js реально
    примет результат — живой пример (найден на визуальном рендере govtech,
    коммит `509f6ff`): ``pie title "T"`` синтаксически валиден (кавычки
    внутри плоской строки — не ошибка грамматики), но рендерится с
    буквальными кавычками. Синтакс-валидаторы (``mermaid-parser-bundle``/
    ``mermaid.parse()``) эту форму пропускали — только фактический рендер
    различает «грамматически валидно» и «примет реальный рендерер». Любой
    сбой (включая недоступность самой библиотеки) -> False, фейл-safe —
    вызывающая сторона просто теряет mermaid-блок, НЕ конвертацию целиком."""
    import mermaidx  # ленивый импорт: pdf/html-путь не платит за chart-специфичный вес

    try:
        mermaidx.render(code).svg()
        return True
    except Exception:  # noqa: BLE001 — любой отказ реального рендера -> честный откат к таблице
        return False


def _mermaid(data: ChartData) -> str | None:
    if data.chart_type in _PIE_LIKE:
        candidate = _mermaid_pie(data)
    elif data.chart_type in _XYCHART_LIKE:
        candidate = _mermaid_xychart(data)
    elif data.chart_type == "radar":
        candidate = _mermaid_radar(data)
    else:
        return None  # scatter/stacked-bar/прочее — mermaid-конструкта нет вовсе
    if candidate is None:
        return None
    code = candidate.removeprefix("```mermaid\n").removesuffix("\n```")
    return candidate if _mermaid_renders(code) else None


def render_chart(data: ChartData) -> str | None:
    """None -> извлечение пустое (нет серий/значений), вызывающая сторона
    зовёт caption-фолбэк (честный маркер, см. ``xlsx_charts.render_chart_marker``/
    ``docx_groups._render_group_marker``). Порядок вывода (§3): подпись ->
    mermaid (если есть) -> таблица."""
    if not data.series or not any(any(v is not None for v in s.values) for s in data.series):
        return None
    parts = [p for p in (_caption(data), _mermaid(data), _table(data)) if p]
    return "\n\n".join(parts)

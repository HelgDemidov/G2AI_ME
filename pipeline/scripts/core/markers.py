"""Грамматика машинных маркеров в теле ``doc.md`` — общая для производителя и читателей.

Маркер VLM-инъекции СТАВИТ слой convert (``figures_vlm``), а ЧИТАЮТ его слои выше по
конвейеру (``index.chunking`` — chunk-провенанс, spec convert-knowledge-seam-hardening
§2). Оставить грамматику у производителя значило бы, что ``index`` импортирует
``convert`` — инверсия слоёв ровно того класса, что аудит нашёл в ``convert.lint``
(Б11) и knowledge-hardening в ``graph``→``discovery`` (А14). Держать по копии у каждого
— класс «строка в N копиях» (Б5, четырежды подтверждённый в проекте). Дом — ``core``,
ниже обоих слоёв.

Здесь же живут ОБЕ стороны каждой пары (рендер + распознавание), поэтому round-trip
producer↔consumer верен по построению, а не по договорённости.
"""
from __future__ import annotations

import re

INJECTION_END_PREFIX = "> [/VLM interpretation "
"""Префикс терминатора блока VLM-инъекции. Терминатор существует, потому что без него
граница «здесь кончилась машинная реконструкция и возобновился verbatim-текст издателя»
не определима ничем: чанковка пакует абзацы независимо, и хвост реконструкции уезжал в
чанк без маркера провенанса (аудит шва Б1 — 16 из 74 чанков корпуса живьём)."""

_INJECTION_OPEN_RE = re.compile(
    r"^> \[.+ — VLM interpretation \(.+\); reconstruction, verify against original\]$"
)
_INJECTION_END_RE = re.compile(r"^> \[/VLM interpretation .+\]$")


def injection_open(head: str, model: str) -> str:
    """Открывающий маркер блока. ``head`` — адрес объекта для человека
    («Figure, p. 6, region abc123def456»)."""
    return f"> [{head} — VLM interpretation ({model}); reconstruction, verify against original]"


def injection_end(address: str) -> str:
    """Терминатор блока. ``address`` — тот же адрес без класса объекта
    («region abc123def456»), чтобы соседние блоки были сопоставимы глазами."""
    return f"{INJECTION_END_PREFIX}{address}]"


def is_injection_open(line: str) -> bool:
    return _INJECTION_OPEN_RE.match(line.strip()) is not None


def is_injection_end(line: str) -> bool:
    return _INJECTION_END_RE.match(line.strip()) is not None

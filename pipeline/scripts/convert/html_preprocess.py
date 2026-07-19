"""Реестр HTML-препроцессоров raw.html -> raw.html (Strategy, spec convert-hardening B1).

Обобщение eli.py: следующий национальный legal-портал (Балканы — ядро фокуса)
регистрируется здесь одной строкой, не правкой ``converters._convert_html``.
Применяется ПЕРВЫЙ сматчившийся препроцессор (не цепочка) — один портал = один
диалект разметки; конфликт диалектов не наш кейс, а порядок в списке — детерминизм.
"""
from __future__ import annotations

from collections.abc import Callable

from convert import eli

Matcher = Callable[[bytes], bool]
Transform = Callable[[bytes], bytes]

_PREPROCESSORS: list[tuple[str, Matcher, Transform]] = [
    ("eli", eli.matches, eli.promote_eli_headings),
]


def apply(html: bytes) -> bytes:
    """Первый сматчившийся препроцессор преобразует HTML; иначе — байт-в-байт как есть."""
    for _name, matcher, transform in _PREPROCESSORS:
        if matcher(html):
            return transform(html)
    return html

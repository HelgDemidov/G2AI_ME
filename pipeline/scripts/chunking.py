"""Разбиение текста документа на канонические чанки ~512 токенов для поиска.

Чанки КАНОНИЧНЫ: одни и те же чанки индексируются и в FTS5, и в векторном слое,
поэтому попадание по ключевому слову и семантическое попадание ссылаются на один chunk.

Логика чанковки НЕ зависит от конкретного токенизатора — функция подсчёта токенов
инжектируется (``count_tokens``). В рантайме передаётся токенизатор bge-m3
(см. bge_tokenizer.py); в тестах — простой счётчик слов. Так логику можно
проверять в CI без модели.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

TokenCounter = Callable[[str], int]

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?;])\s+|\n")


@dataclass(frozen=True)
class Chunk:
    """Канонический чанк: принадлежит документу doc_id, порядковый index."""

    doc_id: str
    index: int
    text: str
    n_tokens: int


def strip_frontmatter(md: str) -> str:
    """Убрать YAML-frontmatter в начале .md (если он есть)."""
    return _FRONTMATTER_RE.sub("", md, count=1)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_RE.split(text) if p.strip()]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _hard_split(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Жёсткая нарезка по словам для аномально длинных предложений (типовой выход
    pdf_to_markdown на дампах диаграмм/таблиц без пунктуации).

    Бюджет накапливается ДЕШЁВОЙ суммой per-word ``count_tokens(word)`` — O(1)
    работы на слово, а не повторной токенизацией всей растущей строки на каждом
    добавленном слове (O(n²) символов: предложение на 5000 слов давало ~12М
    токенизированных "слово-эквивалентов" вместо 5К). Когда оценка превышает
    ``max_tokens`` — ОДНА проверка ``count_tokens`` на полном кандидате; если она
    тоже превышает (subword-склейка на границах слов может дать больше суммы по
    словам, чем реальный совместный счёт) — бинарный поиск точки разреза (O(log n)
    энкодов). Свойство «чанк <= max_tokens» — точное (финальная верификация),
    меняется только СТОИМОСТЬ его достижения.
    """
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    i, n = 0, len(words)
    while i < n:
        j, budget = i, 0
        while j < n:
            w_tokens = count_tokens(words[j])
            if budget + w_tokens > max_tokens and j > i:
                break
            budget += w_tokens
            j += 1
        candidate = " ".join(words[i:j])
        if j == i + 1 or count_tokens(candidate) <= max_tokens:
            # одно слово (дальше резать некуда, даже если оно само больше лимита)
            # либо честная проверка подтвердила оценку — принимаем как есть
            out.append(candidate)
            i = j
            continue
        # оценка соврала (subword-склейка) -> бинарный поиск точки разреза внутри [i, j)
        lo, hi, best = i + 1, j, i + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(" ".join(words[i:mid])) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        out.append(" ".join(words[i:best]))
        i = best
    return out


def _split_long_paragraph(para: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Абзац больше лимита -> нарезать по предложениям (с fallback на слова)."""
    out: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in _sentences(para):
        n = count_tokens(sent)
        if n > max_tokens:
            if current:
                out.append(" ".join(current))
                current, current_tokens = [], 0
            out.extend(_hard_split(sent, count_tokens, max_tokens))
            continue
        if current and current_tokens + n > max_tokens:
            out.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += n
    if current:
        out.append(" ".join(current))
    return out


def chunk_text(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int = 512,
    doc_id: str = "",
) -> list[Chunk]:
    """Разбить текст на чанки <= max_tokens, стараясь не резать абзацы/предложения."""
    raw: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            raw.append("\n\n".join(current))
            current, current_tokens = [], 0

    for para in _paragraphs(text):
        n = count_tokens(para)
        if n > max_tokens:
            flush()
            raw.extend(_split_long_paragraph(para, count_tokens, max_tokens))
            continue
        if current and current_tokens + n > max_tokens:
            flush()
        current.append(para)
        current_tokens += n
    flush()

    # count_tokens(chunk) здесь пересчитывает готовый текст ЗАНОВО, хотя суммы уже
    # накапливались по пути (current_tokens/_split_long_paragraph) — избыточно, но
    # НЕ квадратично (один линейный проход по уже собранным чанкам, не по n²
    # растущих префиксов, как было в _hard_split). Оставлено как есть: чтобы нести
    # накопленное значение через flush()/_split_long_paragraph/_hard_split, все три
    # должны были бы возвращать (text, n_tokens) вместо str — не стоит сложности
    # ради устранения уже-линейной работы (см. spec code-consolidation §5).
    return [Chunk(doc_id, i, chunk, count_tokens(chunk)) for i, chunk in enumerate(raw)]

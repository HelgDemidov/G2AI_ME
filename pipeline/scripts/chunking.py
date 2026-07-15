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
    """Жёсткая нарезка по словам для аномально длинных предложений."""
    out: list[str] = []
    current: list[str] = []
    for word in text.split():
        current.append(word)
        if count_tokens(" ".join(current)) > max_tokens:
            current.pop()
            if current:
                out.append(" ".join(current))
            current = [word]
    if current:
        out.append(" ".join(current))
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

    return [Chunk(doc_id, i, chunk, count_tokens(chunk)) for i, chunk in enumerate(raw)]

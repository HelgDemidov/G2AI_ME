"""Тесты логики чанковки (счётчик токенов = число слов, без модели — CI-safe)."""
from __future__ import annotations

from typing import Any

from chunking import _hard_split, chunk_text, strip_frontmatter


def wc(text: str) -> int:
    """Стаб-счётчик токенов: число слов."""
    return len(text.split())


def test_strip_frontmatter() -> None:
    md = "---\ntitle: X\nid: y-z\n---\n\nBody text here."
    out = strip_frontmatter(md)
    assert "Body text here." in out
    assert "title:" not in out


def test_empty_text() -> None:
    assert chunk_text("", wc) == []


def test_single_short_chunk() -> None:
    chunks = chunk_text("hello world foo", wc, max_tokens=10, doc_id="d")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world foo"
    assert chunks[0].n_tokens == 3
    assert chunks[0].doc_id == "d"
    assert chunks[0].index == 0


def test_paragraph_packing_within_max() -> None:
    text = "\n\n".join("word " * 5 for _ in range(4))  # 4 абзаца по 5 слов
    chunks = chunk_text(text.strip(), wc, max_tokens=12)
    assert all(c.n_tokens <= 12 for c in chunks)
    assert len(chunks) == 2  # 5+5=10<=12, третий не влезает -> 2 чанка по 10


def test_long_paragraph_split_by_sentences() -> None:
    para = "one two three. four five six. seven eight nine."
    chunks = chunk_text(para, wc, max_tokens=4)
    assert len(chunks) == 3
    assert all(c.n_tokens <= 4 for c in chunks)


def test_monster_sentence_hard_split_no_loss() -> None:
    text = ("w " * 20).strip()  # одно «предложение» из 20 слов, без границ
    chunks = chunk_text(text, wc, max_tokens=5)
    assert all(c.n_tokens <= 5 for c in chunks)
    assert sum(c.n_tokens for c in chunks) == 20  # ничего не потеряно


def test_no_word_loss() -> None:
    text = "alpha beta.\n\ngamma delta epsilon.\n\nzeta"
    chunks = chunk_text(text, wc, max_tokens=3)
    joined = " ".join(c.text for c in chunks)
    assert set(joined.split()) == set(text.split())


def test_indices_are_sequential() -> None:
    text = "\n\n".join("word " * 5 for _ in range(6))
    chunks = chunk_text(text.strip(), wc, max_tokens=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))


# --- _hard_split: перф (O(n), не O(n²)) и корректность при "лгущей" per-word оценке ---


def _counting(counter: Any) -> tuple[Any, dict[str, int]]:
    calls = {"n": 0}

    def wrapped(text: str) -> int:
        calls["n"] += 1
        return counter(text)

    return wrapped, calls


def test_hard_split_call_count_is_linear_not_quadratic() -> None:
    """N слов без пунктуации/границ — раньше O(n) ВЫЗОВОВ, каждый над РАСТУЩЕЙ
    строкой (O(n²) символов суммарно). Проверяем, что число ВЫЗОВОВ count_tokens
    растёт линейно с n (удвоение n даёт кратно меньше чем 4x вызовов — сигнатура
    квадратичного алгоритма), не что сама эта метрика была квадратичной раньше
    (звонков и раньше было O(n) — квадратичной была СТОИМОСТЬ каждого)."""
    counted_wc, calls_small = _counting(wc)
    _hard_split(("w " * 100).strip(), counted_wc, max_tokens=5)
    n_small = calls_small["n"]

    counted_wc2, calls_big = _counting(wc)
    _hard_split(("w " * 1000).strip(), counted_wc2, max_tokens=5)
    n_big = calls_big["n"]

    # 10x слов -> не более ~15x вызовов (линейный рост + небольшой оверхед на
    # верификацию по группам); квадратичный рост дал бы ~100x.
    assert n_big < n_small * 15


def test_hard_split_respects_max_tokens_when_word_sum_undercounts() -> None:
    """Синтетический счётчик с "накладными расходами на склейку" (как у реального
    subword-токенизатора: объединённая строка может стоить больше суммы по словам)
    — оценка по сумме слов способна ПРИНЯТЬ на одно слово больше, чем реально
    влезает; бинарный поиск обязан это поймать и не нарушить лимит."""

    def joiny(text: str) -> int:
        words = text.split()
        return len(words) + (1 if len(words) > 1 else 0)  # +1 "токен склейки" при join

    chunks = _hard_split("a b c d e f g h", joiny, max_tokens=3)
    assert all(joiny(c) <= 3 for c in chunks)
    assert " ".join(chunks).split() == "a b c d e f g h".split()  # ничего не потеряно


def test_hard_split_single_oversized_word_kept_alone() -> None:
    """Одно "слово" крупнее лимита — не может быть разрезано дальше, остаётся
    собственным чанком (как и в старой реализации)."""
    chunks = _hard_split("normal aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa normal", wc, max_tokens=1)
    assert chunks == ["normal", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "normal"]


def test_hard_split_empty_text() -> None:
    assert _hard_split("", wc, max_tokens=5) == []
    assert _hard_split("   ", wc, max_tokens=5) == []


def test_hard_split_matches_old_behavior_on_monster_sentence() -> None:
    """Golden-сравнение с прежним O(n²) алгоритмом (wc — точный аддитивный счётчик,
    подмена алгоритма не должна дать другую группировку)."""
    text = ("w " * 37).strip()  # не кратно max_tokens — есть неполный последний чанк
    chunks = _hard_split(text, wc, max_tokens=5)
    assert [c.count(" ") + 1 for c in chunks] == [5, 5, 5, 5, 5, 5, 5, 2]
    assert sum(wc(c) for c in chunks) == 37

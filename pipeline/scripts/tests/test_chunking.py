"""Тесты логики чанковки (счётчик токенов = число слов, без модели — CI-safe)."""
from __future__ import annotations

from chunking import chunk_text, strip_frontmatter


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

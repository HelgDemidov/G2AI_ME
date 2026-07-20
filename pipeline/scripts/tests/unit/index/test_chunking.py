"""Тесты логики чанковки (счётчик токенов = число слов, без модели — CI-safe)."""
from __future__ import annotations

from collections.abc import Callable

from index.chunking import (
    _hard_split,
    _split_table_paragraph,
    _table_header,
    chunk_text,
    embed_input,
    strip_frontmatter,
)


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


def _counting(counter: Callable[[str], int]) -> tuple[Callable[[str], int], dict[str, int]]:
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


# --- breadcrumb-секционирование (spec analyze-retrieval §1) ---


def test_no_headings_single_section_empty_breadcrumb() -> None:
    text = "\n\n".join("word " * 5 for _ in range(3))
    chunks = chunk_text(text.strip(), wc, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].breadcrumb == ""


def test_single_h1_heading_sets_breadcrumb() -> None:
    text = "# Introduction\n\nsome body text here"
    chunks = chunk_text(text, wc, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].breadcrumb == "Introduction"


def test_heading_line_retained_in_chunk_text() -> None:
    text = "# Introduction\n\nbody text"
    chunks = chunk_text(text, wc, max_tokens=100)
    assert "# Introduction" in chunks[0].text
    assert "body text" in chunks[0].text


def test_nested_headings_build_breadcrumb_stack() -> None:
    text = (
        "# H1\n\npara one\n\n"
        "## H2a\n\npara two\n\n"
        "### H3\n\npara three\n\n"
        "## H2b\n\npara four"
    )
    chunks = chunk_text(text, wc, max_tokens=100)
    breadcrumbs = [c.breadcrumb for c in chunks]
    assert breadcrumbs == ["H1", "H1 › H2a", "H1 › H2a › H3", "H1 › H2b"]


def test_section_change_flushes_chunk_even_under_budget() -> None:
    """Два коротких раздела с большим max_tokens всё равно дают ДВА чанка — секция
    не пересекается packing-логикой, даже если оба текста уместились бы в один."""
    text = "# A\n\nshort\n\n# B\n\nshort too"
    chunks = chunk_text(text, wc, max_tokens=1000)
    assert len(chunks) == 2
    assert chunks[0].breadcrumb == "A"
    assert chunks[1].breadcrumb == "B"


def test_indices_sequential_across_sections() -> None:
    text = "# A\n\n" + "\n\n".join("word " * 5 for _ in range(3)) + "\n\n# B\n\nword " * 5
    chunks = chunk_text(text, wc, max_tokens=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_pre_heading_text_has_empty_breadcrumb() -> None:
    text = "intro before any heading\n\n# First Heading\n\nbody"
    chunks = chunk_text(text, wc, max_tokens=100)
    assert chunks[0].breadcrumb == ""
    assert chunks[1].breadcrumb == "First Heading"


def test_sibling_heading_resets_deeper_level() -> None:
    """Заголовок того же уровня сбрасывает более глубокие уровни в стеке (не
    накапливает H3 из предыдущей ветки)."""
    text = "## H2a\n\n### H3\n\npara\n\n## H2b\n\npara"
    chunks = chunk_text(text, wc, max_tokens=100)
    assert chunks[-1].breadcrumb == "H2b"


# --- embed_input (spec analyze-retrieval §1.3) ---


def test_embed_input_prefixes_breadcrumb() -> None:
    assert embed_input("H1 › H2", "body text") == "H1 › H2\nbody text"


def test_embed_input_no_breadcrumb_returns_text_unchanged() -> None:
    assert embed_input("", "body text") == "body text"


# --- фенс-осознанность (_paragraphs): mermaid/код-блоки не рвутся чанковкой
# (живой дефект приёмки convert-cloud-tier чекпоинт 2: VLM-mermaid sg p.6) ---


def _fences_balanced(text: str) -> bool:
    return len([ln for ln in text.splitlines() if ln.strip().startswith("```")]) % 2 == 0


def test_fence_with_blank_lines_stays_in_one_chunk() -> None:
    """Пустые строки ВНУТРИ фенса — ровно живой паттерн VLM-вывода (mermaid с
    пустой строкой между секциями графа): раньше рвали блок на «абзацы»."""
    mermaid = "```mermaid\ngraph TD\n    A[\"X\"] --> B[\"Y\"]\n\n    subgraph S\n    A\n    end\n```"
    text = f"# H\n\nprose before\n\n{mermaid}\n\nprose after"
    chunks = chunk_text(text, wc, max_tokens=100)
    carriers = [c for c in chunks if "```mermaid" in c.text]
    assert len(carriers) == 1
    assert mermaid in carriers[0].text  # блок цел, байт-в-байт


def test_fence_never_split_across_chunks_under_packing_pressure() -> None:
    """Бюджет мал (фенс + проза не влезают вместе) — фенс уходит В СВОЙ чанк
    целиком, а не режется по границе бюджета."""
    mermaid = "```mermaid\ngraph LR\n    " + "\n\n    ".join(f'N{i}["node {i}"]' for i in range(6)) + "\n```"
    prose = "word " * 30
    text = f"{prose}\n\n{mermaid}\n\n{prose}"
    chunks = chunk_text(text, wc, max_tokens=40)
    assert all(_fences_balanced(c.text) for c in chunks)
    assert sum("```mermaid" in c.text for c in chunks) == 1


def test_heading_like_line_inside_fence_not_a_section_boundary() -> None:
    """`# comment` внутри код-фенса — не markdown-заголовок: не должен рвать
    секцию и попадать в breadcrumb."""
    text = "## Real\n\n```\n# not a heading\ncode line\n```\n\ntail prose"
    chunks = chunk_text(text, wc, max_tokens=100)
    assert all(c.breadcrumb == "Real" for c in chunks)
    assert not any("not a heading" in c.breadcrumb for c in chunks)


def test_unclosed_fence_consumed_to_end_without_crash() -> None:
    text = "prose\n\n```mermaid\ngraph TD\n    A --> B"
    chunks = chunk_text(text, wc, max_tokens=100)
    assert chunks  # не упало; хвост не потерян
    assert any("A --> B" in c.text for c in chunks)


def test_oversized_fence_degrades_to_split_not_crash() -> None:
    """Фенс крупнее max_tokens — честная деградация в нарезку (граница
    возможностей задокументирована в _paragraphs), не бесконечный чанк."""
    big = "```mermaid\n" + "\n".join(f'X{i}["{i}"] --> Y{i}["{i}"]' for i in range(50)) + "\n```"
    chunks = chunk_text(big, wc, max_tokens=20)
    assert all(c.n_tokens <= 20 for c in chunks)


# --- таблично-осознанная разрезка (_table_header/_split_table_paragraph):
# оверсайз GFM-таблица не должна ломать синтаксис при разрезке (живой риск:
# _split_long_paragraph резал таблицу «по предложениям» через bare \n, затем
# склеивал результат ЧЕРЕЗ ПРОБЕЛ — строки таблицы съезжались в одну без
# переносов). Реальная форма проверена на doc.md корпуса (PDF Эстонии,
# DOCX-конвертер) — обе дают header+separator+rows. ---


def test_table_header_detects_valid_gfm_table() -> None:
    para = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    result = _table_header(para)
    assert result == ("| a | b |", "| --- | --- |", ["| 1 | 2 |", "| 3 | 4 |"])


def test_table_header_returns_none_for_plain_prose() -> None:
    assert _table_header("one two three.\nfour five six.\nseven eight nine.") is None


def test_table_header_returns_none_for_too_few_lines() -> None:
    assert _table_header("| a | b |\n| --- | --- |") is None  # нет строк данных
    assert _table_header("| a | b |") is None


def test_table_header_returns_none_when_second_line_not_separator() -> None:
    """Первая строка похожа на таблицу, но вторая — не разделитель: не таблица
    (иначе ложное срабатывание на прозе, случайно начинающейся с '|')."""
    para = "| not a real table |\njust some text here\nmore text"
    assert _table_header(para) is None


def test_split_table_paragraph_repeats_header_each_chunk() -> None:
    header, sep, rows = "| a | b |", "| --- | --- |", ["| r1 |", "| r2 |", "| r3 |"]
    chunks = _split_table_paragraph(header, sep, rows, wc, max_tokens=6)
    assert len(chunks) > 1
    assert all(c.startswith(f"{header}\n{sep}\n") for c in chunks)


def test_split_table_paragraph_joins_with_newline_not_space() -> None:
    """Регресс-guard на сам баг: результат должен парситься построчно как
    таблица, НЕ содержать строк вида 'header sep row1 row2' через пробел."""
    header, sep, rows = "| a |", "| --- |", ["| 1 |", "| 2 |", "| 3 |", "| 4 |"]
    chunks = _split_table_paragraph(header, sep, rows, wc, max_tokens=3)
    for chunk in chunks:
        lines = chunk.split("\n")
        assert lines[0] == header
        assert lines[1] == sep
        assert all(ln.startswith("|") for ln in lines[2:])


def test_split_table_paragraph_no_row_loss() -> None:
    header, sep = "| a |", "| --- |"
    rows = [f"| row{i} |" for i in range(20)]
    chunks = _split_table_paragraph(header, sep, rows, wc, max_tokens=5)
    recovered = [ln for c in chunks for ln in c.split("\n")[2:]]
    assert recovered == rows


def test_split_table_paragraph_oversized_single_row_kept_whole() -> None:
    """Одна строка данных крупнее лимита сама по себе — остаётся целой (как
    оверсайз-предложение у _hard_split), не рвётся посреди ячейки."""
    header, sep = "| a |", "| --- |"
    huge_row = "| " + " ".join(f"w{i}" for i in range(10)) + " |"
    chunks = _split_table_paragraph(header, sep, [huge_row], wc, max_tokens=3)
    assert len(chunks) == 1
    assert huge_row in chunks[0]


def test_split_table_paragraph_no_data_rows_returns_header_only() -> None:
    chunks = _split_table_paragraph("| a |", "| --- |", [], wc, max_tokens=10)
    assert chunks == ["| a |\n| --- |"]


def test_oversized_table_splits_into_valid_gfm_chunks_via_chunk_text() -> None:
    """Интеграционный путь: большая таблица через chunk_text (не напрямую
    _split_table_paragraph) — упирается в max_tokens, режется, каждый чанк —
    самостоятельно валидная GFM-таблица."""
    header = "| id | value |"
    sep = "| --- | --- |"
    rows = "\n".join(f"| {i} | value-{i} |" for i in range(30))
    text = f"# Table\n\n{header}\n{sep}\n{rows}"
    chunks = chunk_text(text, wc, max_tokens=15)
    table_chunks = [c for c in chunks if c.text.startswith(header)]
    assert len(table_chunks) > 1
    for c in table_chunks:
        lines = c.text.split("\n")
        assert lines[0] == header
        assert lines[1] == sep
        assert all(ln.startswith("|") and ln.endswith("|") for ln in lines[2:])


def test_oversized_table_no_row_loss_via_chunk_text() -> None:
    header, sep = "| id |", "| --- |"
    rows = [f"| {i} |" for i in range(25)]
    text = f"{header}\n{sep}\n" + "\n".join(rows)
    chunks = chunk_text(text, wc, max_tokens=8)
    recovered = [ln for c in chunks for ln in c.text.split("\n")[2:] if c.text.startswith(header)]
    assert recovered == rows


def test_short_table_stays_intact_as_single_chunk() -> None:
    """Таблица, уместившаяся в бюджет, НЕ проходит через табличный сплиттер
    (packing-путь как для любого короткого абзаца) — регресс-guard, что фикс
    не трогает уже рабочий случай."""
    text = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    chunks = chunk_text(text, wc, max_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].text == text

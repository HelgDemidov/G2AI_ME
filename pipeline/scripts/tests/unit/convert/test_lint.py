"""Тесты convert/lint.py: C1 авто-QA (spec convert-hardening) — чистые функции,
без сети/модели/pdfplumber."""
from __future__ import annotations

from convert.lint import lint_conversion, numeric_delta, token_recall, witness_checks


def test_no_headings_flagged() -> None:
    defects = lint_conversion("Just plain prose, no markdown headings anywhere.", raw_text_chars=None, fmt="pdf")
    assert "no-headings" in defects


def test_headings_present_no_defect() -> None:
    defects = lint_conversion("# Title\n\nSome body text.", raw_text_chars=None, fmt="pdf")
    assert "no-headings" not in defects


def test_text_loss_flagged_when_ratio_below_threshold() -> None:
    md = "# Title\n\nShort."
    defects = lint_conversion(md, raw_text_chars=1000, fmt="pdf")  # md text tiny vs raw
    assert any(d.startswith("text-loss") for d in defects)


def test_text_loss_not_flagged_when_ratio_above_threshold() -> None:
    md = "# Title\n\n" + ("word " * 50)
    defects = lint_conversion(md, raw_text_chars=10, fmt="pdf")  # md text >> raw
    assert not any(d.startswith("text-loss") for d in defects)


def test_text_loss_skipped_when_raw_text_chars_none() -> None:
    """html-путь: raw_text_chars=None -> ratio неинформативен, проверка не выполняется
    вообще, даже если md почти пуст."""
    defects = lint_conversion("# T\n\nx", raw_text_chars=None, fmt="html")
    assert not any(d.startswith("text-loss") for d in defects)


def test_ragged_table_flagged() -> None:
    md = "# Title\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n| only-one-cell |"
    defects = lint_conversion(md, raw_text_chars=None, fmt="pdf")
    assert any(d.startswith("table-ragged") for d in defects)


def test_clean_table_not_flagged() -> None:
    md = "# Title\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    defects = lint_conversion(md, raw_text_chars=None, fmt="pdf")
    assert not any(d.startswith("table-ragged") for d in defects)


def test_clean_document_no_defects() -> None:
    md = "# Title\n\nEnough body prose to comfortably clear the text-loss ratio threshold.\n\n" \
         "| A | B |\n| --- | --- |\n| 1 | 2 |"
    defects = lint_conversion(md, raw_text_chars=10, fmt="pdf")
    assert defects == []


def test_frontmatter_stripped_before_heading_check() -> None:
    """Frontmatter сам начинается с '---', не '#' — не должен создавать ложный
    сигнал «заголовки есть», если тело документа без единого #."""
    md = "---\nid: test-doc\n---\n\nJust prose, no headings in the body."
    defects = lint_conversion(md, raw_text_chars=None, fmt="pdf")
    assert "no-headings" in defects

# --- token_recall / numeric_delta: чистые функции, вынесенные из witness_checks (§4) ---


def test_token_recall_identical_texts_is_one() -> None:
    assert token_recall("Član 1 registar", "Član 1 registar") == 1.0


def test_token_recall_partial_overlap() -> None:
    assert token_recall("alpha beta gamma delta epsilon zeta eta theta iota kappa", "alpha beta") == 0.2


def test_token_recall_reference_without_word_tokens_is_one() -> None:
    """reference из одних чисел/пунктуации — терять нечего, тривиальный recall 1.0."""
    assert token_recall("123 456", "completely different words") == 1.0


def test_token_recall_case_insensitive() -> None:
    assert token_recall("ČLAN Predmet Zakon", "član predmet zakon") == 1.0


def test_token_recall_diacritics_preserved() -> None:
    assert token_recall("vođenje registra Crne Gore", "vođenje registra Crne Gore") == 1.0


def test_numeric_delta_identical_multisets_zero() -> None:
    assert numeric_delta("broj 42 zakona broj 42", "zakona broj 42, opet broj 42") == (0, 0)


def test_numeric_delta_missing_and_added() -> None:
    """8124 (свидетель) распалось на 8/24 (облако) — 1 без пары слева, 2 без пары справа."""
    assert numeric_delta("tač. 8124 ovog zakona", "tač. 8 i 24 ovog zakona") == (1, 2)


def test_numeric_delta_symmetric_swap() -> None:
    missing, added = numeric_delta("8124", "8 24")
    assert (missing, added) == (1, 2)
    added2, missing2 = numeric_delta("8 24", "8124")
    assert (missing2, added2) == (1, 2)  # свап аргументов = свап направления


# --- witness_checks: свидетель (tesseract) vs облачный doc.md (spec convert-cloud-tier §3) ---


def test_witness_identical_texts_no_defects() -> None:
    text = "Član 1. Ovim zakonom uređuje se registracija privrednih subjekata broj 42."
    assert witness_checks(text, text) == []


def test_witness_empty_returns_no_defects() -> None:
    """Свидетель пуст (сбой extract_text) — сигнал неинформативен, не 0.0-recall."""
    assert witness_checks("", "# Cloud Title\n\nBody text.") == []


def test_witness_text_loss_flagged_when_recall_below_threshold() -> None:
    witness = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    cloud = "alpha beta"  # только 2/10 словарных токенов найдены
    defects = witness_checks(witness, cloud)
    assert any(d.startswith("cloud-ocr-text-loss") for d in defects)


def test_witness_text_loss_not_flagged_above_threshold() -> None:
    witness = "alpha beta gamma delta epsilon"
    cloud = "alpha beta gamma delta epsilon zeta"  # 5/5 найдены + облако добавило слово
    defects = witness_checks(witness, cloud)
    assert not any(d.startswith("cloud-ocr-text-loss") for d in defects)


def test_witness_numeric_divergence_reports_divergent_tokens() -> None:
    """Живой класс ошибки чекпоинта 1: tesseract сливает «8 i 24» -> «8124»
    (число «8124» есть только у свидетеля, «8»/«24» — только у облака).
    Формат ocr-eval-harness §8.2: САМИ токены, не только счётчик (боевой
    флаг «-12/+18» без списка не даёт понять, кто прав)."""
    witness = "tač. 8124 ovog zakona"
    cloud = "tač. 8 i 24 ovog zakona"
    defects = witness_checks(witness, cloud)
    assert "cloud-ocr-numeric-divergence: witness_only=[8124] cloud_only=[8,24]" in defects


def test_witness_numeric_divergence_caps_token_list_per_side() -> None:
    """>10 расходящихся чисел на сторону -> список капается, хвост показывает
    остаток (`.state.yaml` не резиновый)."""
    witness = " ".join(str(n) for n in range(1, 13))  # 1..12, все только у свидетеля
    cloud = ""
    defects = witness_checks(witness, cloud)
    (defect,) = [d for d in defects if d.startswith("cloud-ocr-numeric-divergence")]
    assert "witness_only=[1,2,3,4,5,6,7,8,9,10…+2]" in defect
    assert "cloud_only=[none]" in defect


def test_witness_numeric_divergence_absent_when_multisets_equal() -> None:
    assert not any(d.startswith("cloud-ocr-numeric-divergence") for d in witness_checks("broj 42", "broj 42"))


def test_witness_numeric_identical_multisets_no_divergence() -> None:
    """Разный порядок/пунктуация вокруг тех же чисел — мультимножество, не позиция."""
    witness = "broj 42 zakona broj 42"
    cloud = "zakona broj 42, opet broj 42"
    assert witness_checks(witness, cloud) == []


def test_witness_case_insensitive_word_recall() -> None:
    """OCR-регистр не должен создавать ложный text-loss (реальный шум сканов)."""
    witness = "ČLAN Predmet Zakon"
    cloud = "član predmet zakon"
    assert not any(d.startswith("cloud-ocr-text-loss") for d in witness_checks(witness, cloud))


def test_witness_diacritics_preserved_in_tokenization() -> None:
    """Юникодные буквы (đ/č/ž) должны участвовать в токенизации как обычные буквы,
    не рваться в non-word — иначе диакритика тихо искажала бы recall."""
    witness = "vođenje registra Crne Gore"
    cloud = "vođenje registra Crne Gore"
    assert witness_checks(witness, cloud) == []

"""Тесты layout-free восстановления заголовков (ocr_headings.py) — precision-first.

Фикстуры — синтетический «OCR-подобный» плоский текст (не корпусный OCR-скан:
такового ещё нет — см. спек convert-ocr §2.3, precision-first без калибровки)."""
from __future__ import annotations

from convert.ocr_headings import promote_flat_headings

# --- Тир 1: структурные ключевые слова (ANNEX/CHAPTER/TITLE/PART -> #, SECTION/Article/Appendix -> ##) ---


def test_tier1_annex_promoted_to_h1() -> None:
    out = promote_flat_headings("ANNEX I\nSome text.\n")
    assert out.startswith("# ANNEX I\n")


def test_tier1_article_with_title_promoted_to_h2() -> None:
    out = promote_flat_headings("Article 6 Classification rules\nBody text.\n")
    assert out.startswith("## Article 6 Classification rules\n")


def test_tier1_section_promoted_to_h2() -> None:
    out = promote_flat_headings("SECTION 1\nBody text.\n")
    assert out.startswith("## SECTION 1\n")


def test_tier1_guard_rejects_sentence_ending_in_period() -> None:
    """«Article 6 shall apply to…» — тело статьи, не заголовок: trailing-точка гасит
    guard независимо от того, что leading-слово совпадает с ключевым."""
    sentence = "Article 6 shall apply to all providers established in the Union."
    out = promote_flat_headings(f"{sentence}\nNext paragraph.\n")
    assert out.startswith(f"{sentence}\n")
    assert "#" not in out.split("\n")[0]


# --- Тир 2: короткая CAPS-строка + непустое тело следом -> ## ---


def test_tier2_caps_phrase_with_body_promoted_to_h2() -> None:
    out = promote_flat_headings("GENERAL PROVISIONS\nThis chapter establishes...\n")
    assert out.startswith("## GENERAL PROVISIONS\n")


def test_tier2_lone_acronym_not_promoted() -> None:
    """Guard против акронимов: одно слово (даже полностью заглавное) — не заголовок."""
    out = promote_flat_headings("GDPR\nThe regulation defines...\n")
    assert out.startswith("GDPR\n")


def test_tier2_long_caps_line_not_promoted() -> None:
    long_caps = (
        "THIS IS A VERY LONG ALL CAPS LINE DELIBERATELY CONSTRUCTED TO EXCEED THE "
        "HUNDRED CHARACTER CAPS LENGTH LIMIT THAT TIER TWO ENFORCES FOR SAFETY"
    )
    assert len(long_caps) > 100
    out = promote_flat_headings(f"{long_caps}\nBody.\n")
    assert out.startswith(f"{long_caps}\n")


def test_tier2_caps_max_len_raised_to_100_accepts_real_title_length() -> None:
    """Регресс калибровки: реальный заголовок закона/указа на первом скане (2026-07-17)
    был длиной 70 символов и отсекался старым порогом в 60 — ложноотрицательно."""
    title = "UKAZ O PROGLAŠENJU ZAKONA O REGISTRACIJI PRIVREDNIH I DRUGIH SUBJEKATA"
    assert len(title) == 70
    out = promote_flat_headings(f"{title}\nProglašavam Zakon o registraciji...\n")
    assert out.startswith(f"## {title}\n")


def test_tier2_no_following_body_not_promoted() -> None:
    """Guard против обломков: без непустого тела следом — не заголовок."""
    out = promote_flat_headings("GENERAL PROVISIONS\n")
    assert out == "GENERAL PROVISIONS\n"


# --- Тир 3: голая нумерация глубиной <=2 (самый строгий guard) ---


def test_tier3_minor_number_promoted_to_h3() -> None:
    out = promote_flat_headings("1.1 Scope\nThis section defines the scope.\n")
    assert out.startswith("### 1.1 Scope\n")


def test_tier3_major_number_promoted_to_h2() -> None:
    out = promote_flat_headings("1. Definitions\nFor the purposes of this Regulation.\n")
    assert out.startswith("## 1. Definitions\n")


def test_tier3_long_subclause_not_promoted_anti_explosion() -> None:
    """Реальный риск Тир 3: под-клауза статьи с ведущим номером — самый опасный
    ложноположительный случай, ради которого весь тир строится максимально строгим."""
    subclause = "1. The provider shall ensure that the system continuously monitors compliance."
    out = promote_flat_headings(f"{subclause}\nNext clause.\n")
    assert out.startswith(f"{subclause}\n")


def test_tier3_lowercase_title_not_promoted() -> None:
    out = promote_flat_headings("1.1 scope of application\nBody.\n")
    assert out.startswith("1.1 scope of application\n")


def test_tier3_triple_depth_not_matched() -> None:
    """Глубина >2 (1.1.1) вне скоупа Тир 3 — регекс не матчит, строка остаётся телом."""
    out = promote_flat_headings("1.1.1 Sub-point\nBody.\n")
    assert out.startswith("1.1.1 Sub-point\n")


# --- Тир 1 (региональный): ČLAN/CLAN — Черногория/Балканы эквивалент Article ---
# ČLAN подтверждён эмпирически первым реальным сканом (2026-07-17); GLAVA/PRILOG/
# ODJELJAK/DIO добавлены проактивно по общей южнославянской юридической drafting-
# конвенции (Chapter/Annex/Section/Part), см. docstring _TIER1_LEVEL1/2 в модуле.


def test_tier1_clan_ascii_promoted_to_h2() -> None:
    """OCR регулярно роняет диакритику (Član -> Clan) — ASCII-спеллинг тоже ключевое слово."""
    out = promote_flat_headings("Clan 8\nCRPS čine registri.\n")
    assert out.startswith("## Clan 8\n")


def test_tier1_clan_with_diacritic_promoted_to_h2() -> None:
    """Č (U+010C) — Unicode-буква, не ASCII [A-Za-z]: без юникод-regex Тир 1 молча
    не матчился вообще ни на одном «Član»-заголовке (баг, найденный полевым тестом)."""
    out = promote_flat_headings("Član 19\nPostupak registracije pokreće se.\n")
    assert out.startswith("## Član 19\n")


def test_tier1_glava_odjeljak_dio_prilog_recognized() -> None:
    assert promote_flat_headings("GLAVA I\nBody.\n").startswith("# GLAVA I\n")
    assert promote_flat_headings("Odjeljak 1\nBody.\n").startswith("## Odjeljak 1\n")
    assert promote_flat_headings("Dio prvi\nBody.\n").startswith("## Dio prvi\n")
    assert promote_flat_headings("Prilog 1\nBody.\n").startswith("# Prilog 1\n")


# --- Тир 1: деглюинг OCR-расклеенного номера (keyword+цифра [пробел цифра]) ---


def test_degrue_single_digit_glued_to_keyword() -> None:
    """«Član4 CRPS vodi organ...» -> «Član 4 CRPS vodi организ...» (OCR уронил пробел
    перед номером, второй цифры нет — реальный случай из полевого теста, ст. 4)."""
    out = promote_flat_headings("Član4 CRPS vodi organ uprave nadležan za poslove\nBody.\n")
    assert out.startswith("## Član 4 CRPS vodi organ uprave nadležan za poslove\n")


def test_degrue_split_two_digit_number() -> None:
    """«Član3 1 ...» -> «Član 31 ...» (номер статьи 31 распался на 2 отдельных OCR-
    слова — реальный случай из полевого теста, ст. 31)."""
    out = promote_flat_headings("Član3 1\nProtiv rješenja nadležnog organa.\n")
    assert out.startswith("## Član 31\n")


def test_degrue_noop_on_already_clean_heading() -> None:
    """Обычный «Član 19» (пробел уже на месте) — деглюинг не должен ничего менять."""
    out = promote_flat_headings("Član 19\nBody.\n")
    assert out.startswith("## Član 19\n")


# --- Общие инварианты ---


def test_existing_heading_reevaluated_not_blindly_preserved() -> None:
    """Ключевое архитектурное изменение (полевой тест 2026-07-17): существующая
    разметка (символизирует кластеризацию pdf_to_markdown по размеру — ненадёжна на
    OCR bbox) СНИМАЕТСЯ и строка оценивается заново самим ocr_headings, а не слепо
    сохраняется. «# Existing Heading» не матчит ни один тир (не ключевое слово, не
    CAPS, не нумерация) -> становится обычным телом."""
    out = promote_flat_headings("# Existing Heading\nBody.\n")
    assert out == "Existing Heading\nBody.\n"


def test_existing_heading_promoted_at_own_level_not_old_one() -> None:
    """Если снятая с существующей разметки строка САМА матчит тир — уровень берётся
    из ocr_headings (единый источник истины), а не сохраняется старый (демонстрирует
    фикс разнобоя ## vs ### для одного семантического уровня, найденного в реальном
    документе: кластеризация ставила то ##, то ### одному и тому же «Član N»)."""
    out = promote_flat_headings("### Član 8\nCRPS čine registri.\n")
    assert out.startswith("## Član 8\n")  # ocr_headings всегда даёт Тир1-ČLAN уровень ##, не ###


def test_idempotent_double_pass_same_result() -> None:
    text = "ANNEX I\nGENERAL PROVISIONS\n1.1 Scope\nBody text follows.\n"
    once = promote_flat_headings(text)
    twice = promote_flat_headings(once)
    assert once == twice
    assert once == "# ANNEX I\n## GENERAL PROVISIONS\n### 1.1 Scope\nBody text follows.\n"


def test_ordinary_prose_untouched() -> None:
    text = "This is a regular paragraph of body text with no structural markers.\n"
    assert promote_flat_headings(text) == text


def test_empty_string_untouched() -> None:
    assert promote_flat_headings("") == ""


# --- Регресс: реальный markdown-вывод разделяет абзацы ПУСТОЙ строкой (pdf_to_markdown/
# OCR-путь всегда так форматирует, см. golden-документы) — Тир 2/3 должны видеть следующий
# АБЗАЦ, а не буквально следующую строку массива (которая почти всегда пустой разделитель).
# Баг найден живым полевым тестом на реальном скане (Zakon o registraciji, MNE, 2026-07-17):
# каждый Тир 2/3 кандидат имел next_line="" из-за пустой строки-разделителя и не промоутился.


def test_tier2_fires_with_blank_paragraph_separator() -> None:
    text = "I. OSNOVNE ODREDBE\n\nPredmet Clan 1 Ovim zakonom uređuje se nešto.\n"
    out = promote_flat_headings(text)
    assert out.startswith("## I. OSNOVNE ODREDBE\n\n")


def test_tier3_fires_with_blank_paragraph_separator() -> None:
    text = "1.1 Scope\n\nThis section defines the scope of application.\n"
    out = promote_flat_headings(text)
    assert out.startswith("### 1.1 Scope\n\n")


def test_tier2_still_rejects_when_no_body_at_all_after_blanks() -> None:
    """Guard «нет тела следом» должен по-прежнему работать — просто искать нужно
    сквозь пустые строки, а не путать их с отсутствием тела."""
    text = "GENERAL PROVISIONS\n\n\n"
    assert promote_flat_headings(text) == text


def test_idempotent_double_pass_with_blank_separators() -> None:
    """Идемпотентность должна сохраняться и на реалистичном (blank-separated) выводе,
    не только на плотных фикстурах без пустых строк-разделителей."""
    text = "ANNEX I\n\nGENERAL PROVISIONS\n\n1.1 Scope\n\nBody text follows.\n"
    once = promote_flat_headings(text)
    twice = promote_flat_headings(once)
    assert once == twice
    assert once == "# ANNEX I\n\n## GENERAL PROVISIONS\n\n### 1.1 Scope\n\nBody text follows.\n"

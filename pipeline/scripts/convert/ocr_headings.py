"""Layout-free восстановление заголовков из плоского OCR-текста (precision-first).

OCR-нормализация (`_ocr_normalize`) кладёт невидимый текст-слой шрифтом GlyphLessFont,
масштабированным под bounding-box каждого слова — считываемый `pdf_to_markdown` размер
шрифта на таком слое шумная геометрическая подгонка, а не типографский кегль исходника.
Живой полевой тест (Zakon o registraciji, ME, 2026-07-17) подтвердил это эмпирически:
встроенная кластеризация `pdf_to_markdown` по размеру ДЕЙСТВИТЕЛЬНО что-то ловит на OCR-
bbox (не полный шум), но с 1pt-джиттером назначает ОДНОМУ семантическому уровню («Član N»)
то `##`, то `###` — шум geometрии, принятый алгоритмом, калиброванным на чистых цифровых
кеглях, за два разных уровня.

Поэтому этот модуль на OCR-ветке — ЕДИНСТВЕННЫЙ источник истины по заголовкам: любая
существующая разметка (в т.ч. от кластеризации `pdf_to_markdown`) СНИМАЕТСЯ и строка
переоценивается заново по layout-независимым правилам ниже — не два несогласованных
источника, а один. `pdf_to_markdown.py` при этом не трогается ни строкой (цифровой PDF-
путь как был, так и остался на кластеризации — она там калибрована и работает чище).

Precision-first (решение пользователя, спек convert-ocr v2): калибровочного скана в
корпусе ещё не было при проектировании, ложные заголовки (особенно из нумерации под-
клауз закона) дали бы структуру ХУЖЕ честного плоского текста. Три тира по убыванию
точности — при неоднозначности строка остаётся телом, а не промоутится вслепую.
"""
from __future__ import annotations

import re

HEADING_MAX_LEN = 80   # длиннее -> скорее предложение тела, чем заголовок
# Было 60 — калибровка по первому реальному скану (2026-07-17): заголовок закона/указа
# длиной 70 символов отсекался чисто по длине, ложноотрицательно. Поднято с запасом.
CAPS_MAX_LEN = 100      # Тир 2 всё равно строже Тир 1 по остальным guard'ам (см. ниже)

# ANNEX/CHAPTER/TITLE/PART -> #; SECTION/Article/Appendix -> ##.
# CLAN/ČLAN — региональный (Черногория/Балканы) эквивалент Article, ПОДТВЕРЖДЁН эмпирически
# (первый реальный скан, 2026-07-17); GLAVA/PRILOG/ODJELJAK/DIO — та же законодательная
# традиция (Chapter/Annex/Section/Part), добавлены проактивно по общеизвестной конвенции
# южнославянского юридического drafting'а, ЕЩЁ не встречены в живом документе — если
# ошибка (документ таки не использует конкретно этот термин как структурный) — Тир 1
# просто не сработает на нём, ложноположительного риска для НЕ-структурных слов нет
# (сравнение проверяет заголовок листа keyword'ов целиком, не частичное совпадение).
_TIER1_LEVEL1 = frozenset({"ANNEX", "CHAPTER", "TITLE", "PART", "GLAVA", "PRILOG"})
_TIER1_LEVEL2 = frozenset({"SECTION", "ARTICLE", "APPENDIX", "CLAN", "ČLAN", "ODJELJAK", "DIO"})

# Unicode-буквы (НЕ только ASCII A-Za-z) — «Član» начинается с Č (U+010C, LATIN CAPITAL
# LETTER C WITH CARON), которую [A-Za-z] не матчит вовсе (баг, найденный тем же полевым
# тестом: Тир 1 молча не срабатывал на любом слове с диакритикой).
_TIER1_LEADING_WORD = re.compile(r"^([^\W\d_]+)", re.UNICODE)

# «Član3 1» -> «Član 31»: OCR на сканах систематически (не единично — 5 из ~48 статей
# этого документа) роняет пробел между keyword и первой цифрой номера, а между двумя
# цифрами самого номера — наоборот, вставляет лишний (шум individual-bbox между соседними
# символами). Реджойн вызывается ТОЛЬКО из _tier1_heading — то есть ТОЛЬКО когда keyword
# уже подтверждён — не общий текстовый паттерн «слово+цифра» по всему телу документа.
# d1 <= 3 цифр (номера статей вплоть до тройной значности), d2 — ровно одна цифра
# (наблюдаемый паттерн — оторвался последний разряд, не производный самостоятельный номер).
_GLUED_NUMBER_RE = re.compile(r"^([^\W\d_]+)(\d{1,3})(?:\s+(\d))?\s*(.*)$", re.DOTALL)

_TIER3_NUMBERING = re.compile(r"^(\d+)(\.(\d+))?\.?\s+(.+)$")

# Токен из одних римских цифр («XXVIII») — не «настоящее» слово для Тир 2 (см. guard ниже).
_ROMAN_ONLY_RE = re.compile(r"^[IVXLCDM]+$")

_EXISTING_HEADING_RE = re.compile(r"^#{1,6}\s+")


def _degrue_leading_number(stripped: str) -> str:
    """Восстановить «keyword N» из OCR-расклеенного «keywordD [D]» (см. _GLUED_NUMBER_RE).

    No-op на уже чистых заголовках («Član 19 ...») — маска требует, чтобы цифра шла
    СРАЗУ за keyword без пробела, иначе не матчит вовсе."""
    m = _GLUED_NUMBER_RE.match(stripped)
    if not m:
        return stripped
    word, d1, d2, rest = m.group(1), m.group(2), m.group(3), m.group(4)
    number = d1 + d2 if d2 else d1
    return f"{word} {number}" + (f" {rest}" if rest else "")


def _tier1_heading(stripped: str) -> str | None:
    """ANNEX/CHAPTER/TITLE/PART/GLAVA/PRILOG -> #; SECTION/Article/Appendix/ČLAN/ODJELJAK/DIO -> ##.

    Guard против тела-предложения («Article 6 shall apply to…»): короче HEADING_MAX_LEN
    И не оканчивается точкой/точкой-с-запятой. Восстанавливает OCR-расклеенный номер
    (см. _degrue_leading_number) ПОСЛЕ подтверждения keyword'а.
    """
    if len(stripped) >= HEADING_MAX_LEN or stripped.endswith((".", ";")):
        return None
    m = _TIER1_LEADING_WORD.match(stripped)
    if not m:
        return None
    word = m.group(1).upper()
    if word in _TIER1_LEVEL1:
        return f"# {_degrue_leading_number(stripped)}"
    if word in _TIER1_LEVEL2:
        return f"## {_degrue_leading_number(stripped)}"
    return None


def _tier2_heading(stripped: str, next_line: str) -> str | None:
    """Короткая CAPS-строка + непустое тело следом -> ##.

    Guard против акронимов/обломков таблиц: минимум 2 слова (одинокий «GDPR» —
    не заголовок) И хотя бы одно «настоящее» слово из >=2 букв (не только цифры/символы).
    """
    if not stripped or len(stripped) > CAPS_MAX_LEN or not next_line:
        return None
    if stripped[-1] in ".,;:":
        return None
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 2:
        return None
    if sum(1 for c in letters if c.isupper()) / len(letters) < 0.7:
        return None
    words = stripped.split()
    if len(words) < 2:
        return None  # одно слово (акроним/обломок таблицы) — не заголовок
    if not any(sum(1 for c in w if c.isalpha()) >= 2 for w in words):
        return None  # ни одного «настоящего» слова — только цифры/символы/одиночные буквы
    alphas = ("".join(c for c in w if c.isalpha()) for w in words)
    if not any(len(a) >= 4 and not _ROMAN_ONLY_RE.match(a) for a in alphas):
        # Ид-строки из акронимов/чисел/римских цифр («EPA 616 XXVIII» — номер акта
        # Скупштины в подписном блоке КАЖДОГО закона ME; живой false positive приёмки
        # чекпоинта 1 convert-cloud-tier) — настоящий CAPS-заголовок на любом языке
        # корпуса несёт хотя бы одно слово >=4 букв, не являющееся римской цифрой.
        return None
    return f"## {stripped}"


def _tier3_heading(stripped: str, next_line: str) -> str | None:
    """Голая нумерация глубиной <=2 («1.» -> ##, «1.1» -> ###) + тело следом.

    Самый опасный тир: в юридическом корпусе нумерация есть у каждой под-клаузы
    статьи. Guard'ы: тайтл с заглавной, без финальной точки, короче HEADING_MAX_LEN —
    иначе типичная длинная под-клауза («1. The provider shall ensure that…») дала
    бы структуру хуже честного плоского текста.
    """
    if len(stripped) >= HEADING_MAX_LEN or stripped.endswith(".") or not next_line:
        return None
    m = _TIER3_NUMBERING.match(stripped)
    if not m:
        return None
    title = m.group(4)
    if not title or not title[0].isupper():
        return None
    depth = 3 if m.group(2) else 2
    return f"{'#' * depth} {stripped}"


def _next_nonblank(lines: list[str], i: int) -> str:
    """Первая непустая строка ПОСЛЕ индекса ``i``, пропуская пустые строки-разделители
    абзацев. Реальный markdown (pdf_to_markdown/OCR-вывод) всегда разделяет абзацы
    пустой строкой — буквальный ``lines[i + 1]`` почти всегда сама пустая строка-
    разделитель, а не следующий абзац (баг, найденный живым полевым тестом на
    Тир 2/3: гарды «есть тело следом» никогда не срабатывали на реальном выводе)."""
    for candidate in lines[i + 1 :]:
        stripped = candidate.strip()
        if stripped:
            return stripped
    return ""


def merge_missing_headings(md: str) -> str:
    """Additive-режим для ОБЛАЧНОГО OCR-вывода (spec convert-cloud-tier §2.5, v2.1).

    Облачная `#`-разметка НЕПРИКОСНОВЕННА (облако — главный производитель структуры):
    заголовки не снимаются и не переуровниваются; добавляются ТОЛЬКО те, что прецизионные
    Тир 1/Тир 2 находят в строках, оставленных облаком телом. Живой мотиватор (чекпоинт 1):
    8 глав «I. OSNOVNE ODREDBE … VIII.» me-crps — текст verbatim-корректен, но без
    разметки, тогда как CAPS-правило Тира 2 ловит их детерминированно. Тир 3 исключён
    (самый рискованный; нумерованные подклаузы облако оформляет само). Таблицы, цитаты
    и код-фенсы пропускаются; нетронутые строки — байт-в-байт (минимальная инвазивность,
    в отличие от promote_flat_headings, пересобирающего каждую строку). Идемпотентно:
    промоутнутая строка на повторном прогоне уже несёт `#`-префикс и не трогается.
    """
    lines = md.split("\n")
    out: list[str] = []
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or not stripped or stripped.startswith(("#", "|", ">")):
            out.append(line)
            continue
        heading = _tier1_heading(stripped) or _tier2_heading(stripped, _next_nonblank(lines, i))
        out.append(heading if heading is not None else line)
    return "\n".join(out)


def promote_flat_headings(md: str) -> str:
    """Восстановить высокоуверенный скелет заголовков в плоском OCR-markdown.

    Существующая разметка (`#`-префикс от кластеризации `pdf_to_markdown` по размеру
    шрифта — на OCR bbox ненадёжна, см. docstring модуля) СНИМАЕТСЯ, строка оценивается
    заново тремя тирами ниже — единый источник истины вместо двух несогласованных.
    При неоднозначности — строка остаётся телом (precision-first). Идемпотентно: на
    повторном прогоне уже промоутнутая (моими же правилами) строка снова даёт тот же
    результат — тиры детерминированы по содержимому, не по факту наличия `#`.
    """
    lines = md.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = _EXISTING_HEADING_RE.sub("", line.strip(), count=1)
        if not stripped:
            out.append(line)
            continue
        next_line = _next_nonblank(lines, i)
        heading = (
            _tier1_heading(stripped)
            or _tier2_heading(stripped, next_line)
            or _tier3_heading(stripped, next_line)
        )
        out.append(heading if heading is not None else stripped)
    return "\n".join(out)

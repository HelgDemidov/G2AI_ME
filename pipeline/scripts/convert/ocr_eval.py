"""A/B-харнесс качества OCR (spec ocr-eval-harness). Тир 1 — точность против
вручную выверенного эталона (``score_page``); тир 2 — попарные расхождения
кандидатов на всём документе (``diverge``), эталона не требует. Оба тира —
чистые функции, без сети. ``run_document``/``run_pages`` — единственные
функции с сетевым вводом-выводом; обе работают ИСКЛЮЧИТЕЛЬНО через копию
``raw`` в изолированном ``workdir`` (§5): ``cloud_ocr.convert_scan`` пишет
кэш-сайдкары рядом с переданным путём (``cache_path``/``_parts_path`` берут
``raw.parent``), прогон харнесса по оригиналу затёр бы оплаченный
production-кэш документа.

Число расходится с числом свидетеля-эталона через ``convert.lint.numeric_delta``
(та же логика, что у витнесс-гейта, §4 спека) — не дублируется здесь.
"""
from __future__ import annotations

import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

from convert import cloud_ocr, converters
from convert.lint import numeric_counter, numeric_delta

_HEADING_HASH_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)  # только ведущие # строки
_WHITESPACE_RE = re.compile(r"\s+")
_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_DIACRITIC_CHARS = frozenset("čćžšđČĆŽŠĐ")


@dataclass(frozen=True)
class PageScore:
    """Тир 1: точность одного кандидата на одной странице эталона.

    ``page`` не проставляется ``score_page`` (функция не знает номера
    страницы, только тексты) — вызывающая сторона задаёт его через
    ``dataclasses.replace(score, page=n)`` при сборке отчёта постранично.
    """

    page: int
    cer: float
    diacritics_recall: float
    numeric_missing: int  # вхождения эталона без пары в кандидате
    numeric_added: int  # наоборот
    headings_gold: int
    headings_matched: int


@dataclass(frozen=True)
class Divergence:
    """Тир 2: попарное расхождение двух кандидатов на уровне ДОКУМЕНТА
    (постраничного разбиения облачного вывода не существует, spec §2)."""

    left: str
    right: str
    numeric_only_left: tuple[str, ...]  # числа, которые есть у left и нет у right
    numeric_only_right: tuple[str, ...]
    headings_only_left: tuple[str, ...]
    headings_only_right: tuple[str, ...]


@dataclass(frozen=True)
class CandidateResult:
    """Итог прогона одного кандидата (модели либо ``"tesseract"``)."""

    name: str
    document_text: str  # тир 2: весь документ одним прогоном (как в проде)
    page_text: dict[int, str]  # тир 1: только страницы эталона, постранично
    scores: list[PageScore]  # тир 1; пусто, если эталона нет
    failed: str | None = None  # текст отказа; кандидат не роняет прогон


def normalize_for_cer(text: str) -> str:
    """Текст для сравнения по Левенштейну: снять ведущие ``#`` заголовков
    (иначе CER мерил бы разметку, не распознавание), схлопнуть пробелы,
    привести к NFC. Регистр и диакритику НЕ трогать — их-то и меряем
    (``diacritics_recall``/визуальная сверка регистра — отдельные метрики)."""
    text = _HEADING_HASH_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return unicodedata.normalize("NFC", text)


def levenshtein(a: str, b: str) -> int:
    """Редакционное расстояние. Своя реализация (не ``python-Levenshtein``,
    C-расширение — новая зависимость ради страниц в пару КБ не оправдана,
    spec Design rationale). Классический двухрядный DP."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,  # удаление
                current[j - 1] + 1,  # вставка
                previous[j - 1] + cost,  # замена
            )
        previous = current
    return previous[-1]


def extract_headings(md: str) -> list[tuple[int, str]]:
    """Список ``(уровень, текст)`` markdown-заголовков; строки внутри
    fenced-блоков (```...```) игнорируются — код-блок может легитимно
    содержать строку, похожую на заголовок (например, во вложенном mermaid)."""
    headings: list[tuple[int, str]] = []
    in_fence = False
    for line in md.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_LINE_RE.match(line)
        if m:
            headings.append((len(m.group(1)), m.group(2).strip()))
    return headings


def _diacritics_recall(gold: str, candidate: str) -> float:
    """Доля вхождений диакритических символов (``čćžšđ`` + заглавные) эталона,
    воспроизведённых кандидатом — мультимножество (``Counter``), не множество:
    OCR теряет диакритику НЕПОСЛЕДОВАТЕЛЬНО (одно и то же слово то с диакритикой,
    то без — convert-ocr §3.1), позиция/конкретное вхождение неважны, важен счёт.
    Эталон без диакритики -> 1.0 (нечего терять)."""
    gold_counts = Counter(ch for ch in gold if ch in _DIACRITIC_CHARS)
    total = sum(gold_counts.values())
    if total == 0:
        return 1.0
    candidate_counts = Counter(ch for ch in candidate if ch in _DIACRITIC_CHARS)
    matched = sum((gold_counts & candidate_counts).values())
    return matched / total


def score_page(gold: str, candidate: str) -> PageScore:
    """Все четыре метрики тира 1 для одной страницы (``page=0`` — см. докстринг
    ``PageScore``)."""
    norm_gold = normalize_for_cer(gold)
    norm_candidate = normalize_for_cer(candidate)
    if norm_gold:
        cer = levenshtein(norm_gold, norm_candidate) / len(norm_gold)
    else:
        cer = 0.0 if not norm_candidate else 1.0

    missing, added = numeric_delta(gold, candidate)

    gold_headings = extract_headings(gold)
    candidate_headings = Counter(extract_headings(candidate))
    matched = sum((Counter(gold_headings) & candidate_headings).values())

    return PageScore(
        page=0,
        cer=cer,
        diacritics_recall=_diacritics_recall(gold, candidate),
        numeric_missing=missing,
        numeric_added=added,
        headings_gold=len(gold_headings),
        headings_matched=matched,
    )


def _format_tier1_block(results: list[CandidateResult]) -> str:
    pages = sorted({s.page for r in results for s in r.scores})
    header = f"=== Против эталона (стр. {', '.join(str(p) for p in pages)}) ===" if pages \
        else "=== Против эталона (нет страниц) ==="
    lines = [header, ""]
    for r in results:
        if r.failed is not None:
            lines.append(f"  {r.name}: ОТКАЗ — {r.failed}")
            continue
        lines.append(f"  {r.name}:")
        for s in sorted(r.scores, key=lambda s: s.page):
            lines.append(
                f"    p.{s.page}  cer={s.cer:.2f}  diacritics={s.diacritics_recall:.2f}  "
                f"numeric=-{s.numeric_missing}/+{s.numeric_added}  "
                f"headings={s.headings_matched}/{s.headings_gold}"
            )
    return "\n".join(lines)


def _format_divergence_side(tokens: tuple[str, ...]) -> str:
    return ",".join(tokens) if tokens else "none"


def _format_tier2_block(divergences: list[Divergence]) -> str:
    lines = ["=== Расхождения кандидатов (весь документ) ===", ""]
    for d in divergences:
        if not any((d.numeric_only_left, d.numeric_only_right, d.headings_only_left, d.headings_only_right)):
            lines.append(f"  {d.left} vs {d.right}: совпали")
            continue
        lines.append(f"  {d.left} vs {d.right}:")
        if d.numeric_only_left or d.numeric_only_right:
            lines.append(
                f"    numeric: {d.left}_only=[{_format_divergence_side(d.numeric_only_left)}] "
                f"{d.right}_only=[{_format_divergence_side(d.numeric_only_right)}]"
            )
        if d.headings_only_left or d.headings_only_right:
            lines.append(
                f"    headings: {d.left}_only=[{_format_divergence_side(d.headings_only_left)}] "
                f"{d.right}_only=[{_format_divergence_side(d.headings_only_right)}]"
            )
    return "\n".join(lines)


def diverge(results: list[CandidateResult]) -> list[Divergence]:
    """Тир 2: попарное расхождение ``document_text`` всех кандидатов, кроме
    упавших (``failed is not None`` — сравнивать нечего). Порядок пар — по
    порядку ``results`` (``itertools.combinations``), детерминирован входом.

    **Согласие кандидатов не доказывает правильность** (spec §2): модели с
    общей линией обучения ошибаются коррелированно. Расхождение — сильный
    сигнал «здесь кто-то врёт»; пустой список расхождений НЕ значит «оба
    правы», значит только «не разошлись в проверяемом».
    """
    candidates = [r for r in results if r.failed is None]
    out: list[Divergence] = []
    for left, right in combinations(candidates, 2):
        left_nums = numeric_counter(left.document_text)
        right_nums = numeric_counter(right.document_text)
        left_headings = Counter(extract_headings(left.document_text))
        right_headings = Counter(extract_headings(right.document_text))
        out.append(
            Divergence(
                left=left.name,
                right=right.name,
                numeric_only_left=tuple(sorted((left_nums - right_nums).keys(), key=int)),
                numeric_only_right=tuple(sorted((right_nums - left_nums).keys(), key=int)),
                headings_only_left=tuple(f"{lv}:{t}" for lv, t in _sorted_headings(left_headings - right_headings)),
                headings_only_right=tuple(f"{lv}:{t}" for lv, t in _sorted_headings(right_headings - left_headings)),
            )
        )
    return out


def _sorted_headings(counter: Counter[tuple[int, str]]) -> list[tuple[int, str]]:
    """Детерминированный порядок для отчёта: по уровню, затем по тексту."""
    return sorted(counter.keys())


def format_report(results: list[CandidateResult], divergences: list[Divergence]) -> str:
    """Два блока (spec §6): тир 1 (таблица кандидат×страница×метрика) и
    тир 2 (попарные расхождения, пара без расхождений сворачивается в одну
    строку «совпали»). Предупреждения про raw_sha256/OUTLINE_PREAMBLE/слабость
    согласия — уровень CLI (main), не этой функции: она только форматирует
    переданные данные."""
    return _format_tier1_block(results) + "\n\n" + _format_tier2_block(divergences)


# --- run_document / run_pages / run_tesseract: единственные функции с сетевым
# вводом-выводом (кроме tesseract — он локальный и бесплатный). Все три
# работают ИСКЛЮЧИТЕЛЬНО через копию raw в workdir (§5) ---


def _copy_raw(raw: Path, workdir: Path) -> Path:
    """Копия ``raw`` внутри ``workdir`` — единая точка изоляции: все три
    ``run_*`` функции читают/пишут только эту копию, оригинал в ``sources/``
    не открывается на запись НИКЕМ (``cloud_ocr.convert_scan`` пишет
    кэш-сайдкары рядом с переданным путём). Идемпотентна: повторный вызов на
    том же ``workdir`` не копирует заново."""
    copy = workdir / raw.name
    if not copy.exists():
        shutil.copy2(raw, copy)
    return copy


def run_document(raw: Path, language: str, model: str, workdir: Path) -> str:
    """Тир 2: весь документ ОДНИМ вызовом ``convert_scan`` — ровно та
    конфигурация, что работает в проде (13 стр. < ``OCR_BATCH_PAGES=20`` ->
    один запрос, spec §3)."""
    return cloud_ocr.convert_scan(_copy_raw(raw, workdir), language, model=model)


def run_pages(raw: Path, pages: list[int], language: str, model: str, workdir: Path) -> dict[int, str]:
    """Тир 1: КАЖДАЯ страница эталона — отдельный одностраничный PDF и
    отдельный вызов ``convert_scan``. Только так получается однозначное
    соответствие страница <-> текст: постраничных разделителей в общем
    выводе ``convert_scan`` нет (§2). Честная оговорка (§3): одностраничный
    прогон идёт БЕЗ ``_OUTLINE_PREAMBLE`` (тот добавляется только для батча
    N>1 внутри ``convert_scan``) — контекст чуть отличается от
    полнодокументного, для CER несущественно, для заголовков может дать
    расхождение с тем же кандидатом в тире 2. Ожидаемо, не дефект."""
    copy = _copy_raw(raw, workdir)
    result: dict[int, str] = {}
    for page in pages:
        src = pdfium.PdfDocument(str(copy))
        sliced = pdfium.PdfDocument.new()
        sliced.import_pages(src, [page - 1])  # API 0-based, манифест/§1 — 1-based
        page_path = workdir / f"p{page:02d}.pdf"
        sliced.save(str(page_path))
        result[page] = cloud_ocr.convert_scan(page_path, language, model=model)
    return result


def run_tesseract(raw: Path, workdir: Path) -> dict[int, str] | None:
    """Кандидат ``tesseract``: ``pdfplumber.extract_text`` по КАЖДОЙ странице
    копии — без сети, без вызова ``ocrmypdf`` (текст-слой уже вписан in-place
    при первой конвертации документа, convert-ocr §2). Возвращает ``{стр:
    текст}`` для ВСЕХ страниц документа (1-based) одним проходом — и тир 1
    (эталонные страницы), и тир 2 (весь документ) берут срез из ОДНОГО
    результата; в отличие от облачных ``run_document``/``run_pages``,
    которым нужен раздельный сетевой вызов на каждую конфигурацию, здесь
    текст уже физически лежит в PDF и резать на отдельные файлы незачем.

    ``None``, если у документа нет текст-слоя (``converters._was_ocr_normalized``
    вернул ``False``) — кандидат пропускается с сообщением, не роняет прогон
    (spec §3)."""
    copy = _copy_raw(raw, workdir)
    if not converters._was_ocr_normalized(copy):
        return None
    with pdfplumber.open(copy) as pdf:
        return {i: (page.extract_text() or "") for i, page in enumerate(pdf.pages, start=1)}

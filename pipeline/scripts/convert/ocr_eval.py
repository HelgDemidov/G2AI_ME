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

import argparse
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
import yaml

from convert import cloud_ocr, converters
from convert.lint import numeric_counter, numeric_delta
from core import fsio
from core.env import REPO_ROOT, load_dotenv
from core.schema import DEFAULT_SOURCES

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


# --- CLI: сборка кандидатов, эталон, отчёт (spec §6) ---

DEFAULT_GOLD_DIR = REPO_ROOT / "pipeline" / "scripts" / "tests" / "fixtures" / "local" / "ocr_gold"


@dataclass(frozen=True)
class GoldManifest:
    document: str
    raw_sha256: str
    language: str
    pages: list[int]


def _load_manifest(gold_dir: Path) -> GoldManifest:
    data = yaml.safe_load((gold_dir / "manifest.yaml").read_text(encoding="utf-8"))
    return GoldManifest(
        document=str(data["document"]),
        raw_sha256=str(data["raw_sha256"]),
        language=str(data["language"]),
        pages=[int(p) for p in data["pages"]],
    )


def _load_gold_page(gold_dir: Path, page: int) -> str:
    """Файл эталона страницы — по СУФФИКСУ ``-pNN.md``, не по вычисленному из
    ``document`` имени: реальная раскладка (§1) использует короткий человеческий
    префикс (``me-crps-p01.md`` для ``me-crps-registration-law-2025``), который
    код не обязан угадывать."""
    matches = list(gold_dir.glob(f"*-p{page:02d}.md"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"ожидался ровно один файл *-p{page:02d}.md в {gold_dir}, найдено {len(matches)}"
        )
    return matches[0].read_text(encoding="utf-8")


def _resolve_raw(document: str) -> Path:
    """``raw.*`` документа по id — glob по ``sources/``, без загрузки/валидации
    всего реестра: харнесс инструмент курации, а не часть гейта validate_sources."""
    matches = sorted(DEFAULT_SOURCES.rglob(f"{document}/raw.*"))
    if not matches:
        raise FileNotFoundError(f"raw.* не найден для документа {document!r} в {DEFAULT_SOURCES}")
    return matches[0]


def _tesseract_result(raw: Path, workdir: Path, pages: list[int]) -> CandidateResult:
    pages_map = run_tesseract(raw, workdir)
    if pages_map is None:
        return CandidateResult(
            name="tesseract", document_text="", page_text={}, scores=[],
            failed="raw не нормализован OCR (нет текст-слоя, converters._was_ocr_normalized=False)",
        )
    document_text = "\n\n".join(pages_map[p] for p in sorted(pages_map))
    page_text = {p: pages_map[p] for p in pages if p in pages_map}
    return CandidateResult(name="tesseract", document_text=document_text, page_text=page_text, scores=[])


def _build_candidates(
    raw: Path, language: str, pages: list[int], models: list[str], *, include_tesseract: bool, workdir: Path
) -> list[CandidateResult]:
    """Прогнать все облачные модели (§3: каждая — независимая пара сетевых
    вызовов run_document+run_pages) + опционально tesseract. Отказ ОДНОГО
    кандидата не роняет прогон (``failed``), остальные продолжаются."""
    results: list[CandidateResult] = []
    for model in models:
        try:
            document_text = run_document(raw, language, model, workdir)
            page_text = run_pages(raw, pages, language, model, workdir)
        except Exception as exc:  # noqa: BLE001 — намеренно широкий catch: любой отказ кандидата
            results.append(CandidateResult(name=model, document_text="", page_text={}, scores=[], failed=str(exc)))
            continue
        results.append(CandidateResult(name=model, document_text=document_text, page_text=page_text, scores=[]))
    if include_tesseract:
        results.append(_tesseract_result(raw, workdir, pages))
    return results


def _score_candidates(results: list[CandidateResult], gold_pages: dict[int, str]) -> list[CandidateResult]:
    """Тир 1 постфактум: для НЕ упавших кандидатов — score_page по каждой
    странице эталона, которую кандидат реально вернул (частичный отказ
    run_pages на отдельную страницу не предусмотрен run_pages — либо все
    страницы получены, либо весь кандидат упал раньше, на этапе _build_candidates)."""
    scored: list[CandidateResult] = []
    for r in results:
        if r.failed is not None:
            scored.append(r)
            continue
        scores = [
            replace(score_page(gold_pages[p], r.page_text[p]), page=p)
            for p in sorted(gold_pages)
            if p in r.page_text
        ]
        scored.append(replace(r, scores=scores))
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A/B-харнесс качества OCR: тир 1 (эталон) + тир 2 (расхождения кандидатов)"
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument(
        "--models", default=cloud_ocr.ACTIVE_MODEL,
        help="comma-список слагов OpenRouter (дефолт — production-модель convert-cloud-tier)",
    )
    parser.add_argument("--no-tesseract", action="store_true", help="исключить бесплатный tesseract-кандидат")
    parser.add_argument("--dry-run", action="store_true", help="напечатать план, без сетевых вызовов")
    args = parser.parse_args(argv)

    manifest_path = args.gold / "manifest.yaml"
    if not args.gold.exists() or not manifest_path.exists():
        print(f"нет эталона/манифеста в {args.gold} — см. spec ocr-eval-harness §1", file=sys.stderr)
        return 2

    manifest = _load_manifest(args.gold)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    include_tesseract = not args.no_tesseract

    try:
        raw = _resolve_raw(manifest.document)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    n_requests = len(models) * (1 + len(manifest.pages))
    if args.dry_run:
        print(
            f"План: {len(models)} облачных кандидата(ов) x (1 документ + {len(manifest.pages)} стр. эталона) "
            f"= {n_requests} сетевых запросов" + (" + tesseract (бесплатно, без сети)" if include_tesseract else "")
        )
        print(f"Документ: {manifest.document} ({raw})")
        print(f"Страницы эталона: {manifest.pages}")
        return 0

    load_dotenv()
    gold_pages = {p: _load_gold_page(args.gold, p) for p in manifest.pages}
    with tempfile.TemporaryDirectory() as workdir_str:
        results = _build_candidates(
            raw, manifest.language, manifest.pages, models,
            include_tesseract=include_tesseract, workdir=Path(workdir_str),
        )
    results = _score_candidates(results, gold_pages)
    divergences = diverge(results)

    sections = []
    current_sha = fsio.sha256_file(raw)
    if current_sha != manifest.raw_sha256:
        sections.append(
            f"⚠ raw_sha256 разошёлся с манифестом ({manifest.raw_sha256[:12]}… -> {current_sha[:12]}…) "
            "— эталон мог устареть относительно текущего raw (§1)"
        )
    sections.append(format_report(results, divergences))
    sections.append(
        "Примечание: постраничный прогон (тир 1) идёт БЕЗ _OUTLINE_PREAMBLE — контекст чуть "
        "отличается от полнодокументного (§3), для структуры заголовков возможно расхождение с тиром 2."
    )
    sections.append(
        "Напоминание: согласие кандидатов (тир 2) НЕ доказывает правильность — окончательный "
        "источник истины только эталон (тир 1, §2)."
    )
    print("\n\n".join(sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

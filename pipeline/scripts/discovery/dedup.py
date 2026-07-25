"""discovery/dedup.py — кросс-коннекторный dedup кандидатов (spec discovery-core §3).

Ключи сравнения по убыванию надёжности (чартер §4.4): ``normalized_url`` -> ``(issuer,
normalized_title, doc_date)`` -> ``content_hash``. Без fuzzy-библиотек — детерминизм важнее
recall (остаточные дубли дочистит человек на worksheet, discovery-manual).

**Один проход вместо трёх сканов (spec discovery-candidates-sharding §4).** Раньше на
КАЖДОГО нового кандидата шли три последовательных линейных скана по всему пулу — при
масштабе одного харвеста (1790 existing × сотни fresh) это сотни тысяч сравнений, и
росло квадратично с корпусом кандидатов. Теперь пул индексируется ОДИН раз (три dict),
поиск — три точных lookup: O(M+N) вместо O(M×N).

**Каноническая семантика (решение куратора 2026-07-25): строгий приоритет СТРАТЕГИЙ
над пулом.** Кандидат сверяется с ЕДИНЫМ пулом (existing + уже принятые в этом прогоне
fresh) в порядке url -> key -> hash, остановка на первом попадании. Прежняя форма
(``_find_match(cand, existing) or _find_match(cand, fresh)``) давала приоритет ПУЛУ над
стратегией: в кросс-стратегийном углу (новый кандидат совпадает с existing-записью по
ключу-2 И с fresh-записью по URL) она выбирала existing, единый пул выбирает fresh по URL.
Отличие затрагивает ТОЛЬКО выбор цели merge-провенанса (кому дописать
``merged_connector_ids``) — кандидаты не теряются и не дублируются ни в одной из форм;
единый пул убирает двухуровневый порядок из инварианта и проще доказуем (property-тест
сверяет прод-реализацию с независимым наивным оракулом этой же семантики).
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

from core.schema import CandidateRecord

_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_url(url: str) -> str:
    """URL -> ключ сравнения: lower-host, без fragment, без trailing ``/``, http==https.

    Схема нормализуется в фиксированную ``https`` (значение не для перехода по ссылке,
    только для сравнения) — реальный ``source_url`` документа не трогается.
    """
    parts = urlsplit(url)
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return urlunsplit(("https", netloc, path, parts.query, ""))


def normalized_title(title: str) -> str:
    """Заголовок -> ключ сравнения: нижний регистр, только буквы/цифры (юникод-aware).

    Схлопывает и пробелы, и пунктуацию/дефисы разом — "AI Act" / "ai-act" / "AI  Act."
    дают один ключ. Диакритика (č/š/đ) сохраняется как буква, не отбрасывается.
    """
    return _NON_WORD_RE.sub("", title.lower())


def _match_key(cand: CandidateRecord) -> tuple[str, str, str] | None:
    if not (cand.title and cand.issuer):
        return None
    return (cand.issuer, normalized_title(cand.title), str(cand.doc_date))


class _PoolIndex:
    """Три индекса над пулом кандидатов: ``normalized_url`` / ключ-2 / ``content_hash``.

    **First-seen-wins:** если два кандидата пула дают один ключ (легаси-дубли внутри
    ``existing``), индекс хранит ПЕРВОГО в порядке добавления — ровно то, что возвращал
    линейный скан (``for other in pool: return first``). Пул наполняется
    existing-до-fresh, поэтому ВНУТРИ одной стратегии existing по-прежнему выигрывает.
    """

    def __init__(self) -> None:
        self.by_url: dict[str, CandidateRecord] = {}
        self.by_key: dict[tuple[str, str, str], CandidateRecord] = {}
        self.by_hash: dict[str, CandidateRecord] = {}

    def add(self, cand: CandidateRecord) -> None:
        if cand.normalized_url:
            self.by_url.setdefault(cand.normalized_url, cand)
        key = _match_key(cand)
        if key is not None:
            self.by_key.setdefault(key, cand)
        if cand.content_hash:
            self.by_hash.setdefault(cand.content_hash, cand)

    def find(self, cand: CandidateRecord) -> CandidateRecord | None:
        """Строгий порядок стратегий url -> key -> hash, остановка на первом попадании."""
        if cand.normalized_url:
            hit = self.by_url.get(cand.normalized_url)
            if hit is not None:
                return hit
        key = _match_key(cand)
        if key is not None:
            hit = self.by_key.get(key)
            if hit is not None:
                return hit
        if cand.content_hash:
            hit = self.by_hash.get(cand.content_hash)
            if hit is not None:
                return hit
        return None


def _merge_provenance(existing: CandidateRecord, dup: CandidateRecord) -> None:
    """Дописать provenance поглощённого дубля в existing — НИКОГДА не перезаписывая его поля."""
    merged: list[str] = list(getattr(existing, "merged_connector_ids", None) or [])
    if dup.connector_id != existing.connector_id and dup.connector_id not in merged:
        merged.append(dup.connector_id)
        existing.merged_connector_ids = merged  # type: ignore[attr-defined]  # extra="allow"


def dedup(
    new: list[CandidateRecord], existing: list[CandidateRecord]
) -> tuple[list[CandidateRecord], int]:
    """Разложить ``new`` на (свежие-после-dedup, счётчик поглощённых).

    Дубль внутри ``new`` -> первый выигрывает, второй сливается в него. Дубль против
    ``existing`` (включая отклонённых триажем — они персистят с ``rejected_reason`` и
    не должны воскресать как "свежие") -> ``new``-кандидат НЕ добавляется, его
    ``connector_id`` дописывается в ``merged_connector_ids`` существующего (объект
    ``existing`` мутируется на месте — вызывающая сторона персистит его вместе с ``fresh``;
    шардированный ``store.save`` переписывает чужой шард автоматически, см. его §2).
    """
    index = _PoolIndex()
    for cand in existing:
        index.add(cand)

    fresh: list[CandidateRecord] = []
    absorbed = 0

    for cand in new:
        match = index.find(cand)
        if match is not None:
            _merge_provenance(match, cand)
            absorbed += 1
            continue
        fresh.append(cand)
        index.add(cand)  # принятый fresh участвует в сверке следующих (прежний _find_match(cand, fresh))

    return fresh, absorbed

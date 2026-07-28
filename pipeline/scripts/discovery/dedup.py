"""discovery/dedup.py — кросс-коннекторный dedup кандидатов (spec discovery-core §3).

Ключи сравнения по убыванию надёжности (чартер §4.4): URL-идентичность ->
``(issuer, normalized_title, doc_date)`` -> идентичность записи в источнике. Без
fuzzy-библиотек — детерминизм важнее recall (остаточные дубли дочистит человек на
worksheet, discovery-manual).

**Один проход вместо линейных сканов (spec discovery-candidates-sharding §4).** Раньше
на КАЖДОГО нового кандидата шли последовательные линейные сканы по всему пулу — при
масштабе одного харвеста (1790 existing × сотни fresh) это сотни тысяч сравнений, и
росло квадратично с корпусом кандидатов. Теперь пул индексируется ОДИН раз (dict на
стратегию), поиск — точные lookup: O(M+N) вместо O(M×N).

**Каноническая семантика (решение куратора 2026-07-25): строгий приоритет СТРАТЕГИЙ
над пулом.** Кандидат сверяется с ЕДИНЫМ пулом (existing + уже принятые в этом прогоне
fresh) в порядке url -> key, остановка на первом попадании. Прежняя форма
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
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from core.schema import CandidateRecord, SourceFormat, UrlProvenance

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


_EXT_TO_FORMAT = {
    "pdf": SourceFormat.pdf,
    "docx": SourceFormat.docx,
    "xlsx": SourceFormat.xlsx,
    "html": SourceFormat.html,
    "htm": SourceFormat.html,
}


def format_hint_from_url(url: str) -> SourceFormat | None:
    """Генерическая эвристика подсказки формата по URL (spec discovery-acquire-
    seam-hardening §8, Г7): расширение хвоста ``path`` -> ``SourceFormat``, иначе
    ``None``. Query/фрагмент НЕ смотрим (``?format=pdf`` у лендинга и т.п.) —
    точность важнее покрытия, лгущая подсказка (с эхом «по дефолту: …» в сводке
    apply) опаснее отсутствующей: она выглядит авторитетно. URL-утилиты слоя
    живут рядом с ``normalize_url``."""
    segment = urlsplit(url).path.rsplit("/", 1)[-1]
    if "." not in segment:
        return None
    ext = segment.rsplit(".", 1)[-1].lower()
    return _EXT_TO_FORMAT.get(ext)


def _match_key(cand: CandidateRecord) -> tuple[str, str, str, str | None] | None:
    if not (cand.title and cand.issuer):
        return None
    return (cand.issuer, normalized_title(cand.title), str(cand.doc_date), cand.supersedes)


def _identity_url(cand: CandidateRecord) -> str | None:
    """URL как КЛЮЧ ИДЕНТИЧНОСТИ — ``None``, если адресу объявлено недоверие.

    Значение, помеченное ``url_provenance: suspect`` (spec triage-intake-hardening §6 —
    OECD отдаёт один ``website`` нескольким РАЗНЫМ документам), ключом не служит: иначе
    дедуп схлопывает эти документы в одного, и они не доходят до триажа вовсе (замер на
    боевом снапшоте: 23 группы коллизий, 25 поглощённых кандидатов). Метку ставит
    коннектор, но до этого гейта её не читал никто — dedup отрабатывал раньше двери.
    """
    if cand.url_provenance is UrlProvenance.suspect:
        return None
    return normalize_url(cand.source_url) if cand.source_url else None


def _source_identity(cand: CandidateRecord) -> tuple[str, str, str | None]:
    """Стратегия 3 — СОБСТВЕННАЯ идентичность записи в источнике.

    Ключ — ``native_id`` (идентификатор источника), а не ``raw_hash``: последний есть
    дайджест ОПИСАНИЯ и меняется при любом редактировании записи (замер: правка
    ``updatedAt`` обнуляет совпадение 749/749, ``native_id`` переживает 749/749).
    ``raw_hash`` остаётся фолбэком там, где ``native_id`` нет вовсе.
    """
    return (cand.connector_id, cand.native_id or cand.raw_hash, cand.supersedes)


class _PoolIndex:
    """Три индекса над пулом кандидатов: URL-идентичность / ключ-2 / идентичность в источнике.

    **Дискриминатор редакций входит во ВСЕ ключи единообразно** (spec
    discovery-candidates-sharding §5): кандидат с ``supersedes=X`` сопоставляется только
    с записями с тем же ``supersedes=X``. Это единое правило, а не особые случаи —
    редакция есть другая identity по определению, какой бы стратегией её ни ловили.
    Стратегия 2 обычно спасена новой ``doc_date``, но одинаковая дата переиздания —
    реальный кейс; единообразие снимает его.

    ⚠ **Стратегия 3 — идентичность ПОСЛЕДНЕЙ ИНСТАНЦИИ, а не фолбэк промаха первых
    двух.** Применяется только к ``_keyless``-кандидату (нет ни достоверного URL, ни пары
    issuer+title). Кандидат с достоверным URL, который ни с чем не совпал, — это НОВЫЙ
    документ, и добирать его по чему-то ещё нельзя: у snowball ``native_id`` это
    «документ#канал» (227 кандидатов на 55 значений в боевом store), и фолбэк схлопнул бы
    разные ссылки одного документа в одну (найдено живьём при прототипировании).

    **First-seen-wins:** если два кандидата пула дают один ключ (легаси-дубли внутри
    ``existing``), индекс хранит ПЕРВОГО в порядке добавления — ровно то, что возвращал
    линейный скан (``for other in pool: return first``). Пул наполняется
    existing-до-fresh, поэтому ВНУТРИ одной стратегии existing по-прежнему выигрывает.
    """

    def __init__(self) -> None:
        self.by_url: dict[tuple[str, str | None], CandidateRecord] = {}
        self.by_key: dict[tuple[str, str, str, str | None], CandidateRecord] = {}
        self.by_source: dict[tuple[str, str, str | None], CandidateRecord] = {}

    @staticmethod
    def _keyless(cand: CandidateRecord) -> bool:
        return _identity_url(cand) is None and _match_key(cand) is None

    def add(self, cand: CandidateRecord) -> None:
        url = _identity_url(cand)
        if url:
            self.by_url.setdefault((url, cand.supersedes), cand)
        key = _match_key(cand)
        if key is not None:
            self.by_key.setdefault(key, cand)
        if self._keyless(cand):
            self.by_source.setdefault(_source_identity(cand), cand)

    def find(self, cand: CandidateRecord) -> CandidateRecord | None:
        """Строгий порядок стратегий url -> key, остановка на первом попадании.

        Цепочка 1->2 обязана сохраняться: промах по URL добирается парой issuer+title —
        живой сценарий зеркала WAF-заблокированного первоисточника с ДРУГИМ адресом
        (spec discovery-acquire-seam-hardening §5).
        """
        url = _identity_url(cand)
        if url:
            hit = self.by_url.get((url, cand.supersedes))
            if hit is not None:
                return hit
        key = _match_key(cand)
        if key is not None:
            hit = self.by_key.get(key)
            if hit is not None:
                return hit
        if self._keyless(cand):
            return self.by_source.get(_source_identity(cand))
        return None


_MERGE_EXEMPT = (
    # идентичность самой записи-кандидата — переносить её значит подменить запись
    "raw_hash",
    "retrieved_at",
    "connector_id",
    # курируемое/машинное состояние, которого не выставляет НИ ОДИН производитель
    # кандидатов; сторожится AST-гейтом test_dedup_exempt_matches_unproduced_fields
    "admitted_as",
    "rejected_reason",
    "rejected_kind",
    "probe_checked",
    "probe_finding",
)


def _merge_provenance(existing: CandidateRecord, dup: CandidateRecord) -> None:
    """Донести до поглотителя то, что источник узнал о документе заново.

    Правило — про ОТНОШЕНИЕ источников, а не про перечень полей (его не приходится
    править при добавлении поля в схему):

    * свой источник (``connector_id`` совпал) ОБНОВЛЯЕТ произведённое им — переобнаружение
      реестром авторитетнее прежнего снимка, включая понижение доверия
      (``url_provenance: stated -> suspect``);
    * чужой источник ЗАПОЛНЯЕТ только пустое;
    * ``_MERGE_EXEMPT`` неприкосновенно в обоих случаях.

    ``alternate_source_urls`` (spec discovery-acquire-seam-hardening §5, Г4): расходящийся
    адрес копится в extra-поле-списке поглотителя (``extra="allow"`` уже позволяет; дедуп
    внутри списка) — иначе зеркало WAF-заблокированного первоисточника терялось бы нигде,
    тупик ровно на популяции (c), ради которой ``rejected_kind: unacquirable`` существует.
    Копится ВЫТЕСНЯЕМЫЙ адрес: у чужого источника это его собственный (у поглотителя свой
    остаётся), у своего — прежний адрес поглотителя, который сейчас будет переписан.
    """
    merged: list[str] = list(getattr(existing, "merged_connector_ids", None) or [])
    same_source = dup.connector_id == existing.connector_id
    if not same_source and dup.connector_id not in merged:
        merged.append(dup.connector_id)
        existing.merged_connector_ids = merged  # type: ignore[attr-defined]  # extra="allow"

    if dup.source_url and existing.source_url:
        if normalize_url(dup.source_url) != normalize_url(existing.source_url):
            displaced = existing.source_url if same_source else dup.source_url
            alternates: list[str] = list(getattr(existing, "alternate_source_urls", None) or [])
            if displaced not in alternates:
                alternates.append(displaced)
                existing.alternate_source_urls = alternates  # type: ignore[attr-defined]  # extra="allow"

    for name in type(dup).model_fields:
        if name in _MERGE_EXEMPT:
            continue
        value = getattr(dup, name, None)
        if value is None:
            continue
        if same_source or getattr(existing, name, None) is None:
            setattr(existing, name, value)


@dataclass(frozen=True)
class DedupOutcome:
    """Итог ``dedup()`` (spec discovery-acquire-seam-hardening §5, Г4).

    ``fresh``/``absorbed`` — прежняя форма (обратная совместимость счётчика);
    ``absorptions`` — пары ``(дубль, поглотитель)`` по ВСЕМ трём стратегиям —
    честный ответ вызывающей стороне (``inject``: «уже отклонён ранее, вот
    причина», не голое «уже есть»), а не только счётчик поглощённых. Семантика
    сопоставления не меняется ни на бит — меняется только форма возврата.
    """

    fresh: list[CandidateRecord]
    absorbed: int
    absorptions: list[tuple[CandidateRecord, CandidateRecord]]


def dedup(new: list[CandidateRecord], existing: list[CandidateRecord]) -> DedupOutcome:
    """Разложить ``new`` на (свежие-после-dedup, поглощённые) — см. ``DedupOutcome``.

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
    absorptions: list[tuple[CandidateRecord, CandidateRecord]] = []

    for cand in new:
        match = index.find(cand)
        if match is not None:
            _merge_provenance(match, cand)
            absorptions.append((cand, match))
            continue
        fresh.append(cand)
        index.add(cand)  # принятый fresh участвует в сверке следующих (прежний _find_match(cand, fresh))

    return DedupOutcome(fresh=fresh, absorbed=len(absorptions), absorptions=absorptions)

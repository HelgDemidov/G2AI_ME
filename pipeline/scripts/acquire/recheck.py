"""Recheck-контур: события ПОСЛЕ приёма документа (spec post-acquisition-lifecycle §1–§3).

Корпус без этого контура — фотография: документ проходит границу приёма один раз, и
внешний мир для пайплайна перестаёт существовать. Никто не проверяет, жив ли
официальный URL, не подменил ли издатель файл по тому же адресу, не вышла ли новая
редакция. Для юридического корпуса это измеренная реальность, а не гипотеза, и узнать
о расхождении в момент проверки цитат финального пакета — худший из возможных моментов.

Контур — реконсиляционная КОМАНДА (``run_pipeline.py --recheck``), не демон и не cron:
куратор запускает её, когда считает нужным. Дёшево по построению — условный GET
(«изменилось ли с даты X?») почти всегда укладывается в сотни байт ответа.

**Инвариант, зеркальный knowledge-инварианту «проекция не пишет в ядро»: контур НИКОГДА
не мутирует ни meta.yaml, ни raw.** Он пишет findings в операционный ``.state.yaml``
(и probe-поля кандидатов через store), а все содержательные решения — новая редакция?
передобыть? — закрывает человек существующими дверями (``inject --supersedes``,
``run_pipeline --force --only``, decisions.yaml).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from acquire import acquisition
from core import pdfmeta, schema

logger = logging.getLogger("recheck")

# Сколько документов берёт один прогон — НА ПОПУЛЯЦИЮ, не суммарно: популяции живут
# на разных курсорах, и общий потолок обрекал бы малую (недобытые) вечно голодать за
# большой (весь корпус). Константа, не конфиг — дисциплина RRF_K/POOL.
RECHECK_DEFAULT_LIMIT = 20

# Сколько раз валидатор должен подтвердиться ПОДРЯД, прежде чем его смена считается
# сигналом. Порог существует ради html: валидаторы гос-порталов часто волатильны
# (динамические токены в ETag), и без него первый же прогон recheck выдал бы шторм
# ложных drift'ов по всему html-слою корпуса. Эвристика без измерительной базы —
# калибровать по доле ложных drift на первых живых прогонах, не заранее.
HTML_STABLE_CONFIRMS = 2


@dataclass(frozen=True)
class ProbeOutcome:
    """Результат одного условного запроса ДО интерпретации в терминах документа."""

    not_modified: bool  # 304 — тела нет вовсе, и это самый дешёвый честный ответ
    classified: acquisition.ClassifiedResponse | None  # None ровно при not_modified
    digest: str | None = None  # sha256 тела; считается только на --recheck-deep


@dataclass(frozen=True)
class Verdict:
    """Желаемое состояние документа после проверки + человекочитаемая нота.

    Чистый результат: ``apply_verdict`` переносит его в ``OperationalState``, сам
    классификатор ничего не пишет и не ходит в сеть.
    """

    finding: str | None
    etag: str | None
    http_last_modified: str | None
    etag_confirms: int
    note: str


def probe_url(
    url: str,
    *,
    user_agent: str,
    expected: schema.SourceFormat | None,
    etag: str | None = None,
    http_last_modified: str | None = None,
    conditional: bool = True,
) -> ProbeOutcome:
    """Условный GET — **никогда HEAD** (spec §2).

    Урок собственного слоя добычи: блок распознаётся ТОЛЬКО по телу (заголовок
    ``Server: BigIP`` ложноположителен, challenge-страницы ловятся маркерами тела).
    HEAD тела не возвращает — значит ``200``-challenge был бы неотличим от контента, а
    бутстрап валидаторов из HEAD записал бы ETag challenge-страницы как валидатор
    документа: все будущие проверки сравнивали бы мусор с мусором. Условный GET даёт
    нужное сам: ``304`` приходит без тела и безопасен (WAF не отвечает челленджем-304),
    а ``200`` приносит тело ровно тогда, когда его надо классифицировать.

    ``expected=None`` — формат-агностичная классификация (``acquisition.classify_probe``)
    для популяции (c): у кандидата слоя discovery ``source_format`` ещё не объявлен.

    Тело пишется во временный каталог ВНЕ папки документа — не в ``raw.*``: контур не
    мутирует оригинал ни при каком исходе.
    """
    headers: dict[str, str] = {}
    if conditional:
        if etag:
            headers["If-None-Match"] = etag
        if http_last_modified:
            headers["If-Modified-Since"] = http_last_modified

    with tempfile.TemporaryDirectory(prefix="recheck-") as workdir:
        dest = Path(workdir) / "probe.bin"
        raw = acquisition.fetch_raw(url, dest, user_agent=user_agent, extra_headers=headers)
        if raw.unreachable_reason is not None:
            return ProbeOutcome(
                not_modified=False,
                classified=acquisition.ClassifiedResponse(
                    acquisition.AcquisitionOutcome.dead, None, raw.unreachable_reason
                ),
            )
        if raw.status == 304:
            return ProbeOutcome(not_modified=True, classified=None)
        digest = hashlib.sha256(raw.body).hexdigest() if raw.body else None
        classified = (
            acquisition.classify_probe(raw.body, raw.headers_text)
            if expected is None
            else acquisition.classify_response(raw.body, raw.headers_text, expected)
        )
        return ProbeOutcome(not_modified=False, classified=classified, digest=digest)


def classify_recheck(
    state: schema.OperationalState,
    source_format: schema.SourceFormat,
    probe: ProbeOutcome,
    *,
    deep_baseline: str | None = None,
    deep: bool = False,
) -> Verdict:
    """Чистая функция «состояние + ответ -> желаемое состояние» (spec §2).

    Ключевое свойство: **чистый исход НИКОГДА не гасит уже стоящий finding.** Флаг
    ставит машина, снимает — только разрешение человеком (передобыча ``_do_download``
    или суперсидирование, выводящее запись из ротации). Иначе документ с непонятым
    дрейфом молча «выздоравливал» бы на следующей же ротации, ровно после того как мы
    обновили у себя валидатор.
    """
    keep = Verdict(
        finding=state.recheck_finding,
        etag=state.etag,
        http_last_modified=state.http_last_modified,
        etag_confirms=state.etag_confirms,
        note="",
    )

    if probe.not_modified:
        return replace(keep, etag_confirms=state.etag_confirms + 1, note="304 — не изменялся")

    classified = probe.classified
    assert classified is not None  # not_modified=False => ответ классифицирован

    if classified.outcome is acquisition.AcquisitionOutcome.dead:
        if state.fidelity is schema.Fidelity.archived_snapshot:
            # source_url был подтверждённо мёртв ещё при добыче — смерть тут ОЖИДАЕМОЕ
            # состояние, а не событие. Finding означал бы вечный link-rot-шум на каждой
            # ротации по всем архивным записям корпуса.
            return replace(keep, note=f"мёртв — ожидаемо для archived_snapshot ({classified.reason})")
        return replace(keep, finding=f"link-rot: {classified.reason}", note="официальный URL умер")

    if classified.outcome is acquisition.AcquisitionOutcome.blocked:
        # Блок — известное состояние КАНАЛА, не событие документа: сказать про сам
        # документ мы ничего не можем, поэтому честное «непроверяемо» и bump курсора.
        return replace(keep, note=f"unverifiable: blocked ({classified.reason})")

    if state.fidelity is schema.Fidelity.archived_snapshot:
        # Обратное событие: URL, добытый когда-то только из архива, отвечает живым
        # контентом. Сводка предложит передобыть живую редакцию (--force --only).
        return replace(keep, finding=f"resurrected: {classified.reason}", note="источник ожил")

    if deep:
        return _verdict_deep(keep, probe, deep_baseline)
    return _verdict_validators(keep, state, source_format, classified)


def _verdict_deep(keep: Verdict, probe: ProbeOutcome, baseline: str | None) -> Verdict:
    """Глубокая сверка (--recheck-deep): дайджест тела против эталона.

    ``baseline is None`` — честное «непроверяемо», а не молчаливый вердикт «чисто»:
    у скана, нормализованного OCR до появления ``original_sha256``, издательского
    оригинала на диске больше нет, и сравнивать текущий ответ не с чем в принципе
    (``sha256`` в состоянии описывает файл ПОСЛЕ вшивания текст-слоя).
    """
    if baseline is None:
        return replace(keep, note="unverifiable: нет эталонного дайджеста (OCR-нормализованный raw)")
    if probe.digest == baseline:
        return replace(keep, note="дайджест совпал с эталоном")
    observed = (probe.digest or "")[:12] or "пусто"
    return replace(
        keep,
        finding=f"drift: дайджест {baseline[:12]} -> {observed}",
        note="содержимое официального URL изменилось",
    )


def _verdict_validators(
    keep: Verdict,
    state: schema.OperationalState,
    source_format: schema.SourceFormat,
    classified: acquisition.ClassifiedResponse,
) -> Verdict:
    """Сравнение серверных валидаторов (обычный режим).

    Первый recheck по легаси-записи (валидаторов нет — их не захватывали до этого
    спека) БЕСПЛАТНО вооружает её: ok-классифицированный ответ уже принёс заголовки,
    остаётся их запомнить.
    """
    if state.etag is None and state.http_last_modified is None:
        if classified.etag is None and classified.last_modified is None:
            return replace(keep, note="валидаторов нет ни у нас, ни в ответе (глубже — --recheck-deep)")
        return replace(
            keep,
            etag=classified.etag,
            http_last_modified=classified.last_modified,
            etag_confirms=0,
            note="валидаторы забутстраплены",
        )

    stored: str | None
    observed: str | None
    label: str
    if state.etag is not None:
        stored, observed, label = state.etag, classified.etag, "ETag"
    else:
        stored, observed, label = state.http_last_modified, classified.last_modified, "Last-Modified"

    if observed is None:
        return replace(keep, note=f"{label} исчез из ответа — сравнивать не с чем")
    if observed == stored:
        return replace(keep, etag_confirms=state.etag_confirms + 1, note=f"{label} совпал")

    refreshed = replace(
        keep,
        etag=classified.etag,
        http_last_modified=classified.last_modified,
        etag_confirms=0,  # у НОВОГО валидатора подтверждений ещё нет
    )
    if source_format is schema.SourceFormat.html and state.etag_confirms < HTML_STABLE_CONFIRMS:
        return replace(
            refreshed,
            note=(
                f"html: {label} нестабилен (подтверждений {state.etag_confirms} < "
                f"{HTML_STABLE_CONFIRMS}) — drift не ставится"
            ),
        )
    return replace(
        refreshed,
        finding=f"drift: {label} {stored} -> {observed}",
        note="валидатор изменился",
    )


def apply_verdict(state: schema.OperationalState, verdict: Verdict, today: _dt.date) -> None:
    """Перенести вердикт в состояние. **Курсор бампается при ЛЮБОМ исходе** — иначе
    документ с непогашенным finding навсегда остался бы «самым давно не проверенным»,
    пере-probe'ился бы каждым прогоном и на каждом дёргал бы SavePageNow."""
    state.recheck_finding = verdict.finding
    state.etag = verdict.etag
    state.http_last_modified = verdict.http_last_modified
    state.etag_confirms = verdict.etag_confirms
    state.acquisition_checked = today


def probe_url_for(rec: schema.SourceRecord, state: schema.OperationalState) -> str:
    """С какого URL сняты валидаторы — с тем и сравнивать (spec §2).

    Отдельного поля «URL валидаторов» нет намеренно: правка ``official_alt_url`` в
    meta между добычей и проверкой даёт максимум один ложный drift, который разбирает
    человек, — дешевле, чем ещё одно поле состояния, живущее ради края.
    """
    if state.acquisition_method is schema.AcquisitionMethod.official_alt and rec.official_alt_url:
        return rec.official_alt_url
    return rec.source_url


def deep_baseline(rec: schema.SourceRecord, root: Path, state: schema.OperationalState) -> str | None:
    """Эталонный дайджест для ``--recheck-deep``: издательские байты, какими мы их получили.

    ``original_sha256`` (снят до OCR-мутации) — точный ответ. Для born-digital raw
    подходит обычный ``sha256`` (файл не мутировал). Для скана БЕЗ ``original_sha256``
    (нормализован до появления поля) эталона не существует — None, и вердикт честно
    скажет «непроверяемо» вместо ложного drift на каждом прогоне.
    """
    if state.original_sha256 is not None:
        return state.original_sha256
    if rec.source_format is not schema.SourceFormat.pdf:
        return state.sha256  # OCR-путь существует только для PDF
    raw = schema.raw_file(rec, root)
    if raw is None or not raw.exists():
        return state.sha256
    try:
        return None if pdfmeta.was_ocr_normalized(raw) else state.sha256
    except Exception:  # noqa: BLE001 — диагностический проход не должен ронять проверку
        logger.debug("не удалось определить OCR-происхождение %s", raw, exc_info=True)
        return None


# --- ротация: кого проверяем в этот прогон (§1) ---


def _checked_sort_key(checked: _dt.date | None, doc_id: str) -> tuple[bool, _dt.date, str]:
    """None (никогда не проверялся) — первым; дальше по возрастанию даты, затем id
    (детерминизм при равных датах: без него порядок зависел бы от обхода ФС)."""
    return (checked is not None, checked or _dt.date.min, doc_id)


def due_records(
    records: list[schema.SourceRecord], root: Path, *, limit: int
) -> tuple[list[schema.SourceRecord], list[schema.SourceRecord]]:
    """Популяции (a) и (b) прогона: ``(с raw, без raw но с провалом добычи)``.

    Суперсидированные записи выбывают из (a) целиком: их дрейф — ожидаемое состояние
    (издатель работает над действующей редакцией), а не сигнал. Множество считает
    общий ``schema.superseded_ids`` — то же определение, которым будет пользоваться
    фасет ``superseded`` graph-v2, поэтому разойтись они не могут.

    ``sensitivity: confidential`` из (a) НЕ исключается: условный запрос идёт к тому
    же официальному источнику, что и добыча, третьих сторон в нём нет (в отличие от
    SavePageNow, который гейтится).
    """
    superseded = schema.superseded_ids(records)
    with_raw: list[tuple[tuple[bool, _dt.date, str], schema.SourceRecord]] = []
    without_raw: list[tuple[tuple[bool, _dt.date, str], schema.SourceRecord]] = []
    for rec in records:
        if rec.id in superseded:
            continue
        try:
            raw = schema.raw_file(rec, root)
        except ValueError as exc:  # несколько raw.* — проблема раскладки, не повод рвать прогон
            logger.warning("  ✗ %s: пропущен (%s)", rec.id, exc)
            continue
        state = schema.load_state(schema.state_file(rec, root))
        if raw is not None:
            with_raw.append((_checked_sort_key(state.acquisition_checked, rec.id), rec))
        elif state.acquisition_failed is not None:
            without_raw.append((_checked_sort_key(state.acquisition_failed, rec.id), rec))
    with_raw.sort(key=lambda pair: pair[0])
    without_raw.sort(key=lambda pair: pair[0])
    return [r for _, r in with_raw[:limit]], [r for _, r in without_raw[:limit]]


def due_candidates(
    candidates: list[schema.CandidateRecord], *, limit: int
) -> list[schema.CandidateRecord]:
    """Популяция (c): кандидаты, отклонённые как ``unacquirable`` (§5).

    «Не нужен» и «нужен, но недобываем» — разные состояния с разной судьбой: первое
    терминально, второе живёт очередью ожидания обстоятельств (WAF снимут, появится
    зеркало). Без этого различения второй закрывался бы навсегда по ошибке, потому что
    заметить смену обстоятельств некому.
    """
    due = [
        c
        for c in candidates
        if c.rejected_kind is schema.RejectionKind.unacquirable and c.source_url
    ]
    due.sort(key=lambda c: _checked_sort_key(c.probe_checked, c.raw_hash))
    return due[:limit]


# --- прогон ---


@dataclass
class RecheckItem:
    doc_id: str
    note: str
    finding: str | None = None
    error: str | None = None


@dataclass
class RecheckSummary:
    items: list[RecheckItem]
    candidates_changed: bool = False

    @property
    def findings(self) -> list[RecheckItem]:
        return [i for i in self.items if i.finding is not None]

    @property
    def errors(self) -> list[RecheckItem]:
        return [i for i in self.items if i.error is not None]


def _recheck_one(
    rec: schema.SourceRecord,
    root: Path,
    *,
    user_agent: str,
    deep: bool,
    today: _dt.date,
) -> RecheckItem:
    """Популяция (a): запись с raw. Один документ, один условный запрос."""
    state_path = schema.state_file(rec, root)
    state = schema.load_state(state_path)
    probe = probe_url(
        probe_url_for(rec, state),
        user_agent=user_agent,
        expected=rec.source_format,
        etag=state.etag,
        http_last_modified=state.http_last_modified,
        conditional=not deep,  # глубокая сверка обязана получить ТЕЛО, а не 304
    )
    verdict = classify_recheck(
        state,
        rec.source_format,
        probe,
        deep=deep,
        deep_baseline=deep_baseline(rec, root, state) if deep else None,
    )
    previous_finding = state.recheck_finding
    apply_verdict(state, verdict, today)
    # SPN на дрейфе (§4): снять ИЗМЕНИВШУЮСЯ редакцию, пока она жива, — до того как
    # куратор дошёл до разбора. Строго при УСТАНОВКЕ/ИЗМЕНЕНИИ строки, не при её
    # повторном подтверждении: иначе каждая ротация стреляла бы в Wayback заново.
    # Цель снимка — probe_url_for(rec, state) (spec acquire-convert-seam-hardening
    # §7, В9-код), НЕ голый rec.source_url: дрейф НАБЛЮДЁН на том URL, с которого
    # сняты валидаторы (source_url либо official_alt_url, если добыча шла этой
    # ступенью) — снимать нужно ТУ редакцию, что реально изменилась, а не всегда
    # каноническую ссылку издателя.
    if (
        verdict.finding is not None
        and verdict.finding != previous_finding
        and verdict.finding.startswith("drift:")
        and schema.external_disclosure_allowed(rec.sensitivity)
    ):
        acquisition.request_snapshot(probe_url_for(rec, state))
    schema.save_state(state_path, state)
    return RecheckItem(rec.id, verdict.note, finding=verdict.finding)


def _reprobe_unacquired(
    rec: schema.SourceRecord, root: Path, *, user_agent: str, today: _dt.date
) -> RecheckItem:
    """Популяция (b): допущен триажем, добыть не удалось — «а не открылось ли?».

    Успех НЕ скачивает документ здесь: контур только снимает backoff, а добирает его
    ближайший штатный прогон ``run_pipeline`` — одна дверь к добыче, а не две."""
    state_path = schema.state_file(rec, root)
    state = schema.load_state(state_path)
    probe = probe_url(
        probe_url_for(rec, state), user_agent=user_agent, expected=rec.source_format, conditional=False
    )
    classified = probe.classified
    assert classified is not None  # conditional=False => 304 невозможен
    if classified.outcome is acquisition.AcquisitionOutcome.ok:
        state.acquisition_failed = None
        state.acquisition_failure_reason = None
        note = "стало добываемо — ближайший прогон run_pipeline доберёт"
    else:
        state.acquisition_failed = today
        state.acquisition_failure_reason = classified.reason
        note = f"по-прежнему недобываем: {classified.reason}"
    schema.save_state(state_path, state)
    return RecheckItem(rec.id, note)


def _probe_unacquirable(
    cand: schema.CandidateRecord, *, user_agent: str, today: _dt.date
) -> RecheckItem:
    """Популяция (c): «а не открылось ли?» по URL недобываемого кандидата.

    Мутирует переданную запись; персист — забота вызывающей стороны (оркестратор
    сшивает слои: сам ACQUIRE в store слоя discovery не лезет).
    """
    assert cand.source_url is not None  # гарантирует отбор в due_candidates
    probe = probe_url(cand.source_url, user_agent=user_agent, expected=None, conditional=False)
    classified = probe.classified
    assert classified is not None  # conditional=False => 304 невозможен
    kind = {
        acquisition.AcquisitionOutcome.ok: "acquirable",
        acquisition.AcquisitionOutcome.dead: "dead",
        acquisition.AcquisitionOutcome.blocked: "blocked",
    }[classified.outcome]
    cand.probe_checked = today
    cand.probe_finding = f"{kind}: {classified.reason}"
    label = f"{cand.raw_hash[:12]} {cand.title or cand.source_url}"
    actionable = classified.outcome is acquisition.AcquisitionOutcome.ok
    return RecheckItem(
        label,
        cand.probe_finding,
        finding=f"acquirable: {classified.reason}" if actionable else None,
    )


def run_recheck(
    records: list[schema.SourceRecord],
    root: Path,
    *,
    user_agent: str,
    limit: int = RECHECK_DEFAULT_LIMIT,
    deep: bool = False,
    today: _dt.date | None = None,
    candidates: list[schema.CandidateRecord] | None = None,
) -> RecheckSummary:
    """Прогон контура по трём популяциям. Отказ одного документа не рвёт прогон
    (изоляция как в ``process_docs``): упавший остаётся с прежним курсором и будет
    взят следующим прогоном первым же.

    ``candidates`` (популяция (c)) мутируется НА МЕСТЕ; загрузку и сохранение store
    делает вызывающая сторона — слой ACQUIRE не знает о раскладке слоя DISCOVERY.
    ``RecheckSummary.candidates_changed`` говорит, есть ли что сохранять.
    """
    today = today or _dt.date.today()
    with_raw, without_raw = due_records(records, root, limit=limit)
    items: list[RecheckItem] = []

    for rec in with_raw:
        try:
            items.append(_recheck_one(rec, root, user_agent=user_agent, deep=deep, today=today))
        except Exception as exc:  # noqa: BLE001 — изоляция отказа документа
            logger.error("  ✗ %s: %s", rec.id, exc)
            items.append(RecheckItem(rec.id, "", error=str(exc)))

    for rec in without_raw:
        try:
            items.append(_reprobe_unacquired(rec, root, user_agent=user_agent, today=today))
        except Exception as exc:  # noqa: BLE001 — изоляция отказа документа
            logger.error("  ✗ %s: %s", rec.id, exc)
            items.append(RecheckItem(rec.id, "", error=str(exc)))

    changed = False
    for cand in due_candidates(candidates or [], limit=limit):
        try:
            items.append(_probe_unacquirable(cand, user_agent=user_agent, today=today))
            changed = True
        except Exception as exc:  # noqa: BLE001 — изоляция отказа кандидата
            logger.error("  ✗ %s: %s", cand.raw_hash[:12], exc)
            items.append(RecheckItem(cand.raw_hash[:12], "", error=str(exc)))

    return RecheckSummary(items, candidates_changed=changed)


# Подсказки разрешения (§6): контур замыкается ДВУМЯ существующими дверями, выбор
# между ними — содержательное суждение о документе, автоматизировать его спек
# сознательно отказывается (юридический корпус, человек в петле).
_RESOLUTION_HINTS = {
    "drift": (
        "новая редакция -> discover.py inject --supersedes <doc-id> --url <тот же URL> …, "
        "затем штатный триаж; правка на месте -> run_pipeline.py --force --only <doc-id>"
    ),
    "link-rot": "URL умер -> run_pipeline.py --force --only <doc-id> (лестница уйдёт в archive)",
    "resurrected": "источник ожил -> run_pipeline.py --force --only <doc-id> (заберёт живую редакцию)",
    "acquirable": (
        "недобываемый кандидат открылся -> discover.py worksheet (секция недобываемых), "
        "затем решение `action: revive` в decisions.yaml"
    ),
}


def report(summary: RecheckSummary) -> int:
    """Сводка прогона. Findings — НЕ ошибки: ненулевой код только при отказе самого
    прогона (сеть/ФС), иначе нормальная работа контура делала бы прогон красным."""
    for item in summary.items:
        if item.error is not None:
            continue
        logger.info("• %s: %s", item.doc_id, item.note)
    for item in summary.findings:
        prefix = (item.finding or "").split(":", 1)[0]
        logger.warning("  ⚑ %s — %s", item.doc_id, item.finding)
        hint = _RESOLUTION_HINTS.get(prefix)
        if hint:
            logger.info("     %s", hint)
    logger.info(
        "Recheck: проверено %d | findings %d | отказов %d",
        len(summary.items) - len(summary.errors), len(summary.findings), len(summary.errors),
    )
    return 1 if summary.errors else 0

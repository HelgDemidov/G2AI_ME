"""Голден экстракции цитат на ФИКСИРОВАННОЙ паре документов — spec graph-hardening §4.

Recall паттернов мы меряем, precision — нет: ложное ребро в юридическом графе дороже
пропущенного, а сторожили мы до сих пор только вторую ошибку. Полный эталонный харнесс
(образец ``ocr_eval.py``, 497 строк) для шести регексов несоразмерен и, главное, кладёт
на куратора ПОВТОРЯЮЩУЮСЯ разметку. Здесь применён дешёвый приём ``test_convert_golden``:
эталон — зафиксированный список, новый ложноположительный паттерн немедленно даёт diff.

**Пара документов фиксирована намеренно.** Голден по всему корпусу переписывался бы при
каждом новом документе, куратор перестал бы читать diff'ы — классическая смерть
голден-тестов. ``eu-ai-act-2024`` даёт цитатонасыщенность (``eu_act``/``eu_act_slash``),
``me-crps-registration-law-2025`` — структурно самый рискованный регекс
``sluzbeni_list_cg`` (типографика кавычек, списки актов одной строкой, гейт двузначных
годов). Живьём эти два документа покрывают 3 правила из 6: ``celex``/``iso``/``nist`` в
корпусе не встречаются НИГДЕ и остаются на юнитах/hypothesis до появления носителей.

Экстракция гоняется **pattern-only, без алиасов**: голден сторожит стабильность РЕГЕКСОВ
и не должен стрелять от правки курируемого ``identifiers.yaml``, который куратор законно
меняет в любой момент (алиас-канал покрыт герметичными юнитами в ``test_cite_mining``).

⚠ ЧЕСТНАЯ ГРАНИЦА: голден меряет СТАБИЛЬНОСТЬ, а не правильность — он не доказывает, что
128 идентификаторов верны, только что они перестали меняться. Одна человеческая сверка
списка нужна, но ОДНА, а не повторяющаяся разметка. Diff законно даёт не только правка
паттернов, но и смена самого текста: реконверсия (бамп версии конвертера) или передобыча
``eu-ai-act-2024`` (он за AWS WAF, бэклог §25 — передобыча ручная). Это ожидаемое
поведение интеграционного сторожа, а не поломка: сверить причину и перегенерировать.

Требует локальный корпус (``sources/``); в CI пропускается (``@pytest.mark.corpus`` →
``heavy``). Перегенерация: ``.venv/bin/python -m pytest <этот файл> --regenerate-cite-golden``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from core import schema
from graph import cite_mining

pytestmark = pytest.mark.corpus

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "cite_golden.yaml"
GOLDEN_DOCS = ("eu-ai-act-2024", "me-crps-registration-law-2025")


def _extract_pattern_only(doc_id: str) -> dict[str, str] | None:
    """``{идентификатор: правило}`` документа корпуса; ``None`` — нет doc.md (свежий клон)."""
    if not schema.DEFAULT_SOURCES.exists():
        return None
    for rec in schema.load_records(schema.DEFAULT_SOURCES):
        if rec.id != doc_id:
            continue
        md = schema.md_file(rec, schema.DEFAULT_SOURCES)
        if not md.exists():
            return None
        return dict(cite_mining.extract_identifiers(md.read_text(encoding="utf-8")))
    return None


def _load_golden() -> dict[str, dict[str, str]]:
    data: Any = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8")) if GOLDEN_PATH.exists() else {}
    return data if isinstance(data, dict) else {}


@pytest.mark.parametrize("doc_id", GOLDEN_DOCS)
def test_extraction_matches_golden(doc_id: str, pytestconfig: pytest.Config) -> None:
    actual = _extract_pattern_only(doc_id)
    if actual is None:
        pytest.skip(f"{doc_id}: нет локального doc.md — голден нечем сверять")

    if pytestconfig.getoption("--regenerate-cite-golden"):
        golden = _load_golden()
        golden[doc_id] = actual
        GOLDEN_PATH.write_text(
            yaml.safe_dump(golden, allow_unicode=True, sort_keys=True), encoding="utf-8"
        )
        pytest.skip(f"{doc_id}: голден перегенерирован ({len(actual)} идентификаторов)")

    expected = _load_golden().get(doc_id)
    assert expected is not None, (
        f"{doc_id}: записи нет в {GOLDEN_PATH.name} — перегенерировать "
        f"(--regenerate-cite-golden) и СВЕРИТЬ список глазами перед коммитом"
    )
    new = {i: r for i, r in actual.items() if i not in expected}
    gone = {i: r for i, r in expected.items() if i not in actual}
    retagged = {i: (expected[i], r) for i, r in actual.items() if i in expected and expected[i] != r}
    assert not (new or gone or retagged), (
        f"{doc_id}: экстракция изменилась — новые {new}, пропавшие {gone}, "
        f"сменившие правило {retagged}. Причина — правка паттернов ЛИБО смена самого "
        f"текста (реконверсия/передобыча); сверить и перегенерировать осознанно"
    )


def test_golden_covers_the_riskiest_pattern() -> None:
    """Пара выбрана не «два любых документа»: без SLCG-носителя голден не сторожил бы
    самый хрупкий регекс реестра, и правка его типографики прошла бы незамеченной."""
    golden = _load_golden()
    if not golden:
        pytest.skip("голден ещё не сгенерирован")
    rules = {rule for doc in golden.values() for rule in doc.values()}
    assert {"eu_act", "eu_act_slash", "sluzbeni_list_cg"} <= rules

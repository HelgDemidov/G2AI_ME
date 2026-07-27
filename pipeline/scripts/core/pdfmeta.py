"""PDF-метаданные, нужные слоям ВЫШЕ convert (spec discovery-acquire-seam-hardening §1).

До этого спека ``was_ocr_normalized`` жила в ``convert/converters.py``, но читалась и
``acquire/recheck.py`` (ленивый импорт, задокументированная инверсия PR #53), и
``discovery/connectors/snowball.py`` (модульный импорт discovery→convert) — второй
случай был ровно тем «вторым импортом», при котором PR #53 предписывал пересмотр.
``core`` — единственный слой, легальный для ОБОИХ потребителей: ``pdfplumber`` уже
общая runtime-зависимость репо (``requirements.txt``), и прецедент использования вне
convert уже есть (``acquire/acquisition.py::_looks_like_candidate_pdf``).
"""
from __future__ import annotations

from pathlib import Path

import pdfplumber


def was_ocr_normalized(raw: Path) -> bool:
    """PDF уже прошёл OCR-нормализацию РАНЬШЕ (по метаданным ocrmypdf).

    ``convert/converters._ocr_normalize`` мутирует ``raw`` in-place (один файл, не
    сайдкар) — после первого успеха текст-слой уже есть, и детекция скана больше НЕ
    поднимет «нужен OCR» на повторных конвертациях (``--force``, бамп версии
    конвертера). Без этой проверки постпроцесс заголовков OCR-ветки перестал бы
    применяться после первого прогона — метаданные ocrmypdf (``Creator: ocrmypdf ...``)
    переживают мутацию текст-слоя и остаются надёжным маркером.

    Потребители вне convert-слоя (``acquire/recheck.py::deep_baseline`` — эталон для
    ``--recheck-deep``; ``discovery/connectors/snowball.py`` — пометка ``ocr-text-url``
    для URL, извлечённых из OCR-текста, подверженного искажению цифр/диакритики;
    ``run_pipeline.scan_fallback_counts``; ``convert/ocr_eval.py``) читают этот факт,
    не дублируя логику.
    """
    with pdfplumber.open(raw) as pdf:
        creator = (pdf.metadata.get("Creator") or "").lower()
    return "ocrmypdf" in creator

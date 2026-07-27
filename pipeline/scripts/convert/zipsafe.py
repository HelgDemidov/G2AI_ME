"""Потолок разжатия OOXML-архивов (spec convert-knowledge-seam-hardening §8).

docx/xlsx — zip, и весь конвейер читает их парты целиком в память
(``ZipFile.read``), нигде не сверяясь с заявленным распакованным размером. Живой
замер аудита: архив 199 КБ разворачивается в 438 МБ пикового RAM на ОДНОМ
``z.read`` — на машине с 8 ГБ это OOM всего прогона, а не отказ одного документа.
Вектор не гипотетический: docx приходят и из batch-каналов discovery, то есть от
третьих сторон.

Гейт — один проход по ``infolist()`` ДО первого чтения парта: заявленный
``file_size`` покрывает главный класс (целиком-в-память ``read``), а занижение
размера у намеренно сломанного архива ломает сам разжим контролируемым
исключением, изолированным per-doc. Побайтового контроля потока нет намеренно
(Design rationale спека): обвязывать каждое чтение ради края, который и так
отказывает громко, — сложность без выигрыша.

Прецедент индустрии — ``ZipSecureFile`` в Apache POI (min inflate ratio + потолок
размера) как штатная защита OOXML-пайплайнов.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

MAX_MEMBER_BYTES = 128 * 1024 * 1024   # один парт (крупная media/лист) — с большим запасом
MAX_TOTAL_BYTES = 512 * 1024 * 1024    # весь архив в распакованном виде
# Стартовая калибровка: легитимный гос-документ на порядки меньше (самый тяжёлый в
# корпусе — единицы МБ), а 8 ГБ RAM машины держат 512 МБ распакованного без риска.
# Как все численные пороги проекта — пересматриваются по факту живой приёмки.


class ArchiveBombSuspected(RuntimeError):
    """Архив заявляет распакованный размер выше потолка — читать его не начинаем.

    Наследует ``RuntimeError``, а НЕ ``converters.ConversionError``: гейт зовут два
    независимых входа (``converters`` и ``figures_vlm``), и импорт ``converters``
    отсюда замкнул бы цикл. На маршрутизацию это не влияет — изоляция отказа документа
    в ``run_pipeline.process_docs`` ловит ``Exception``; тип нужен для читаемости и
    тестов.
    """


def check_archive(
    path: Path, *, max_member: int = MAX_MEMBER_BYTES, max_total: int = MAX_TOTAL_BYTES
) -> None:
    """Проверить заявленные размеры членов архива ДО чтения парта. Тихо возвращает
    None, если всё в пределах; иначе ``ArchiveBombSuspected`` с виновником."""
    with zipfile.ZipFile(path) as z:
        total = 0
        for info in z.infolist():
            if info.file_size > max_member:
                raise ArchiveBombSuspected(
                    f"{path.name}: член {info.filename} заявляет {info.file_size} байт "
                    f"распакованного (потолок {max_member}) — архив не читается"
                )
            total += info.file_size
            if total > max_total:
                raise ArchiveBombSuspected(
                    f"{path.name}: суммарный распакованный размер превысил {max_total} байт "
                    f"— архив не читается"
                )

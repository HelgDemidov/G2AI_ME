"""Тесты потолка разжатия OOXML (spec convert-knowledge-seam-hardening §8).

Архив-бомба конструируется дешёвыми нулями: zip сжимает их на три порядка, поэтому
фикстура весит килобайты, а заявляет сотни мегабайт — ровно тот класс, который живой
замер аудита показал как 199 КБ -> 438 МБ пикового RAM на одном ``z.read``.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest

from convert import zipsafe
from convert.converters import _convert_docx, _convert_xlsx
from convert.zipsafe import ArchiveBombSuspected, check_archive


def _zip_with(path: Path, members: dict[str, int]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, size in members.items():
            z.writestr(name, b"\0" * size)
    return path


def test_oversize_member_rejected(tmp_path: Path) -> None:
    raw = _zip_with(tmp_path / "bomb.docx", {"word/media/img1.png": 4 * 1024 * 1024})
    with pytest.raises(ArchiveBombSuspected, match="img1.png"):
        check_archive(raw, max_member=1024 * 1024, max_total=64 * 1024 * 1024)


def test_oversize_total_rejected(tmp_path: Path) -> None:
    raw = _zip_with(
        tmp_path / "bomb.docx",
        {f"word/media/img{i}.png": 1024 * 1024 for i in range(8)},
    )
    with pytest.raises(ArchiveBombSuspected, match="суммарный"):
        check_archive(raw, max_member=4 * 1024 * 1024, max_total=4 * 1024 * 1024)


def test_ordinary_archive_passes(tmp_path: Path) -> None:
    raw = _zip_with(tmp_path / "ok.docx", {"word/document.xml": 4096, "word/media/i.png": 20_000})
    check_archive(raw)  # не бросает — это и есть утверждение


def test_defaults_are_generous_enough_for_real_documents(tmp_path: Path) -> None:
    """Регресс против слишком тугого потолка: легитимный гос-документ (единицы МБ)
    обязан проходить с дефолтами."""
    raw = _zip_with(tmp_path / "real.docx", {"word/media/photo.png": 8 * 1024 * 1024})
    check_archive(raw)


def test_docx_converter_gated_before_reading_parts(tmp_path: Path, monkeypatch: Any) -> None:
    raw = _zip_with(tmp_path / "bomb.docx", {"word/document.xml": 2 * 1024 * 1024})
    # дефолты берутся из сигнатуры, поэтому патчим саму функцию-гейт
    monkeypatch.setattr(
        zipsafe, "check_archive",
        lambda p, **kw: (_ for _ in ()).throw(ArchiveBombSuspected(f"{p.name}: стоп")),
    )
    with pytest.raises(ArchiveBombSuspected):
        _convert_docx(raw, tmp_path / "out.md", "en")


def test_xlsx_converter_gated_before_reading_parts(tmp_path: Path, monkeypatch: Any) -> None:
    raw = _zip_with(tmp_path / "bomb.xlsx", {"xl/workbook.xml": 2 * 1024 * 1024})
    monkeypatch.setattr(
        zipsafe, "check_archive",
        lambda p, **kw: (_ for _ in ()).throw(ArchiveBombSuspected(f"{p.name}: стоп")),
    )
    with pytest.raises(ArchiveBombSuspected):
        _convert_xlsx(raw, tmp_path / "out.md", "en")


def test_zip_member_sizes_are_read_without_decompressing(tmp_path: Path) -> None:
    """Гейт смотрит ЗАЯВЛЕННЫЙ размер (infolist), а не разжимает — иначе он был бы той
    самой атакой, от которой защищает."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.bin", b"\0" * (32 * 1024 * 1024))
    raw = tmp_path / "b.docx"
    raw.write_bytes(buf.getvalue())
    assert raw.stat().st_size < 1024 * 1024  # фикстура физически мала
    with pytest.raises(ArchiveBombSuspected):
        check_archive(raw, max_member=1024 * 1024)

"""core/pdfmeta.py — spec discovery-acquire-seam-hardening §1.

``was_ocr_normalized`` переехала сюда из ``convert/converters.py``: она нужна слоям
ВЫШЕ convert (``acquire/recheck.py``, ``discovery/connectors/snowball.py``), которые
раньше читали её импортом-инверсией. Тождество реэкспорта (``converters.was_ocr_normalized
is pdfmeta.was_ocr_normalized``) и guard-тест «convert.* не импортируется в acquire/
discovery» живут здесь же.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from convert import converters
from core import pdfmeta
from core.env import REPO_ROOT

_PIPELINE_DIR = REPO_ROOT / "pipeline" / "scripts"


class _FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class _FakePdf:
    def __init__(self, metadata: dict[str, str] | None = None) -> None:
        self.pages: list[_FakePage] = []
        self.metadata = metadata or {}

    def __enter__(self) -> "_FakePdf":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _patch_open(monkeypatch: Any, metadata: dict[str, str] | None = None) -> None:
    monkeypatch.setattr("core.pdfmeta.pdfplumber.open", lambda path: _FakePdf(metadata))


# --- was_ocr_normalized: метка ocrmypdf в метаданных переживает мутацию raw ---


def test_was_ocr_normalized_true_when_creator_mentions_ocrmypdf(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_open(monkeypatch, metadata={"Creator": "ocrmypdf 15.2.0+dfsg1 / Tesseract OCR-PDF 5.3.4"})
    assert pdfmeta.was_ocr_normalized(tmp_path / "raw.pdf") is True


def test_was_ocr_normalized_false_for_born_digital_pdf(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_open(monkeypatch, metadata={"Creator": "Microsoft® Word 2019"})
    assert pdfmeta.was_ocr_normalized(tmp_path / "raw.pdf") is False


def test_was_ocr_normalized_false_when_no_metadata_at_all(monkeypatch: Any, tmp_path: Path) -> None:
    _patch_open(monkeypatch)  # metadata=None -> {}
    assert pdfmeta.was_ocr_normalized(tmp_path / "raw.pdf") is False


# --- тождество реэкспорта (convert/converters.py) ---


def test_converters_reexport_is_identical_function() -> None:
    """`convert.converters.was_ocr_normalized`/`_was_ocr_normalized` — тот же объект
    функции, не копия: рефактор одной из сторон не может разойтись с другой незамеченным."""
    assert converters.was_ocr_normalized is pdfmeta.was_ocr_normalized
    assert converters._was_ocr_normalized is pdfmeta.was_ocr_normalized


# --- guard: convert.* не импортируется в acquire/discovery (зеркало К6 convert-
# knowledge-seam-hardening) ---


def _convert_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return {m for m in imported if m == "convert" or m.startswith("convert.")}


def test_acquire_and_discovery_do_not_import_convert_layer() -> None:
    """Оба слоевых импорта convert (recheck.deep_baseline, snowball.py:37) закрыты
    переездом в core/pdfmeta.py (spec §1) — регрессия сюда не должна проходить
    ревью молча. Тесты (``tests/``) исключены: они легитимно тестируют convert
    напрямую."""
    offenders: dict[str, set[str]] = {}
    for layer_dir in ("acquire", "discovery"):
        for path in sorted((_PIPELINE_DIR / layer_dir).rglob("*.py")):
            found = _convert_imports(path)
            if found:
                offenders[str(path.relative_to(_PIPELINE_DIR))] = found
    assert not offenders, f"convert-слой импортирован вне convert: {offenders}"

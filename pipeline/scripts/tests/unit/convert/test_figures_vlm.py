"""Тесты figures_vlm.py (spec convert-cloud-tier §5): скан маркеров, кэш
.figures.yaml, детерминированная инъекция, идемпотентность. Сеть/pdfplumber
рендер — мокнуты (сеть — только @cloud, в CI пропускается)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from convert import pdf_graphics, pdf_to_markdown
from convert.figures_vlm import (
    FIG_PROMPT,
    _content_bbox,
    _docx_media_uri,
    _render_docx_group,
    _render_xlsx_chart,
    _soffice_available,
    apply_figures_pass,
    has_bare_markers,
)
from tests.support import build_docx_with_shape_group, build_minimal_docx


def _region(rid: str, bbox: pdf_graphics.BBox = (0.0, 0.0, 10.0, 10.0), kind: str = "opaque") -> pdf_graphics.Region:
    return pdf_graphics.Region(bbox=bbox, elements=[], words=[], kind=kind, id=rid)


def _image(iid_source: str, bbox: pdf_graphics.BBox = (0.0, 0.0, 10.0, 10.0)) -> pdf_graphics.Element:
    x0, top, x1, bottom = bbox
    return pdf_graphics.Element("image", x0, top, x1, bottom, content_hash=iid_source)


def _doc_graphics(
    n_pages: int,
    *,
    regions: list[tuple[int, Any]] | None = None,
    raster: list[tuple[int, Any]] | None = None,
) -> Any:
    """``regions``/``raster`` — список (page_num, объект): по умолчанию задаётся
    страница явно (не «последняя»/«единственная»), т.к. один документ может
    нести маркеры на РАЗНЫХ страницах одновременно."""
    pages = [pdf_to_markdown.PageGraphics(i, 600.0, 800.0, [], [], [], []) for i in range(1, n_pages + 1)]
    for page_num, region in regions or []:
        pages[page_num - 1].regions.append(region)
    for page_num, image in raster or []:
        pages[page_num - 1].raster_targets.append(image)
    return pdf_to_markdown.DocGraphics(stats=None, pages=pages)  # type: ignore[arg-type]


class _FakePage:
    bbox = (0.0, 0.0, 612.0, 792.0)

    def __init__(self) -> None:
        self.cropped_bboxes: list[Any] = []

    def crop(self, bbox: Any) -> "_FakePage":
        self.cropped_bboxes.append(bbox)
        return self

    def to_image(self, resolution: int) -> "_FakePage":
        return self

    @property
    def original(self) -> Any:
        from PIL import Image

        return Image.new("RGB", (4, 4), color="white")


class _FakePdf:
    def __init__(self, n_pages: int) -> None:
        self.pages = [_FakePage() for _ in range(n_pages)]
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _patch_pdfplumber(monkeypatch: Any, n_pages: int = 10) -> None:
    monkeypatch.setattr("convert.figures_vlm.pdfplumber.open", lambda raw: _FakePdf(n_pages))


def _patch_key(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def _write_doc(tmp_path: Path, text: str) -> tuple[Path, Path]:
    raw = tmp_path / "raw.pdf"
    raw.write_bytes(b"pdf bytes")
    md = tmp_path / "doc.md"
    md.write_text(text, encoding="utf-8")
    return md, raw


FIGURE_MD = (
    "# Title\n\n"
    "> [Figure, p. 6, region 6eb947f5358b — structure not reconstructed]\n"
    "> Labels (reading order not guaranteed): MCP; Protocols; Agent\n\n"
    "Body text follows.\n"
)

IMAGE_MD = (
    "# Title\n\n"
    "> [Image, p. 20, image bbde82b91e13 — raster content not analyzed]\n\n"
    "Body text follows.\n"
)


def test_noop_when_no_markers(tmp_path: Path, monkeypatch: Any) -> None:
    md, raw = _write_doc(tmp_path, "# Title\n\nJust prose, no markers.\n")
    _patch_key(monkeypatch)
    assert apply_figures_pass(md, raw, model="m") is False
    assert md.read_text(encoding="utf-8") == "# Title\n\nJust prose, no markers.\n"


def test_missing_key_raises(tmp_path: Path, monkeypatch: Any) -> None:
    md, raw = _write_doc(tmp_path, FIGURE_MD)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        apply_figures_pass(md, raw, model="m")


def test_cache_hit_skips_vlm_call_and_injects(tmp_path: Path, monkeypatch: Any) -> None:
    md, raw = _write_doc(tmp_path, FIGURE_MD)
    _patch_key(monkeypatch)
    cache_path = raw.parent / ".figures.yaml"
    cache_path.write_text(
        yaml.safe_dump({"6eb947f5358b": {"model": "cached-model", "markdown": "Cached prose.", "requested": "2026-01-01"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("сеть не должна была вызываться на кэш-хите")),
    )
    monkeypatch.setattr(
        "convert.figures_vlm.pdf_to_markdown.compute_page_graphics",
        lambda raw_: (_ for _ in ()).throw(AssertionError("пере-детекция не нужна на кэш-хите")),
    )

    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    text = md.read_text(encoding="utf-8")
    assert "> [Figure, p. 6, region 6eb947f5358b — VLM interpretation (cached-model); " \
        "reconstruction, verify against original]\n\nCached prose." in text
    assert "structure not reconstructed" not in text


def test_cache_miss_calls_vlm_once_and_persists(tmp_path: Path, monkeypatch: Any) -> None:
    md, raw = _write_doc(tmp_path, FIGURE_MD)
    _patch_key(monkeypatch)
    _patch_pdfplumber(monkeypatch)
    monkeypatch.setattr(
        "convert.figures_vlm.pdf_to_markdown.compute_page_graphics",
        lambda raw_: _doc_graphics(6, regions=[(6, _region("6eb947f5358b"))]),
    )
    calls: list[dict[str, Any]] = []

    def fake_chat(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        calls.append(payload)
        return {"choices": [{"message": {"content": "Fresh prose about protocols."}}]}

    monkeypatch.setattr("convert.figures_vlm.openrouter.chat_request", fake_chat)

    changed = apply_figures_pass(md, raw, model="fresh-model")
    assert changed is True
    assert len(calls) == 1
    assert calls[0]["model"] == "fresh-model"
    text = md.read_text(encoding="utf-8")
    assert "Fresh prose about protocols." in text
    assert "region 6eb947f5358b — VLM interpretation (fresh-model)" in text

    cache = yaml.safe_load((raw.parent / ".figures.yaml").read_text(encoding="utf-8"))
    assert cache["6eb947f5358b"]["model"] == "fresh-model"
    assert cache["6eb947f5358b"]["markdown"] == "Fresh prose about protocols."


def test_image_marker_matched_and_injected(tmp_path: Path, monkeypatch: Any) -> None:
    md, raw = _write_doc(tmp_path, IMAGE_MD)
    _patch_key(monkeypatch)
    _patch_pdfplumber(monkeypatch, n_pages=20)
    monkeypatch.setattr(
        "convert.figures_vlm.pdf_to_markdown.compute_page_graphics",
        lambda raw_: _doc_graphics(20, raster=[(20, _image("bbde82b91e13" + "0" * 52))]),
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda payload, *, api_key, timeout=1800.0: {"choices": [{"message": {"content": "A raster chart."}}]},
    )

    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    text = md.read_text(encoding="utf-8")
    assert "image bbde82b91e13 — VLM interpretation (m); reconstruction, verify against original" in text
    assert "A raster chart." in text
    assert "raster content not analyzed" not in text


def test_region_not_found_on_redetection_warns_and_skips(tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    md, raw = _write_doc(tmp_path, FIGURE_MD)
    _patch_key(monkeypatch)
    monkeypatch.setattr(
        "convert.figures_vlm.pdf_to_markdown.compute_page_graphics",
        lambda raw_: _doc_graphics(6, regions=[]),  # id не найден
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("VLM не должен был вызываться")),
    )

    import logging

    with caplog.at_level(logging.WARNING):
        changed = apply_figures_pass(md, raw, model="m")
    assert changed is False
    assert md.read_text(encoding="utf-8") == FIGURE_MD  # маркер не тронут
    assert "не найден при пере-детекции" in caplog.text


def test_vlm_failure_leaves_marker_unchanged(tmp_path: Path, monkeypatch: Any) -> None:
    md, raw = _write_doc(tmp_path, FIGURE_MD)
    _patch_key(monkeypatch)
    _patch_pdfplumber(monkeypatch)
    monkeypatch.setattr(
        "convert.figures_vlm.pdf_to_markdown.compute_page_graphics",
        lambda raw_: _doc_graphics(6, regions=[(6, _region("6eb947f5358b"))]),
    )

    def failing_chat(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("OpenRouter: исчерпаны попытки")

    monkeypatch.setattr("convert.figures_vlm.openrouter.chat_request", failing_chat)

    changed = apply_figures_pass(md, raw, model="m")
    assert changed is False
    assert md.read_text(encoding="utf-8") == FIGURE_MD
    assert not (raw.parent / ".figures.yaml").exists()


def test_idempotent_second_run_is_byte_identical_noop(tmp_path: Path, monkeypatch: Any) -> None:
    md, raw = _write_doc(tmp_path, FIGURE_MD)
    _patch_key(monkeypatch)
    _patch_pdfplumber(monkeypatch)
    monkeypatch.setattr(
        "convert.figures_vlm.pdf_to_markdown.compute_page_graphics",
        lambda raw_: _doc_graphics(6, regions=[(6, _region("6eb947f5358b"))]),
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda payload, *, api_key, timeout=1800.0: {"choices": [{"message": {"content": "Prose."}}]},
    )

    assert apply_figures_pass(md, raw, model="m") is True
    once = md.read_text(encoding="utf-8")

    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("двойной прогон не должен звать сеть")),
    )
    monkeypatch.setattr(
        "convert.figures_vlm.pdf_to_markdown.compute_page_graphics",
        lambda raw_: (_ for _ in ()).throw(AssertionError("двойной прогон не должен пере-детектировать")),
    )
    changed_again = apply_figures_pass(md, raw, model="m")
    assert changed_again is False
    assert md.read_text(encoding="utf-8") == once  # байт-в-байт


def test_multiple_markers_processed_independently(tmp_path: Path, monkeypatch: Any) -> None:
    """Один маркер успешен, другой не находит регион — первый инъецируется,
    второй остаётся честным маркером; документ в итоге изменён (True)."""
    text = FIGURE_MD + "\n> [Figure, p. 10, region deadbeefcafe — structure not reconstructed]\n" \
        "> Labels (reading order not guaranteed): X\n"
    md, raw = _write_doc(tmp_path, text)
    _patch_key(monkeypatch)
    _patch_pdfplumber(monkeypatch, n_pages=10)
    monkeypatch.setattr(
        "convert.figures_vlm.pdf_to_markdown.compute_page_graphics",
        lambda raw_: _doc_graphics(10, regions=[(6, _region("6eb947f5358b"))]),  # deadbeefcafe отсутствует
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda payload, *, api_key, timeout=1800.0: {"choices": [{"message": {"content": "Prose."}}]},
    )

    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    out = md.read_text(encoding="utf-8")
    assert "region 6eb947f5358b — VLM interpretation" in out
    assert "region deadbeefcafe — structure not reconstructed" in out  # непойманный маркер остался как есть


def test_prompt_requires_quoted_mermaid_labels_and_visible_edges_only() -> None:
    """Грамматика промпта — строковый тест (spec §5, урок пилота: непроцитированные
    лейблы со скобками ломают mermaid-парсер; выдуманные рёбра — известный класс
    ошибки VLM на фигурах, пилот)."""
    assert "double quotes" in FIG_PROMPT
    assert "visually present" in FIG_PROMPT
    assert "never infer or guess" in FIG_PROMPT
    assert "verbatim" in FIG_PROMPT


def test_render_crop_clamps_bbox_to_page_bounds(tmp_path: Path, monkeypatch: Any) -> None:
    """Живой случай приёмки чекпоинта 2 (обложка sg): bbox изображения выходит за
    MediaBox — без клампа pdfplumber.crop поднимает ValueError, и маркер честно, но
    НАВСЕГДА оставался бы необработанным. Кламп к границам страницы чинит класс."""
    md, raw = _write_doc(tmp_path, IMAGE_MD)
    _patch_key(monkeypatch)
    fake_pdf = _FakePdf(n_pages=20)
    monkeypatch.setattr("convert.figures_vlm.pdfplumber.open", lambda raw_: fake_pdf)
    oob = _image("bbde82b91e13" + "0" * 52, bbox=(-1.25, -0.65, 611.74, 806.45))
    monkeypatch.setattr(
        "convert.figures_vlm.pdf_to_markdown.compute_page_graphics",
        lambda raw_: _doc_graphics(20, raster=[(20, oob)]),
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda payload, *, api_key, timeout=1800.0: {"choices": [{"message": {"content": "Cover art."}}]},
    )

    assert apply_figures_pass(md, raw, model="m") is True
    assert fake_pdf.pages[19].cropped_bboxes == [(0.0, 0.0, 611.74, 792.0)]


def test_warm_cache_injection_works_offline_without_key(tmp_path: Path, monkeypatch: Any) -> None:
    """Ключ требуется ЛЕНИВО (только на cache-miss): реинъекция с тёплым кэшем —
    полностью офлайн и без ключа. На этом стоит golden-самосверка @corpus."""
    md, raw = _write_doc(tmp_path, FIGURE_MD)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (raw.parent / ".figures.yaml").write_text(
        yaml.safe_dump({"6eb947f5358b": {"model": "m", "markdown": "Cached.", "requested": "2026-01-01"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("сеть недопустима")),
    )
    assert apply_figures_pass(md, raw, model="m") is True
    assert "Cached." in md.read_text(encoding="utf-8")


# --- docx-ветка (spec convert-docx §2-bis): маркер без номера страницы, рендер =
# извлечение из zip по id (без кропа), мимо-тип по расширению ---


def _docx_image_md(marker_id: str) -> str:
    return (
        "# Title\n\nBody prose.\n\n"
        "## Figures (position unknown)\n\n"
        f"> [Image, docx media {marker_id} — raster content not analyzed]\n"
    )


def _write_docx_doc(tmp_path: Path, text: str, *, media: dict[str, bytes] | None = None) -> tuple[Path, Path]:
    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_minimal_docx(["placeholder"], media=media or {}))
    md = tmp_path / "doc.md"
    md.write_text(text, encoding="utf-8")
    return md, raw


def test_docx_image_marker_detected_by_has_bare_markers() -> None:
    marker_id = hashlib.sha256(b"x" * 6000).hexdigest()[:12]
    assert has_bare_markers(_docx_image_md(marker_id)) is True


def test_docx_media_uri_finds_by_id_and_encodes_png_mime(tmp_path: Path) -> None:
    data = b"z" * 6000
    marker_id = hashlib.sha256(data).hexdigest()[:12]
    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_minimal_docx(["Body."], media={"chart.png": data}))
    uri = _docx_media_uri(raw, marker_id)
    assert uri is not None
    assert uri.startswith("data:image/png;base64,")


def test_docx_media_uri_jpeg_mime(tmp_path: Path) -> None:
    data = b"j" * 6000
    marker_id = hashlib.sha256(data).hexdigest()[:12]
    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_minimal_docx(["Body."], media={"photo.jpg": data}))
    uri = _docx_media_uri(raw, marker_id)
    assert uri is not None
    assert uri.startswith("data:image/jpeg;base64,")


def test_docx_media_uri_non_raster_format_returns_none_and_warns(tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    import logging

    data = b"s" * 6000
    marker_id = hashlib.sha256(data).hexdigest()[:12]
    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_minimal_docx(["Body."], media={"vector.svg": data}))
    with caplog.at_level(logging.WARNING):
        uri = _docx_media_uri(raw, marker_id)
    assert uri is None
    assert "не растр" in caplog.text


def test_docx_media_uri_not_found_returns_none_and_warns(tmp_path: Path, caplog: Any) -> None:
    import logging

    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_minimal_docx(["Body."]))
    with caplog.at_level(logging.WARNING):
        uri = _docx_media_uri(raw, "0" * 12)
    assert uri is None
    assert "не найдено" in caplog.text


def test_apply_figures_pass_docx_cache_miss_calls_vlm_with_media_bytes(tmp_path: Path, monkeypatch: Any) -> None:
    data = b"c" * 6000
    marker_id = hashlib.sha256(data).hexdigest()[:12]
    md, raw = _write_docx_doc(tmp_path, _docx_image_md(marker_id), media={"chart.png": data})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_chat(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        calls.append(payload)
        return {"choices": [{"message": {"content": "Docx chart description."}}]}

    monkeypatch.setattr("convert.figures_vlm.openrouter.chat_request", fake_chat)
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    assert len(calls) == 1
    img_parts = [p for p in calls[0]["messages"][0]["content"] if p["type"] == "image_url"]
    assert len(img_parts) == 1
    assert img_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    text = md.read_text(encoding="utf-8")
    assert (
        f"> [Image, docx media {marker_id} — VLM interpretation (m); "
        "reconstruction, verify against original]" in text
    )
    assert "Docx chart description." in text
    assert "raster content not analyzed" not in text


def test_apply_figures_pass_docx_cache_hit_skips_network(tmp_path: Path, monkeypatch: Any) -> None:
    data = b"h" * 6000
    marker_id = hashlib.sha256(data).hexdigest()[:12]
    md, raw = _write_docx_doc(tmp_path, _docx_image_md(marker_id), media={"chart.png": data})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    (raw.parent / ".figures.yaml").write_text(
        yaml.safe_dump({marker_id: {"model": "cached", "markdown": "Cached docx figure.", "requested": "2026-01-01"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("сеть не должна была вызываться")),
    )
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    assert "Cached docx figure." in md.read_text(encoding="utf-8")


def test_apply_figures_pass_docx_media_not_found_warns_and_skips(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    import logging

    marker_id = "0" * 12
    text = _docx_image_md(marker_id)
    md, raw = _write_docx_doc(tmp_path, text)  # media вовсе нет
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("VLM не должен был вызываться")),
    )
    with caplog.at_level(logging.WARNING):
        changed = apply_figures_pass(md, raw, model="m")
    assert changed is False
    assert md.read_text(encoding="utf-8") == text
    assert "не найдено" in caplog.text


def test_apply_figures_pass_docx_idempotent_second_run(tmp_path: Path, monkeypatch: Any) -> None:
    data = b"i" * 6000
    marker_id = hashlib.sha256(data).hexdigest()[:12]
    md, raw = _write_docx_doc(tmp_path, _docx_image_md(marker_id), media={"chart.png": data})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda payload, *, api_key, timeout=1800.0: {"choices": [{"message": {"content": "Prose."}}]},
    )
    assert apply_figures_pass(md, raw, model="m") is True
    once = md.read_text(encoding="utf-8")

    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("двойной прогон не должен звать сеть")),
    )
    assert apply_figures_pass(md, raw, model="m") is False
    assert md.read_text(encoding="utf-8") == once


def test_apply_figures_pass_docx_duplicate_id_single_vlm_call(tmp_path: Path, monkeypatch: Any) -> None:
    """Два media-файла с одинаковыми байтами -> одинаковый id, ДВА вхождения
    маркера в тексте -> ОДИН вызов VLM (кэш заполняется на первом вхождении,
    второе — из кэша в рамках того же прогона; та же логика, что pdf-путь)."""
    data = b"d" * 6000
    marker_id = hashlib.sha256(data).hexdigest()[:12]
    text = _docx_image_md(marker_id) + f"\n> [Image, docx media {marker_id} — raster content not analyzed]\n"
    md, raw = _write_docx_doc(tmp_path, text, media={"a.png": data, "b.jpg": data})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: list[dict[str, Any]] = []

    def fake_chat(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        calls.append(payload)
        return {"choices": [{"message": {"content": "Shared figure."}}]}

    monkeypatch.setattr("convert.figures_vlm.openrouter.chat_request", fake_chat)
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    assert len(calls) == 1
    assert md.read_text(encoding="utf-8").count("Shared figure.") == 2


# --- docx composite-группы (spec convert-docx §2-ter): маркер + captions,
# рендер = изолированный мини-docx -> soffice -> PDF -> кроп bbox -> data-URI.
# soffice/pdfplumber — мокнуты; реальный рендер — integration-тест отдельным
# файлом (tests/integration/test_docx_groups_live.py, требует системный soffice) ---


def _docx_group_md(group_id: str, captions: str = "Cap A; Cap B") -> str:
    return (
        "# Title\n\nBody prose.\n\n"
        f"> [Figure, docx group {group_id} — composite content not analyzed]\n"
        f"> captions: {captions}\n"
    )


def _build_group_raw(tmp_path: Path, captions: list[str], images: dict[str, bytes]) -> tuple[Path, str]:
    from convert import docx_groups

    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_docx_with_shape_group([], captions, images, []))
    _rewritten, groups = docx_groups.extract_and_strip_groups(raw)
    return raw, groups[0].id12


class _FakeGroupPage:
    rects: list[Any] = []
    curves: list[Any] = []
    images: list[Any] = []
    chars: list[Any] = []
    bbox = (0.0, 0.0, 595.0, 842.0)

    def crop(self, bbox: Any) -> "_FakeGroupPage":
        return self

    def to_image(self, resolution: int) -> "_FakeGroupPage":
        return self

    @property
    def original(self) -> Any:
        from PIL import Image

        return Image.new("RGB", (4, 4), color="white")


class _FakeGroupPdf:
    def __init__(self) -> None:
        self.pages = [_FakeGroupPage()]

    def __enter__(self) -> "_FakeGroupPdf":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


def test_docx_group_marker_detected_by_has_bare_markers() -> None:
    assert has_bare_markers(_docx_group_md("a" * 12)) is True


def test_soffice_available_reflects_shutil_which(monkeypatch: Any) -> None:
    monkeypatch.setattr("convert.figures_vlm.shutil.which", lambda name: "/usr/bin/soffice")
    assert _soffice_available() is True
    monkeypatch.setattr("convert.figures_vlm.shutil.which", lambda name: None)
    assert _soffice_available() is False


def test_content_bbox_empty_page_returns_none() -> None:
    assert _content_bbox(_FakeGroupPage()) is None


def test_content_bbox_unions_elements() -> None:
    class _Page:
        rects = [{"x0": 10.0, "x1": 20.0, "top": 5.0, "bottom": 15.0}]
        curves: list[Any] = []
        images = [{"x0": 0.0, "x1": 30.0, "top": 2.0, "bottom": 8.0}]
        chars: list[Any] = []
        bbox = (0.0, 0.0, 595.0, 842.0)

    bbox = _content_bbox(_Page())
    assert bbox == (0.0, 2.0, 30.0, 15.0)


def test_content_bbox_clamped_to_page_box() -> None:
    """LO может выложить фантомный элемент мини-docx частично за край страницы
    (живой кейс ultimate-теста: top=-57pt у чарта) — объединение клэмпится к
    bbox страницы, иначе page.crop() падает; видимого контента за страницей
    нет по построению (PDF обрезает)."""

    class _Page:
        rects = [{"x0": 79.0, "x1": 559.0, "top": -57.3, "bottom": 422.6}]
        curves: list[Any] = []
        images: list[Any] = []
        chars: list[Any] = []
        bbox = (0.0, 0.0, 595.3, 841.9)

    assert _content_bbox(_Page()) == (79.0, 0.0, 559.0, 422.6)


def test_content_bbox_fully_offpage_returns_none() -> None:
    class _Page:
        rects = [{"x0": -50.0, "x1": -10.0, "top": 5.0, "bottom": 15.0}]
        curves: list[Any] = []
        images: list[Any] = []
        chars: list[Any] = []
        bbox = (0.0, 0.0, 595.0, 842.0)

    assert _content_bbox(_Page()) is None


def test_render_docx_group_no_soffice_returns_none_and_warns(tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    import logging

    raw, gid = _build_group_raw(tmp_path, ["Cap"], {})
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        result = _render_docx_group(raw, gid)
    assert result is None
    assert "soffice не установлен" in caplog.text


def test_render_docx_group_extraction_miss_returns_none_and_warns(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    import logging

    raw, _gid = _build_group_raw(tmp_path, ["Cap"], {})
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: True)
    with caplog.at_level(logging.WARNING):
        result = _render_docx_group(raw, "0" * 12)  # такого id в документе нет
    assert result is None
    assert "не найдена при пере-детекции" in caplog.text


def test_render_docx_group_soffice_nonzero_exit_returns_none_and_warns(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    import logging

    raw, gid = _build_group_raw(tmp_path, ["Cap"], {})
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: True)

    class _Result:
        returncode = 1
        stderr = "boom"

    monkeypatch.setattr("convert.figures_vlm.subprocess.run", lambda *a, **kw: _Result())
    with caplog.at_level(logging.WARNING):
        result = _render_docx_group(raw, gid)
    assert result is None
    assert "не смог отрендерить" in caplog.text


def test_render_docx_group_soffice_timeout_returns_none_and_warns(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    import logging
    import subprocess as sp

    raw, gid = _build_group_raw(tmp_path, ["Cap"], {})
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: True)

    def fake_run(*a: Any, **kw: Any) -> Any:
        raise sp.TimeoutExpired(cmd="soffice", timeout=60)

    monkeypatch.setattr("convert.figures_vlm.subprocess.run", fake_run)
    with caplog.at_level(logging.WARNING):
        result = _render_docx_group(raw, gid)
    assert result is None
    assert "не уложился" in caplog.text


def test_render_docx_group_success_returns_data_uri(tmp_path: Path, monkeypatch: Any) -> None:
    raw, gid = _build_group_raw(tmp_path, ["Cap"], {})
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: True)

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd: list[str], *, check: bool, capture_output: bool, text: bool, timeout: float) -> Any:
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "obj.pdf").write_bytes(b"%PDF-fake")
        return _Result()

    monkeypatch.setattr("convert.figures_vlm.subprocess.run", fake_run)
    monkeypatch.setattr("convert.figures_vlm.pdfplumber.open", lambda path: _FakeGroupPdf())
    result = _render_docx_group(raw, gid)
    assert result is not None
    assert result.startswith("data:image/jpeg;base64,")


def test_apply_figures_pass_docx_group_cache_miss_calls_render_and_vlm(tmp_path: Path, monkeypatch: Any) -> None:
    raw, gid = _build_group_raw(tmp_path, ["Cap A", "Cap B"], {})
    md = tmp_path / "doc.md"
    md.write_text(_docx_group_md(gid, "Cap A; Cap B"), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "convert.figures_vlm._render_docx_group", lambda raw_, id12: "data:image/jpeg;base64,AAA"
    )
    calls: list[dict[str, Any]] = []

    def fake_chat(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        calls.append(payload)
        return {"choices": [{"message": {"content": "Group description."}}]}

    monkeypatch.setattr("convert.figures_vlm.openrouter.chat_request", fake_chat)
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    assert len(calls) == 1
    text = md.read_text(encoding="utf-8")
    assert (
        f"> [Figure, docx group {gid} — VLM interpretation (m); "
        "reconstruction, verify against original]" in text
    )
    assert "Group description." in text
    assert "composite content not analyzed" not in text


def _docx_chart_md(chart_id: str, captions: str = "Chart title") -> str:
    return (
        "# Title\n\nBody prose.\n\n"
        f"> [Figure, docx chart {chart_id} — chart content not analyzed]\n"
        f"> captions: {captions}\n"
    )


def test_docx_chart_marker_detected_by_has_bare_markers() -> None:
    assert has_bare_markers(_docx_chart_md("b" * 12)) is True


def test_apply_figures_pass_docx_chart_injects_chart_noun(tmp_path: Path, monkeypatch: Any) -> None:
    """kind="chart" (§2-ter, нативный c:chart): единый цикл обработки с
    группами, но существительное маркера сохраняется и в инъецированном виде."""
    from convert import docx_groups
    from tests.support import build_docx_with_inline_chart

    raw = tmp_path / "raw.docx"
    raw.write_bytes(build_docx_with_inline_chart([], ["Chart title"], []))
    _rewritten, groups = docx_groups.extract_and_strip_groups(raw)
    cid = groups[0].id12
    md = tmp_path / "doc.md"
    md.write_text(_docx_chart_md(cid), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "convert.figures_vlm._render_docx_group", lambda raw_, id12: "data:image/jpeg;base64,AAA"
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda payload, *, api_key, timeout=1800.0: {
            "choices": [{"message": {"content": "Chart description."}}]
        },
    )
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    text = md.read_text(encoding="utf-8")
    assert (
        f"> [Figure, docx chart {cid} — VLM interpretation (m); "
        "reconstruction, verify against original]" in text
    )
    assert "Chart description." in text
    assert "chart content not analyzed" not in text


def test_apply_figures_pass_docx_group_cache_hit_skips_render(tmp_path: Path, monkeypatch: Any) -> None:
    raw, gid = _build_group_raw(tmp_path, ["Cap"], {})
    md = tmp_path / "doc.md"
    md.write_text(_docx_group_md(gid), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    (raw.parent / ".figures.yaml").write_text(
        yaml.safe_dump({gid: {"model": "cached", "markdown": "Cached group.", "requested": "2026-01-01"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "convert.figures_vlm._render_docx_group",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("рендер не должен был вызываться")),
    )
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    assert "Cached group." in md.read_text(encoding="utf-8")


def test_apply_figures_pass_docx_group_render_failure_leaves_marker_unchanged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    raw, gid = _build_group_raw(tmp_path, ["Cap"], {})
    text = _docx_group_md(gid)
    md = tmp_path / "doc.md"
    md.write_text(text, encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("convert.figures_vlm._render_docx_group", lambda raw_, id12: None)
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("VLM не должен был вызываться")),
    )
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is False
    assert md.read_text(encoding="utf-8") == text


def test_apply_figures_pass_docx_group_idempotent_second_run(tmp_path: Path, monkeypatch: Any) -> None:
    raw, gid = _build_group_raw(tmp_path, ["Cap"], {})
    md = tmp_path / "doc.md"
    md.write_text(_docx_group_md(gid), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "convert.figures_vlm._render_docx_group", lambda raw_, id12: "data:image/jpeg;base64,AAA"
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda payload, *, api_key, timeout=1800.0: {"choices": [{"message": {"content": "Prose."}}]},
    )
    assert apply_figures_pass(md, raw, model="m") is True
    once = md.read_text(encoding="utf-8")

    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("двойной прогон не должен звать сеть")),
    )
    assert apply_figures_pass(md, raw, model="m") is False
    assert md.read_text(encoding="utf-8") == once


# --- xlsx-чарты (spec convert-xlsx §3, 5-я грамматика маркера): рендер —
# soffice/pdfplumber мокнуты (та же схема, что docx-группы, _render_via_soffice
# общий); реальный рендер — integration-тест отдельным файлом
# (tests/integration/test_xlsx_charts_live.py, требует системный soffice) ---


def _xlsx_chart_md(chart_id: str, sheet: str = "Data", anchor: str = "D2", captions: str = "Chart Title") -> str:
    return (
        "## Data\n\n| Cat | Val |\n| --- | --- |\n| A | 1 |\n\n"
        f"> [Figure, xlsx chart {chart_id} on {sheet}!{anchor} — chart content not analyzed]\n"
        f"> captions: {captions}\n"
    )


def _build_chart_raw(
    tmp_path: Path, *, title: str = "Chart Title", anchor: str = "D2", sheet_name: str = "Data"
) -> tuple[Path, str]:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference

    from convert import xlsx_charts

    wb = Workbook()
    ws: Any = wb.active
    assert ws is not None
    ws.title = sheet_name
    ws.append(["Cat", "Val"])
    ws.append(["A", 1])
    chart = BarChart()
    chart.title = title
    chart.add_data(Reference(ws, min_col=2, min_row=1, max_row=2), titles_from_data=True)
    ws.add_chart(chart, anchor)
    raw = tmp_path / "raw.xlsx"
    wb.save(raw)
    return raw, xlsx_charts.extract_charts(raw)[0].id12


def test_xlsx_chart_marker_detected_by_has_bare_markers() -> None:
    assert has_bare_markers(_xlsx_chart_md("a" * 12)) is True


def test_render_xlsx_chart_no_soffice_returns_none_and_warns(tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    import logging

    raw, cid = _build_chart_raw(tmp_path)
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        result = _render_xlsx_chart(raw, cid)
    assert result is None
    assert "soffice не установлен" in caplog.text


def test_render_xlsx_chart_extraction_miss_returns_none_and_warns(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    import logging

    raw, _cid = _build_chart_raw(tmp_path)
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: True)
    with caplog.at_level(logging.WARNING):
        result = _render_xlsx_chart(raw, "0" * 12)  # такого id в книге нет
    assert result is None
    assert "не найден при пере-детекции" in caplog.text


def test_render_xlsx_chart_soffice_nonzero_exit_returns_none_and_warns(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    import logging

    raw, cid = _build_chart_raw(tmp_path)
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: True)

    class _Result:
        returncode = 1
        stderr = "boom"

    monkeypatch.setattr("convert.figures_vlm.subprocess.run", lambda *a, **kw: _Result())
    with caplog.at_level(logging.WARNING):
        result = _render_xlsx_chart(raw, cid)
    assert result is None
    assert "не смог отрендерить" in caplog.text


def test_render_xlsx_chart_soffice_timeout_returns_none_and_warns(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    import logging
    import subprocess as sp

    raw, cid = _build_chart_raw(tmp_path)
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: True)

    def fake_run(*a: Any, **kw: Any) -> Any:
        raise sp.TimeoutExpired(cmd="soffice", timeout=60)

    monkeypatch.setattr("convert.figures_vlm.subprocess.run", fake_run)
    with caplog.at_level(logging.WARNING):
        result = _render_xlsx_chart(raw, cid)
    assert result is None
    assert "не уложился" in caplog.text


def test_render_xlsx_chart_success_returns_data_uri(tmp_path: Path, monkeypatch: Any) -> None:
    raw, cid = _build_chart_raw(tmp_path)
    monkeypatch.setattr("convert.figures_vlm._soffice_available", lambda: True)

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd: list[str], *, check: bool, capture_output: bool, text: bool, timeout: float) -> Any:
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "obj.pdf").write_bytes(b"%PDF-fake")
        return _Result()

    monkeypatch.setattr("convert.figures_vlm.subprocess.run", fake_run)
    monkeypatch.setattr("convert.figures_vlm.pdfplumber.open", lambda path: _FakeGroupPdf())
    result = _render_xlsx_chart(raw, cid)
    assert result is not None
    assert result.startswith("data:image/jpeg;base64,")


def test_apply_figures_pass_xlsx_chart_cache_miss_calls_render_and_vlm(tmp_path: Path, monkeypatch: Any) -> None:
    raw, cid = _build_chart_raw(tmp_path)
    md = tmp_path / "doc.md"
    md.write_text(_xlsx_chart_md(cid), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "convert.figures_vlm._render_xlsx_chart", lambda raw_, id12: "data:image/jpeg;base64,AAA"
    )
    calls: list[dict[str, Any]] = []

    def fake_chat(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
        calls.append(payload)
        return {"choices": [{"message": {"content": "Chart description."}}]}

    monkeypatch.setattr("convert.figures_vlm.openrouter.chat_request", fake_chat)
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    assert len(calls) == 1
    text = md.read_text(encoding="utf-8")
    assert (
        f"> [Figure, xlsx chart {cid} on Data!D2 — VLM interpretation (m); "
        "reconstruction, verify against original]" in text
    )
    assert "Chart description." in text
    assert "chart content not analyzed" not in text


def test_apply_figures_pass_xlsx_chart_cache_hit_skips_render(tmp_path: Path, monkeypatch: Any) -> None:
    raw, cid = _build_chart_raw(tmp_path)
    md = tmp_path / "doc.md"
    md.write_text(_xlsx_chart_md(cid), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    (raw.parent / ".figures.yaml").write_text(
        yaml.safe_dump({cid: {"model": "cached", "markdown": "Cached chart.", "requested": "2026-01-01"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "convert.figures_vlm._render_xlsx_chart",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("рендер не должен был вызываться")),
    )
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is True
    assert "Cached chart." in md.read_text(encoding="utf-8")


def test_apply_figures_pass_xlsx_chart_render_failure_leaves_marker_unchanged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    raw, cid = _build_chart_raw(tmp_path)
    text = _xlsx_chart_md(cid)
    md = tmp_path / "doc.md"
    md.write_text(text, encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("convert.figures_vlm._render_xlsx_chart", lambda raw_, id12: None)
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("VLM не должен был вызываться")),
    )
    changed = apply_figures_pass(md, raw, model="m")
    assert changed is False
    assert md.read_text(encoding="utf-8") == text


def test_apply_figures_pass_xlsx_chart_idempotent_second_run(tmp_path: Path, monkeypatch: Any) -> None:
    raw, cid = _build_chart_raw(tmp_path)
    md = tmp_path / "doc.md"
    md.write_text(_xlsx_chart_md(cid), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "convert.figures_vlm._render_xlsx_chart", lambda raw_, id12: "data:image/jpeg;base64,AAA"
    )
    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda payload, *, api_key, timeout=1800.0: {"choices": [{"message": {"content": "Prose."}}]},
    )
    assert apply_figures_pass(md, raw, model="m") is True
    once = md.read_text(encoding="utf-8")

    monkeypatch.setattr(
        "convert.figures_vlm.openrouter.chat_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("двойной прогон не должен звать сеть")),
    )
    assert apply_figures_pass(md, raw, model="m") is False
    assert md.read_text(encoding="utf-8") == once

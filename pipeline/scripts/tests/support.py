"""Общие фабрики тестовых данных: валидная запись корпуса + папка-документ на диске.

Выделены из test_schema.py при введении слоевой иерархии тестов (feature/repo-layout):
хелперы, разделяемые тестами разных слоёв, живут в support-модуле пакета tests,
а не импортируются из чужого тест-файла.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import yaml

_DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_DOCX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def build_minimal_docx(paragraphs: list[str], *, media: dict[str, bytes] | None = None) -> bytes:
    """Минимальный валидный OOXML (spec convert-docx §Тестовое покрытие): ровно три
    члена zip-архива (``[Content_Types].xml``/``_rels/.rels``/``word/document.xml``) —
    без ``styles.xml``/``fontTable.xml``/``docProps`` и т.п., которые Word пишет, но
    markitdown/mammoth для чтения не требуют (проверено эмпирически: минимальный
    3-member docx парсится безошибочно). Ни одного бинарника в git — фикстура рождается
    в тесте каждый раз заново.

    ``media`` (spec §2-bis) — произвольные ``{имя: байты}`` под ``word/media/`` —
    маркер-код листингует эту папку НАПРЯМУЮ, без сверки с relationships (§2-bis
    design), поэтому синтетические байты не обязаны декодироваться как настоящая
    картинка."""
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paragraphs)
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        z.writestr("_rels/.rels", _DOCX_RELS)
        z.writestr("word/document.xml", document)
        for name, data in (media or {}).items():
            z.writestr(f"word/media/{name}", data)
    return buf.getvalue()


_DOCX_CONTENT_TYPES_IMG = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_OOXML_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_OOXML_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_OOXML_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_OOXML_PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
_OOXML_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_OOXML_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def _docx_para(text: str) -> str:
    return f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


def _docx_drawing(rid: str) -> str:
    return (
        f'<w:drawing xmlns:wp="{_OOXML_WP}"><wp:inline>'
        f'<wp:docPr id="1" name="Picture"/>'
        f'<a:graphic xmlns:a="{_OOXML_A}"><a:graphicData uri="{_OOXML_PIC}">'
        f'<pic:pic xmlns:pic="{_OOXML_PIC}"><pic:blipFill>'
        f'<a:blip r:embed="{rid}" xmlns:r="{_OOXML_R}"/></pic:blipFill></pic:pic>'
        f'</a:graphicData></a:graphic></wp:inline></w:drawing>'
    )


def _docx_zip(body: str, images: dict[str, bytes]) -> bytes:
    """Собрать docx, где каждый файл images ПОДКЛЮЧЁН relationship'ом rId100+i
    (в отличие от build_minimal_docx(media=...), чьи файлы — сироты по построению)."""
    rels_items = "".join(
        f'<Relationship Id="rId{100 + i}" Type="{_OOXML_R}/image" Target="media/{name}"/>'
        for i, name in enumerate(images)
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels_items}</Relationships>"
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_OOXML_W}"><w:body>{body}</w:body></w:document>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES_IMG)
        z.writestr("_rels/.rels", _DOCX_RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        for name, data in images.items():
            z.writestr(f"word/media/{name}", data)
    return buf.getvalue()


def build_docx_with_inline_image(
    before: list[str], image: bytes, after: list[str], image_name: str = "image1.png"
) -> bytes:
    """Картинка, по-настоящему вписанная в поток (DrawingML wp:inline + a:blip
    r:embed + rels) — mammoth инлайнит её на месте (позиционный путь v2)."""
    body = "".join(_docx_para(p) for p in before)
    body += f"<w:p><w:r>{_docx_drawing('rId100')}</w:r></w:p>"
    body += "".join(_docx_para(p) for p in after)
    return _docx_zip(body, {image_name: image})


def build_docx_with_choice_only_images(paragraphs: list[str], images: dict[str, bytes]) -> bytes:
    """Каждая картинка — ТОЛЬКО в mc:Choice при пустом mc:Fallback: mammoth
    (читающий Fallback) её НЕ инлайнит, но ссылка в document.xml есть —
    ровно класс «referenced-but-not-walked», который держит фолбэк-секцию."""
    body = "".join(_docx_para(p) for p in paragraphs)
    for i, _name in enumerate(images):
        body += (
            f'<w:p><w:r><mc:AlternateContent xmlns:mc="{_OOXML_MC}">'
            f'<mc:Choice Requires="wpg">{_docx_drawing(f"rId{100 + i}")}</mc:Choice>'
            f"<mc:Fallback/></mc:AlternateContent></w:r></w:p>"
        )
    return _docx_zip(body, images)


_OOXML_WPG = "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"


def _docx_group_ac(captions: list[str], n_images: int, *, rid_offset: int = 100) -> str:
    """mc:AlternateContent, чей mc:Choice содержит wpg:wgp (детект docx_groups
    смотрит РОВНО на это) — внутри вольные текстовые узлы (captions) и
    pic-элементы с r:embed=rId{rid_offset+i} (media_ids); mc:Fallback пуст
    (детект не смотрит туда). Не претендует на подлинность реальной
    wpg-разметки Word — минимум, достаточный для extract_and_strip_groups."""
    pics = "".join(
        f'<pic:pic xmlns:pic="{_OOXML_PIC}"><pic:blipFill>'
        f'<a:blip r:embed="rId{rid_offset + i}" xmlns:r="{_OOXML_R}"/></pic:blipFill></pic:pic>'
        for i in range(n_images)
    )
    caption_nodes = "".join(f"<a:t>{c}</a:t>" for c in captions)
    return (
        f'<mc:AlternateContent xmlns:mc="{_OOXML_MC}"><mc:Choice Requires="wpg">'
        f'<w:drawing xmlns:wp="{_OOXML_WP}"><wp:inline><wp:docPr id="1" name="Group"/>'
        f'<a:graphic xmlns:a="{_OOXML_A}"><a:graphicData uri="{_OOXML_WPG}">'
        f'<wpg:wgp xmlns:wpg="{_OOXML_WPG}">{caption_nodes}{pics}</wpg:wgp>'
        f"</a:graphicData></a:graphic></wp:inline></w:drawing>"
        f"</mc:Choice><mc:Fallback/></mc:AlternateContent>"
    )


def build_docx_with_shape_group(
    before: list[str], captions: list[str], images: dict[str, bytes], after: list[str]
) -> bytes:
    """docx с ОДНОЙ composite-группой (spec convert-docx §2-ter) — см.
    ``_docx_group_ac``."""
    group_ac = _docx_group_ac(captions, len(images))
    body = "".join(_docx_para(p) for p in before)
    body += f"<w:p><w:r>{group_ac}</w:r></w:p>"
    body += "".join(_docx_para(p) for p in after)
    return _docx_zip(body, images)


_OOXML_C = "http://schemas.openxmlformats.org/drawingml/2006/chart"


def _docx_chart_drawing(rid: str) -> str:
    """Голый w:drawing с c:chart-анкером (kind="chart" в docx_groups): нативный
    Word-чарт БЕЗ AlternateContent/Fallback — класс «mammoth молча теряет»."""
    return (
        f'<w:drawing xmlns:wp="{_OOXML_WP}"><wp:inline>'
        f'<wp:docPr id="2" name="Chart"/>'
        f'<a:graphic xmlns:a="{_OOXML_A}"><a:graphicData uri="{_OOXML_C}">'
        f'<c:chart xmlns:c="{_OOXML_C}" r:id="{rid}" xmlns:r="{_OOXML_R}"/>'
        f"</a:graphicData></a:graphic></wp:inline></w:drawing>"
    )


def _docx_chart_part(title_texts: list[str]) -> str:
    """Минимальный chart-парт: c:title с rich-текстом (источник captions
    маркера) — плюс пустой plotArea для структурной правдоподобности."""
    runs = "".join(f'<a:r xmlns:a="{_OOXML_A}"><a:t>{t}</a:t></a:r>' for t in title_texts)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<c:chartSpace xmlns:c="{_OOXML_C}"><c:chart>'
        f'<c:title><c:tx><c:rich><a:p xmlns:a="{_OOXML_A}">{runs}</a:p></c:rich></c:tx></c:title>'
        f"<c:plotArea/></c:chart></c:chartSpace>"
    )


def _docx_chart_zip(before: list[str], after: list[str], chart_part_xml: str) -> bytes:
    """Общая сборка zip под ОДИН нативный ``c:chart`` (spec convert-docx
    §2-ter, kind="chart"): drawing-анкер в body + произвольный chart-парт +
    rels/[Content_Types] — используется и минимальным (``_docx_chart_part``,
    только title) и data-driven (``_docx_chart_part_with_series``, numCache)
    билдерами."""
    body = "".join(_docx_para(p) for p in before)
    body += f"<w:p><w:r>{_docx_chart_drawing('rId200')}</w:r></w:p>"
    body += "".join(_docx_para(p) for p in after)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_OOXML_W}"><w:body>{body}</w:body></w:document>'
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId200" Type="{_OOXML_R}/chart" Target="charts/chart1.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
        "</Types>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", _DOCX_RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/charts/chart1.xml", chart_part_xml)
    return buf.getvalue()


def build_docx_with_inline_chart(
    before: list[str], title_texts: list[str], after: list[str]
) -> bytes:
    """docx с ОДНИМ нативным c:chart (spec convert-docx §2-ter, kind="chart"):
    drawing-анкер в body + chart-парт с заголовком (БЕЗ numCache — captions-only
    фикстура; для data-driven резолюции см. ``build_docx_with_inline_chart_data``)
    + rels/[Content_Types]."""
    return _docx_chart_zip(before, after, _docx_chart_part(title_texts))


def _docx_chart_part_with_series(
    title: str, categories: list[str], values: list[str], value_format: str
) -> str:
    """chart-парт с РЕАЛЬНЫМ ``c:numCache``/``c:strCache`` (spec
    chart-data-extraction §4.2) — та же DrawingML-схема ``c:chart``, что у
    xlsx (``xl/charts/*.xml`` vs ``word/charts/*.xml``); один bar-серия чарт,
    достаточный для сквозной сверки data-driven резолюции chart-kind."""
    cat_pts = "".join(f'<c:pt idx="{i}"><c:v>{c}</c:v></c:pt>' for i, c in enumerate(categories))
    val_pts = "".join(f'<c:pt idx="{i}"><c:v>{v}</c:v></c:pt>' for i, v in enumerate(values))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<c:chartSpace xmlns:c="{_OOXML_C}" xmlns:a="{_OOXML_A}"><c:chart>'
        f'<c:title><c:tx><c:rich><a:p><a:r><a:t>{title}</a:t></a:r></a:p></c:rich></c:tx></c:title>'
        '<c:plotArea><c:barChart><c:barDir val="col"/><c:grouping val="clustered"/>'
        '<c:ser><c:idx val="0"/><c:order val="0"/>'
        f'<c:cat><c:strRef><c:f>Sheet1!$A$2:$A${len(categories) + 1}</c:f>'
        f'<c:strCache><c:ptCount val="{len(categories)}"/>{cat_pts}</c:strCache></c:strRef></c:cat>'
        f'<c:val><c:numRef><c:f>Sheet1!$B$2:$B${len(values) + 1}</c:f>'
        f'<c:numCache><c:formatCode>{value_format}</c:formatCode>'
        f'<c:ptCount val="{len(values)}"/>{val_pts}</c:numCache></c:numRef></c:val>'
        "</c:ser></c:barChart></c:plotArea></c:chart></c:chartSpace>"
    )


def build_docx_with_inline_chart_data(
    before: list[str],
    after: list[str],
    *,
    title: str = "My Chart",
    categories: list[str] | None = None,
    values: list[str] | None = None,
    value_format: str = "0.0%",
) -> bytes:
    """Как ``build_docx_with_inline_chart``, но chart-парт несёт РЕАЛЬНЫЙ
    numCache — под тесты data-driven резолюции (chart-data-extraction §4.2),
    а не только caption-фолбэка."""
    cats, vals = categories or ["A", "B"], values or ["1", "2"]
    return _docx_chart_zip(
        before, after, _docx_chart_part_with_series(title, cats, vals, value_format)
    )


def build_docx_with_group_and_standalone_image(
    group_captions: list[str], group_image: bytes, standalone_image: bytes
) -> bytes:
    """Композит: одна composite-группа (§2-ter) + одна ОБЫЧНАЯ инлайн-картинка
    вне группы (§2-bis/v2, DrawingML wp:inline вне AlternateContent) — под
    регресс-guard на отсутствие перекрёстного заражения между двумя
    механизмами позиционирования в одном документе."""
    images = {"group.png": group_image, "standalone.png": standalone_image}
    group_ac = _docx_group_ac(group_captions, 1, rid_offset=100)
    body = f"<w:p><w:r>{group_ac}</w:r></w:p>"
    body += f"<w:p><w:r>{_docx_drawing('rId101')}</w:r></w:p>"
    return _docx_zip(body, images)


def build_docx_with_shape_group_and_inline_chart(
    group_captions: list[str], group_images: dict[str, bytes], chart_title_texts: list[str]
) -> bytes:
    """Композит: одна composite-группа (kind="group") + один нативный c:chart
    (kind="chart") в ОДНОМ документе (spec chart-data-extraction §2) —
    характеризационная фикстура: детект/вырезка/сентинел обоих kind делят
    общий код (``docx_groups._iter_objects``), а РЕЗОЛЮЦИЯ chart-kind меняется
    на data-driven — этот билдер даёт регресс-guard, что изменение резолюции
    chart НЕ задевает group-путь (и наоборот)."""
    group_ac = _docx_group_ac(group_captions, len(group_images))
    body = f"<w:p><w:r>{group_ac}</w:r></w:p>"
    body += f"<w:p><w:r>{_docx_chart_drawing('rId200')}</w:r></w:p>"
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_OOXML_W}"><w:body>{body}</w:body></w:document>'
    )
    image_rels = "".join(
        f'<Relationship Id="rId{100 + i}" Type="{_OOXML_R}/image" Target="media/{name}"/>'
        for i, name in enumerate(group_images)
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{image_rels}<Relationship Id="rId200" Type="{_OOXML_R}/chart" Target="charts/chart1.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
        "</Types>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", _DOCX_RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/charts/chart1.xml", _docx_chart_part(chart_title_texts))
        for name, data in group_images.items():
            z.writestr(f"word/media/{name}", data)
    return buf.getvalue()


def build_pdf(
    *,
    lines: list[tuple[str, float, float, float]] | None = None,
    table: tuple[list[list[str]], float, float, float, float] | None = None,
    page_size: tuple[float, float] = (612.0, 792.0),
) -> bytes:
    """Синтетический однострочичный PDF через reportlab (test-coverage-hardening) — реальный
    файл для тестов, которым нужны настоящие объекты ``pdfplumber.Page``/``Table``, не мок
    (пере-реализация геометрии pdfplumber мокoм тестировала бы мок, не код).

    ``lines`` — ``(text, x, y_from_top, font_size)``, ``y_from_top`` в системе координат
    pdfplumber (0 у верхнего края страницы) — функция сама переводит в bottom-up систему
    reportlab. ``table`` — ``(data, x, y_from_top, col_width, row_height)`` — таблица с
    реальной сеткой линий (``GRID``), которую ``pdfplumber.find_tables()`` детектирует своей
    штатной lines-стратегией (не наш код — сама библиотека)."""
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Table, TableStyle

    width, height = page_size
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size)
    for text, x, y_from_top, size in lines or []:
        c.setFont("Helvetica", size)
        c.drawString(x, height - y_from_top - size, text)
    if table is not None:
        data, tx, ty_from_top, col_width, row_height = table
        t = Table(data, colWidths=col_width, rowHeights=row_height)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.75, colors.black)]))
        _, th = t.wrap(0, 0)
        t.drawOn(c, tx, height - ty_from_top - th)
    c.showPage()
    c.save()
    return buf.getvalue()


def valid_record() -> dict[str, Any]:
    """Минимально валидная запись (термины — из реальных словарей pipeline/vocab/)."""
    return {
        "id": "sg-imda-mgf-agentic-2026",
        "entity_id": "sg",
        "track": "intl-xperience",
        "title": "Model AI Governance Framework for Agentic AI",
        "issuer": "Infocomm Media Development Authority (IMDA)",
        "issuer_type": "government",
        "geo_scope": "national",
        "language": "en",
        "dates": {"published": "2026-05-20", "retrieved": "2026-07-15"},
        "doc_type": "framework",
        "authority": "soft_law",
        "topics": ["ai-governance", "agentic-ai"],
        "g2ai_pattern": ["agent-governance-framework"],
        "source_url": "https://example.org/doc.pdf",
        "relevance": {
            "target_fit": "primary",
            "axis": "agentic_g2ai",
            "assessed_stage": "confirmed",
            "rationale": "эталонный агентный G2AI-документ",
            "assessed_date": "2026-07-15",
        },
    }


def write_doc(
    root: Path,
    rec: dict[str, Any],
    *,
    raw: bytes | None = None,
    md: str | None = None,
    state: dict[str, Any] | None = None,
) -> Path:
    """Создать папку-документ sources/<track>/<entity>/<id>/ + meta.yaml (+ raw.pdf/doc.md/.state.yaml)."""
    d = root / str(rec["track"]) / str(rec["entity_id"]) / str(rec["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.yaml").write_text(yaml.safe_dump(rec, allow_unicode=True), encoding="utf-8")
    if raw is not None:
        (d / "raw.pdf").write_bytes(raw)
    if md is not None:
        (d / "doc.md").write_text(md, encoding="utf-8")
    if state is not None:
        (d / ".state.yaml").write_text(yaml.safe_dump(state, allow_unicode=True), encoding="utf-8")
    return d

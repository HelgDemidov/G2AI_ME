"""ELI-разметка (European Legislation Identifier, CELLAR/Publications Office ЕС) ->
настоящие <hN>-заголовки ДО передачи в trafilatura.

EUR-Lex (и порталы стран-членов/кандидатов ЕС, постепенно принимающие ELI) не несёт
ни одного семантического <h1>-<h6> — вся иерархия (CHAPTER/SECTION/Article/ANNEX)
выражена CSS-классами на <p>/<div> (см. https://eur-lex.europa.eu/eli-register/).
trafilatura промоутит в markdown только настоящие <hN> — без этого прохода вся
структура падает в плоскую прозу. Классы переиспользуются МЕЖДУ уровнями
(oj-ti-section-1/2 — и глава, и секция; oj-doc-ti — и заголовок документа, и
заголовок приложения), поэтому уровень определяется по id-префиксу структурных
div (cpt_/cpt_X.sct_/art_/anx_), не по одному классу.

Безопасный no-op на любом HTML без этой разметки — дешёвая проверка байт-маркера
до полного парсинга, остальные (не-ELI) источники не платят цену прохода.
"""
from __future__ import annotations

import re
from typing import cast

from lxml import html as lhtml

_ELI_MARKERS = (b"eli-subdivision", b"eli-container", b"oj-ti-section-1")

_CPT_ID = re.compile(r"^cpt_[IVXLCDM]+(\.sct_\d+)?$")


def _text(el: lhtml.HtmlElement) -> str:
    return re.sub(r"\s+", " ", el.text_content()).strip()


def _make_heading(level: int, text: str, tail: str | None) -> lhtml.HtmlElement:
    heading = lhtml.Element(f"h{level}")
    heading.text = text
    heading.tail = tail
    return heading


def _promote_numbered_block(
    container: lhtml.HtmlElement,
    *,
    number_xpath: str,
    title_xpath: str,
    level: int,
) -> None:
    """CHAPTER/SECTION/Article: заменить прямой child «номер» на <hN> «номер —
    заголовок», удалить обёртку eli-title с самим заголовком (текст уже в <hN>).
    Молча пропускает, если ожидаемая форма (номер-параграф) не найдена —
    сомнение -> оставить как есть, а не гадать (принцип чартера §2.5).
    """
    number_matches = container.xpath(number_xpath)
    if not number_matches:
        return
    number_el = number_matches[0]
    title_matches = container.xpath(title_xpath)
    title_text = _text(title_matches[0]) if title_matches else ""
    number_text = _text(number_el)
    heading_text = f"{number_text} — {title_text}" if title_text else number_text

    parent = number_el.getparent()
    heading = _make_heading(level, heading_text, number_el.tail)
    parent.replace(number_el, heading)
    if title_matches:
        wrapper = title_matches[0].getparent()
        wrapper.getparent().remove(wrapper)


def _promote_chapters_and_sections(tree: lhtml.HtmlElement) -> None:
    for div in tree.xpath('//div[@id]'):
        div_id = div.get("id", "")
        if not _CPT_ID.match(div_id):
            continue
        level = 2 if ".sct_" in div_id else 1
        _promote_numbered_block(
            div,
            number_xpath='./p[@class="oj-ti-section-1"]',
            title_xpath='./div[@class="eli-title"]/p[@class="oj-ti-section-2"]',
            level=level,
        )


def _promote_articles(tree: lhtml.HtmlElement) -> None:
    for div in tree.xpath('//div[@class="eli-subdivision" and starts-with(@id, "art_")]'):
        ancestors = div.xpath('ancestor::div[@id]')
        under_section = any(".sct_" in a.get("id", "") for a in ancestors)
        level = 3 if under_section else 2
        _promote_numbered_block(
            div,
            number_xpath='./p[@class="oj-ti-art"]',
            title_xpath='./div[@class="eli-title"]/p[@class="oj-sti-art"]',
            level=level,
        )


def _promote_annexes(tree: lhtml.HtmlElement) -> None:
    for div in tree.xpath('//div[@class="eli-container" and starts-with(@id, "anx_")]'):
        title_ps = div.xpath('./p[@class="oj-doc-ti"]')
        if not title_ps:
            continue
        heading_text = " — ".join(_text(p) for p in title_ps)
        first, rest = title_ps[0], title_ps[1:]
        parent = first.getparent()
        heading = _make_heading(1, heading_text, first.tail)
        parent.replace(first, heading)
        for p in rest:
            p.getparent().remove(p)


def _promote_annex_subsections(tree: lhtml.HtmlElement) -> None:
    for p in tree.xpath('//p[@class="oj-ti-grseq-1"]'):
        heading = _make_heading(2, _text(p), p.tail)
        p.getparent().replace(p, heading)


def _wrap_body_in_article(tree: lhtml.HtmlElement) -> None:
    """Оборачивает содержимое <body> в <article> (in-place).

    Без этого trafilatura находит <hN> корректно (см. вручную проверенное
    trafilatura.main_extractor.extract_content), но теряет их: несколько
    BODY_XPATH-кандидатов (trafilatura/xpaths.py) поочерёдно пробуют разные
    поддеревья «главного контента»; наш EUR-Lex-документ не матчит ни один
    ранний кандидат (нет ни <article>, ни узнаваемых id/class типа content/main),
    поэтому парсер откатывается к более поздним кандидатам через несколько
    проходов — и заголовки (тег head после конверсии <hN>) теряются в процессе
    (эмпирически подтверждено трассировкой extract_content: 5 head на входе,
    0 на выходе), хотя обычный текст выживает через recall-фолбэк
    (recover_wild_text — восстанавливает p/table/div, но НЕ head).
    <article> — первый (наиболее специфичный) BODY_XPATH-кандидат, матчится
    сразу за один проход — обходной путь без патчинга trafilatura.
    """
    body = tree.find(".//body")
    if body is None:
        return
    article = lhtml.Element("article")
    for child in list(body):
        article.append(child)
    body.append(article)


def matches(html: bytes) -> bool:
    """Дешёвая проверка маркера-подстроки ДО полного парсинга — используется и
    здесь (быстрый ранний выход), и реестром ``convert/html_preprocess.py``
    (spec convert-hardening B1) для диспетчеризации без парсинга каждого
    кандидата HTML."""
    return any(marker in html for marker in _ELI_MARKERS)


def promote_eli_headings(html: bytes) -> bytes:
    """Продвинуть ELI-структурные маркеры (CHAPTER/SECTION/Article/ANNEX) в
    настоящие <hN> внутри HTML-дерева. Не-ELI документы возвращаются байт-в-байт
    неизменными (дешёвая проверка маркера-подстроки до полного парсинга).
    """
    if not matches(html):
        return html
    tree = lhtml.fromstring(html)
    _promote_chapters_and_sections(tree)
    _promote_articles(tree)
    _promote_annexes(tree)
    _promote_annex_subsections(tree)
    _wrap_body_in_article(tree)
    return cast(bytes, lhtml.tostring(tree, encoding="utf-8"))

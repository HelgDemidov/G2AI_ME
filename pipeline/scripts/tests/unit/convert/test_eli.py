"""Тесты ELI-промоутера (eli.py): CHAPTER/SECTION/Article/ANNEX -> <hN>.

Фикстуры — минимальные синтетические HTML-обрубки, повторяющие реальную
разметку EUR-Lex (проверено на живом документе AI Act, PR convert-html), а не
сам корпусный документ (тела документов не версионируются в git)."""
from __future__ import annotations

from lxml import html as lhtml

from convert.eli import promote_eli_headings

_HEAD = b'<head><meta charset="utf-8"></head>'  # явный charset — как в реальном EUR-Lex

_NON_ELI_HTML = b"<html><body><p>Just a regular paragraph, nothing ELI here.</p></body></html>"

_CHAPTER_WITH_ARTICLE = b"""<html>""" + _HEAD + b"""<body>
<div id="cpt_I">
  <p class="oj-ti-section-1">CHAPTER I</p>
  <div class="eli-title" id="cpt_I.tit_1">
    <p class="oj-ti-section-2"><span class="oj-bold">GENERAL PROVISIONS</span></p>
  </div>
  <div class="eli-subdivision" id="art_1">
    <p class="oj-ti-art">Article 1</p>
    <div class="eli-title" id="art_1.tit_1">
      <p class="oj-sti-art">Subject matter</p>
    </div>
    <div id="001.001">
      <p class="oj-normal">The purpose of this Regulation is to improve things.</p>
    </div>
  </div>
</div>
</body></html>"""

_CHAPTER_WITH_SECTION_AND_ARTICLE = b"""<html>""" + _HEAD + b"""<body>
<div id="cpt_III">
  <p class="oj-ti-section-1">CHAPTER III</p>
  <div class="eli-title" id="cpt_III.tit_1">
    <p class="oj-ti-section-2"><span class="oj-bold">HIGH-RISK AI SYSTEMS</span></p>
  </div>
  <div id="cpt_III.sct_1">
    <p class="oj-ti-section-1"><span class="oj-italic">SECTION 1</span></p>
    <div class="eli-title" id="cpt_III.sct_1.tit_1">
      <p class="oj-ti-section-2"><span class="oj-bold"><span class="oj-italic">Classification</span></span></p>
    </div>
    <div class="eli-subdivision" id="art_6">
      <p class="oj-ti-art">Article 6</p>
      <div class="eli-title" id="art_6.tit_1">
        <p class="oj-sti-art">Classification rules</p>
      </div>
      <div id="006.001">
        <p class="oj-normal">An AI system shall be considered high-risk.</p>
      </div>
    </div>
  </div>
</div>
</body></html>"""

_ANNEX = b"""<html>""" + _HEAD + b"""<body>
<div class="eli-container" id="anx_I">
  <p class="oj-doc-ti" id="d1e38-124-1">ANNEX I</p>
  <p class="oj-doc-ti">List of Union harmonisation legislation</p>
  <p class="oj-ti-grseq-1">Section A. List based on the New Legislative Framework</p>
  <p class="oj-normal">Some annex body text here.</p>
</div>
</body></html>"""

_CHAPTER_WITHOUT_TITLE = b"""<html>""" + _HEAD + b"""<body>
<div id="cpt_I">
  <p class="oj-ti-section-1">CHAPTER I</p>
  <p class="oj-normal">Body text directly, no eli-title wrapper.</p>
</div>
</body></html>"""


def test_non_eli_html_returned_unchanged() -> None:
    assert promote_eli_headings(_NON_ELI_HTML) == _NON_ELI_HTML


def test_chapter_promoted_to_h1_with_number_and_title() -> None:
    out = promote_eli_headings(_CHAPTER_WITH_ARTICLE)
    tree = lhtml.fromstring(out)
    h1 = tree.xpath("//h1")
    assert len(h1) == 1
    assert h1[0].text == "CHAPTER I — GENERAL PROVISIONS"


def test_eli_title_wrapper_removed_after_promotion() -> None:
    """Текст заголовка ушёл в <hN> — обёртка eli-title не должна дублировать его прозой."""
    out = promote_eli_headings(_CHAPTER_WITH_ARTICLE)
    tree = lhtml.fromstring(out)
    assert tree.xpath('//div[@class="eli-title"]') == []


def test_article_directly_under_chapter_is_h2() -> None:
    out = promote_eli_headings(_CHAPTER_WITH_ARTICLE)
    tree = lhtml.fromstring(out)
    h2 = tree.xpath("//h2")
    assert len(h2) == 1
    assert h2[0].text == "Article 1 — Subject matter"


def test_article_body_text_preserved() -> None:
    out = promote_eli_headings(_CHAPTER_WITH_ARTICLE)
    tree = lhtml.fromstring(out)
    assert "purpose of this Regulation" in tree.text_content()


def test_section_nested_in_chapter_is_h2_and_article_under_section_is_h3() -> None:
    out = promote_eli_headings(_CHAPTER_WITH_SECTION_AND_ARTICLE)
    tree = lhtml.fromstring(out)
    h1 = tree.xpath("//h1")
    h2 = tree.xpath("//h2")
    h3 = tree.xpath("//h3")
    assert len(h1) == 1 and "CHAPTER III" in h1[0].text
    assert len(h2) == 1 and "SECTION 1" in h2[0].text
    assert len(h3) == 1 and h3[0].text == "Article 6 — Classification rules"


def test_annex_promoted_to_h1_merging_all_doc_ti_paragraphs() -> None:
    out = promote_eli_headings(_ANNEX)
    tree = lhtml.fromstring(out)
    h1 = tree.xpath("//h1")
    assert len(h1) == 1
    assert h1[0].text == "ANNEX I — List of Union harmonisation legislation"
    # ровно один oj-doc-ti "съеден" в заголовок, второй удалён -> ни одного не осталось прозой
    assert tree.xpath('//p[@class="oj-doc-ti"]') == []


def test_annex_subsection_grseq_promoted_to_h2() -> None:
    out = promote_eli_headings(_ANNEX)
    tree = lhtml.fromstring(out)
    h2 = tree.xpath("//h2")
    assert len(h2) == 1
    assert "Section A" in h2[0].text


def test_annex_body_text_preserved() -> None:
    out = promote_eli_headings(_ANNEX)
    tree = lhtml.fromstring(out)
    assert "Some annex body text here" in tree.text_content()


def test_chapter_without_title_wrapper_still_promotes_number_only() -> None:
    """Нет eli-title-обёртки — заголовок = только номер, без гадания за текст."""
    out = promote_eli_headings(_CHAPTER_WITHOUT_TITLE)
    tree = lhtml.fromstring(out)
    h1 = tree.xpath("//h1")
    assert len(h1) == 1
    assert h1[0].text == "CHAPTER I"


def test_body_content_wrapped_in_article() -> None:
    """<article> — первый BODY_XPATH-кандидат trafilatura; без него заголовки
    теряются при мульти-проходном отборе главного контента (см. docstring eli.py)."""
    out = promote_eli_headings(_CHAPTER_WITH_ARTICLE)
    tree = lhtml.fromstring(out)
    articles = tree.xpath("//body/article")
    assert len(articles) == 1
    assert articles[0].xpath(".//h1")


def test_chapter_only_document_without_article_or_annex_marker_still_promoted() -> None:
    """Регресс дешёвого fast-path: документ БЕЗ eli-subdivision/eli-container (только
    глава, без статей/приложений) не должен молча пройти как «нет ELI-разметки» —
    у главы свой класс-маркер (oj-ti-section-1), не пересекающийся с article/annex."""
    assert any(m in _CHAPTER_WITHOUT_TITLE for m in (b"eli-subdivision", b"eli-container")) is False
    out = promote_eli_headings(_CHAPTER_WITHOUT_TITLE)
    assert lhtml.fromstring(out).xpath("//h1")

"""Тесты pdf_graphics: элементы, дедуп, фильтр, кластеризация, guards. Вся
геометрия — синтетика, без единого реального PDF (spec convert-graphics §1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from convert.pdf_graphics import (
    Element,
    Region,
    classify_images,
    cluster_elements,
    dedupe_elements,
    detect_regions,
    document_hash_counts,
    filter_elements,
    region_guards_ok,
    region_hash,
    region_id,
    region_words,
    render_raster_marker,
    render_region_block,
    try_grid,
    try_sequence,
    union_bbox,
)

PAGE_W, PAGE_H = 600.0, 800.0
PAGE_AREA = PAGE_W * PAGE_H


@dataclass
class _Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


def _rect(x0: float, top: float, x1: float, bottom: float) -> Element:
    return Element("rect", x0, top, x1, bottom)


# --- dedupe_elements ---


def test_dedupe_collapses_stroke_fill_pair_within_tolerance() -> None:
    a = _rect(10.0, 10.0, 100.0, 60.0)
    b = _rect(10.2, 10.1, 100.1, 59.9)  # тот же бокс, stroke+fill вариация < 0.5pt
    assert dedupe_elements([a, b]) == [a]


def test_dedupe_keeps_distinct_boxes() -> None:
    a = _rect(10.0, 10.0, 100.0, 60.0)
    b = _rect(200.0, 10.0, 300.0, 60.0)  # далеко — не дубль
    assert dedupe_elements([a, b]) == [a, b]


def test_dedupe_does_not_collapse_across_kinds() -> None:
    rect = _rect(10.0, 10.0, 100.0, 60.0)
    image = Element("image", 10.0, 10.0, 100.0, 60.0)  # тот же bbox, другой kind
    assert dedupe_elements([rect, image]) == [rect, image]


# --- filter_elements ---


def test_filter_excludes_thin_ruler() -> None:
    ruler = _rect(10.0, 10.0, 400.0, 12.0)  # высота 2pt < ELEMENT_MIN_SIDE_PT (4pt)
    assert filter_elements([ruler], PAGE_AREA) == []


def test_filter_excludes_page_background() -> None:
    bg = _rect(0.0, 0.0, PAGE_W, PAGE_H)  # 100% площади > ELEMENT_MAX_PAGE_FRACTION (85%)
    assert filter_elements([bg], PAGE_AREA) == []


def test_filter_keeps_normal_box() -> None:
    box = _rect(10.0, 10.0, 110.0, 60.0)  # 100x50pt, обычный бокс фигуры
    assert filter_elements([box], PAGE_AREA) == [box]


# --- cluster_elements ---


def test_cluster_splits_two_distant_groups() -> None:
    near_a = _rect(10.0, 10.0, 50.0, 50.0)
    near_b = _rect(55.0, 10.0, 95.0, 50.0)  # 5pt зазор < CLUSTER_GAP_PT (8pt) — связан с near_a
    far = _rect(500.0, 500.0, 540.0, 540.0)  # далеко — отдельный кластер
    clusters = cluster_elements([near_a, near_b, far])
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_gap_threshold_determines_connectivity() -> None:
    a = _rect(0.0, 0.0, 10.0, 10.0)
    b = _rect(30.0, 0.0, 40.0, 10.0)  # зазор 20pt между bbox
    connected = cluster_elements([a, b], gap=15.0)  # 15*2=30 >= 20pt зазора -> связаны
    disconnected = cluster_elements([a, b], gap=5.0)  # 5*2=10 < 20pt -> раздельны
    assert len(connected) == 1
    assert len(disconnected) == 2


def test_cluster_single_element_is_its_own_cluster() -> None:
    a = _rect(0.0, 0.0, 10.0, 10.0)
    assert cluster_elements([a]) == [[a]]


# --- region_words / union_bbox ---


def test_union_bbox_spans_all_elements() -> None:
    a = _rect(10.0, 20.0, 30.0, 40.0)
    b = _rect(50.0, 5.0, 70.0, 25.0)
    assert union_bbox([a, b]) == (10.0, 5.0, 70.0, 40.0)


def test_region_words_selects_by_center() -> None:
    inside = _Word("inside", x0=15.0, x1=25.0, top=22.0, bottom=30.0)  # центр (20,26) внутри
    outside = _Word("outside", x0=200.0, x1=220.0, top=200.0, bottom=210.0)
    bbox = (10.0, 20.0, 30.0, 40.0)
    assert region_words([inside, outside], bbox) == [inside]


# --- region_guards_ok: три guard'а (§1, чартер §7.1) ---


def test_guards_reject_too_few_elements() -> None:
    """< REGION_MIN_ELEMENTS (3) — callout/плашка, не фигура."""
    elements = [_rect(0.0, 0.0, 50.0, 20.0), _rect(0.0, 25.0, 50.0, 45.0)]  # 2 элемента
    words: list[Any] = []
    assert region_guards_ok(elements, words, PAGE_AREA) is False


def test_guards_reject_near_full_page_region() -> None:
    """Регион на ~всю страницу — рамка/подложка, не фигура."""
    elements = [
        _rect(0.0, 0.0, PAGE_W, PAGE_H * 0.4),
        _rect(0.0, PAGE_H * 0.4, PAGE_W, PAGE_H * 0.7),
        _rect(0.0, PAGE_H * 0.7, PAGE_W, PAGE_H * 0.98),
    ]
    assert region_guards_ok(elements, [], PAGE_AREA) is False


def test_guards_reject_dense_prose_callout() -> None:
    """Плотный текст (> REGION_MAX_CHARS_PER_ELEMENT/элемент) — боксированная
    проза (callout/кейс-стади), не инфографика."""
    elements = [_rect(0.0, 0.0, 300.0, 100.0), _rect(0.0, 110.0, 300.0, 210.0), _rect(0.0, 220.0, 300.0, 320.0)]
    long_text = "x" * 800  # 800 симв. / 3 элемента = 267 > 250
    words = [_Word(long_text, x0=10.0, x1=290.0, top=50.0, bottom=60.0)]
    assert region_guards_ok(elements, words, PAGE_AREA) is False


def test_guards_reject_region_with_zero_words() -> None:
    """Регион без единого слова в bbox (декоративная иконка из curve-сегментов,
    напр. Сингапур p.26 аудита) — маркер с пустым Labels был бы чистым шумом;
    распустить дёшево (слов и так нет, возвращать в прозу нечего)."""
    elements = [
        _rect(0.0, 0.0, 10.0, 10.0), _rect(15.0, 0.0, 25.0, 10.0), _rect(30.0, 0.0, 40.0, 10.0),
    ]
    assert region_guards_ok(elements, [], PAGE_AREA) is False


def test_guards_accept_plausible_swot_region() -> None:
    """Правдоподобная SWOT-подобная фигура (4 бокса, короткие подписи) проходит
    все три guard'а."""
    elements = [
        _rect(0.0, 0.0, 100.0, 100.0), _rect(110.0, 0.0, 210.0, 100.0),
        _rect(0.0, 110.0, 100.0, 210.0), _rect(110.0, 110.0, 210.0, 210.0),
    ]
    words = [_Word("Strength", x0=10.0, x1=60.0, top=10.0, bottom=20.0)]
    assert region_guards_ok(elements, words, PAGE_AREA) is True


# --- try_grid: 2x2 SWOT-подобная матрица (§1.1) ---


def _swot_2x2() -> tuple[list[Element], list[_Word]]:
    elements = [
        _rect(0.0, 0.0, 100.0, 100.0),      # (row0, col0)
        _rect(110.0, 0.0, 210.0, 100.0),    # (row0, col1)
        _rect(0.0, 110.0, 100.0, 210.0),    # (row1, col0)
        _rect(110.0, 110.0, 210.0, 210.0),  # (row1, col1)
    ]
    words = [
        _Word("Strength", x0=10.0, x1=60.0, top=10.0, bottom=20.0),
        _Word("Weakness", x0=120.0, x1=170.0, top=10.0, bottom=20.0),
        _Word("Opportunity", x0=10.0, x1=60.0, top=120.0, bottom=130.0),
        _Word("Threat", x0=120.0, x1=170.0, top=120.0, bottom=130.0),
    ]
    return elements, words


def test_try_grid_reconstructs_2x2_matrix_with_correct_cell_text() -> None:
    elements, words = _swot_2x2()
    assert try_grid(elements, words) == [["Strength", "Weakness"], ["Opportunity", "Threat"]]


def test_try_grid_empty_cell_renders_as_empty_string() -> None:
    elements, words = _swot_2x2()
    words = words[:3]  # без слова в (row1,col1) — ячейка должна остаться ""
    cells = try_grid(elements, words)
    assert cells is not None
    assert cells[1][1] == ""


def test_try_grid_none_when_occupancy_below_threshold() -> None:
    """3 из 4 ячеек 2x2 — заполненность 75% < GRID_MIN_OCCUPANCY (90%) -> None
    («дырка» в матрице — не почти-грид, честный отказ)."""
    elements, words = _swot_2x2()
    assert try_grid(elements[:3], words) is None


def test_try_grid_none_when_too_many_columns() -> None:
    """7 колонок > GRID_MAX_COLS (6) -> None."""
    elements = [
        _rect(col * 110.0, row * 110.0, col * 110.0 + 100.0, row * 110.0 + 100.0)
        for row in range(2) for col in range(7)
    ]
    assert try_grid(elements, []) is None


def test_try_grid_none_when_two_rects_share_a_cell() -> None:
    """Пятый rect попадает в ту же ячейку (row0,col0), что и первый -> None
    (грид не фабрикуется при коллизии). x0/top=1.0 — внутри GRID_SNAP_PT (3pt)
    от кластера "0", поэтому схлопывается в ТУ ЖЕ колонку/строку, а не в новую."""
    elements, words = _swot_2x2()
    collision = _rect(1.0, 1.0, 90.0, 90.0)
    assert try_grid([*elements, collision], words) is None


def test_try_grid_none_without_any_rects() -> None:
    curve = Element("curve", 0.0, 0.0, 50.0, 50.0)
    assert try_grid([curve, curve, curve], []) is None


# --- try_sequence: одноосевая последовательность боксов (§1.2) ---


def test_try_sequence_horizontal_chain_orders_by_x0() -> None:
    elements = [
        _rect(100.0, 0.0, 140.0, 40.0), _rect(0.0, 0.0, 40.0, 40.0),  # намеренно не по порядку
        _rect(150.0, 0.0, 190.0, 40.0), _rect(50.0, 0.0, 90.0, 40.0),
    ]
    words = [
        _Word("Build", x0=105.0, x1=135.0, top=15.0, bottom=25.0),
        _Word("Discover", x0=5.0, x1=35.0, top=15.0, bottom=25.0),
        _Word("Launch", x0=155.0, x1=185.0, top=15.0, bottom=25.0),
        _Word("Design", x0=55.0, x1=85.0, top=15.0, bottom=25.0),
    ]
    assert try_sequence(elements, words) == ["Discover", "Design", "Build", "Launch"]


def test_try_sequence_vertical_chain_orders_by_top() -> None:
    elements = [
        _rect(0.0, 100.0, 100.0, 140.0), _rect(0.0, 0.0, 100.0, 40.0),
        _rect(0.0, 50.0, 100.0, 90.0),
    ]
    words = [
        _Word("Third", x0=10.0, x1=40.0, top=110.0, bottom=120.0),
        _Word("First", x0=10.0, x1=40.0, top=10.0, bottom=20.0),
        _Word("Second", x0=10.0, x1=40.0, top=60.0, bottom=70.0),
    ]
    assert try_sequence(elements, words) == ["First", "Second", "Third"]


def test_try_sequence_none_when_boxes_overlap() -> None:
    elements = [
        _rect(0.0, 0.0, 50.0, 40.0), _rect(30.0, 0.0, 80.0, 40.0),  # пересекаются по x
        _rect(100.0, 0.0, 140.0, 40.0),
    ]
    assert try_sequence(elements, []) is None


def test_try_sequence_none_when_too_few_boxes() -> None:
    elements = [_rect(0.0, 0.0, 40.0, 40.0), _rect(50.0, 0.0, 90.0, 40.0)]  # 2 < SEQ_MIN_BOXES
    assert try_sequence(elements, []) is None


def test_try_sequence_none_when_axis_overlap_insufficient() -> None:
    """Боксы не пересекаются, но и не выровнены ни по одной оси (диагональная
    россыпь) — ни горизонтальная, ни вертикальная цепь не набирают перекрытие."""
    elements = [
        _rect(0.0, 0.0, 40.0, 40.0),
        _rect(60.0, 60.0, 100.0, 100.0),
        _rect(120.0, 120.0, 160.0, 160.0),
    ]
    assert try_sequence(elements, []) is None


# --- region_hash / region_id: детерминизм VLM-шва (чартер §4.3) ---


def test_region_hash_stable_to_text_order_and_subpixel_noise() -> None:
    a = region_hash(1, (10.3, 20.2, 50.1, 60.4), ["alpha", "beta"])
    b = region_hash(1, (10.4, 20.1, 50.4, 60.2), ["beta", "alpha"])
    assert a == b


def test_region_hash_differs_by_page() -> None:
    assert region_hash(1, (0.0, 0.0, 10.0, 10.0), ["x"]) != region_hash(2, (0.0, 0.0, 10.0, 10.0), ["x"])


def test_region_id_is_12_char_prefix_of_hash() -> None:
    h = region_hash(3, (0.0, 0.0, 10.0, 10.0), ["x"])
    assert region_id(3, (0.0, 0.0, 10.0, 10.0), ["x"]) == h[:12]


# --- render_region_block: точная грамматика маркеров (§2) ---


def test_render_grid_block_matches_grammar() -> None:
    region = Region(
        bbox=(0.0, 0.0, 210.0, 210.0), elements=[], words=[],
        kind="grid", id="abc123def456",
        cells=[["Strength", "Weakness"], ["Opportunity", "Threat"]],
    )
    expected = (
        "> [Reconstructed infographic, p. 5, region abc123def456]\n"
        "\n"
        "| Strength | Weakness |\n"
        "| --- | --- |\n"
        "| Opportunity | Threat |"
    )
    assert render_region_block(region, page=5) == expected


def test_render_sequence_block_matches_grammar() -> None:
    region = Region(
        bbox=(0.0, 0.0, 190.0, 40.0), elements=[], words=[],
        kind="sequence", id="seq000111222", items=["Discover", "Design", "Build"],
    )
    expected = (
        "> [Reconstructed infographic, p. 3, region seq000111222]\n"
        "\n"
        "1. Discover\n"
        "2. Design\n"
        "3. Build"
    )
    assert render_region_block(region, page=3) == expected


def test_render_opaque_block_matches_grammar() -> None:
    words = [
        _Word("Node A", x0=10.0, x1=50.0, top=10.0, bottom=20.0),
        _Word("Node B", x0=10.0, x1=50.0, top=60.0, bottom=70.0),
    ]
    region = Region(
        bbox=(0.0, 0.0, 100.0, 100.0), elements=[], words=words,
        kind="opaque", id="opq333444555",
    )
    expected = (
        "> [Figure, p. 6, region opq333444555 — structure not reconstructed]\n"
        "> Labels (reading order not guaranteed): Node A; Node B"
    )
    assert render_region_block(region, page=6) == expected


# --- detect_regions: сквозной конвейер (guards + классификация + изъятие слов) ---


def test_detect_regions_end_to_end_grid_removes_words_from_prose() -> None:
    elements, words = _swot_2x2()
    prose_word = _Word("unrelated prose", x0=300.0, x1=400.0, top=300.0, bottom=310.0)
    regions, remaining = detect_regions(
        page=1, elements=elements, words=[*words, prose_word],
        page_width=PAGE_W, page_height=PAGE_H, table_bboxes=[],
    )
    assert len(regions) == 1
    assert regions[0].kind == "grid"
    assert remaining == [prose_word]  # слова региона изъяты, посторонняя проза осталась


def test_detect_regions_dissolved_cluster_keeps_words_in_prose() -> None:
    """Кластер из 2 элементов (< REGION_MIN_ELEMENTS) — регион не создаётся,
    его слова НЕ изымаются из words."""
    elements = [_rect(0.0, 0.0, 50.0, 20.0), _rect(0.0, 25.0, 50.0, 45.0)]
    word = _Word("callout text", x0=5.0, x1=45.0, top=5.0, bottom=15.0)
    regions, remaining = detect_regions(
        page=1, elements=elements, words=[word],
        page_width=PAGE_W, page_height=PAGE_H, table_bboxes=[],
    )
    assert regions == []
    assert remaining == [word]


def _image(x0: float, top: float, x1: float, bottom: float, content_hash: str | None) -> Element:
    return Element("image", x0, top, x1, bottom, content_hash=content_hash)


# --- растровая политика: document_hash_counts / classify_images (§1.4) ---


def test_document_hash_counts_counts_across_whole_document() -> None:
    """Частота — ПО ДОКУМЕНТУ (все страницы в одном плоском списке), не по
    странице: логотип, встреченный по разу на 3 разных страницах, должен
    засчитаться 3 раза, а не остаться по 1 на страницу."""
    images = [_image(0, 0, 10, 10, "logo"), _image(0, 0, 10, 10, "logo"), _image(0, 0, 10, 10, "logo")]
    assert document_hash_counts(images) == {"logo": 3}


def test_document_hash_counts_ignores_missing_hash() -> None:
    images = [_image(0, 0, 10, 10, None), _image(0, 0, 10, 10, None)]
    assert document_hash_counts(images) == {}


def test_classify_images_excludes_repeated_decor() -> None:
    """Повторяется >= RASTER_DECOR_MIN_REPEATS (3) раз по документу — декор,
    молча, даже если крупное на этой конкретной странице."""
    big_logo = _image(0, 0, PAGE_W, PAGE_H, "logo")  # 100% площади — крупное
    marked = classify_images([big_logo], consumed_bboxes=[], hash_counts={"logo": 3}, page_area=PAGE_AREA)
    assert marked == []


def test_classify_images_marks_unique_large_image() -> None:
    big_unique = _image(0.0, 0.0, PAGE_W, PAGE_H * 0.5, "unique-chart")  # 50% площади
    marked = classify_images(
        [big_unique], consumed_bboxes=[], hash_counts={"unique-chart": 1}, page_area=PAGE_AREA
    )
    assert marked == [big_unique]


def test_classify_images_silent_for_small_icon() -> None:
    icon = _image(0.0, 0.0, 20.0, 20.0, "icon")  # << 8% площади страницы
    marked = classify_images([icon], consumed_bboxes=[], hash_counts={"icon": 1}, page_area=PAGE_AREA)
    assert marked == []


def test_classify_images_missing_hash_treated_as_unique() -> None:
    """Битый/зашифрованный поток (content_hash=None) — НЕ декор по построению
    (никогда не совпадёт по hash_counts), крупное -> помечается."""
    broken_stream = _image(0.0, 0.0, PAGE_W, PAGE_H * 0.5, None)
    marked = classify_images([broken_stream], consumed_bboxes=[], hash_counts={}, page_area=PAGE_AREA)
    assert marked == [broken_stream]


def test_classify_images_excludes_image_already_inside_a_region() -> None:
    """Изображение — часть уже детектированного региона (напр. флоучарт из
    img+rect+curve) — не получает ОТДЕЛЬНЫЙ растровый маркер поверх маркера
    региона (не дублируем)."""
    inside = _image(10.0, 10.0, 90.0, 90.0, "unique-in-flowchart")
    region_bbox = (0.0, 0.0, 200.0, 200.0)
    marked = classify_images(
        [inside], consumed_bboxes=[region_bbox], hash_counts={"unique-in-flowchart": 1}, page_area=PAGE_AREA
    )
    assert marked == []


def test_render_raster_marker_matches_grammar() -> None:
    assert render_raster_marker(page=2) == "> [Image, p. 2 — raster content not analyzed]"


def test_detect_regions_excludes_elements_inside_table_bbox() -> None:
    """Элементы внутри уже забранной таблицы не участвуют в детекции регионов
    (приоритет у get_real_tables — интейк-шаг §1)."""
    elements, words = _swot_2x2()
    table_bbox = (0.0, 0.0, 210.0, 210.0)  # накрывает весь SWOT-кластер
    regions, remaining = detect_regions(
        page=1, elements=elements, words=words,
        page_width=PAGE_W, page_height=PAGE_H, table_bboxes=[table_bbox],
    )
    assert regions == []
    assert remaining == words

"""Тесты линта мёртвых якорей (memory_lint.py, spec docs/memory/memory-lint/spec.md).
Полностью герметичны: тестовые деревья/файлы строятся в tmp_path, реальные CLAUDE.md
и внешний Claude-каталог памяти не читаются."""
from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import memory_lint as ml


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pipeline" / "scripts" / "convert").mkdir(parents=True)
    (root / "pipeline" / "scripts" / "convert" / "converters.py").write_text("", encoding="utf-8")
    (root / "docs" / "pipeline" / "general").mkdir(parents=True)
    (root / "docs" / "pipeline" / "general" / "pipeline_improvements.md").write_text("", encoding="utf-8")
    (root / "sources" / "me" / "doc1").mkdir(parents=True)
    (root / "sources" / "me" / "doc1" / "raw.pdf").write_text("", encoding="utf-8")
    (root / ".claude" / "commands").mkdir(parents=True)
    (root / ".claude" / "commands" / "tech-spec.md").write_text("", encoding="utf-8")
    (root / "CLAUDE.md").write_text("", encoding="utf-8")
    return root


# --- classify(): 5 базовых классов спека ---

def test_classify_path_with_slash() -> None:
    a = ml.classify("convert/converters.py")
    assert a is not None and a.kind == "path"


def test_classify_path_bare_extension() -> None:
    a = ml.classify("schema.py")
    assert a is not None and a.kind == "path"


def test_classify_const_with_equals() -> None:
    a = ml.classify("RRF_K=60")
    assert a is not None and a.kind == "const"


def test_classify_const_with_colon_and_flexible_spaces() -> None:
    a = ml.classify("CHUNK_MAX   =   512")
    assert a is not None and a.kind == "const"


def test_classify_qualified_dotted() -> None:
    a = ml.classify("schema.promote_candidate")
    assert a is not None
    assert a.kind == "qualified" and a.check_value == "promote_candidate"


def test_classify_qualified_double_colon() -> None:
    a = ml.classify("discover.py::inject")
    assert a is not None
    assert a.kind == "qualified" and a.check_value == "inject"


def test_classify_bare_snake_case() -> None:
    a = ml.classify("RRF_K")
    assert a is not None and a.kind == "bare"


def test_classify_bare_camel_case() -> None:
    a = ml.classify("SourceRecord")
    assert a is not None and a.kind == "bare"


# --- classify(): границы класса 5 (прочее — пропуск) ---

def test_classify_skips_single_segment_lowercase_word() -> None:
    assert ml.classify("build") is None
    assert ml.classify("search") is None


def test_classify_skips_cli_flag() -> None:
    assert ml.classify("--force") is None


def test_classify_skips_span_with_space() -> None:
    assert ml.classify("PR #25") is None


def test_classify_skips_numeric_looking_span() -> None:
    assert ml.classify("3.07x") is None


# --- classify(): гварды, найденные живьём на первом смок-тесте ---

def test_classify_skips_brace_placeholder() -> None:
    assert ml.classify("{track}") is None
    assert ml.classify("sources/{track}/{entity}/raw.*") is None


def test_classify_skips_angle_bracket_placeholder() -> None:
    assert ml.classify("docs/pipeline/<блок>/spec.md") is None


def test_classify_skips_gitignore_negation() -> None:
    assert ml.classify("!/.claude/skills/") is None


def test_classify_skips_bare_extension_mention() -> None:
    assert ml.classify(".md") is None
    assert ml.classify(".py") is None


def test_classify_skips_external_domain() -> None:
    assert ml.classify("agora.eto.tech") is None
    assert ml.classify("publications.europa.eu/webapi/rdf/sparql") is None


def test_classify_does_not_treat_known_repo_root_as_domain() -> None:
    a = ml.classify(".claude/commands/tech-spec.md")
    assert a is not None and a.kind == "path"


def test_classify_skips_git_branch_ref() -> None:
    assert ml.classify("feature/discovery-manual") is None
    assert ml.classify("origin/main") is None


def test_classify_skips_slash_separated_flag_list() -> None:
    assert ml.classify("--force/--only") is None


def test_classify_skips_slash_separated_numeric_delta() -> None:
    assert ml.classify("-12/+18") is None


def test_classify_skips_doi() -> None:
    assert ml.classify("10.5281/zenodo.13883066") is None


def test_classify_slash_command_maps_to_command_file() -> None:
    a = ml.classify("/tech-spec")
    assert a is not None
    assert a.kind == "path" and a.check_value == ".claude/commands/tech-spec.md"


def test_classify_leading_slash_repo_relative_strips_slash() -> None:
    a = ml.classify("/sources/candidates.yaml")
    assert a is not None and a.check_value == "sources/candidates.yaml"


def test_classify_leading_slash_os_path_is_skipped() -> None:
    assert ml.classify("/proc") is None


# --- extract_anchors(): fenced-блоки исключены ---

def test_extract_anchors_finds_inline_spans() -> None:
    text = "текст с `span_one` и `span_two` тут"
    result = list(ml.extract_anchors(text))
    assert [s for s, _ in result] == ["span_one", "span_two"]


def test_extract_anchors_skips_fenced_block_content() -> None:
    text = "до\n```\n`fenced_span`\n```\nпосле `real_span`"
    result = list(ml.extract_anchors(text))
    assert [s for s, _ in result] == ["real_span"]


# --- is_historical_mention() ---

@pytest.mark.parametrize("line", [
    "файл удалён 2026-07-22",
    "путь переименован в новый",
    "класс упразднён PR #9",
    "легаси-имя guides/",
    "механизм deprecated с прошлой версии",
])
def test_is_historical_mention_true(line: str) -> None:
    assert ml.is_historical_mention(line)


def test_is_historical_mention_false_on_neutral_line() -> None:
    assert not ml.is_historical_mention("обычное описание функции без маркеров")


# --- load_allowlist() ---

def test_load_allowlist_missing_file_returns_empty(tmp_path: Path) -> None:
    assert ml.load_allowlist(tmp_path / "nope.yaml") == set()


def test_load_allowlist_reads_spans(tmp_path: Path) -> None:
    p = tmp_path / "allow.yaml"
    p.write_text("allow:\n  - span: 'foo_bar'\n    reason: 'test'\n", encoding="utf-8")
    assert ml.load_allowlist(p) == {"foo_bar"}


def test_load_allowlist_empty_list_is_valid(tmp_path: Path) -> None:
    p = tmp_path / "allow.yaml"
    p.write_text("allow: []\n", encoding="utf-8")
    assert ml.load_allowlist(p) == set()


# --- path_exists(): базы, глоб, trailing-slash, bare-anywhere ---

def test_path_exists_direct_from_root(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert ml.path_exists("CLAUDE.md", root)


def test_path_exists_via_pipeline_scripts_base(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert ml.path_exists("convert/converters.py", root)


def test_path_exists_via_docs_pipeline_base(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert ml.path_exists("general/pipeline_improvements.md", root)


def test_path_exists_glob_pattern_matches(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert ml.path_exists("raw.*", root)


def test_path_exists_glob_pattern_no_match(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert not ml.path_exists("nonexistent_ext.*", root)


def test_path_exists_trailing_slash_directory(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert ml.path_exists("convert/", root)


def test_path_exists_trailing_slash_missing_directory(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert not ml.path_exists("ghost_dir/", root)


def test_path_exists_missing_returns_false(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert not ml.path_exists("ghost/nonexistent.py", root)


def test_path_exists_bare_filename_search_anywhere(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert ml.path_exists("converters.py", root)


def test_path_exists_finds_shallow_file_in_pipeline_index(tmp_path: Path) -> None:
    """pipeline/index/ — бинарный SQLite-артефакт, обычно 1 файл; неглубокая (не
    рекурсивная) проверка находит его напрямую без риска ложных совпадений в
    большом дереве."""
    root = _make_repo(tmp_path)
    (root / "pipeline" / "index").mkdir(parents=True)
    (root / "pipeline" / "index" / "corpus.db").write_bytes(b"")
    assert ml.path_exists("corpus.db", root)


def test_path_exists_does_not_recurse_into_pipeline_index_subdirs(tmp_path: Path) -> None:
    """Неглубокая проверка НЕ рекурсирует внутрь pipeline/index/ — файл в
    поддиректории (гипотетический WAL/журнал) не должен всплывать общим
    bare-поиском наравне с реальным кодом."""
    root = _make_repo(tmp_path)
    (root / "pipeline" / "index" / "nested").mkdir(parents=True)
    (root / "pipeline" / "index" / "nested" / "deep_file.db").write_bytes(b"")
    assert not ml.path_exists("deep_file.db", root)


def test_path_exists_does_not_ignore_scripts_index_subpackage(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "pipeline" / "scripts" / "index").mkdir(parents=True)
    (root / "pipeline" / "scripts" / "index" / "embed.py").write_text("", encoding="utf-8")
    assert ml.path_exists("embed.py", root)


# --- anchor_found() / build_corpus_text() ---

def test_anchor_found_const_match(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "pipeline" / "scripts" / "core").mkdir(parents=True)
    (root / "pipeline" / "scripts" / "core" / "x.py").write_text("RRF_K = 60\n", encoding="utf-8")
    corpus = ml.build_corpus_text(root)
    anchor = ml.classify("RRF_K = 60")
    assert anchor is not None
    assert ml.anchor_found(anchor, root, corpus)


def test_anchor_found_const_value_mismatch_is_dead(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "pipeline" / "scripts" / "core").mkdir(parents=True)
    (root / "pipeline" / "scripts" / "core" / "x.py").write_text("RRF_K = 60\n", encoding="utf-8")
    corpus = ml.build_corpus_text(root)
    anchor = ml.classify("RRF_K = 99")
    assert anchor is not None
    assert not ml.anchor_found(anchor, root, corpus)


def test_anchor_found_bare_identifier_match(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "pipeline" / "scripts" / "core").mkdir(parents=True)
    (root / "pipeline" / "scripts" / "core" / "x.py").write_text("class SourceRecord:\n    pass\n", encoding="utf-8")
    corpus = ml.build_corpus_text(root)
    anchor = ml.classify("SourceRecord")
    assert anchor is not None
    assert ml.anchor_found(anchor, root, corpus)


def test_anchor_found_bare_identifier_missing(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    corpus = ml.build_corpus_text(root)
    anchor = ml.classify("GhostClassName")
    assert anchor is not None
    assert not ml.anchor_found(anchor, root, corpus)


# --- scan_file(): интеграция классификации+проверки+allowlist+historical ---

def test_scan_file_finds_dead_anchor_as_warning(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    src = root / "CLAUDE.md"
    src.write_text("Ссылка на `ghost_module.py` тут.\n", encoding="utf-8")
    corpus = ml.build_corpus_text(root)
    findings = ml.scan_file(src, root, corpus, set())
    assert len(findings) == 1
    assert findings[0].span == "ghost_module.py"
    assert findings[0].level == "WARNING"


def test_scan_file_marks_historical_mention_as_info(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    src = root / "CLAUDE.md"
    src.write_text("Файл `ghost_module.py` удалён куратором.\n", encoding="utf-8")
    corpus = ml.build_corpus_text(root)
    findings = ml.scan_file(src, root, corpus, set())
    assert len(findings) == 1 and findings[0].level == "INFO"


def test_scan_file_allowlisted_span_produces_no_finding(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    src = root / "CLAUDE.md"
    src.write_text("Ссылка на `ghost_module.py` тут.\n", encoding="utf-8")
    corpus = ml.build_corpus_text(root)
    findings = ml.scan_file(src, root, corpus, {"ghost_module.py"})
    assert findings == []


def test_scan_file_live_anchor_produces_no_finding(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    src = root / "CLAUDE.md"
    src.write_text("Реестр в `convert/converters.py`.\n", encoding="utf-8")
    corpus = ml.build_corpus_text(root)
    findings = ml.scan_file(src, root, corpus, set())
    assert findings == []


# --- run() / main(): end-to-end на синтетическом дереве, --fail-on-dead ---

def _make_memory_dir(tmp_path: Path, content: str = "Ссылка на `ghost.py` тут.\n") -> Path:
    d = tmp_path / "memory"
    d.mkdir()
    (d / "project_foo.md").write_text(content, encoding="utf-8")
    return d


def test_run_finds_dead_anchor_in_memory_file(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    mem = _make_memory_dir(tmp_path)
    findings = ml.run(root, mem, False, tmp_path / "no_allow.yaml")
    assert any(f.span == "ghost.py" for f in findings)


def test_run_respects_allowlist(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    mem = _make_memory_dir(tmp_path)
    allow = tmp_path / "allow.yaml"
    allow.write_text("allow:\n  - span: 'ghost.py'\n    reason: 'test'\n", encoding="utf-8")
    findings = ml.run(root, mem, False, allow)
    assert not any(f.span == "ghost.py" for f in findings)


def test_run_include_commands_scans_claude_commands(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / ".claude" / "commands" / "ghost-cmd.md").write_text(
        "Ссылка на `nonexistent_thing_xyz` тут.\n", encoding="utf-8"
    )
    mem = tmp_path / "empty_memory"
    mem.mkdir()
    findings = ml.run(root, mem, True, tmp_path / "no_allow.yaml")
    assert any(f.span == "nonexistent_thing_xyz" for f in findings)


def test_run_without_include_commands_skips_claude_commands(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / ".claude" / "commands" / "ghost-cmd.md").write_text(
        "Ссылка на `nonexistent_thing_xyz` тут.\n", encoding="utf-8"
    )
    mem = tmp_path / "empty_memory"
    mem.mkdir()
    findings = ml.run(root, mem, False, tmp_path / "no_allow.yaml")
    assert not any(f.span == "nonexistent_thing_xyz" for f in findings)


# --- skip_stats() / --verbose ---

def test_skip_stats_counts_classified_and_skipped(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    src = root / "CLAUDE.md"
    src.write_text("Путь `convert/converters.py` и мусор `build` и `--force`.\n", encoding="utf-8")
    total, skipped = ml.skip_stats([src])
    assert total == 3  # 3 бэктик-спана
    assert skipped == 2  # "build" (односегментное слово) и "--force" (CLI-флаг) — класс 5


def test_skip_stats_empty_sources() -> None:
    assert ml.skip_stats([]) == (0, 0)


def test_main_verbose_prints_skip_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _make_repo(tmp_path)
    mem = tmp_path / "empty_memory"
    mem.mkdir()
    code = ml.main(["--repo-root", str(root), "--memory-dir", str(mem), "--verbose"])
    assert code == 0
    assert "[verbose] спанов извлечено" in capsys.readouterr().out


def test_main_without_verbose_omits_skip_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _make_repo(tmp_path)
    mem = tmp_path / "empty_memory"
    mem.mkdir()
    ml.main(["--repo-root", str(root), "--memory-dir", str(mem)])
    assert "[verbose]" not in capsys.readouterr().out


def test_main_exit_zero_by_default_even_with_dead_anchors(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    mem = _make_memory_dir(tmp_path)
    code = ml.main(["--repo-root", str(root), "--memory-dir", str(mem)])
    assert code == 0


def test_main_fail_on_dead_returns_nonzero_when_findings_exist(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    mem = _make_memory_dir(tmp_path)
    code = ml.main(["--repo-root", str(root), "--memory-dir", str(mem), "--fail-on-dead"])
    assert code == 1


def test_main_fail_on_dead_returns_zero_when_clean(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    mem = tmp_path / "empty_memory"
    mem.mkdir()
    code = ml.main(["--repo-root", str(root), "--memory-dir", str(mem), "--fail-on-dead"])
    assert code == 0


def test_main_missing_memory_dir_scans_only_claude_md(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    code = ml.main(["--repo-root", str(root), "--memory-dir", str(tmp_path / "does_not_exist")])
    assert code == 0


# --- format_report() ---

def test_format_report_includes_totals() -> None:
    findings = [ml.Finding("f.md", "x", "bare", "line", "WARNING")]
    report = ml.format_report(findings)
    assert "1 WARNING" in report and "0 INFO" in report


def test_format_report_empty_findings() -> None:
    assert "0 WARNING" in ml.format_report([])


# --- hypothesis: извлечение спанов не падает на произвольном markdown ---

@given(st.text())
@settings(max_examples=200)
def test_extract_anchors_never_crashes(text: str) -> None:
    list(ml.extract_anchors(text))


@given(st.text(min_size=1, max_size=60))
@settings(max_examples=200)
def test_classify_never_crashes(span: str) -> None:
    ml.classify(span)

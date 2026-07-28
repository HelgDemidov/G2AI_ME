"""Тесты discover.py CLI: подкоманда `discover` — argparse + вызов orchestrate.discover
(spec discovery-core §5)."""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from core import fsio, schema
from discover import main
from discovery import registry, store
from discovery.base import DiscoverResult
from discovery.connectors import snowball


class _StaticConnector:
    def __init__(self, cid: str) -> None:
        self.id = cid
        self.kind = schema.ConnectorKind.manual
        self.enabled = True

    def discover(self) -> DiscoverResult:
        cand = schema.CandidateRecord.model_validate(
            {
                "connector_id": self.id,
                "retrieved_at": dt.date(2026, 7, 21),
                "raw_hash": f"doc-{self.id}",
            }
        )
        return DiscoverResult(candidates=[cand])


class _BoomConnector:
    id = "boom"
    kind = schema.ConnectorKind.manual
    enabled = True

    def discover(self) -> DiscoverResult:
        raise RuntimeError("down")


@pytest.fixture(autouse=True)
def _isolated_snowball_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-путь не принимает ``cache_dir`` параметром, поэтому изоляция делается модульной
    константой — та читается В МОМЕНТ ВЫЗОВА именно ради этого. Без фикстуры подкоманда
    `snowball` писала бы в БОЕВОЙ кэш, и гейт герметичности краснеет (справедливо)."""
    monkeypatch.setattr(snowball, "CACHE_DIR", tmp_path / "snowball_cache")


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    saved = dict(registry.CONNECTORS)
    registry.CONNECTORS.clear()
    yield
    registry.CONNECTORS.clear()
    registry.CONNECTORS.update(saved)


def test_discover_subcommand_runs_and_persists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry.register(_StaticConnector("a"))

    code = main(["discover", "--root", str(tmp_path)])

    assert code == 0
    assert len(store.load(tmp_path)) == 1
    assert "1 новых кандидат" in capsys.readouterr().out


def test_discover_subcommand_dry_run_does_not_write(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a"))

    code = main(["discover", "--root", str(tmp_path), "--dry-run"])

    assert code == 0
    assert store.load(tmp_path) == []


def test_discover_subcommand_only_narrows_connectors(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a"))
    registry.register(_StaticConnector("b"))

    main(["discover", "--root", str(tmp_path), "--only", "a"])

    loaded = store.load(tmp_path)
    assert [c.connector_id for c in loaded] == ["a"]


def test_discover_subcommand_nonzero_exit_on_connector_failure(tmp_path: Path) -> None:
    registry.register(_BoomConnector())

    assert main(["discover", "--root", str(tmp_path)]) == 1


def test_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])


# --- inject (spec discovery-manual §2) ---


def test_inject_subcommand_adds_candidate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "inject",
            "--root",
            str(tmp_path),
            "--url",
            "https://gov.example.org/strategy.pdf",
            "--title",
            "National AI Strategy",
            "--issuer",
            "Ministry",
            "--language",
            "en",
        ]
    )
    assert code == 0
    assert len(store.load(tmp_path)) == 1
    assert "добавлен кандидат" in capsys.readouterr().out


def test_inject_subcommand_directed_search_missing_campaign_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "inject",
            "--root",
            str(tmp_path),
            "--url",
            "https://gov.example.org/a.pdf",
            "--title",
            "T",
            "--issuer",
            "I",
            "--language",
            "en",
            "--kind",
            "directed_search",
            "--query",
            "ai strategy",
        ]
    )
    assert code == 1
    assert "campaign" in capsys.readouterr().out


def test_inject_subcommand_duplicate_is_noop_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "inject",
        "--root",
        str(tmp_path),
        "--url",
        "https://gov.example.org/a.pdf",
        "--title",
        "T",
        "--issuer",
        "I",
        "--language",
        "en",
    ]
    assert main(argv) == 0
    code = main(argv)
    assert code == 0
    assert "уже присутствует" in capsys.readouterr().out
    assert len(store.load(tmp_path)) == 1


def test_inject_subcommand_parses_optional_flags(tmp_path: Path) -> None:
    code = main(
        [
            "inject",
            "--root",
            str(tmp_path),
            "--url",
            "https://gov.example.org/a.pdf",
            "--title",
            "T",
            "--issuer",
            "I",
            "--language",
            "en",
            "--jurisdiction",
            "me",
            "--date",
            "2026-03-01",
            "--summary",
            "short summary",
            "--rights",
            "cc-by",
            "--sensitivity",
            "confidential",
        ]
    )
    assert code == 0
    cand = store.load(tmp_path)[0]
    assert cand.jurisdiction == "me"
    assert cand.doc_date is not None and cand.doc_date.isoformat() == "2026-03-01"
    assert cand.native_summary == "short summary"
    assert cand.rights == schema.Rights.cc_by
    assert cand.sensitivity == schema.Sensitivity.confidential


def test_inject_subcommand_supersedes_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--supersedes` заводит РЕДАКЦИЮ на том же URL (spec discovery-candidates-sharding §5):
    dedup её не поглощает, сводка честно говорит, что это редакция."""
    from tests.support import valid_record

    rec = schema.SourceRecord.model_validate(
        valid_record() | {"source_url": "https://gov.example.org/law.pdf"}
    )
    schema.save_record(rec, tmp_path)
    common = [
        "inject", "--root", str(tmp_path),
        "--url", "https://gov.example.org/law.pdf",
        "--title", "Registration Law", "--issuer", "Ministry", "--language", "en",
    ]
    assert main(common) == 0  # обычный кандидат на том же URL

    code = main([*common, "--supersedes", rec.id])

    assert code == 0
    out = capsys.readouterr().out
    assert f"редакция, заменяет {rec.id}" in out
    editions = [c for c in store.load(tmp_path) if c.supersedes == rec.id]
    assert len(editions) == 1


def test_inject_subcommand_supersedes_unknown_doc_id_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "inject", "--root", str(tmp_path),
            "--url", "https://gov.example.org/law.pdf",
            "--title", "T", "--issuer", "I", "--language", "en",
            "--supersedes", "me-no-such-doc-2026",
        ]
    )

    assert code == 1
    assert "нет в реестре корпуса" in capsys.readouterr().out
    assert store.load(tmp_path) == []


# --- worksheet (spec discovery-manual §3) ---


def test_worksheet_subcommand_prints_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "inject",
            "--root",
            str(tmp_path),
            "--url",
            "https://gov.example.org/a.pdf",
            "--title",
            "T",
            "--issuer",
            "I",
            "--language",
            "en",
        ]
    )
    code = main(["worksheet", "--root", str(tmp_path), "--format", "md"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Триаж-worksheet" in out
    assert "gov.example.org/a.pdf" in out


def test_worksheet_subcommand_writes_to_out_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "inject",
            "--root",
            str(tmp_path),
            "--url",
            "https://gov.example.org/a.pdf",
            "--title",
            "T",
            "--issuer",
            "I",
            "--language",
            "en",
        ]
    )
    out_path = tmp_path / "triage_worksheet.md"
    code = main(
        ["worksheet", "--root", str(tmp_path), "--out", str(out_path), "--format", "md"]
    )
    assert code == 0
    assert out_path.exists()
    assert "Триаж-worksheet" in out_path.read_text(encoding="utf-8")
    assert "1 ждущих" in capsys.readouterr().out


def test_worksheet_subcommand_empty_root_no_candidates(tmp_path: Path) -> None:
    code = main(["worksheet", "--root", str(tmp_path)])
    assert code == 0


# --- --format/--connector/--limit (spec triage-intake-hardening §2) ---


def test_worksheet_subcommand_default_format_is_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Смена дефолта (spec §2): без --format машинный контракт, не человеческая таблица."""
    main(
        [
            "inject", "--root", str(tmp_path), "--url", "https://gov.example.org/a.pdf",
            "--title", "T", "--issuer", "I", "--language", "en",
        ]
    )
    capsys.readouterr()  # смыть вывод inject — worksheet проверяем изолированно
    code = main(["worksheet", "--root", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["shown"] == 1
    assert "Триаж-worksheet" not in out


def test_worksheet_subcommand_connector_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "inject", "--root", str(tmp_path), "--url", "https://gov.example.org/a.pdf",
            "--title", "T", "--issuer", "I", "--language", "en",
        ]
    )
    existing = store.load(tmp_path)
    existing.append(
        schema.CandidateRecord.model_validate(
            {
                "connector_id": "oecd", "retrieved_at": "2026-07-21", "raw_hash": "o" * 64,
                "title": "Other", "issuer": "Gov", "language": "en",
                "source_url": "https://oecd.example.org/x.pdf",
            }
        )
    )
    store.save(existing, tmp_path)
    capsys.readouterr()  # смыть вывод inject — worksheet проверяем изолированно
    code = main(["worksheet", "--root", str(tmp_path), "--connector", "manual"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shown"] == 1
    assert payload["candidates"][0]["connector_id"] == "manual"


def test_worksheet_subcommand_limit_truncates_and_reports_total(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for i in range(3):
        main(
            [
                "inject", "--root", str(tmp_path), "--url", f"https://gov.example.org/{i}.pdf",
                "--title", f"T{i}", "--issuer", "I", "--language", "en",
            ]
        )
    capsys.readouterr()  # смыть вывод трёх inject — worksheet проверяем изолированно
    code = main(["worksheet", "--root", str(tmp_path), "--limit", "2"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shown"] == 2
    assert payload["pending_total"] == 3


def test_worksheet_subcommand_connector_without_limit_reports_no_truncation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Регресс дефекта `/review` PR #56: `--connector` без `--limit` ничего не усекает,
    поэтому `shown == pending_total` и строки «Показано N из M» быть не должно."""
    main(
        [
            "inject", "--root", str(tmp_path), "--url", "https://gov.example.org/a.pdf",
            "--title", "T", "--issuer", "I", "--language", "en",
        ]
    )
    capsys.readouterr()
    code = main(["worksheet", "--root", str(tmp_path), "--connector", "manual", "--format", "md"])
    assert code == 0
    assert "Показано" not in capsys.readouterr().out


def test_worksheet_subcommand_limit_cuts_unacquirable_section_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--limit N` — «дай партию», и это относится к ОБЕИМ очередям: иначе `--limit 20`
    при сотне недобываемых отдавал бы 20 + 100."""
    cands = [
        schema.CandidateRecord.model_validate(
            {
                "connector_id": "manual", "retrieved_at": "2026-07-28", "raw_hash": str(i) * 64,
                "title": f"T{i}", "issuer": "I", "language": "en",
                "source_url": f"https://gov.example.org/{i}.pdf",
                "rejected_reason": "WAF", "rejected_kind": "unacquirable",
            }
        )
        for i in range(1, 4)
    ]
    store.save(cands, tmp_path)
    code = main(["worksheet", "--root", str(tmp_path), "--limit", "2"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.split("\n(+", 1)[0])
    assert payload["unacquirable_shown"] == 2
    assert payload["unacquirable_total"] == 3


def test_worksheet_subcommand_no_selection_omits_total_annotation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "inject", "--root", str(tmp_path), "--url", "https://gov.example.org/a.pdf",
            "--title", "T", "--issuer", "I", "--language", "en",
        ]
    )
    capsys.readouterr()  # смыть вывод inject — worksheet проверяем изолированно
    code = main(["worksheet", "--root", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pending_total"] == payload["shown"] == 1


# --- apply (spec discovery-manual §4) ---


_DECISIONS_YAML = """\
- raw_hash: "{raw_hash}"
  action: admit
  id: me-example-strategy-2026
  entity_id: me
  track: target-entity
  issuer_type: government
  geo_scope: national
  doc_type: national_strategy
  authority: soft_law
  admission: {{axis: agentic_g2ai, rationale: "matches axis"}}
"""


def test_apply_subcommand_admits_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "inject",
            "--root",
            str(tmp_path),
            "--url",
            "https://gov.example.org/strategy.pdf",
            "--title",
            "T",
            "--issuer",
            "I",
            "--language",
            "en",
        ]
    )
    raw_hash = store.load(tmp_path)[0].raw_hash
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text(_DECISIONS_YAML.format(raw_hash=raw_hash), encoding="utf-8")

    code = main(["apply", str(decisions_path), "--root", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Следующий шаг" in out
    assert (tmp_path / "target-entity" / "me" / "me-example-strategy-2026" / "meta.yaml").exists()


def test_apply_subcommand_error_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text("- raw_hash: 'unknownhash12'\n  action: reject\n", encoding="utf-8")

    code = main(["apply", str(decisions_path), "--root", str(tmp_path)])
    assert code == 1
    assert "✗" in capsys.readouterr().out


def test_apply_subcommand_dry_run_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "inject",
            "--root",
            str(tmp_path),
            "--url",
            "https://gov.example.org/strategy.pdf",
            "--title",
            "T",
            "--issuer",
            "I",
            "--language",
            "en",
        ]
    )
    raw_hash = store.load(tmp_path)[0].raw_hash
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text(_DECISIONS_YAML.format(raw_hash=raw_hash), encoding="utf-8")

    code = main(["apply", str(decisions_path), "--root", str(tmp_path), "--dry-run"])
    assert code == 0
    assert "dry-run" in capsys.readouterr().out
    assert not (tmp_path / "target-entity" / "me" / "me-example-strategy-2026" / "meta.yaml").exists()


def test_apply_subcommand_rejects_non_list_decisions_file(tmp_path: Path) -> None:
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text("not_a_list: true\n", encoding="utf-8")

    code = main(["apply", str(decisions_path), "--root", str(tmp_path)])
    assert code == 1


_DECISIONS_YAML_BAD_AXIS = """\
- raw_hash: "{raw_hash}"
  action: admit
  id: me-example-strategy-2026
  entity_id: me
  track: target-entity
  issuer_type: government
  geo_scope: national
  doc_type: national_strategy
  authority: soft_law
  admission: {{axis: economy, rationale: "matches axis"}}
"""


def test_apply_subcommand_flags_invalid_axis_after_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Спек vocab-axes, rationale «слабое место свапа»: опечатка в словарном поле
    (axis вне vocab_axes.yaml) ловится сразу после apply, не только следующим
    отдельным запуском validate_sources/run_pipeline."""
    main(
        [
            "inject",
            "--root",
            str(tmp_path),
            "--url",
            "https://gov.example.org/strategy.pdf",
            "--title",
            "T",
            "--issuer",
            "I",
            "--language",
            "en",
        ]
    )
    raw_hash = store.load(tmp_path)[0].raw_hash
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text(
        _DECISIONS_YAML_BAD_AXIS.format(raw_hash=raw_hash), encoding="utf-8"
    )

    code = main(["apply", str(decisions_path), "--root", str(tmp_path)])
    assert code == 1
    out = capsys.readouterr().out
    assert "невалиден" in out
    assert "admission.axis" in out and "вне словаря" in out
    # meta.yaml уже записан — гейт здесь постфактум, не блокирует запись (см. rationale)
    assert (tmp_path / "target-entity" / "me" / "me-example-strategy-2026" / "meta.yaml").exists()


def test_apply_subcommand_dry_run_skips_post_batch_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """dry-run не пишет meta.yaml — постбатчевая валидация не запускается вовсе."""
    main(
        [
            "inject",
            "--root",
            str(tmp_path),
            "--url",
            "https://gov.example.org/strategy.pdf",
            "--title",
            "T",
            "--issuer",
            "I",
            "--language",
            "en",
        ]
    )
    raw_hash = store.load(tmp_path)[0].raw_hash
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text(
        _DECISIONS_YAML_BAD_AXIS.format(raw_hash=raw_hash), encoding="utf-8"
    )

    code = main(["apply", str(decisions_path), "--root", str(tmp_path), "--dry-run"])
    assert code == 0
    assert "невалиден" not in capsys.readouterr().out


# --- `discover.py snowball` — полный проход через main(argv) (spec discovery-snowball §3,
# коммит 5). Единственный внешний ресурс snowball — уже принятый корпус на диске; никакой
# сети/модели — реальный CI-safe "интеграционный" тест этого слоя (см. spec §Тестовое
# покрытие: конвенция проекта не заводит для этого отдельную integration/-папку). ---


def test_snowball_subcommand_dry_run_finds_link_but_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.support import build_pdf, valid_record, write_doc

    data = valid_record() | {"id": "snowball-cli-doc", "entity_id": "me", "track": "target-entity"}
    raw_bytes = build_pdf(
        lines=[("Egypt AI Strategy", 50.0, 60.0, 12.0)],
        links=[("https://ai.gov.eg/strategy.pdf", 50.0, 55.0, 300.0, 80.0)],
    )
    write_doc(tmp_path, data, raw=raw_bytes, md="no printed urls", state={"sha256": "a" * 64})

    code = main(
        ["snowball", "--doc", "snowball-cli-doc", "--root", str(tmp_path), "--dry-run"]
    )

    assert code == 0
    assert store.load(tmp_path) == []
    out = capsys.readouterr().out
    assert "snowball: найдено 1 | свежих 1 | слито 0" in out


def test_snowball_subcommand_with_citations_loads_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Живой дефект (2026-07-25): ``discover.py snowball --with-citations`` падал с
    «нет OPENROUTER_API_KEY» на чистой оболочке — в отличие от ``run_pipeline.py``
    (``if embed: load_dotenv()``), ``discover.py`` нигде не читал ``.env`` сам,
    полагаясь на то, что ключ уже есть в окружении. Пустой корпус (``tmp_path`` без
    документов) — ноль реальных обращений к OpenRouter, проверяется только сам факт
    вызова ``load_dotenv``."""
    import discover

    calls: list[None] = []
    monkeypatch.setattr(discover, "load_dotenv", lambda: calls.append(None))

    code = main(["snowball", "--with-citations", "--root", str(tmp_path), "--dry-run"])

    assert code == 0
    assert len(calls) == 1


def test_snowball_subcommand_without_citations_does_not_load_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Обычный прогон (без ``--with-citations``) НЕ должен трогать окружение вовсе —
    зеркало ``run_pipeline.py``'s ``if embed: load_dotenv()`` (условно, не всегда)."""
    import discover

    calls: list[None] = []
    monkeypatch.setattr(discover, "load_dotenv", lambda: calls.append(None))

    code = main(["snowball", "--root", str(tmp_path), "--dry-run"])

    assert code == 0
    assert calls == []


def test_snowball_subcommand_persists_candidate(tmp_path: Path) -> None:
    from tests.support import build_pdf, valid_record, write_doc

    data = valid_record() | {"id": "snowball-cli-persist-doc", "entity_id": "me", "track": "target-entity"}
    raw_bytes = build_pdf(
        lines=[("Egypt AI Strategy", 50.0, 60.0, 12.0)],
        links=[("https://ai.gov.eg/strategy.pdf", 50.0, 55.0, 300.0, 80.0)],
    )
    write_doc(tmp_path, data, raw=raw_bytes, md="no printed urls", state={"sha256": "a" * 64})

    code = main(["snowball", "--doc", "snowball-cli-persist-doc", "--root", str(tmp_path)])

    assert code == 0
    loaded = store.load(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].source_url == "https://ai.gov.eg/strategy.pdf"
    assert loaded[0].connector_id == "snowball"


def test_snowball_subcommand_doc_filter_excludes_other_documents(tmp_path: Path) -> None:
    from tests.support import build_pdf, valid_record, write_doc

    data_a = valid_record() | {"id": "snowball-doc-a", "entity_id": "me", "track": "target-entity"}
    data_b = valid_record() | {"id": "snowball-doc-b", "entity_id": "me", "track": "target-entity"}
    write_doc(
        tmp_path,
        data_a,
        raw=build_pdf(
            lines=[("A link", 50.0, 60.0, 12.0)],
            links=[("https://example.org/only-a", 50.0, 55.0, 300.0, 80.0)],
        ),
        md="x",
        state={"sha256": "a" * 64},
    )
    write_doc(
        tmp_path,
        data_b,
        raw=build_pdf(
            lines=[("B link", 50.0, 60.0, 12.0)],
            links=[("https://example.org/only-b", 50.0, 55.0, 300.0, 80.0)],
        ),
        md="x",
        state={"sha256": "b" * 64},
    )

    main(["snowball", "--doc", "snowball-doc-a", "--root", str(tmp_path)])

    loaded = store.load(tmp_path)
    assert [c.source_url for c in loaded] == ["https://example.org/only-a"]


# --- corpus_mutation.lock (spec discovery-acquire-seam-hardening §2, Г1) ---
#
# Живая проба PR #53 (fsio.exclusive_flock): два независимых open+flock конфликтуют
# и ВНУТРИ одного процесса (per-open-file-description locking) — держим лок в тесте
# через `with fsio.exclusive_flock(...)`, вызов `main()` внутри блока симулирует
# конкурентный прогон (recheck/run_pipeline) без второго процесса.


def test_inject_subcommand_blocked_while_lock_held(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Регресс репро A аудита: inject в окне, где другой мутатор (recheck/discover)
    держит корпусный лок, обязан быть отвергнут, а не потерян тихой перезаписью store."""
    with fsio.exclusive_flock(schema.corpus_lock_path(tmp_path)):
        code = main(
            [
                "inject", "--root", str(tmp_path),
                "--url", "https://example.org/doc", "--title", "Doc",
                "--issuer", "Issuer", "--language", "en",
            ]
        )
    assert code == 1
    assert "другой прогон" in capsys.readouterr().out
    assert store.load(tmp_path) == []  # не просочилось мимо лока


def test_apply_subcommand_blocked_while_lock_held(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text("[]", encoding="utf-8")
    with fsio.exclusive_flock(schema.corpus_lock_path(tmp_path)):
        code = main(["apply", str(decisions), "--root", str(tmp_path)])
    assert code == 1


def test_apply_subcommand_dry_run_not_blocked_while_lock_held(tmp_path: Path) -> None:
    """dry-run обязан быть no-op и не мешать живому прогону — лок его не касается."""
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text("[]", encoding="utf-8")
    with fsio.exclusive_flock(schema.corpus_lock_path(tmp_path)):
        code = main(["apply", str(decisions), "--root", str(tmp_path), "--dry-run"])
    assert code == 0


def test_discover_subcommand_blocked_while_lock_held(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a"))
    with fsio.exclusive_flock(schema.corpus_lock_path(tmp_path)):
        code = main(["discover", "--root", str(tmp_path)])
    assert code == 1
    assert store.load(tmp_path) == []


def test_discover_subcommand_dry_run_not_blocked_while_lock_held(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a"))
    with fsio.exclusive_flock(schema.corpus_lock_path(tmp_path)):
        code = main(["discover", "--root", str(tmp_path), "--dry-run"])
    assert code == 0


def test_worksheet_subcommand_not_blocked_while_lock_held(tmp_path: Path) -> None:
    """worksheet — read-only, лок никогда не берёт, удержанный лок ему не мешает."""
    with fsio.exclusive_flock(schema.corpus_lock_path(tmp_path)):
        code = main(["worksheet", "--root", str(tmp_path)])
    assert code == 0


def test_snowball_subcommand_blocked_while_lock_held(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with fsio.exclusive_flock(schema.corpus_lock_path(tmp_path)):
        code = main(["snowball", "--root", str(tmp_path)])
    assert code == 1
    assert "другой прогон" in capsys.readouterr().out


def test_snowball_subcommand_dry_run_not_blocked_while_lock_held(tmp_path: Path) -> None:
    with fsio.exclusive_flock(schema.corpus_lock_path(tmp_path)):
        code = main(["snowball", "--root", str(tmp_path), "--dry-run"])
    assert code == 0

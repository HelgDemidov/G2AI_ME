"""Тесты discover.py CLI: подкоманда `discover` — argparse + вызов orchestrate.discover
(spec discovery-core §5)."""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest

from core import schema
from discover import main
from discovery import registry, store
from discovery.base import ConnectorCursor, DiscoverResult


class _StaticConnector:
    def __init__(self, cid: str) -> None:
        self.id = cid
        self.kind = schema.ConnectorKind.manual
        self.enabled = True

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        cand = schema.CandidateRecord.model_validate(
            {
                "connector_id": self.id,
                "retrieved_at": dt.date(2026, 7, 21),
                "raw_hash": f"doc-{self.id}",
            }
        )
        return DiscoverResult(candidates=[cand], cursor={})


class _BoomConnector:
    id = "boom"
    kind = schema.ConnectorKind.manual
    enabled = True

    def discover(self, cursor: ConnectorCursor | None) -> DiscoverResult:
        raise RuntimeError("down")


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
    assert len(store.load(tmp_path / "candidates.yaml")) == 1
    assert "1 новых кандидат" in capsys.readouterr().out


def test_discover_subcommand_dry_run_does_not_write(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a"))

    code = main(["discover", "--root", str(tmp_path), "--dry-run"])

    assert code == 0
    assert not (tmp_path / "candidates.yaml").exists()


def test_discover_subcommand_only_narrows_connectors(tmp_path: Path) -> None:
    registry.register(_StaticConnector("a"))
    registry.register(_StaticConnector("b"))

    main(["discover", "--root", str(tmp_path), "--only", "a"])

    loaded = store.load(tmp_path / "candidates.yaml")
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
    assert len(store.load(tmp_path / "candidates.yaml")) == 1
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
    assert len(store.load(tmp_path / "candidates.yaml")) == 1


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
    cand = store.load(tmp_path / "candidates.yaml")[0]
    assert cand.jurisdiction == "me"
    assert cand.doc_date is not None and cand.doc_date.isoformat() == "2026-03-01"
    assert cand.native_summary == "short summary"
    assert cand.rights == schema.Rights.cc_by
    assert cand.sensitivity == schema.Sensitivity.confidential


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
    code = main(["worksheet", "--root", str(tmp_path)])
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
    code = main(["worksheet", "--root", str(tmp_path), "--out", str(out_path)])
    assert code == 0
    assert out_path.exists()
    assert "Триаж-worksheet" in out_path.read_text(encoding="utf-8")
    assert "1 ждущих" in capsys.readouterr().out


def test_worksheet_subcommand_empty_root_no_candidates(tmp_path: Path) -> None:
    code = main(["worksheet", "--root", str(tmp_path)])
    assert code == 0


# --- apply (spec discovery-manual §4) ---


_DECISIONS_YAML = """\
- raw_hash: "{raw_hash}"
  action: admit
  id: me-example-strategy-2026
  entity_id: me
  track: montenegro
  issuer_type: government
  geo_scope: national
  doc_type: national_strategy
  authority: soft_law
  relevance: {{target_fit: primary, axis: agentic_g2ai, assessed_stage: triage,
              rationale: "matches axis", assessed_date: 2026-07-21}}
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
    raw_hash = store.load(tmp_path / "candidates.yaml")[0].raw_hash
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text(_DECISIONS_YAML.format(raw_hash=raw_hash), encoding="utf-8")

    code = main(["apply", str(decisions_path), "--root", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Следующий шаг" in out
    assert (tmp_path / "montenegro" / "me" / "me-example-strategy-2026" / "meta.yaml").exists()


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
    raw_hash = store.load(tmp_path / "candidates.yaml")[0].raw_hash
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text(_DECISIONS_YAML.format(raw_hash=raw_hash), encoding="utf-8")

    code = main(["apply", str(decisions_path), "--root", str(tmp_path), "--dry-run"])
    assert code == 0
    assert "dry-run" in capsys.readouterr().out
    assert not (tmp_path / "montenegro" / "me" / "me-example-strategy-2026" / "meta.yaml").exists()


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
  track: montenegro
  issuer_type: government
  geo_scope: national
  doc_type: national_strategy
  authority: soft_law
  relevance: {{target_fit: primary, axis: economy, assessed_stage: triage,
              rationale: "matches axis", assessed_date: 2026-07-21}}
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
    raw_hash = store.load(tmp_path / "candidates.yaml")[0].raw_hash
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text(
        _DECISIONS_YAML_BAD_AXIS.format(raw_hash=raw_hash), encoding="utf-8"
    )

    code = main(["apply", str(decisions_path), "--root", str(tmp_path)])
    assert code == 1
    out = capsys.readouterr().out
    assert "невалиден" in out
    assert "relevance.axis" in out and "вне словаря" in out
    # meta.yaml уже записан — гейт здесь постфактум, не блокирует запись (см. rationale)
    assert (tmp_path / "montenegro" / "me" / "me-example-strategy-2026" / "meta.yaml").exists()


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
    raw_hash = store.load(tmp_path / "candidates.yaml")[0].raw_hash
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

    data = valid_record() | {"id": "snowball-cli-doc", "entity_id": "me", "track": "montenegro"}
    raw_bytes = build_pdf(
        lines=[("Egypt AI Strategy", 50.0, 60.0, 12.0)],
        links=[("https://ai.gov.eg/strategy.pdf", 50.0, 55.0, 300.0, 80.0)],
    )
    write_doc(tmp_path, data, raw=raw_bytes, md="no printed urls", state={"sha256": "a" * 64})

    code = main(
        ["snowball", "--doc", "snowball-cli-doc", "--root", str(tmp_path), "--dry-run"]
    )

    assert code == 0
    assert not (tmp_path / "candidates.yaml").exists()
    out = capsys.readouterr().out
    assert "snowball: найдено 1 | свежих 1 | слито 0" in out
    assert "Итого: 1 новых кандидат" in out
    assert "dry-run" in out


def test_snowball_subcommand_persists_candidate(tmp_path: Path) -> None:
    from tests.support import build_pdf, valid_record, write_doc

    data = valid_record() | {"id": "snowball-cli-persist-doc", "entity_id": "me", "track": "montenegro"}
    raw_bytes = build_pdf(
        lines=[("Egypt AI Strategy", 50.0, 60.0, 12.0)],
        links=[("https://ai.gov.eg/strategy.pdf", 50.0, 55.0, 300.0, 80.0)],
    )
    write_doc(tmp_path, data, raw=raw_bytes, md="no printed urls", state={"sha256": "a" * 64})

    code = main(["snowball", "--doc", "snowball-cli-persist-doc", "--root", str(tmp_path)])

    assert code == 0
    loaded = store.load(tmp_path / "candidates.yaml")
    assert len(loaded) == 1
    assert loaded[0].source_url == "https://ai.gov.eg/strategy.pdf"
    assert loaded[0].connector_id == "snowball"


def test_snowball_subcommand_doc_filter_excludes_other_documents(tmp_path: Path) -> None:
    from tests.support import build_pdf, valid_record, write_doc

    data_a = valid_record() | {"id": "snowball-doc-a", "entity_id": "me", "track": "montenegro"}
    data_b = valid_record() | {"id": "snowball-doc-b", "entity_id": "me", "track": "montenegro"}
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

    loaded = store.load(tmp_path / "candidates.yaml")
    assert [c.source_url for c in loaded] == ["https://example.org/only-a"]

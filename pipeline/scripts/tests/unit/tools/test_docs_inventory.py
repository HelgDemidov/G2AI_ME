"""Тесты генератора ROADMAP-инвентаризации (docs_inventory.py). Полностью герметичны:
фикстурное docs-дерево и оверлей строятся в tmp_path, реальный docs/ не читается."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import docs_inventory as di

STATUS_A = "черновик v1 · 2026-07-17"
STATUS_CHARTER = "чартер v1.0 · ФИНАЛИЗИРОВАН"


def make_docs(tmp_path: Path) -> Path:
    """Фикстурное дерево: блок alpha (чартер + спек), блок beta (спек без статуса,
    спек со статусом с '|')."""
    root = tmp_path / "docs"
    (root / "alpha" / "charters").mkdir(parents=True)
    (root / "alpha" / "charters" / "architecture.md").write_text(
        f"# Чартер\n\nСтатус: {STATUS_CHARTER}\nТип: зонтик\n", encoding="utf-8"
    )
    for block, slug, body in [
        ("alpha", "spec-a", f"# Спек A\n\nСтатус: {STATUS_A}\nВетка: x\n"),
        ("beta", "spec-b", "# Спек B\n\nбез статус-строки вовсе\n"),
        ("beta", "spec-pipe", "# C\n\nСтатус: реализовано (PR #1) | хвост с пайпом\n"),
    ]:
        d = root / block / "tech_specs" / slug
        d.mkdir(parents=True)
        (d / "spec.md").write_text(body, encoding="utf-8")
    return root


OVERLAY: dict[str, object] = {
    "blocks": [
        {"key": "beta", "title": "BETA — второй блок первым"},
        {"key": "alpha", "title": "ALPHA — блок"},
    ],
    "queue": {"spec-a": "**1** · вперёд"},
    "extra": [
        {"block": "beta", "name": "jit-x", "kind": "спек", "path": "— (не написан)",
         "status": "JIT", "queue": "**2** · после spec-a"},
    ],
}


# --- extract_status ---

def test_extract_status_found_and_collapsed() -> None:
    assert di.extract_status("шапка\nСтатус:   a   b \nхвост") == "a b"


def test_extract_status_missing() -> None:
    assert di.extract_status("документ без статуса") == di.NO_STATUS


def test_extract_status_truncates_long() -> None:
    long = "x" * (di.MAX_STATUS_LEN + 40)
    out = di.extract_status(f"Статус: {long}")
    assert len(out) == di.MAX_STATUS_LEN + 1 and out.endswith("…")


# --- scan ---

def test_scan_rows(tmp_path: Path) -> None:
    rows = di.scan(make_docs(tmp_path))
    by_name = {r.name: r for r in rows}
    assert set(by_name) == {"Архитектура alpha", "spec-a", "spec-b", "spec-pipe"}
    charter = by_name["Архитектура alpha"]
    assert (charter.block, charter.kind, charter.status) == ("alpha", "чартер", STATUS_CHARTER)
    assert charter.path == "alpha/charters/architecture.md"
    assert by_name["spec-a"].path == "alpha/tech_specs/spec-a/spec.md"
    assert by_name["spec-b"].status == di.NO_STATUS
    assert rows == di.scan(make_docs(tmp_path / "again"))  # детерминизм порядка


# --- render ---

def test_render_order_queue_extra_and_escaping(tmp_path: Path) -> None:
    out = di.render(di.scan(make_docs(tmp_path)), dict(OVERLAY))
    beta_pos, alpha_pos = out.index("## BETA"), out.index("## ALPHA")
    assert beta_pos < alpha_pos  # порядок блоков — из оверлея
    assert "| spec-a | спек |" in out and "| **1** · вперёд |" in out
    assert "| spec-b | спек |" in out and f"| {di.NO_STATUS} | — |" in out  # дефолт очереди
    assert "| jit-x | спек | — (не написан) | JIT | **2** · после spec-a |" in out
    assert "PR #1) \\| хвост" in out  # '|' в статусе экранирован
    assert out.index("| spec-b") < out.index("| jit-x")  # extra после сканированных


def test_render_unknown_block_appended(tmp_path: Path) -> None:
    overlay = {"blocks": [{"key": "alpha", "title": "ALPHA"}]}
    out = di.render(di.scan(make_docs(tmp_path)), overlay)
    assert "## BETA" in out and out.index("## ALPHA") < out.index("## BETA")


def test_render_empty_block_skipped(tmp_path: Path) -> None:
    blocks = [{"key": "beta", "title": "BETA"}, {"key": "alpha", "title": "ALPHA"},
              {"key": "ghost", "title": "GHOST"}]
    overlay = dict(OVERLAY) | {"blocks": blocks}
    assert "GHOST" not in di.render(di.scan(make_docs(tmp_path)), overlay)


# --- replace_auto_section ---

def test_replace_preserves_outside() -> None:
    text = f"шапка\n{di.BEGIN_MARK}\nстарое\n{di.END_MARK}\nхвост"
    out = di.replace_auto_section(text, "НОВОЕ")
    assert out.startswith("шапка\n") and out.endswith("\nхвост")
    assert "НОВОЕ" in out and "старое" not in out


@pytest.mark.parametrize("text", ["без маркеров", f"{di.END_MARK}\nпотом\n{di.BEGIN_MARK}"])
def test_replace_bad_markers_raise(text: str) -> None:
    with pytest.raises(ValueError):
        di.replace_auto_section(text, "x")


# --- main (CLI) ---

def _cli_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    docs = make_docs(tmp_path)
    overlay = tmp_path / "roadmap.yaml"
    overlay.write_text(yaml.safe_dump(OVERLAY, allow_unicode=True), encoding="utf-8")
    target = tmp_path / "ROADMAP.md"
    target.write_text(f"# R\n\n{di.BEGIN_MARK}\n{di.END_MARK}\n\nхвост\n", encoding="utf-8")
    return docs, overlay, target


def _argv(docs: Path, overlay: Path, target: Path, *extra: str) -> list[str]:
    return ["--docs-root", str(docs), "--overlay", str(overlay), "--target", str(target), *extra]


def test_main_generates_then_idempotent(tmp_path: Path) -> None:
    docs, overlay, target = _cli_env(tmp_path)
    assert di.main(_argv(docs, overlay, target)) == 0
    first = target.read_text(encoding="utf-8")
    assert "## BETA" in first and first.endswith("хвост\n")  # хвост цел
    assert di.main(_argv(docs, overlay, target)) == 0  # повтор — no-op
    assert target.read_text(encoding="utf-8") == first


def test_main_check_detects_stale_and_fresh(tmp_path: Path) -> None:
    docs, overlay, target = _cli_env(tmp_path)
    assert di.main(_argv(docs, overlay, target, "--check")) == 1  # пустая секция != желаемая
    assert di.main(_argv(docs, overlay, target)) == 0
    assert di.main(_argv(docs, overlay, target, "--check")) == 0  # актуален
    spec = docs / "alpha" / "tech_specs" / "spec-a" / "spec.md"
    spec.write_text(spec.read_text(encoding="utf-8").replace(STATUS_A, "реализовано (PR #99)"),
                    encoding="utf-8")
    assert di.main(_argv(docs, overlay, target, "--check")) == 1  # статус сменился — устарел
    assert di.main(_argv(docs, overlay, target)) == 0  # перегенерация подтягивает статус
    assert "PR #99" in target.read_text(encoding="utf-8")


def test_main_bad_overlay_type(tmp_path: Path) -> None:
    docs, overlay, target = _cli_env(tmp_path)
    overlay.write_text("- список\n- а не mapping\n", encoding="utf-8")
    assert di.main(_argv(docs, overlay, target)) == 2


def test_main_target_without_markers(tmp_path: Path) -> None:
    docs, overlay, target = _cli_env(tmp_path)
    target.write_text("файл без маркеров\n", encoding="utf-8")
    assert di.main(_argv(docs, overlay, target)) == 2

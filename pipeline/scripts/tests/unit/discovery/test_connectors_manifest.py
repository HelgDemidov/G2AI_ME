"""Тесты регистрации через манифест `discovery/connectors/__init__.py` (spec discovery-agora §6).

Подпроцесс — намеренно: `discover.py`/`discovery.connectors.agora` уже импортированы другими
тестовыми модулями этой сессии (Python кэширует импорт, повторный import — no-op, top-level
код agora.py не перезапустится) — только свежий интерпретатор честно проверяет ЦЕПОЧКУ
`discover.py` -> `discovery.connectors` (манифест) -> `agora.py` -> `registry.register()`.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

from core.env import REPO_ROOT

_SCRIPTS_DIR = REPO_ROOT / "pipeline" / "scripts"


# --- _load_all: изоляция отказа загрузки одного коннектора (spec discovery-
# acquire-seam-hardening §7, Г6) ---


def test_load_all_isolates_broken_connector_module(monkeypatch: Any, caplog: Any) -> None:
    """Опечатка в ЛЮБОМ из пяти ``discovery_*.yaml`` (файлы правятся руками по
    назначению — это тюнинг-конфиги) раньше валила ВЕСЬ импорт пакета
    ``discovery.connectors`` — тщательно выстроенная изоляция отказов оркестратора
    (``orchestrate.discover``) обходилась этажом ниже, на импорт-тайме. Прецедент —
    stevedore (OpenStack): битый плагин репортится ``on_load_failure_callback`` и
    изолируется, не валя приложение."""
    import importlib

    from discovery import connectors, registry

    real_import = importlib.import_module

    def fake_import(name: str) -> Any:
        if name.endswith(".broken_connector"):
            raise RuntimeError("YAML синтаксис сломан")
        return real_import(name)

    monkeypatch.setattr("discovery.connectors.importlib.import_module", fake_import)

    with caplog.at_level("WARNING", logger="discovery.connectors"):
        connectors._load_all(("broken_connector", "agora"))

    assert any("broken_connector" in r.message for r in caplog.records)
    assert "agora" in registry.CONNECTORS  # остальные коннекторы регистрируются штатно


def _run_check(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPTS_DIR),
        timeout=30,
    )


def test_importing_discover_cli_registers_agora_via_manifest() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "assert 'agora' in registry.CONNECTORS, sorted(registry.CONNECTORS)\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_agora_registered_as_registry_kind_and_config_gated_enabled() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "from core import schema\n"
        "conn = registry.CONNECTORS['agora']\n"
        "assert conn.kind == schema.ConnectorKind.registry\n"
        "assert conn.enabled is True\n"  # discovery_agora.yaml: enabled: true
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_agora_reachable_via_enabled_connectors_only_filter() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "found = registry.enabled_connectors(only=['agora'])\n"
        "assert len(found) == 1 and found[0].id == 'agora'\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_importing_discover_cli_registers_eurlex_via_manifest() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "assert 'eurlex' in registry.CONNECTORS, sorted(registry.CONNECTORS)\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_eurlex_registered_as_registry_kind_and_config_gated_enabled() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "from core import schema\n"
        "conn = registry.CONNECTORS['eurlex']\n"
        "assert conn.kind == schema.ConnectorKind.registry\n"
        "assert conn.enabled is True\n"  # discovery_eurlex.yaml: enabled: true
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_eurlex_reachable_via_enabled_connectors_only_filter() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "found = registry.enabled_connectors(only=['eurlex'])\n"
        "assert len(found) == 1 and found[0].id == 'eurlex'\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_both_agora_and_eurlex_coexist_in_registry() -> None:
    """Второй registry-коннектор не вытесняет первый (register() отказал бы на дубль
    id, но agora/eurlex — разные id) — оба доступны одновременно после импорта манифеста."""
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "assert {'agora', 'eurlex'} <= set(registry.CONNECTORS)\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_importing_discover_cli_registers_aiforgood_via_manifest() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "assert 'aiforgood' in registry.CONNECTORS, sorted(registry.CONNECTORS)\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_aiforgood_registered_as_registry_kind_and_config_gated_enabled() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "from core import schema\n"
        "conn = registry.CONNECTORS['aiforgood']\n"
        "assert conn.kind == schema.ConnectorKind.registry\n"
        "assert conn.enabled is True\n"  # discovery_aiforgood.yaml: enabled: true
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_aiforgood_reachable_via_enabled_connectors_only_filter() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "found = registry.enabled_connectors(only=['aiforgood'])\n"
        "assert len(found) == 1 and found[0].id == 'aiforgood'\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_all_three_registry_connectors_coexist() -> None:
    """Третий registry-коннектор не вытесняет первые два — все доступны одновременно."""
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "assert {'agora', 'eurlex', 'aiforgood'} <= set(registry.CONNECTORS)\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_importing_discover_cli_registers_snowball_via_manifest() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "assert 'snowball' in registry.CONNECTORS, sorted(registry.CONNECTORS)\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_snowball_registered_as_snowball_kind_and_config_gated_enabled() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "from core import schema\n"
        "conn = registry.CONNECTORS['snowball']\n"
        "assert conn.kind == schema.ConnectorKind.snowball\n"
        "assert conn.enabled is True\n"  # discovery_snowball.yaml: enabled: true
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_snowball_reachable_via_enabled_connectors_only_filter() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "found = registry.enabled_connectors(only=['snowball'])\n"
        "assert len(found) == 1 and found[0].id == 'snowball'\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_snowball_coexists_with_three_registry_connectors() -> None:
    """Пятый архетип (snowball) не вытесняет три registry-коннектора — все доступны разом."""
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "assert {'agora', 'eurlex', 'aiforgood', 'snowball'} <= set(registry.CONNECTORS)\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_importing_discover_cli_registers_oecd_via_manifest() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "assert 'oecd' in registry.CONNECTORS, sorted(registry.CONNECTORS)\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_oecd_registered_as_registry_kind_and_config_gated_enabled() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "from core import schema\n"
        "conn = registry.CONNECTORS['oecd']\n"
        "assert conn.kind == schema.ConnectorKind.registry\n"
        "assert conn.enabled is True\n"  # discovery_oecd.yaml: enabled: true
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_oecd_reachable_via_enabled_connectors_only_filter() -> None:
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "found = registry.enabled_connectors(only=['oecd'])\n"
        "assert len(found) == 1 and found[0].id == 'oecd'\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr


def test_all_four_registry_connectors_and_snowball_coexist() -> None:
    """Четвёртый registry-коннектор (oecd) не вытесняет никого из предыдущих четырёх —
    все пять доступны одновременно после импорта манифеста."""
    code = (
        "from discover import main\n"
        "from discovery import registry\n"
        "assert {'agora', 'eurlex', 'aiforgood', 'snowball', 'oecd'} <= set(registry.CONNECTORS)\n"
    )
    result = _run_check(code)
    assert result.returncode == 0, result.stderr

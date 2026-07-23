"""Тесты регистрации через манифест `discovery/connectors/__init__.py` (spec discovery-agora §6).

Подпроцесс — намеренно: `discover.py`/`discovery.connectors.agora` уже импортированы другими
тестовыми модулями этой сессии (Python кэширует импорт, повторный import — no-op, top-level
код agora.py не перезапустится) — только свежий интерпретатор честно проверяет ЦЕПОЧКУ
`discover.py` -> `discovery.connectors` (манифест) -> `agora.py` -> `registry.register()`.
"""
from __future__ import annotations

import subprocess
import sys

from core.env import REPO_ROOT

_SCRIPTS_DIR = REPO_ROOT / "pipeline" / "scripts"


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

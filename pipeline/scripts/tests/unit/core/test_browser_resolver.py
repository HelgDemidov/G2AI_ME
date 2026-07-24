"""Тесты core/browser_resolver.py: герметично — subprocess замокан, реальный
Node/lightpanda не запускается (тот же паттерн test_openrouter.py — фейковый
транспорт вместо реальной сети/процесса). Живой прогон resolve.mjs —
tests/integration/test_browser_resolver_live.py (маркер browser)."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any

import pytest

from core import browser_resolver
from core.browser_resolver import BrowserResolverUnavailable, is_available, resolve


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_is_available_true_when_node_and_binary_present(monkeypatch: Any, tmp_path: Any) -> None:
    fake_binary = tmp_path / "lightpanda"
    fake_binary.write_bytes(b"")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(browser_resolver, "LIGHTPANDA_BINARY", fake_binary)
    assert is_available() is True


def test_is_available_false_when_node_missing(monkeypatch: Any, tmp_path: Any) -> None:
    fake_binary = tmp_path / "lightpanda"
    fake_binary.write_bytes(b"")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(browser_resolver, "LIGHTPANDA_BINARY", fake_binary)
    assert is_available() is False


def test_is_available_false_when_binary_missing(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(browser_resolver, "LIGHTPANDA_BINARY", tmp_path / "no-such-binary")
    assert is_available() is False


def test_resolve_raises_unavailable_when_not_available(monkeypatch: Any) -> None:
    monkeypatch.setattr(browser_resolver, "is_available", lambda: False)
    with pytest.raises(BrowserResolverUnavailable):
        resolve("https://example.org")


def test_resolve_success_parses_ok_payload(monkeypatch: Any) -> None:
    monkeypatch.setattr(browser_resolver, "is_available", lambda: True)
    payload = '{"ok":true,"html":"<html>hi</html>","url":"https://example.org/landed"}'
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(0, stdout=payload))
    result = resolve("https://example.org")
    assert result.ok is True
    assert result.html == "<html>hi</html>"
    assert result.final_url == "https://example.org/landed"


def test_resolve_content_failure_returns_ok_false_not_exception(monkeypatch: Any) -> None:
    """resolve.mjs само сообщает {"ok":false,...} (страница не отдалась) — это
    содержательный результат, не инструментальный отказ."""
    monkeypatch.setattr(browser_resolver, "is_available", lambda: True)
    payload = '{"ok":false,"error":"Navigation timeout"}'
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(0, stdout=payload))
    result = resolve("https://example.org")
    assert result.ok is False
    assert result.error == "Navigation timeout"


def test_resolve_nonzero_exit_raises_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setattr(browser_resolver, "is_available", lambda: True)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(1, stderr="node: module not found")
    )
    with pytest.raises(BrowserResolverUnavailable, match="node: module not found"):
        resolve("https://example.org")


def test_resolve_malformed_json_raises_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setattr(browser_resolver, "is_available", lambda: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(0, stdout="not json at all"))
    with pytest.raises(BrowserResolverUnavailable, match="невалидный JSON"):
        resolve("https://example.org")


def test_resolve_empty_stdout_raises_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setattr(browser_resolver, "is_available", lambda: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(0, stdout=""))
    with pytest.raises(BrowserResolverUnavailable):
        resolve("https://example.org")


def test_resolve_timeout_raises_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setattr(browser_resolver, "is_available", lambda: True)

    def _raise_timeout(*a: Any, **kw: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="node resolve.mjs", timeout=45)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    with pytest.raises(BrowserResolverUnavailable, match="не ответил"):
        resolve("https://example.org")

"""Тесты core/openrouter.py: ретраи (spec convert-cloud-tier §1) — фейковый
urllib, без реальной сети/ключа. Зеркалит test_embed.py (та же броня)."""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from core.openrouter import RETRY_SCHEDULE, chat_request


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _chat_payload(text: str = "hello") -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}], "usage": {"cost": 0.01}}


def _http_error(code: int, body: bytes = b"{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/chat/completions", code, "err", Message(), io.BytesIO(body)
    )


def _call() -> dict[str, Any]:
    return chat_request({"model": "m", "messages": []}, api_key="test-key")


# --- ретраи ---


def test_retry_succeeds_after_two_429s(monkeypatch: Any) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: float = 1800.0) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _http_error(429)
        return _FakeResponse(_chat_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = _call()
    assert out["choices"][0]["message"]["content"] == "hello"
    assert calls["n"] == 3
    assert sleeps == [RETRY_SCHEDULE[0], RETRY_SCHEDULE[1]]


def test_non_retryable_4xx_raises_immediately_without_sleep(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=1800.0: (_ for _ in ()).throw(_http_error(400, b"bad request")))
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="400"):
        _call()
    assert sleeps == []


def test_413_payload_too_large_is_non_retryable(monkeypatch: Any) -> None:
    """§6 спека: 413 PayloadTooLarge — неретраябельный 4xx, сигнал калибровки
    батча, не ретраев."""
    sleeps: list[float] = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=1800.0: (_ for _ in ()).throw(_http_error(413, b"payload too large")))
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="413"):
        _call()
    assert sleeps == []


def test_5xx_is_retried_then_exhausts(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=1800.0: (_ for _ in ()).throw(_http_error(503)))
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="исчерпаны попытки"):
        _call()
    assert sleeps == list(RETRY_SCHEDULE)  # все 4 паузы, 5 попыток


def test_network_errors_are_retried(monkeypatch: Any) -> None:
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: float = 1800.0) -> Any:
        raise urllib.error.URLError("сеть недоступна")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="исчерпаны попытки"):
        _call()
    assert sleeps == list(RETRY_SCHEDULE)


def test_timeout_error_is_retried(monkeypatch: Any) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: float = 1800.0) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("read timed out")
        return _FakeResponse(_chat_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = _call()
    assert out["choices"][0]["message"]["content"] == "hello"
    assert sleeps == [RETRY_SCHEDULE[0]]


# --- ошибка в теле HTTP-200 (OpenRouter заворачивает провайдерские отказы) ---


def test_inband_error_400_raises_immediately_without_sleep(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=1800.0: _FakeResponse({"error": {"message": "bad request", "code": 400}}),
    )
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="ошибка в теле 200"):
        _call()
    assert sleeps == []


def test_inband_error_429_is_retried_then_succeeds(monkeypatch: Any) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: float = 1800.0) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse({"error": {"message": "rate limited", "code": 429}})
        return _FakeResponse(_chat_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = _call()
    assert out["choices"][0]["message"]["content"] == "hello"
    assert sleeps == [RETRY_SCHEDULE[0]]


# --- ключ не в тексте исключений ---


def test_api_key_never_appears_in_exception_text(monkeypatch: Any) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=1800.0: (_ for _ in ()).throw(_http_error(400, b"bad request")))
    with pytest.raises(RuntimeError) as exc_info:
        chat_request({"model": "m", "messages": []}, api_key="super-secret-key-12345")
    assert "super-secret-key-12345" not in str(exc_info.value)

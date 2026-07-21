"""Тесты OpenRouterEmbedder: ретраи (spec embed-api-first §2) + MRL-усечение
размерности (§2-bis) — фейковый urllib, без реальной сети/ключа."""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from index.embed import RETRY_SCHEDULE, OnnxBgeEmbedder, OpenRouterEmbedder, get_embedder


def _make_embedder(monkeypatch: Any, **kwargs: Any) -> OpenRouterEmbedder:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return OpenRouterEmbedder(**kwargs)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _payload(dim: int, n: int = 1) -> dict[str, Any]:
    return {"data": [{"index": i, "embedding": [1.0] * dim} for i in range(n)]}


def _http_error(code: int, body: bytes = b"{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/embeddings", code, "err", Message(), io.BytesIO(body)
    )


# --- ретраи (§2) ---


def test_retry_succeeds_after_two_429s(monkeypatch: Any) -> None:
    embedder = _make_embedder(monkeypatch, dims=None)
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: int = 120) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _http_error(429)
        return _FakeResponse(_payload(3))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = embedder.embed(["a"])
    assert out.shape == (1, 3)
    assert calls["n"] == 3
    assert sleeps == [RETRY_SCHEDULE[0], RETRY_SCHEDULE[1]]


def test_non_retryable_4xx_raises_immediately_without_sleep(monkeypatch: Any) -> None:
    embedder = _make_embedder(monkeypatch, dims=None)
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: int = 120) -> Any:
        raise _http_error(400, b"bad request")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="400"):
        embedder.embed(["a"])
    assert sleeps == []


def test_network_errors_exhaust_schedule_and_raise_with_fallback_hint(monkeypatch: Any) -> None:
    embedder = _make_embedder(monkeypatch, dims=None)
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: int = 120) -> Any:
        raise urllib.error.URLError("сеть недоступна")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="--backend bge"):
        embedder.embed(["a"])
    assert sleeps == list(RETRY_SCHEDULE)  # все 4 паузы исчерпаны, 5 попыток


# --- dims (§2-bis) ---


def test_dims_truncates_and_renormalizes(monkeypatch: Any) -> None:
    embedder = _make_embedder(monkeypatch, dims=1024)
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=120: _FakeResponse(_payload(4096)),
    )
    out = embedder.embed(["a"])
    assert out.shape == (1, 1024)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert embedder.name.endswith("@1024")


def test_dims_none_keeps_full_vector_and_unsuffixed_name(monkeypatch: Any) -> None:
    embedder = _make_embedder(monkeypatch, model="google/gemini-embedding-001", dims=None)
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=120: _FakeResponse(_payload(3072)),
    )
    out = embedder.embed(["a"])
    assert out.shape == (1, 3072)
    assert embedder.name == "google/gemini-embedding-001"


def test_dims_never_sent_in_request_payload(monkeypatch: Any) -> None:
    """Живой факт 2026-07-18 (nemotron: «dimensions must be one of 2048»): провайдер
    с фиксированной размерностью ОТВЕРГАЕТ неподдержанное значение, а не игнорирует.
    Усечение — ТОЛЬКО клиентским срезом; payload без "dimensions" при любом dims."""
    for dims in (1024, None):
        embedder = _make_embedder(monkeypatch, dims=dims)
        captured: dict[str, Any] = {}

        def fake_urlopen(req: Any, timeout: int = 120) -> Any:
            captured["body"] = json.loads(req.data)
            return _FakeResponse(_payload(2048))

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        embedder.embed(["a"])
        assert "dimensions" not in captured["body"]


# --- ошибка в теле HTTP-200 (OpenRouter заворачивает провайдерские отказы) ---


def test_inband_error_400_raises_immediately_without_sleep(monkeypatch: Any) -> None:
    embedder = _make_embedder(monkeypatch, dims=1024)
    sleeps: list[float] = []
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=120: _FakeResponse(
            {"error": {"message": "dimensions must be one of 2048", "code": 400}}
        ),
    )
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="ошибка в теле 200"):
        embedder.embed(["a"])
    assert sleeps == []


def test_inband_error_429_is_retried_then_succeeds(monkeypatch: Any) -> None:
    embedder = _make_embedder(monkeypatch, dims=None)
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req: Any, timeout: int = 120) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse({"error": {"message": "rate limited", "code": 429}})
        return _FakeResponse(_payload(3))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = embedder.embed(["a"])
    assert out.shape == (1, 3)
    assert sleeps == [RETRY_SCHEDULE[0]]


def test_missing_api_key_raises(monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterEmbedder()


# --- пустой ввод: короткое замыкание без сетевого вызова ---


def test_embed_empty_list_returns_empty_array_without_request(monkeypatch: Any) -> None:
    embedder = _make_embedder(monkeypatch, dims=None)

    def tripwire(req: Any, timeout: int = 120) -> Any:
        raise AssertionError("urlopen не должен вызываться на пустом списке текстов")

    monkeypatch.setattr(urllib.request, "urlopen", tripwire)
    out = embedder.embed([])
    assert out.shape == (0, 0)


# --- get_embedder: диспетчеризация backend -> Embedder ---


def test_get_embedder_openrouter_returns_openrouter_embedder(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    embedder = get_embedder("openrouter")
    assert isinstance(embedder, OpenRouterEmbedder)


def test_get_embedder_bge_dispatches_to_onnx_bge_embedder(tmp_path: Path) -> None:
    """Дошли до OnnxBgeEmbedder.__init__ (не какой-то другой класс) — подтверждается тем,
    что заведомо несуществующий model_path даёт ИМЕННО её FileNotFoundError, без реальной
    модели bge-m3."""
    with pytest.raises(FileNotFoundError, match="модель не найдена"):
        get_embedder("bge", model_path=tmp_path / "nonexistent.onnx")


def test_get_embedder_unknown_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="неизвестный бэкенд эмбеддера"):
        get_embedder("nonexistent-backend")


# --- OnnxBgeEmbedder: guard на отсутствующий файл модели (без реальной модели bge-m3) ---


def test_onnx_bge_embedder_raises_when_model_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="модель не найдена"):
        OnnxBgeEmbedder(model_path=tmp_path / "nonexistent.onnx")

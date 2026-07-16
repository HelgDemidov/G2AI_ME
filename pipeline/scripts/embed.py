"""Модель-агностичный интерфейс эмбеддингов для семантического поиска.

Бэкенды:
  - OnnxBgeEmbedder — локальный bge-m3 int8 ONNX (CPU, приватно, бесплатно):
    CLS-pooling (last_hidden_state[:, 0]) + L2-нормализация -> 1024-мерный вектор.
  - OpenRouterEmbedder — эталон (gemini-embedding-001 и др.) через OpenRouter.

Векторы ВСЕГДА L2-нормализованы, поэтому косинус = скалярное произведение.
ВНИМАНИЕ: векторы разных моделей несравнимы — на корпусе живёт одна модель за раз.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from bge_tokenizer import EMBED_MAX_TOKENS, MODEL_DIR, TOKENIZER_JSON, load_tokenizer

FloatArray = NDArray[np.float32]

DEFAULT_ONNX = MODEL_DIR / "model_int8.onnx"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


def l2_normalize(mat: FloatArray) -> FloatArray:
    """L2-нормализация по строкам (нулевые строки остаются нулевыми)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (mat / norms).astype(np.float32)


class Embedder(Protocol):
    """Общий интерфейс: name (идентификатор модели), dim, embed(texts) -> (n, dim)."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> FloatArray: ...


class OnnxBgeEmbedder:
    """Локальный bge-m3 (int8 ONNX) через onnxruntime + tokenizers."""

    name = "bge-m3-onnx-int8"
    dim = 1024

    def __init__(
        self,
        model_path: Path = DEFAULT_ONNX,
        tokenizer_path: Path = TOKENIZER_JSON,
        max_tokens: int = EMBED_MAX_TOKENS,
        batch_size: int = 16,
    ) -> None:
        import onnxruntime as ort

        if not model_path.exists():
            raise FileNotFoundError(f"модель не найдена: {model_path} — скачать bge-m3?")
        self._session: Any = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._tok: Any = load_tokenizer(tokenizer_path)
        pad_id = self._tok.token_to_id("<pad>")
        self._tok.enable_truncation(max_length=max_tokens)
        self._tok.enable_padding(pad_id=pad_id if pad_id is not None else 1, pad_token="<pad>")
        self._batch = batch_size

    def embed(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[FloatArray] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            enc = self._tok.encode_batch(batch)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            (last_hidden,) = self._session.run(
                ["last_hidden_state"], {"input_ids": ids, "attention_mask": mask}
            )
            cls = np.asarray(last_hidden, dtype=np.float32)[:, 0, :]
            out.append(l2_normalize(cls))
        return np.vstack(out).astype(np.float32)


class OpenRouterEmbedder:
    """Эталонный эмбеддер через OpenRouter (OpenAI-совместимый /embeddings)."""

    def __init__(
        self,
        model: str = "google/gemini-embedding-001",
        api_key: str | None = None,
        batch_size: int = 32,
        url: str = OPENROUTER_URL,
    ) -> None:
        self.name = model
        self.dim = 0  # станет известно после первого ответа
        self._model = model
        self._batch = batch_size
        self._url = url
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("нет OPENROUTER_API_KEY (см. .env / .env.example)")
        self._key = key

    def _request(self, batch: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self._model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data: Any = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
        items = sorted(data["data"], key=lambda x: x["index"])
        return [list(item["embedding"]) for item in items]

    def embed(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            vecs.extend(self._request(texts[start : start + self._batch]))
        mat = np.asarray(vecs, dtype=np.float32)
        self.dim = int(mat.shape[1])
        return l2_normalize(mat)


def get_embedder(backend: str = "bge", **kwargs: Any) -> Embedder:
    """Фабрика: 'bge' -> локальный ONNX, 'openrouter' -> облачный эталон."""
    if backend == "bge":
        return OnnxBgeEmbedder(**kwargs)
    if backend == "openrouter":
        return OpenRouterEmbedder(**kwargs)
    raise ValueError(f"неизвестный бэкенд эмбеддера: {backend!r}")

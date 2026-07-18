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
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from index.bge_tokenizer import EMBED_MAX_TOKENS, MODEL_DIR, TOKENIZER_JSON, load_tokenizer

FloatArray = NDArray[np.float32]

DEFAULT_ONNX = MODEL_DIR / "model_int8.onnx"
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
INTRA_OP_THREADS = 4  # физический лимит машины: 2 ядра/4 потока (spec embed-local-swap §3) —
# лечит документированную оверсабскрипцию (onnxruntime иначе создаёт потоков больше ядер)
RETRY_SCHEDULE = (1.0, 4.0, 15.0, 60.0)  # паузы (сек) МЕЖДУ попытками; всего ≤5 попыток
# (spec embed-api-first §2) — 429/5xx/сетевые обрывы ретраятся, прочие 4xx неисправимы
DEFAULT_CLOUD_MODEL = "qwen/qwen3-embedding-8b"  # ПЛЕЙСХОЛДЕР до A/B-чекпоинта §1 спека
# embed-api-first (Fable 5 + пользователь) — коммит 4 подтверждает или меняет по решению


def l2_normalize(mat: FloatArray) -> FloatArray:
    """L2-нормализация по строкам (нулевые строки остаются нулевыми)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (mat / norms).astype(np.float32)


class Embedder(Protocol):
    """Общий интерфейс: name (идентификатор модели), dim, max_tokens (бюджет в
    СОБСТВЕННЫХ токенах модели; None — гейт неприменим/неизвестен, напр. облачный
    эмбеддер без фиксированного лимита), embed(texts, kind) -> (n, dim).

    ``kind`` — асимметрия документ/запрос: модели с промпт-префиксами (напр.
    EmbeddingGemma) кодируют запрос и документ по-разному и без префикса теряют
    качество (spec embed-local-swap §2); симметричные бэкенды параметр принимают,
    но игнорируют — сигнатура едина для ВСЕХ реализаций."""

    name: str
    dim: int
    max_tokens: int | None

    def embed(self, texts: list[str], *, kind: Literal["doc", "query"] = "doc") -> FloatArray: ...


class OnnxBgeEmbedder:
    """Локальный bge-m3 (int8 ONNX) через onnxruntime + tokenizers."""

    name = "bge-m3-onnx-int8"
    dim = 1024
    # тип аннотирован явно: Protocol.max_tokens инвариантен для изменяемых атрибутов
    # (mypy иначе выводит голый int и ругается на несовместимость с int | None)
    max_tokens: int | None = EMBED_MAX_TOKENS

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
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = INTRA_OP_THREADS  # лечит оверсабскрипцию (bge_tokenizer §11)
        self._session: Any = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._tok: Any = load_tokenizer(tokenizer_path)
        pad_id = self._tok.token_to_id("<pad>")
        self._tok.enable_truncation(max_length=max_tokens)
        self._tok.enable_padding(pad_id=pad_id if pad_id is not None else 1, pad_token="<pad>")
        self._batch = batch_size

    def embed(self, texts: list[str], *, kind: Literal["doc", "query"] = "doc") -> FloatArray:
        # bge-m3 симметрична — kind игнорируется, сигнатура едина с протоколом
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


class _InbandError(Exception):
    """Ошибка, пришедшая в ТЕЛЕ HTTP-200 ответа OpenRouter ({"error": {...}}) —
    транспортного HTTPError нет, ретраябельность решается по коду из тела той же
    логикой, что для HTTP-кодов: 429/5xx — временное, прочее — неисправимо."""

    def __init__(self, code: Any, body: str) -> None:
        super().__init__(body)
        self.body = body
        self.retryable = code == 429 or (isinstance(code, int) and code >= 500)


class OpenRouterEmbedder:
    """Production/эталонный эмбеддер через OpenRouter (OpenAI-совместимый /embeddings).

    Ретраи (spec embed-api-first §2): 429/5xx/сетевые обрывы (URLError/TimeoutError,
    напр. обрыв LTE) — до 5 попыток с паузами из ``RETRY_SCHEDULE``; прочие HTTP 4xx —
    немедленный отказ (неисправимо, ретраить бессмысленно).

    Размерность (§2-bis, MRL-усечение): ``dims`` срезает вектор ответа НА КЛИЕНТЕ и
    ре-нормализует — держит RAM векторного индекса в узде (нативные 3072-4096d моделей-
    кандидатов не влезают в бюджет 8ГБ-машины на масштабе целевого корпуса). Усечённые и
    полные векторы — РАЗНЫЕ неймспейсы (``name`` получает суффикс ``@<dims>``), иначе они
    бы смешались под одним ключом ``model`` в таблице ``vectors``.
    """

    def __init__(
        self,
        model: str = "google/gemini-embedding-001",
        api_key: str | None = None,
        batch_size: int = 32,
        url: str = OPENROUTER_URL,
        dims: int | None = 1024,
    ) -> None:
        self.name = f"{model}@{dims}" if dims is not None else model
        self.dim = 0  # станет известно после первого ответа
        self.max_tokens: int | None = None  # облачный лимит не фиксирован здесь — гейт неприменим
        self._model = model
        self._dims = dims
        self._batch = batch_size
        self._url = url
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("нет OPENROUTER_API_KEY (см. .env / .env.example)")
        self._key = key

    def _request(self, batch: list[str]) -> list[list[float]]:
        # "dimensions" в payload НЕ передаётся (отступление от spec §2-bis, живой факт
        # 2026-07-18): провайдер с фиксированной нативной размерностью не игнорирует
        # неподдержанное значение, а ОТВЕРГАЕТ запрос (nemotron: «dimensions must be
        # one of 2048»). Клиентский срез в embed() — единственный механизм усечения,
        # работает поверх любого провайдера.
        payload = json.dumps({"model": self._model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data: Any = json.loads(resp.read())
        if "data" not in data:
            # OpenRouter заворачивает провайдерские ошибки в HTTP 200 с {"error": …} —
            # транспортного HTTPError нет, класс отказа маппится на код ИЗ ТЕЛА
            err = data.get("error") or {}
            raise _InbandError(err.get("code"), json.dumps(data, ensure_ascii=False)[:500])
        items = sorted(data["data"], key=lambda x: x["index"])
        return [list(item["embedding"]) for item in items]

    def _request_with_retry(self, batch: list[str]) -> list[list[float]]:
        reason = ""
        total_attempts = len(RETRY_SCHEDULE) + 1
        for attempt in range(1, total_attempts + 1):
            try:
                return self._request(batch)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                if exc.code != 429 and exc.code < 500:
                    raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
                reason = f"HTTP {exc.code}: {body}"
            except _InbandError as exc:
                if not exc.retryable:
                    raise RuntimeError(f"OpenRouter (ошибка в теле 200): {exc.body}") from exc
                reason = f"ошибка в теле 200: {exc.body}"
            except (urllib.error.URLError, TimeoutError) as exc:
                reason = str(exc)
            if attempt == total_attempts:
                break
            delay = RETRY_SCHEDULE[attempt - 1]
            print(f"попытка {attempt}/{total_attempts} через {delay:.0f}s: {reason}", file=sys.stderr)
            time.sleep(delay)
        raise RuntimeError(
            f"OpenRouter: исчерпаны попытки ({total_attempts}) — {reason}. "
            "сеть/ключ? локальный фолбэк: --backend bge"
        )

    def embed(self, texts: list[str], *, kind: Literal["doc", "query"] = "doc") -> FloatArray:
        # OpenRouter-модели (текущий каталог) симметричны — kind игнорируется
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            vecs.extend(self._request_with_retry(texts[start : start + self._batch]))
        mat = np.asarray(vecs, dtype=np.float32)
        if self._dims is not None:
            mat = mat[:, : self._dims]
        self.dim = int(mat.shape[1])
        return l2_normalize(mat)


def get_embedder(backend: str = "bge", **kwargs: Any) -> Embedder:
    """Фабрика: 'bge' -> локальный ONNX, 'openrouter' -> облачный эталон."""
    if backend == "bge":
        return OnnxBgeEmbedder(**kwargs)
    if backend == "openrouter":
        return OpenRouterEmbedder(**kwargs)
    raise ValueError(f"неизвестный бэкенд эмбеддера: {backend!r}")

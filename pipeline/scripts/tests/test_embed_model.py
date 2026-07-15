"""Тесты локального bge-m3 ONNX-эмбеддера — требуют скачанную модель (@pytest.mark.model).

В CI модель отсутствует -> пропускаются (skipif) и отфильтровываются (-m 'not model').
"""
from __future__ import annotations

import numpy as np
import pytest

from bge_tokenizer import TOKENIZER_JSON
from embed import DEFAULT_ONNX, OnnxBgeEmbedder

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not (DEFAULT_ONNX.exists() and TOKENIZER_JSON.exists()),
        reason="модель bge-m3 не скачана (pipeline/models/bge-m3-onnx-int8/)",
    ),
]


@pytest.fixture(scope="module")
def embedder() -> OnnxBgeEmbedder:
    return OnnxBgeEmbedder()


def test_shape_and_normalized(embedder: OnnxBgeEmbedder) -> None:
    vecs = embedder.embed(["hello world", "another sentence here"])
    assert vecs.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)


def test_empty_input(embedder: OnnxBgeEmbedder) -> None:
    assert embedder.embed([]).shape == (0, 1024)


def test_deterministic(embedder: OnnxBgeEmbedder) -> None:
    a = embedder.embed(["same text"])
    b = embedder.embed(["same text"])
    assert np.allclose(a, b, atol=1e-4)


def test_semantic_sanity(embedder: OnnxBgeEmbedder) -> None:
    vecs = embedder.embed(
        [
            "AI governance and human oversight framework",
            "policy for artificial intelligence accountability",
            "chocolate cake baking recipe",
        ]
    )
    related = float(vecs[0] @ vecs[1])
    unrelated = float(vecs[0] @ vecs[2])
    assert related > unrelated

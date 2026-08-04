"""bge-m3 tokenizer loading (tokenizer.json) — shared by chunking and embeddings.

Needs a locally downloaded model directory plus the lightweight ``tokenizers`` library
(Rust backend, no torch). Model-dependent: absent in CI, so calling tests carry
``@pytest.mark.model``.

Model location is RESOLVED, not hardcoded (``model_dir``), in this order:

1. ``BGE_MODEL_DIR`` from the environment or ``.env`` — the artifacts live outside the
   repository and are shared between checkouts (559 MB, re-downloading them per clone is
   pure waste on a metered link).
2. ``pipeline/models/bge-m3-onnx-int8`` inside the repository — backward-compatible
   fallback, so a checkout that already holds the model keeps working with no config.

Resolution is LAZY on purpose. ``.env`` is read by CLI entry points, never at import
time, so a module-level constant would be computed before ``load_dotenv`` had a chance to
run and the variable would be silently ignored.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.env import load_dotenv

MODEL_DIR_ENV = "BGE_MODEL_DIR"
REPO_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "bge-m3-onnx-int8"

# Единый бюджет чанка/эмбеддинга корпуса. Модельный контекст bge-m3 — 8192
# токена, но наш бюджет — 512: внимание растёт квадратично, целый документ на
# 8192 на 2-ядерном CPU взорвался бы по памяти/времени (см. spec
# knowledge-graph-metadata §2). Единственный источник этого числа — chunking,
# corpus_index (CLI-дефолт) и embed.OnnxBgeEmbedder читают ЕГО, не хардкодят своё.
EMBED_MAX_TOKENS = 512


def model_dir() -> Path:
    """Directory holding the bge-m3 artifacts; resolution order is in the module docstring."""
    load_dotenv()
    raw = os.environ.get(MODEL_DIR_ENV, "").strip()
    return Path(raw).expanduser() if raw else REPO_MODEL_DIR


def tokenizer_json() -> Path:
    """Path to ``tokenizer.json`` inside the resolved model directory."""
    return model_dir() / "tokenizer.json"


def load_tokenizer(path: Path | None = None) -> Any:
    """Load the bge-m3 fast tokenizer from ``tokenizer.json`` (resolved when omitted)."""
    from tokenizers import Tokenizer

    target = path if path is not None else tokenizer_json()
    if not target.exists():
        raise FileNotFoundError(
            f"токенизатор bge-m3 не найден: {target} — скачать модель "
            f"(см. docs/pipeline/knowledge/tech_specs/knowledge-graph-metadata/spec.md) "
            f"или указать каталог в {MODEL_DIR_ENV}"
        )
    return Tokenizer.from_file(str(target))


def token_counter(path: Path | None = None) -> Callable[[str], int]:
    """Функция подсчёта токенов bge-m3 (для чанковки)."""
    tokenizer = load_tokenizer(path)

    def count(text: str) -> int:
        return len(tokenizer.encode(text).ids)

    return count

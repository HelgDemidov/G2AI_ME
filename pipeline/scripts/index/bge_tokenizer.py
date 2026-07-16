"""Загрузка токенизатора bge-m3 (tokenizer.json) — общий для чанковки и эмбеддингов.

Требует локально скачанную модель в pipeline/models/bge-m3-onnx-int8/ и лёгкую
библиотеку `tokenizers` (Rust-бэкенд, без torch). Модель-зависимо: в CI отсутствует,
поэтому вызывающие тесты помечаются @pytest.mark.model.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "bge-m3-onnx-int8"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"

# Единый бюджет чанка/эмбеддинга корпуса. Модельный контекст bge-m3 — 8192
# токена, но наш бюджет — 512: внимание растёт квадратично, целый документ на
# 8192 на 2-ядерном CPU взорвался бы по памяти/времени (см. spec
# knowledge-graph-metadata §2). Единственный источник этого числа — chunking,
# corpus_index (CLI-дефолт) и embed.OnnxBgeEmbedder читают ЕГО, не хардкодят своё.
EMBED_MAX_TOKENS = 512


def load_tokenizer(path: Path = TOKENIZER_JSON) -> Any:
    """Загрузить fast-токенизатор bge-m3 из tokenizer.json."""
    from tokenizers import Tokenizer

    if not path.exists():
        raise FileNotFoundError(
            f"токенизатор bge-m3 не найден: {path} — скачать модель "
            "(см. pipeline/setup/knowledge-graph-metadata/spec.md)"
        )
    return Tokenizer.from_file(str(path))


def token_counter(path: Path = TOKENIZER_JSON) -> Callable[[str], int]:
    """Функция подсчёта токенов bge-m3 (для чанковки)."""
    tokenizer = load_tokenizer(path)

    def count(text: str) -> int:
        return len(tokenizer.encode(text).ids)

    return count

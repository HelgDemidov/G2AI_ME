"""Клиент R2 — spec: бэклог §9 (инфраструктура, миграция не начата).

Тесты герметичны: ни одного сетевого вызова и ни одного чтения боевого `.env` —
конфиг всегда передаётся явным словарём, иначе исход зависел бы от того, что лежит
в `.env` у конкретного разработчика (и падал бы в CI, где его нет вовсе).
"""
from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from core import r2

_ENV = {
    "R2_S3_ENDPOINT": "https://acc.r2.cloudflarestorage.com",
    "R2_ACCESS_KEY_ID": "0" * 32,
    "R2_SECRET_ACCESS_KEY": "f" * 64,
}


def test_load_config_reads_all_three_variables() -> None:
    cfg = r2.load_config(env=_ENV)
    assert cfg.endpoint == _ENV["R2_S3_ENDPOINT"]
    assert cfg.access_key_id == _ENV["R2_ACCESS_KEY_ID"]
    assert cfg.secret_access_key == _ENV["R2_SECRET_ACCESS_KEY"]
    assert cfg.bucket == r2.DEFAULT_BUCKET


def test_bucket_is_overridable() -> None:
    assert r2.load_config("other-bucket", env=_ENV).bucket == "other-bucket"


@pytest.mark.parametrize("missing", sorted(_ENV))
def test_missing_variable_fails_loudly_and_names_it(missing: str) -> None:
    """Отказ до первого сетевого вызова, с именем недостающей переменной: иначе куратор
    увидел бы невнятную ошибку подписи вместо «не задан R2_ACCESS_KEY_ID»."""
    env = {k: v for k, v in _ENV.items() if k != missing}
    with pytest.raises(r2.R2ConfigError, match=missing):
        r2.load_config(env=env)


def test_empty_value_counts_as_missing() -> None:
    """Пустая строка в .env (`R2_ACCESS_KEY_ID=`) — не «значение», а забытая правка."""
    with pytest.raises(r2.R2ConfigError, match="R2_ACCESS_KEY_ID"):
        r2.load_config(env={**_ENV, "R2_ACCESS_KEY_ID": ""})


def test_default_path_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Боевой путь вызова — без аргумента ``env``. Тест остаётся детерминированным:
    переменные уже выставлены в процессе, а ``load_dotenv`` использует ``setdefault`` и
    поэтому не может их перебить содержимым чьего-то локального `.env`."""
    for name, value in _ENV.items():
        monkeypatch.setenv(name, value)
    assert r2.load_config().endpoint == _ENV["R2_S3_ENDPOINT"]


def test_client_is_built_without_network() -> None:
    """Создание клиента не должно ходить в сеть — иначе любой импорт-тест стал бы
    сетевым, а в CI доступа нет."""
    s3 = r2.client(r2.load_config(env=_ENV))
    assert s3.meta.endpoint_url == _ENV["R2_S3_ENDPOINT"]
    assert s3.meta.region_name == "auto"  # требование R2: реального региона нет


class _FakeS3:
    """Минимальный дубль: интересует только поведение вокруг кода ответа."""

    def __init__(self, status: int | None) -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.status is None:
            return {}
        raise ClientError({"ResponseMetadata": {"HTTPStatusCode": self.status}}, "HeadObject")


def test_object_exists_true_and_passes_bucket_and_key() -> None:
    fake = _FakeS3(None)
    cfg = r2.load_config(env=_ENV)
    assert r2.object_exists("sources/a/raw.pdf", cfg, s3=fake) is True
    assert fake.calls == [{"Bucket": r2.DEFAULT_BUCKET, "Key": "sources/a/raw.pdf"}]


def test_object_exists_false_on_404() -> None:
    cfg = r2.load_config(env=_ENV)
    assert r2.object_exists("нет-такого", cfg, s3=_FakeS3(404)) is False


@pytest.mark.parametrize("status", [403, 500])
def test_object_exists_reraises_non_404(status: int) -> None:
    """403 (неверный скоуп токена) и 5xx — НЕ «файла нет». Молчаливый False здесь означал
    бы «не смогли посмотреть» под видом «посмотрели и пусто»; для слоя, который однажды
    будет решать «нужно ли перезаливать оригинал», это худший вид ошибки."""
    cfg = r2.load_config(env=_ENV)
    with pytest.raises(ClientError):
        r2.object_exists("k", cfg, s3=_FakeS3(status))

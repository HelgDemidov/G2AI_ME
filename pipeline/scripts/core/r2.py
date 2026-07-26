"""Клиент Cloudflare R2 (S3-совместимое хранилище) — инфраструктура под миграцию §9.

⚠ **Пайплайн этим модулем пока НЕ пользуется, и это осознанно.** Оригиналы `raw.*`
продолжают жить локально; перенос хранения в облако (бэклог §9) не триггернут — корпус
18 МБ при бесплатных 10 ГБ. Здесь только фундамент, чтобы миграция начиналась с готового
и проверенного клиента, а не с настройки доступа.

Действующий канал бэкапа — `pipeline/tools/r2_backup.sh` (rclone). Он остаётся основным
для МАССОВОЙ заливки и после миграции: гонять гигабайты через Python-процесс на этой
машине незачем, у rclone это профильная задача с докачкой после обрыва. Этот модуль — для
пообъектных операций из кода (проверить наличие, положить/забрать один файл).

Конфигурация берётся из `.env` (те же переменные, что у скрипта бэкапа) — единственный
источник истины по секретам; ни своего конфига, ни аргументов командной строки, где
секрет был бы виден в `ps`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from core.env import load_dotenv

if TYPE_CHECKING:  # boto3 не поставляет типы; аннотация — только для читателя
    S3Client = Any

DEFAULT_BUCKET = "g2ai-corpus-raw"
"""Бакет offsite-бэкапа: EEUR, юрисдикция default, публичный доступ выключен."""

_ENV_VARS = ("R2_S3_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")

# Токен намеренно сужен до `Object Read & Write` на ОДИН бакет, поэтому операции уровня
# аккаунта (ListBuckets) отдают 403 — это признак правильного скоупа, а не поломки.
# Отсюда же: не вызывать head_bucket/create_bucket для «проверки доступности».
_RETRY_ATTEMPTS = 5


class R2ConfigError(RuntimeError):
    """Не заданы переменные доступа к R2 — отказ громкий, до первого сетевого вызова."""


@dataclass(frozen=True)
class R2Config:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str = DEFAULT_BUCKET


def load_config(bucket: str = DEFAULT_BUCKET, *, env: dict[str, str] | None = None) -> R2Config:
    """Собрать конфиг из окружения (`.env` подхватывается, если переменных ещё нет).

    ``env`` — только для тестов: явный словарь отключает чтение процесса и диска, поэтому
    исход теста не зависит от того, что лежит в `.env` у конкретного разработчика.
    """
    if env is None:
        load_dotenv()
        env = dict(os.environ)
    missing = [name for name in _ENV_VARS if not env.get(name)]
    if missing:
        raise R2ConfigError(
            f"не заданы {', '.join(missing)} — см. .env.example (раздел Cloudflare R2)"
        )
    return R2Config(
        endpoint=env["R2_S3_ENDPOINT"],
        access_key_id=env["R2_ACCESS_KEY_ID"],
        secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        bucket=bucket,
    )


def client(config: R2Config | None = None) -> Any:
    """Готовый S3-клиент к R2. Сетевых вызовов при создании не делает.

    ``region_name="auto"`` — требование R2: реального региона у бакета нет, а botocore
    обязан чем-то подписать запрос (SigV4 включает регион в область подписи).
    """
    cfg = config or load_config()
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": _RETRY_ATTEMPTS, "mode": "standard"},
        ),
    )


def object_exists(key: str, config: R2Config | None = None, *, s3: Any = None) -> bool:
    """Есть ли объект с таким ключом. ``404`` — это ответ «нет», а не ошибка.

    Прочие коды (403 при неверном скоупе, 5xx) пробрасываются: молча возвращать False на
    отказ доступа значило бы сообщать «файла нет» там, где мы просто не смогли посмотреть.
    """
    cfg = config or load_config()
    s3 = s3 or client(cfg)
    try:
        s3.head_object(Bucket=cfg.bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return False
        raise
    return True

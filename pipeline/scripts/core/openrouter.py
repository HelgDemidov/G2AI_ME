"""Общий chat/VLM-клиент OpenRouter (spec convert-cloud-tier §1): scan-OCR +
figures-VLM — два потребителя одного шва (chat/completions с image_url-вложением).

Броня — зеркало проверенной в ``index/embed.py`` (PR #16, spec embed-api-first §2),
но для chat/completions вместо /embeddings. ``RETRY_SCHEDULE``/``InbandError``
СОЗНАТЕЛЬНО скопированы, а не импортированы: ``core/`` не должен тянуть ``index/``
(направление слоёв). Консолидация — отдельный будущий спек при третьем потребителе
(design rationale convert-cloud-tier).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_SCHEDULE = (1.0, 4.0, 15.0, 60.0)  # копия index/embed.py — см. docstring модуля


class InbandError(Exception):
    """Ошибка, пришедшая в ТЕЛЕ HTTP-200 ответа OpenRouter (``{"error": {...}}``) —
    транспортного HTTPError нет, ретраябельность решается по коду из тела той же
    логикой, что для HTTP-кодов: 429/5xx — временное, прочее — неисправимо."""

    def __init__(self, code: Any, body: str) -> None:
        super().__init__(body)
        self.body = body
        self.retryable = code == 429 or (isinstance(code, int) and code >= 500)


def _request(payload: dict[str, Any], *, api_key: str, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_CHAT_URL,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body: Any = json.loads(resp.read())
    if "choices" not in body:
        # OpenRouter заворачивает провайдерские ошибки в HTTP 200 с {"error": …} —
        # транспортного HTTPError нет, класс отказа маппится на код ИЗ ТЕЛА (тот
        # же живой факт, что embed.py, PR #16).
        err = body.get("error") or {}
        raise InbandError(err.get("code"), json.dumps(body, ensure_ascii=False)[:500])
    return body  # type: ignore[no-any-return]


def chat_request(payload: dict[str, Any], *, api_key: str, timeout: float = 1800.0) -> dict[str, Any]:
    """POST + retry-лестница: 429/5xx/``URLError``/``TimeoutError``/``InbandError``
    (retryable) -> до 5 попыток; прочие 4xx (включая 413 PayloadTooLarge, §6
    спека) — немедленный ``RuntimeError`` с телом. Возврат — полный JSON
    (вызывающий сам берёт ``choices``/``usage``). Ключ ни в лог, ни в исключение
    не попадает. ``timeout=1800`` — урок пилота: не-стримовая генерация 10k+
    токенов на медленном провайдере превышает 900 с (см. Design rationale спека)."""
    reason = ""
    total_attempts = len(RETRY_SCHEDULE) + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return _request(payload, api_key=api_key, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code != 429 and exc.code < 500:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
            reason = f"HTTP {exc.code}: {body}"
        except InbandError as exc:
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
    raise RuntimeError(f"OpenRouter: исчерпаны попытки ({total_attempts}) — {reason}")

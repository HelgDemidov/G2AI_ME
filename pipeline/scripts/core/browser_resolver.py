"""Headless-browser resolver: Puppeteer-core (Node) драйвит движок Lightpanda
через CDP, отдаёт отрендеренный HTML одной страницы. Обходит WAF (F5/BigIP,
Akamai — НЕ Cloudflare, см. спек §3) и дорендеривает JS-SPA.

Спек: docs/pipeline/core/tech_specs/headless-browser-resolver/spec.md.

Субпроцесс-мост к Node-скрипту (``pipeline/browser/resolve.mjs``) — тот же
паттерн, что ``ocrmypdf``/``soffice`` в convert-слое: внешний бинарь/рантайм,
логика не переносится в Python. Требует Node >=20 + ``puppeteer-core``
(``pipeline/browser/node_modules``) + бинарь ``lightpanda`` (``pipeline/browser/``)
— все опциональны и gitignored (README в той же папке — команды установки);
``is_available()`` позволяет вызывающему коду деградировать без них
(см. ``acquire/acquisition.py`` — ступень лестницы пропускается, если недоступно).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

from core.env import REPO_ROOT

BROWSER_DIR = REPO_ROOT / "pipeline" / "browser"
LIGHTPANDA_BINARY = BROWSER_DIR / "lightpanda"
RESOLVE_SCRIPT = BROWSER_DIR / "resolve.mjs"

DEFAULT_WAIT_MS = 9000
DEFAULT_TIMEOUT_S = 45  # запас над wait_ms + goto-таймаут + запуск/остановка lightpanda (см. resolve.mjs)


class BrowserResolverUnavailable(RuntimeError):
    """Инструментальный отказ: Node/lightpanda не установлены, resolve.mjs упал,
    завис или вернул не-JSON. НЕ означает «страница вернула WAF-блок» — это
    другой, содержательный исход (``BrowserResult(ok=False, ...)``), который
    вызывающий код классифицирует как обычный отказ, не как недоступность инструмента.
    """


@dataclass
class BrowserResult:
    ok: bool
    html: str
    final_url: str
    error: str


def is_available() -> bool:
    """Дешёвая проверка окружения (без запуска процесса): Node в PATH и бинарь
    lightpanda на диске. ``node_modules``/``puppeteer-core`` отдельно не
    проверяется — его отсутствие проявится как явная ошибка при первом резолве,
    что достаточно диагностично и не стоит второй stat-проверки на каждый вызов.
    """
    return shutil.which("node") is not None and LIGHTPANDA_BINARY.exists()


def resolve(url: str, *, wait_ms: int = DEFAULT_WAIT_MS, timeout: int = DEFAULT_TIMEOUT_S) -> BrowserResult:
    """Отрендерить ``url`` движком Lightpanda через Puppeteer-core, вернуть HTML.

    Поднимает ``BrowserResolverUnavailable``, если Node/lightpanda не установлены
    (дешевле проверить заранее через ``is_available()``, чем ловить исключение на
    каждый URL батча) — либо если сам Node-процесс не ответил/упал/вернул
    невалидный JSON за отведённое время. Содержательный отказ страницы (WAF
    всё же заблокировал, таймаут навигации на стороне Lightpanda) приходит как
    обычное возвращаемое значение ``BrowserResult(ok=False, error=...)``, не исключением.
    """
    if not is_available():
        raise BrowserResolverUnavailable("Node и/или pipeline/browser/lightpanda не найдены")
    try:
        proc = subprocess.run(
            ["node", str(RESOLVE_SCRIPT), url, str(wait_ms)],
            cwd=BROWSER_DIR, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BrowserResolverUnavailable(f"resolve.mjs не ответил за {timeout}с") from exc
    if proc.returncode != 0:
        raise BrowserResolverUnavailable(
            f"resolve.mjs завершился с кодом {proc.returncode}: {proc.stderr[:300]}"
        )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise BrowserResolverUnavailable(f"resolve.mjs вернул невалидный JSON: {proc.stdout[:300]}") from exc
    if payload.get("ok"):
        return BrowserResult(ok=True, html=payload.get("html", ""), final_url=payload.get("url", url), error="")
    return BrowserResult(ok=False, html="", final_url=url, error=str(payload.get("error", "unknown")))

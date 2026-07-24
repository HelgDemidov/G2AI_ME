# pipeline/browser — headless-browser resolver (Lightpanda + Puppeteer)

Рабочая связка проекта для обхода WAF (F5/BigIP, Akamai) и рендеринга JS-страниц на
этапах discovery/acquire: **Puppeteer-core драйвит движок Lightpanda через CDP**.
Обоснование выбора и сравнение с альтернативами — `docs/pipeline/core/tech_specs/headless-browser-resolver/spec.md`.

## Что трекается / что нет
- `package.json` + `package-lock.json` — трекаются (манифест Node-зависимости).
- `node_modules/` — gitignored (ставится `npm ci`).
- `lightpanda` — бинарь ~146 МБ, gitignored (качается отдельно, как модели в `pipeline/models/`).

## Установка (свежий клон)
```bash
# 1. Node-драйвер (puppeteer-core, БЕЗ Chromium)
cd pipeline/browser && PUPPETEER_SKIP_DOWNLOAD=1 npm ci

# 2. Движок Lightpanda (nightly, один бинарь Linux x86_64)
curl -L -o lightpanda \
  https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-x86_64-linux
chmod +x lightpanda
```

## Использование
`lightpanda serve` поднимает CDP-сервер (по умолчанию `ws://127.0.0.1:9222/`), Puppeteer
подключается `puppeteer.connect({ browserWSEndpoint })`. Обязательный паттерн — создавать
страницу через `browser.createBrowserContext() → newPage()` (не `browser.pages()[0]`),
иначе навигация зависает. Требуется Node ≥ 20.

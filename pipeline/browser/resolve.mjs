#!/usr/bin/env node
// Резолвер: Puppeteer-core драйвит движок Lightpanda через CDP, отдаёт
// отрендеренный HTML одной страницы. Субпроцесс-мост для core/browser_resolver.py
// (прецедент ocrmypdf/soffice — внешний бинарь/рантайм, логика не переносится в Python).
//
// Использование: node resolve.mjs <url> [waitMs]
// Стдаут — ОДНА строка JSON: {"ok":true,"html":"...","url":"..."} | {"ok":false,"error":"..."}
//
// Самодостаточен: сам поднимает `lightpanda serve` на свободном порту и сам
// его гасит по завершении — вызывающему коду (Python) не нужно управлять
// жизненным циклом движка. Обязательный паттерн создания страницы —
// createBrowserContext()->newPage() (НЕ browser.pages()[0]/фантомный
// browser-таргет — иначе навигация зависает, см. спек headless-browser-resolver §4).
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import net from 'node:net';
import puppeteer from 'puppeteer-core';

const __dirname = dirname(fileURLToPath(import.meta.url));
const LIGHTPANDA_BIN = join(__dirname, 'lightpanda');

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
    srv.on('error', reject);
  });
}

function waitForCdp(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    (function tick() {
      fetch(`http://127.0.0.1:${port}/json/version`).then(() => resolve()).catch(() => {
        if (Date.now() > deadline) reject(new Error('lightpanda serve не поднялся вовремя'));
        else setTimeout(tick, 150);
      });
    })();
  });
}

async function main() {
  const url = process.argv[2];
  const waitMs = parseInt(process.argv[3] || '9000', 10);
  if (!url) {
    console.log(JSON.stringify({ ok: false, error: 'URL не передан' }));
    return;
  }

  const port = await freePort();
  const lp = spawn(LIGHTPANDA_BIN, ['serve', '--host', '127.0.0.1', '--port', String(port)], { stdio: 'ignore' });
  try {
    await waitForCdp(port, 8000);
    const browser = await puppeteer.connect({
      browserWSEndpoint: `ws://127.0.0.1:${port}/`,
      protocolTimeout: waitMs + 10000,
    });
    try {
      const ctx = await browser.createBrowserContext();
      const page = await ctx.newPage();
      // goto может легитимно не дождаться lifecycle-события на тяжёлом SPA —
      // это не отказ: контент всё равно читается ниже после явного ожидания.
      await page.goto(url, { timeout: waitMs + 5000 }).catch(() => {});
      await new Promise((r) => setTimeout(r, waitMs));
      const html = await page.content();
      const finalUrl = page.url();
      console.log(JSON.stringify({ ok: true, html, url: finalUrl }));
    } finally {
      await browser.disconnect();
    }
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: String((e && e.message) || e).slice(0, 300) }));
  } finally {
    lp.kill();
  }
}

main();

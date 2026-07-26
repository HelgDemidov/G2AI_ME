#!/usr/bin/env bash
# Offsite-бэкап корпуса в Cloudflare R2 (бэклог §9).
#
# Закрывает риск ЕДИНСТВЕННОЙ копии: в git из `sources/` версионируется только
# `meta.yaml`, а всё остальное существует в одном экземпляре и переживает потерю диска
# по-разному:
#   raw.*                     — оригиналы; часть источников передобыть тяжело или нельзя
#                               (напр. eu-ai-act-2024 за AWS WAF, бэклог §25)
#   .state.yaml               — sha256/провенанс; `original_sha256` восстановим ТОЛЬКО передобычей
#   .figures.yaml/.cloudocr.* — кэши облачного VLM: регенерация стоит реальных денег
#   candidates/*.yaml         — слой кандидатов: потеря = повторный прогон всех харвестов
#   .state/cursors.yaml       — курсоры коннекторов: потеря = коннекторы сканируют заново
# Поэтому копируется весь `sources/` целиком, а не только `raw.*`.
#
# Три неочевидных решения, каждое обосновано:
#
# 1. `copy`, а НЕ `sync`. У R2 нет версионирования объектов (проверено 2026-07-26:
#    ни S3-совместимости, ни собственного API), поэтому удалённая копия — единственная.
#    `sync` удаляет в облаке то, чего нет локально, то есть одной командой уничтожает
#    бэкап при повреждении локального каталога. `copy` не удаляет никогда.
# 2. Секреты — через переменные окружения, а не аргументами командной строки: аргументы
#    видны в `ps` любому процессу системы.
# 3. `no_check_bucket=true` обязателен: токен намеренно сужен до `Object Read & Write`
#    на один бакет, поэтому проверка уровня бакета возвращает 403. Это признак
#    правильного скоупа, а не ошибка (ListBuckets так же отдаёт 403 — так и задумано).
# 4. `no_head=true` — иначе КАЖДЫЙ файл даёт ложную ошибку (найдено живым прогоном
#    2026-07-26). Механика: R2 возвращает на успешный PUT заголовок `x-amz-version-id`,
#    rclone 1.60 честно использует его в проверочном `HEAD ?versionId=...`, а чтение по
#    версии R2 не реализует -> 501 Not Implemented -> rclone считает передачу
#    неудавшейся и ретраит транзакцию целиком. Файлы при этом заливались (со второй
#    попытки), то есть симптом — шум из ошибок при формально успешном результате.
#    Целостность от этого НЕ страдает: rclone шлёт `Content-MD5` в самом PUT, и R2
#    проверяет его на стороне сервера, так что битая загрузка отвергается независимо
#    от HEAD. Сверить копию можно в любой момент:
#      rclone check sources/ r2:g2ai-corpus-raw/sources/ --one-way
#
# ACL здесь СОЗНАТЕЛЬНО не задаётся: R2 не поддерживает ACL вообще, параметр был бы
# молчаливым no-op, создающим ложное впечатление настроенного доступа. Приватность
# бакета обеспечивается тем, что публичный доступ (r2.dev/custom domain) не включён.
#
# Конфиг rclone (`~/.config/rclone/rclone.conf`) НЕ создаётся: секрет остаётся ровно в
# одном месте — в `.env`.
set -euo pipefail

BUCKET="${R2_BUCKET:-g2ai-corpus-raw}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

command -v rclone >/dev/null || { echo "rclone не установлен: sudo apt install -y rclone" >&2; exit 1; }
[[ -f .env ]] || { echo "нет .env в корне репозитория ($REPO_ROOT)" >&2; exit 1; }

set -a; . ./.env; set +a
for var in R2_S3_ENDPOINT R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; do
    [[ -n "${!var:-}" ]] || { echo "в .env не задан $var (см. .env.example)" >&2; exit 1; }
done

# Пустой конфиг ЯВНО: иначе rclone на каждый запуск печатает NOTICE об отсутствии
# ~/.config/rclone/rclone.conf — файла, которого здесь не должно быть по построению.
export RCLONE_CONFIG=/dev/null \
       RCLONE_CONFIG_R2_TYPE=s3 \
       RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
       RCLONE_CONFIG_R2_REGION=auto \
       RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
       RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
       RCLONE_CONFIG_R2_ENDPOINT="$R2_S3_ENDPOINT" \
       RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true \
       RCLONE_CONFIG_R2_NO_HEAD=true

# Живой прогресс — только в интерактивном терминале; при перенаправлении в файл/пайп он
# превращается в километр перерисовок, поэтому там одна строка статистики.
if [[ -t 1 ]]; then PROGRESS=(--progress); else PROGRESS=(--stats-one-line); fi

# --transfers 2 вместо дефолтных 4 — рабочий канал LTE, полная параллельность его забивает.
# Все аргументы скрипта пробрасываются в rclone: `--dry-run`, `-v`, `--exclude` и т.п.
exec rclone copy sources/ "r2:${BUCKET}/sources/" --transfers 2 "${PROGRESS[@]}" "$@"

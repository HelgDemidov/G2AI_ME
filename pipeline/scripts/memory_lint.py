#!/usr/bin/env python3
"""Линт мёртвых якорей в CLAUDE.md и памяти ассистента (report-only, spec memory-lint).

Извлекает инлайновые `бэктик`-спаны (fenced-блоки вне скоупа v1) из CLAUDE.md и
memory/*.md, классифицирует каждый (путь / константа-со-значением / квалифицированный
идентификатор / голый идентификатор / прочее-пропуск) и сверяет с живым репозиторием.
Мёртвый якорь в строке с явным историческим маркером («удалён»/«упразднён»/…) —
уровень INFO (текст сам утверждает отсутствие, это не дефект); иначе WARNING.

НИКОГДА не гейт: exit 0 всегда, если не передан --fail-on-dead (задел под будущую
автоматизацию, нигде не включён по умолчанию). Смысловые классы C3-C5 (статусы,
нормы, история) — вне скоупа, ими занимается /memory-sync, не регекс.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.env import REPO_ROOT

DEFAULT_MEMORY_DIR = Path(
    "/home/fastcentrifuge/.claude/projects/"
    "-home-fastcentrifuge--------------Projects-Dev-G2AI-ME/memory"
)
DEFAULT_ALLOWLIST_REL = Path("pipeline/config/memory_lint_allow.yaml")

_CORPUS_GLOBS = (
    "pipeline/scripts/**/*.py",
    "pipeline/vocab/**/*.yaml",
    "pipeline/config/**/*.yaml",
    "pyproject.toml",
    ".github/workflows/**/*.yml",
)
# Расширения известны спеком как (.py/.yaml/.md/.toml/.db); ``.yml`` добавлен —
# .github/workflows сам корпус сверки состоит из .yml-файлов, не признавать их
# путём было бы внутренне противоречиво.
#
# ``.claude/**/*.md`` НЕ в текстовом корпусе (отклонение от буквального текста спека,
# найдено живьём при написании тестов): при --include-commands эти же файлы ОДНОВРЕМЕННО
# источник сканируемых якорей — bare/const/qualified-идентификатор в прозе команды
# тривиально "подтверждал бы сам себя" словопоиском по СВОЕМУ ЖЕ тексту (self-reference,
# ложноотрицательный результат). Существование путей ВНУТРИ .claude/ (напр. slash-команда
# → .claude/commands/<name>.md) проверяется отдельно через path_exists() — файловую
# систему, не текстовый корпус — там self-reference не возникает.
_PATH_EXTS = (".py", ".yaml", ".yml", ".md", ".toml", ".db")
_BARE_PATH_SEARCH_ROOTS = ("pipeline", "sources", ".claude", ".github", "docs")
_IGNORED_DIR_NAMES = {"__pycache__", ".git", "node_modules"}
# Точные путевые префиксы (не имена каталогов!) — pipeline/scripts/index/ и .../models/
# ЛЕГИТИМНЫЕ кодовые подпакеты, их нельзя отсечь по имени "index"/"models" наравне с
# бинарными pipeline/index/ (SQLite) и pipeline/models/ (веса bge-m3).
_IGNORED_PATH_PREFIXES = ("pipeline/index", "pipeline/models")

_ANCHOR_RE = re.compile(r"`([^`\n]+)`")
_CONST_SPAN_RE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*([:=])\s*(.+)$")
_QUALIFIED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:(?:\.[A-Za-z_][A-Za-z0-9_]*)|(?:::[A-Za-z_][A-Za-z0-9_]*))+$")
_BARE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HISTORICAL_RE = re.compile(
    r"упраздн|убран|удал[её]н|замен[её]н|прежн|бывш|deprecated"
    r"|переименован|легаси|legacy|не существ|устарел|перенес[её]н"
    # .claude/commands|skills — протокольный текст на английском (project convention)
    r"|removed|renamed|moved to|no longer|obsolete|superseded",
    re.IGNORECASE,
)
_BARE_EXTENSIONS = {f".{ext}" for ext in ("py", "yaml", "yml", "md", "toml", "db")}
_KNOWN_REPO_ROOTS = {
    "sources", "pipeline", "docs", "tests", ".github", ".claude", ".venv",
    "requirements.txt", "requirements-dev.txt", "pyproject.toml", "CLAUDE.md", "README.md",
}
_DOMAIN_LABEL_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,10}$", re.IGNORECASE)
# Известные базы, относительно которых CLAUDE.md/память пишут пути БЕЗ префикса
# ("convert/converters.py" вместо "pipeline/scripts/convert/converters.py",
# "general/pipeline_improvements.md" вместо "docs/pipeline/general/…") — перебираются
# все базы, не только корень репо.
_PATH_SEARCH_BASES = ("", "pipeline/scripts", "pipeline", "docs/pipeline")


@dataclass(frozen=True)
class Anchor:
    span: str
    kind: str  # "path" | "const" | "qualified" | "bare"
    check_value: str  # что ищем: span целиком (path/const) либо хвостовой сегмент (qualified/bare)


@dataclass(frozen=True)
class Finding:
    source_file: str
    span: str
    kind: str
    line_snippet: str
    level: str  # "WARNING" | "INFO"


_GIT_REF_RE = re.compile(r"^(?:feature|origin)/[\w./-]+$|^origin/main$")


def _looks_like_git_ref(span: str) -> bool:
    """Ссылка на git-ветку ("feature/discovery-manual", "origin/main") — не файловый
    путь: ветки не обязаны существовать как каталоги в рабочем дереве, а C3-семантика
    их жизненного цикла (создана/смерджена/удалена) — вне скоупа v1 (§ Вне скоупа спека)."""
    return bool(_GIT_REF_RE.match(span))


def _looks_like_slash_separated_list(span: str) -> bool:
    """"/" как разделитель перечисления в прозе, не путь: список CLI-флагов
    ("--force/--only"), список слов ("direct/live"), числовая дельта ("-12/+18")."""
    segments = span.split("/")
    if len(segments) < 2:
        return False
    if all(seg.startswith("--") for seg in segments):
        return True
    if all(re.match(r"^[+-]?\d+$", seg) for seg in segments):
        return True
    return False


_SLASH_COMMAND_RE = re.compile(r"^/([a-z][a-z0-9-]*)$")
# Известные не-командные однословные абсолютные пути ("/proc") — коллизия формы с
# "/tech-spec"-подобной ссылкой на кастомную команду неразличима одним regex'ом.
_UNIX_TOP_LEVEL_DIRS = {
    "proc", "sys", "dev", "tmp", "etc", "var", "bin", "usr", "home",
    "root", "opt", "mnt", "boot", "lib", "sbin", "srv", "run",
}


def slash_command_anchor(span: str) -> Anchor | None:
    """"/tech-spec" и т.п. — ссылка на кастомную команду, не файловый путь по себе;
    реальная проверяемая claim — существование `.claude/commands/<name>.md`."""
    m = _SLASH_COMMAND_RE.match(span)
    if not m or m.group(1) in _UNIX_TOP_LEVEL_DIRS:
        return None
    return Anchor(span, "path", f".claude/commands/{m.group(1)}.md")


def _strip_leading_slash_if_repo_relative(span: str) -> str | None:
    """Ведущий "/" в прозе этого проекта означает "от корня репо" ("/sources/"), НЕ
    OS-абсолютный путь — pathlib иначе трактует `Path() / "/x"` как сброс на абсолютный
    путь (баг, если не перехватить). Возвращает None, если первый сегмент не похож на
    известный корень репо (напр. `/proc` — реальный OS-путь, не наш, вне скоупа)."""
    if not span.startswith("/") or span == "/":
        return span
    rest = span[1:]
    first = rest.split("/")[0]
    if first in _KNOWN_REPO_ROOTS or first == ".claude" or first == ".github":
        return rest
    return None


def _looks_like_external_url(span: str) -> bool:
    """Внешний домен/URL ("agora.eto.tech", "publications.europa.eu/webapi/…") —
    не репозиторный якорь ни при какой классификации; веб-верификация reference_* —
    отдельная задача (вне скоупа §7 /memory-sync), сюда не относится."""
    if span.endswith(_PATH_EXTS):
        return False  # известное расширение сильнее домен-эвристики: .py=Парагвай,
        # .md=Молдова, .io/.ai — реальные ccTLD, совпадающие с нашими путями
    first = span.split("/")[0]
    if first in _KNOWN_REPO_ROOTS or first.startswith("."):
        return False
    return bool(_DOMAIN_LABEL_RE.match(first))


def classify(span: str) -> Anchor | None:
    """Классификация спана; None — пропуск (прочее/шум, класс 5 спека).

    Консервативно: при сомнении — пропуск, не проверка (precision-first, §3 спека).
    """
    span = span.strip()
    if not span or "\n" in span:
        return None
    if any(c in span for c in "{}<>"):
        return None  # шаблонный плейсхолдер ("{track}", "<блок>", "KEY=<...>") — не код,
        # проверяется ДО const-класса: "NAME=<...>" иначе матчил бы const-паттерн буквально

    m = _CONST_SPAN_RE.match(span)
    if m:
        # const-проверка ДО общего guard'а на пробелы: спек явно требует гибких
        # пробелов вокруг разделителя ("NAME = value"), не только "NAME=value".
        return Anchor(span, "const", span)
    if " " in span:
        return None
    if span.startswith("!"):
        return None  # gitignore-негация ("!/.claude/skills/"), не путь
    if span.startswith("~"):
        return None  # home-directory путь (машино-специфичный, вне репо) — не проверяем
    if span in _BARE_EXTENSIONS:
        return None  # голое упоминание расширения ("`.md`"), не конкретный файл
    if _looks_like_external_url(span):
        return None  # внешний домен/URL, не репозиторный якорь
    if _looks_like_git_ref(span):
        return None  # ссылка на git-ветку, не файловый путь
    if _looks_like_slash_separated_list(span):
        return None  # "/" как разделитель перечисления в прозе, не путь
    if re.match(r"^10\.\d{4,}/", span):
        return None  # DOI ("10.5281/zenodo…"), не путь

    slash_cmd = slash_command_anchor(span)
    if slash_cmd is not None:
        return slash_cmd

    if "::" in span and "/" in span:
        # "analyze/retrieve.py::retrieve()" — путь-префикс к модулю + вызов функции;
        # содержательная проверяемая claim — существование ИМЕНИ после "::", не файла.
        tail = span.split("::")[-1].split("(")[0]
        if len(tail) >= 2 and _BARE_RE.match(tail):
            return Anchor(span, "qualified", tail)
        return None

    if "/" in span or span.endswith(_PATH_EXTS):
        if span.startswith("/"):
            stripped = _strip_leading_slash_if_repo_relative(span)
            if stripped is None:
                return None  # похоже на OS-абсолютный путь вне репо ("/proc") — не наш
            return Anchor(span, "path", stripped)
        return Anchor(span, "path", span)

    if _QUALIFIED_RE.match(span) and not span[0].isdigit():
        tail = re.split(r"\.|::", span)[-1]
        if len(tail) >= 2:
            return Anchor(span, "qualified", tail)
        return None

    if _BARE_RE.match(span):
        segments = [s for s in span.split("_") if s]
        is_snake_multi = "_" in span and len(segments) >= 2
        # ВСЕ-ЗАГЛАВНЫЕ слова ("UNRESOLVED") — акцент в прозе, не CamelCase-идентификатор;
        # настоящий CamelCase требует СМЕШАННОГО регистра (строчные буквы тоже есть).
        is_camel = (
            span[0].isupper()
            and any(c.isupper() for c in span[1:])
            and any(c.islower() for c in span)
        )
        if is_snake_multi or is_camel:
            return Anchor(span, "bare", span)

    return None


def extract_anchors(text: str) -> Iterator[tuple[str, str]]:
    """(span, содержащая строка) для каждого инлайнового `бэктик`-спана.

    Fenced-блоки (```...```) целиком пропускаются построчным тоггл-состоянием —
    вне скоупа v1 (§2 спека: парс дерева репозитория внутри них хрупок).
    """
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in _ANCHOR_RE.finditer(line):
            yield m.group(1), line


def is_historical_mention(line: str) -> bool:
    return bool(_HISTORICAL_RE.search(line))


def load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("allow") or []
    return {str(e["span"]) for e in entries if isinstance(e, dict) and "span" in e}


def _iter_corpus_files(repo_root: Path, roots: Iterable[str]) -> Iterator[Path]:
    for root_name in roots:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(repo_root).as_posix()
            if any(rel.startswith(f"{prefix}/") for prefix in _IGNORED_PATH_PREFIXES):
                continue
            if any(part in _IGNORED_DIR_NAMES for part in p.relative_to(root).parts[:-1]):
                continue
            yield p


def build_corpus_text(repo_root: Path) -> str:
    """Текст файлов корпуса сверки (const/qualified/bare ищутся в нём словограницей)."""
    chunks: list[str] = []
    for pattern in _CORPUS_GLOBS:
        for p in sorted(repo_root.glob(pattern)):
            if p.is_file():
                chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def path_exists(span: str, repo_root: Path) -> bool:
    """Существование path-якоря. Со слэшем — точный путь, перебирая известные базы
    (корень репо И `pipeline/scripts/` — CLAUDE.md часто пишет пакет-относительный
    путь без префикса, напр. "convert/converters.py"); trailing-slash ("commands/")
    трактуется как ссылка на директорию целиком. Без слэша (голое имя/глоб `raw.*`)
    — поиск ГДЕ УГОДНО под известными корнями репо (спек: существование в ФС от
    корня репо, шире узкого корпуса rg-поиска других классов)."""
    if span.endswith("/"):
        span = span[:-1]
        for base in _PATH_SEARCH_BASES:
            if (repo_root / base / span).is_dir():
                return True
        return False

    if "/" in span:
        parts = span.split("/")
        pattern = parts[-1]
        dir_parts = parts[:-1]
        for base in _PATH_SEARCH_BASES:
            dir_path = repo_root.joinpath(base, *dir_parts) if base else repo_root.joinpath(*dir_parts)
            if not dir_path.is_dir():
                continue
            if "*" in pattern or "?" in pattern:
                if any(dir_path.glob(pattern)):
                    return True
            elif (dir_path / pattern).exists():
                return True
        return False

    if any(fnmatch.fnmatch(p.name, span) for p in repo_root.iterdir() if p.is_file()):
        return True  # файлы прямо в корне репо (CLAUDE.md, README.md, pyproject.toml…)
    for prefix in _IGNORED_PATH_PREFIXES:
        # неглубокая (не rglob) проверка внутри исключённых деревьев — pipeline/index/
        # содержит буквально один файл (corpus.db), pipeline/models/ — считанные веса;
        # легитимные точечные упоминания резолвятся без риска false-positive от
        # случайного совпадения имени где-то в большом бинарном дереве.
        d = repo_root / prefix
        if d.is_dir() and any(fnmatch.fnmatch(p.name, span) for p in d.iterdir() if p.is_file()):
            return True
    for p in _iter_corpus_files(repo_root, _BARE_PATH_SEARCH_ROOTS):
        if fnmatch.fnmatch(p.name, span):
            return True
    return False


def const_pattern(span: str) -> re.Pattern[str] | None:
    m = _CONST_SPAN_RE.match(span)
    if not m:
        return None
    name, _op, value = m.groups()
    return re.compile(rf"{re.escape(name)}\s*[:=]\s*{re.escape(value)}")


def word_boundary_pattern(word: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(word)}\b")


def anchor_found(anchor: Anchor, repo_root: Path, corpus_text: str) -> bool:
    if anchor.kind == "path":
        return path_exists(anchor.check_value, repo_root)
    if anchor.kind == "const":
        pattern = const_pattern(anchor.span)
        return bool(pattern and pattern.search(corpus_text))
    return bool(word_boundary_pattern(anchor.check_value).search(corpus_text))


def scan_file(path: Path, repo_root: Path, corpus_text: str, allowlist: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8")
    for span, line in extract_anchors(text):
        if span in allowlist:
            continue
        anchor = classify(span)
        if anchor is None:
            continue
        if anchor_found(anchor, repo_root, corpus_text):
            continue
        level = "INFO" if is_historical_mention(line) else "WARNING"
        findings.append(Finding(str(path), span, anchor.kind, line.strip()[:160], level))
    return findings


def collect_sources(repo_root: Path, memory_dir: Path, include_commands: bool) -> list[Path]:
    sources = [repo_root / "CLAUDE.md"]
    if memory_dir.is_dir():
        sources += sorted(memory_dir.glob("*.md"))
    if include_commands:
        commands_dir = repo_root / ".claude" / "commands"
        if commands_dir.is_dir():
            sources += sorted(commands_dir.glob("*.md"))
        skills_dir = repo_root / ".claude" / "skills"
        if skills_dir.is_dir():
            sources += sorted(skills_dir.glob("**/SKILL.md"))
    return [s for s in sources if s.exists()]


def run(repo_root: Path, memory_dir: Path, include_commands: bool, allowlist_path: Path) -> list[Finding]:
    allowlist = load_allowlist(allowlist_path)
    corpus_text = build_corpus_text(repo_root)
    sources = collect_sources(repo_root, memory_dir, include_commands)
    findings: list[Finding] = []
    for src in sources:
        findings += scan_file(src, repo_root, corpus_text, allowlist)
    return findings


def format_report(findings: list[Finding]) -> str:
    by_file: dict[str, list[Finding]] = {}
    for f in findings:
        by_file.setdefault(f.source_file, []).append(f)

    lines: list[str] = []
    for src, items in by_file.items():
        lines.append(f"\n{src}:")
        for f in items:
            lines.append(f"  [{f.level}] `{f.span}` ({f.kind}) — NOT FOUND")
            lines.append(f"    | {f.line_snippet}")

    warnings = sum(1 for f in findings if f.level == "WARNING")
    infos = sum(1 for f in findings if f.level == "INFO")
    lines.append(f"\nИтого: {warnings} WARNING, {infos} INFO")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Линт мёртвых якорей CLAUDE.md/памяти (report-only)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument("--include-commands", action="store_true")
    parser.add_argument("--fail-on-dead", action="store_true")
    args = parser.parse_args(argv)

    allowlist_path = args.allowlist or (args.repo_root / DEFAULT_ALLOWLIST_REL)
    findings = run(args.repo_root, args.memory_dir, args.include_commands, allowlist_path)
    print(format_report(findings))

    if args.fail_on_dead and any(f.level == "WARNING" for f in findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

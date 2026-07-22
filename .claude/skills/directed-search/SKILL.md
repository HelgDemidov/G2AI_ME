---
name: directed-search
description: Run a reproducible directed-search campaign that feeds source candidates into the discovery layer via discover.py inject.
---

Run a reproducible directed-search campaign: web-search sessions that feed source candidates into the discovery layer via `discover.py inject` (G2AI_ME).

This is a PROTOCOL, not a code connector: the "heavy" part (query design, snippet judgement, publisher verification) is this agent session; provenance lands in the existing `CandidateRecord` fields (`connector_id: search:<campaign>` — the id grammar IS the channel archetype — plus `matched_query` = the query as executed). Example invocation (Claude Code): `/directed-search <campaign-slug> [axis] [jurisdictions...]` — invocation syntax depends on your harness; see `README.md` next to this file for per-harness instructions.

## Runtime requirements

This protocol needs an agent with:
1. A web search capability.
2. Shell access at the repo root (venv binaries are invoked as `.venv/bin/...`).
3. Read access to repo files (the vocab/config files referenced below).

If your agent did not auto-discover this skill, point it at this file and instruct it to follow the protocol.

## Inputs (resolve before the first query)

1. **Campaign slug** — kebab-case, stable for the whole campaign (it becomes `connector_id: search:<slug>` on every candidate; reuse the SAME slug when resuming a campaign across sessions).
2. **Axis** — pick one from `pipeline/vocab/vocab_axes.yaml` (currently: `agentic_g2ai` — narrow, agentic services/MCP-like protocols/agent governance; `digital_sovereignty` — broad, national AI/digital policy capacity; see the file for the canonical descriptions and any axes added since). Determines query vocabulary, NOT a triage verdict.
3. **Jurisdiction map** — explicit list for this campaign. Small states are the charter's priority, not an exclusivity rule. Use ISO 3166-1 alpha-2 codes for `--jurisdiction`.
4. **Vocabulary** — read `pipeline/vocab/vocab_topics.yaml` and `pipeline/vocab/vocab_g2ai_patterns.yaml`; query templates are built from these terms × jurisdictions. Read `frontier_year` in `pipeline/config/triage.yaml` — freshness is a SIGNAL (prefer recent; older foundational documents are still injectable), not a gate.

## Query discipline

- Build a template grid up front (term × jurisdiction × language), then walk it. Ad-hoc improvisation is allowed ON TOP of the grid (following a lead), never INSTEAD of it — coverage must be auditable.
- Query in English AND in the jurisdiction's own language(s) — national primary sources are routinely invisible to English-only queries (live precedents: Estonian `et`, Montenegrin `cnr`).
- **Record every query verbatim** — including the ones that found nothing (a summary line each). The query that found a candidate goes into `--query` exactly as executed.
- Per-jurisdiction stop rule: after 2 consecutive grid queries with zero new leads, move to the next jurisdiction; note the early exit in the session summary.

## What counts as a hit

A PRIMARY document: strategy / law / regulation / framework / guidance / official report issued by a government, IGO, standards body, or (for the `research-papers` track) a named research publisher. News/blog coverage is a LEAD, not a candidate — follow it to the primary source and inject THAT. If the primary source cannot be located, it is not a hit; log the lead in the session summary instead.

## Verification before inject (metadata only)

The seed-list lesson (charter §8) is the reason this section exists: aggregator claims and secondary reporting routinely attribute titles that do not exist (live precedents: fictional UAE and Estonia entries). Before every inject:

1. **Publisher** — the URL must be the publisher's own domain (or its official CDN/uploads path). An aggregator/rehost link is not injectable; find the official page.
2. **Title** — as the publisher states it, not as a news article paraphrases it.
3. **Date** — from the publisher's page or official announcement. If only secondary reporting dates the document, use the best-supported date and say so in `--summary`.
4. Fetching METADATA pages (landing pages, press releases) is allowed. **Never fetch the document body**  — it passes through a small model and is not verbatim (CLAUDE.md rule); body acquisition belongs to `pipeline/scripts/run_pipeline.py`, after triage.
5. Capture `--rights`/`--sensitivity` best-effort if the page states them; otherwise omit (triage finalizes).

## Inject (one command per candidate)

```
.venv/bin/python pipeline/scripts/discover.py inject \
  --url "<official URL>" --title "<verbatim title>" --issuer "<publisher>" \
  --language <ll> --kind directed_search --campaign <slug> --query "<query as executed>" \
  [--jurisdiction <cc>] [--date YYYY-MM-DD] [--summary "<1-2 sentences EN>"]
```

- `--language`: ISO 639-1; 639-3 only where no 639-1 code exists (Montenegrin `cnr`).
- `--summary`: 2-3 sentences EN, hard cap 600 characters (`schema.CANDIDATE_SUMMARY_MAX` — check if it changed; schema rejects longer — shorten, don't fight it).
- Re-injecting a known URL is a safe no-op (dedup; rejected candidates do not resurrect) — do not pre-filter against `sources/candidates.yaml` manually, just inject.
- A `кандидат уже присутствует (уже отклонён ранее: …)` response is a FINDING: record it in the summary, do not argue with it.

## Explicit prohibitions

- Do NOT download or fetch document bodies (acquisition is `pipeline/scripts/run_pipeline.py`, post-triage).
- Do NOT assign `target_fit`/axis verdicts or edit `relevance` — that is triage (`worksheet`/`apply`), a separate session.
- Do NOT create candidates by editing `sources/candidates.yaml` directly or via ad-hoc scripts — `inject` is the only entry (provenance + dedup by construction).
- Do NOT copy seed-list titles into injects without the full verification above — such leads (the historical `small_states_v01.md`, removed by the curator 2026-07-22 after being fully mined) are the lowest tier.

## Session summary (end of every campaign session)

Report to chat: queries executed (count + the verbatim list), hits found, injected (new), dedup no-ops, rejected-earlier collisions, leads without a locatable primary source, jurisdictions closed early by the stop rule. Point the user to the next step: `discover.py worksheet` → batch triage.

# Repository Guidelines

## Purpose
This repo contains a hackathon project for **Statement Two: Multi-Agent Collaboration**. The product concept (BabyHandoff) is a “village of agents” that turns quick parent notes into (1) a handoff summary and (2) a rubric-scored decision between competing priorities (sleep vs errands vs admin), backed by MongoDB Atlas for durable context and auditability.

## Project Structure
- `AGENTS.md`: contributor guidelines + the project PRD (keep this as the canonical PRD).
- `docs/`: supporting docs (optional).
- `server/`: Python API + web UI (FastAPI recommended).
- `server/templates/`: server-rendered pages (minimal JS; HTMX-friendly).
- `server/static/`: static assets (CSS, icons).
- `tests/`: unit/integration tests.

## Build, Test, and Development Commands
Recommended local workflow (adjust if tooling changes):
- Create env: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt` (or `pip install -e .` if using `pyproject.toml`)
- Run dev server: `PYTHONPYCACHEPREFIX=.pycache uvicorn server.main:app --reload`
- Run tests: `pytest`

## Coding Style & Naming
- Python: 4-space indentation, type hints on public functions, prefer small pure functions.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep “agent” behavior modular (one file/module per agent role) and keep MongoDB access behind a thin repository layer.

## Testing Guidelines
- Use `pytest`; name tests `tests/test_*.py`.
- Prefer deterministic tests (seeded “sample shift” fixtures) over network-dependent tests.

## Commit & Pull Requests
- No established history yet; use Conventional Commits (e.g., `feat: add decision card`, `fix: handle empty shift`).
- PRs: include a short description, demo steps (commands + sample input), and a screenshot/GIF of the UI change when relevant.

## Config & Security
- Store secrets in `.env` (never commit). Provide `.env.example` with placeholders like `MONGODB_URI`.
- Avoid medical recommendations; keep content scoped to planning/checklists and include a short “not medical advice” disclaimer in the UI.

---

# PRD: BabyHandoff (“It takes a village.”)

## Summary
New parents face constant **competing priorities** (sleep, errands, admin) under **fatigue**, leading to decision paralysis and dropped tasks. BabyHandoff is a minimalist web app that turns quick notes into a rubric-scored **Decision Card** (three competing plans + auto-selected winner) plus a clean handoff export, powered by a “village” of specialized agents with MongoDB Atlas as the context and audit engine.

## Problem Statement
When you’re exhausted, you don’t need more advice—you need a fast, low-friction way to:
- capture messy reality in seconds
- decide what to prioritize next (without re-planning from scratch)
- avoid missed deadlines and “we ran out of X” surprises
- hand off context to “future you” (and optionally to a partner/helper)

## Goals (Punchlines)
- **It takes a village** to raise a child → a village of agents helps plan.
- **Competing priorities** under fatigue → three scenario planners debate and a referee scores.
- **Stop re-planning from scratch** → every run is versioned, replayable, and auditable in MongoDB.

## Target User
Primary: one solo operator (one caregiver juggling work/school + home). Sharing is optional via export.

## MVP User Flow
1. User starts/opens a shift and adds messy notes (text; optional voice-to-text fills the same input).
2. User clicks **Generate Plan (next 12h)**.
3. Agents propose 3 plans: **Sleep-first**, **Errands-first**, **Admin-first**.
4. Referee scores plans against a rubric; Coordinator auto-selects and materializes tasks.
5. UI shows: selected plan + rationale with evidence links; other plans are viewable.
6. User exports a Markdown handoff for Obsidian (`.md`) and/or shares it.
7. Optional: user changes one “what-if” dropdown (Energy) → re-run → see v1→v2 diff.

## Outputs (Artifacts)
- **Decision Card (hero):** 3 proposals + rubric scores + chosen plan + evidence-linked rationale.
- **Next 12h Plan:** time blocks (simple schedule), Top 5 priorities, admin checklist, running-low list.
- **24h Outlook (lightweight):** upcoming deadlines + “prep now to reduce tomorrow’s chaos” (not a full plan).
- **Handoff Markdown:** a clean `.md` export suitable for Obsidian (and rendered in-app from the same source).

## Proposed Tech Stack (MVP)
- Backend: Python `FastAPI` + `uvicorn`
- UI: server-rendered HTML (`Jinja2`) + `HTMX` for interactions; `Tailwind` (CDN) for a sleek minimal UI
- Realtime: `HTMX` polling for Risk Radar; watcher triggers via MongoDB change streams (with fallback mode)
- Database: MongoDB Atlas (Motor async driver) as durable context + audit log
- Models: deterministic heuristic council by default; optional OpenAI proposals via `COUNCIL_MODE=openai` + `OPENAI_MODEL=gpt-5.2`
- Voice-to-text: browser Web Speech API (fills the same text input; graceful fallback to typing)

## Core Screens (Desktop-first)
- **Shift Capture**: one input box + timeline of notes; `Load sample shift` button for demo.
- **Run / Council**: proposals, rubric scores, event stream, and selected plan.
- **Handoff Artifact**: in-app rendered view sourced from Markdown + download `.md`; version dropdown and diff view.

## Inputs (Low-Friction Tagging)
Single text box + optional “chips” that tag the entry (default: `Note`):
- `Note`: freeform context
- `Task`: an explicit action item
- `Deadline`: time-sensitive admin item
- `Running low`: inventory/supplies risk
Chips should be optional; agents still normalize untagged notes.

## Decision Card (Hero Artifact)
Shows:
- Selected plan (winner) + confidence score.
- Three proposals (A/B/C) with short plan blocks.
- Rubric table (scores + short justifications).
- Evidence-linked rationale (“because of” links to specific notes/logs).

### Plan Archetypes
- **Sleep-first**: protect rest windows, reduce workload, add buffers, defer non-urgent errands.
- **Errands-first**: batch outside tasks, optimize by time windows and dependencies, minimize trips.
- **Admin-first**: handle deadlines/forms/scheduling/school/work logistics; keep other tasks minimal.

### Rubric (Default)
Top 5 scoring dimensions (each 1–5):
1. Fixed commitments fit (appointments, work/school blocks)
2. Deadline risk (must-do today / overdue)
3. Energy match (user-reported or inferred)
4. Time-window feasibility (available blocks, travel assumptions)
5. Inventory risk (running-low items that can break the next shift)

## What-If Replan (Minimal)
Goal: demonstrate that “best plan” changes when constraints change, without introducing a complex settings UI.
- Dropdown: `Energy` = `High | Medium | Low`
- Action: `Re-run council`
- Result: persists a new run + decision (`v2`) and shows a compact diff vs `v1`
Notes:
- Default to inferred energy from notes when possible; the dropdown is an override.
- Keep the demo scenario where `Energy: Medium → Low` flips the winner or meaningfully changes the schedule blocks.

## Multi-Agent Collaboration Design
Agents register with skills and communicate via task-scoped context packs (token-budgeted). In the MVP, each agent is a specialized role with deterministic heuristics and/or an optional shared LLM provider (`COUNCIL_MODE=openai`).

### LLM Provider (MVP)
- Default: `COUNCIL_MODE=heuristic` (no external LLM calls; fast + deterministic for demos)
- Optional: `COUNCIL_MODE=openai` uses OpenAI Responses API for the three planner proposals (Sleep/Errands/Admin) with `AGENT_MAX_RETRIES` + `AGENT_TIMEOUT_SECONDS`, and falls back to heuristic proposals on failure

### Orchestration (How the Coordinator Invokes Agents)
For each council run, the Coordinator executes a pipeline with **parallel fan-out**:
1. **Build + persist a context pack** (summary + trimmed log refs; token budget via `CONTEXT_PACK_TOKEN_BUDGET`)
2. **Parallel proposals**: `SleepPlanner`, `ErrandsPlanner`, `AdminPlanner` run concurrently from the same context pack
3. **Referee**: deterministic rubric scoring selects a winner
4. **HandoffWriter**: writes the handoff Markdown for the selected plan
5. **Persist + version**: store proposals/decision/artifacts, and append an auditable council transcript derived from `messages`

Implementation note: runs execute asynchronously; the run page auto-refreshes while `run.status=running` so judges can watch `tasks` + `events` update live from MongoDB.

### Collaboration Visibility (MVP)
The MVP makes collaboration “judge-visible” via MongoDB-backed audit trails:
- `tasks` show per-agent status, attempts, and outputs (including OpenAI fallback signals)
- `messages` store the coordinator context pack + each agent’s output
- `context_packs` persist token-budgeted inputs shared across agents

### Task + Message Format (MongoDB Auditability)
We persist both the work graph and the “debate” so judges can replay the run:
- `tasks` (one doc per agent invocation):
  - `shift_id`, `run_id`
  - `name` (e.g., `plan_sleep_first`, `score_rubric`, `write_handoff`)
  - `agent_name` (e.g., `SleepPlanner`)
  - `status`: `queued | running | completed | failed`
  - `attempt`: integer, `max_attempts`: integer (computed as `1 + AGENT_MAX_RETRIES`)
  - `started_at`, `finished_at`, `updated_at`
  - `inputs`: `{ context_pack_id, proposal_ids, decision_id }` (as applicable)
  - `outputs`: `{ proposal_id, decision_id, artifact_id, llm_used, attempts, fallback, error }` (as applicable)
  - `error`: `{ type, message }` (optional)
- `messages` (append-only “chat log”):
  - `shift_id`, `run_id`, `task_id`
  - `sender`: `Coordinator | <AgentName>`
  - `role`: `system | developer | user | assistant` (LLM-compatible)
  - `content`: string
  - `meta`: dict (optional; e.g., `{ context_pack_id, model, attempts }`)
  - `created_at`

In addition, every run writes a **Council Transcript Markdown** artifact suitable for Obsidian export, sourced from `messages` (not from ad-hoc strings).

- `context_packs` (token-budgeted shared inputs):
  - `shift_id`, `run_id`, `token_budget`
  - `summary` + `compiled_text` + `log_refs[]`
  - `created_at`

### LLM Provider (MVP Implementation)
- Default: `COUNCIL_MODE=heuristic` (no external calls)
- Optional: `COUNCIL_MODE=openai` uses OpenAI for the three planner proposals (Sleep/Errands/Admin) and falls back to heuristic proposals if the API call fails
- Retries + timeout are controlled by `AGENT_MAX_RETRIES` and `AGENT_TIMEOUT_SECONDS` and are persisted in `tasks.outputs` for audit

Stretch goal: swap in additional providers behind a unified interface without changing the MongoDB persistence model (`tasks` + `messages` + `artifacts`).

### Token Budget (Context Packs)
Default context pack budget: **~4k tokens** (configurable).
MVP trimming strategy:
- Persist one `context_pack` per run with `summary` + `compiled_text` + `log_refs[]`
- Include the most recent ~40 logs (keeps prompts small and demos reliable)

Upgrade path (post-MVP):
- Prioritize `Deadline` + `Running low` over generic notes when trimming
- Keep stable log IDs in `log_refs` so evidence links remain valid

### Rubric Scoring (Deterministic MVP)
The Referee uses a deterministic rubric scorer so runs are fast and replayable:
- Dimensions (1–5): commitments_fit, deadline_risk, energy_match, time_window, inventory_risk
- Winner: sum of scores (equal weights)
- Tie-breakers: higher `deadline_risk`, then `inventory_risk`, then `energy_match`

Upgrade path: optionally swap the Referee to an LLM judge (with JSON output) while keeping the same persistence model (`decisions` + `messages` + `artifacts`).

## Triggered Autonomy (Wow Factor, Minimal)
BabyHandoff includes always-on “watcher” agents that react to new information without a button click:
- **Risk Radar**: when a new log is added, watcher agents recompute `shift_state` (energy inferred, deadline risk, inventory risk, suggested next actions).
- **Change-stream wakeups**: watchers can be driven by MongoDB change streams (Atlas) so agent activity is event-driven and persistent; the UI shows recent “agent woke up” events.
Keep the full **Village Council** run on an explicit user action (`Generate plan`) to avoid surprise complexity.

**MVP stance:** Risk Radar is MVP-critical because it demonstrates “triggered autonomy” (agents acting without a button click) and makes the app feel alive even before running the council.

### Error Handling (Retries + Fallbacks)
Agents can time out or fail; the Coordinator must still produce a usable output:
- Per-agent timeout: `AGENT_TIMEOUT_SECONDS` (applies to OpenAI planner calls)
- Retries: up to `1 + AGENT_MAX_RETRIES` attempts with exponential backoff
- Fallback: if a planner’s LLM call fails, the system uses the deterministic heuristic proposal for that archetype and records the error in `tasks.outputs`
- Run completion: the council still completes and produces `handoff_markdown` + `council_transcript_markdown`

## Skill-Based Delegation (Hackathon Visible)
The Coordinator assigns each council step to a specialized agent by querying the agent registry (skills) and records a task list + event stream in MongoDB, so judges can see delegation and collaboration—not a single monolithic prompt.

### Roles
- **Coordinator**: decomposes work, assigns tasks by skill/availability, merges outputs, versions artifacts.
- **Normalizer**: converts raw notes into structured events (entities, timestamps, intents).
- **Scenario Planners (3)**: generate competing plans (Sleep/Errands/Admin).
- **Referee**: scores proposals with rubric + explanations.
- **Handoff Writer**: generates the final Markdown artifact.

### Token-Limited Context Sharing
Each agent receives a `context_pack` (summary + references), not the full history. Packs are persisted so runs are reproducible and replayable.

## MongoDB Atlas (Required) Usage
MongoDB is the system of record for durable context, collaboration, and audit:
- `agents`: skill registry (who can do what)
- `shifts`: operator context + window
- `logs`: raw notes + metadata
- `shift_state`: computed Risk Radar state
- `runs`: each council execution (status, timings)
- `tasks`: agent tasks + retries + assignments + model used
- `events`: append-only run timeline (demo-friendly “audit log”)
- `messages`: inter-agent communications
- `context_packs`: token-budgeted bundles used per task
- `proposals`: the three scenario plans
- `decisions`: rubric scores + selected plan + rationale
- `artifacts`: handoff markdown versions + diffs

## Non-Goals / Safety
- No medical diagnosis, symptom triage, or treatment recommendations.
- Allowed: non-medical routines, checklists, and “planning help” (e.g., create a grocery list, block a focus window, reminder to prepare bottles) as long as it avoids medical claims.
- No integrations required for MVP (calendar, messaging, etc.).
- No online research required for MVP (avoid demo fragility).

## Disclaimer (UI Copy)
“BabyHandoff is a planning tool. It does not provide medical advice. For health concerns, contact a licensed professional.”

## Demo Plan (2–3 minutes)
1. Click `Load sample shift` (pre-seeded notes: low sleep, admin deadline, running-low).
2. Add one new note → show the Risk Radar update + watcher “wakeup” event (change streams; fallback still works).
3. Click `Generate Plan` → show delegation (`tasks`) + event stream (`events`) updating, then the three proposals appear.
4. Open `Council Transcript` → show the persisted context pack + each agent’s output (MongoDB-backed audit log).
5. Open Decision Card → show rubric + evidence-linked rationale; highlight the auto-selected winner.
6. Export `.md` → show rendered preview + download.
7. What-if: set `Energy: Auto → Low` → re-run → show v1→v2 diff and new winner (if applicable).
8. Flash the run replay/audit trail view to prove persistence (MongoDB-backed): runs, tasks, messages, artifacts.

### Sample Shift (Seed Data)
Use a realistic, slightly chaotic set of notes that forces tradeoffs:
- `Note`: “Slept ~3–4 hours broken; feeling low energy.”
- `Deadline`: “Submit daycare deposit form by tomorrow 5pm.”
- `Task`: “Pick up diapers + wipes today.”
- `Running low`: “Wipes < 1 day; diaper cream almost out.”
- `Note`: “Part-time class assignment due in 2 days; need 60–90 min focus block.”
- `Task`: “Groceries: protein + easy snacks (15 min list).”

## Success Criteria (MVP)
- <45 seconds from sample input to selected plan + handoff artifact.
- Decision Card shows 3 proposals + rubric + evidence links.
- Exported Markdown is clean and readable in Obsidian.
- All run steps are persisted and replayable from MongoDB.

## Risks & Mitigations (Hackathon-Realistic)
- **Looks like “just an LLM”** → make the run graph, proposals, rubric, and decision log front-and-center.
- **Unsafe/medical content** → hard disclaimer + “medical safety” guardrails; keep outputs in planning/checklist mode.
- **Latency / demo fragility** → seeded sample shift + deterministic demo mode; avoid network dependencies beyond Atlas.
- **Token bloat** → context packs with strict budget and explicit references back to stored logs/artifacts.

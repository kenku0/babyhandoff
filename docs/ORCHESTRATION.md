# BabyHandoff Orchestration (MVP)

This document explains how the Coordinator runs the “Village Council” and how outputs become judge-visible artifacts, with MongoDB as the system of record.

## Execution Model

- A **run** is created (`runs`), then a background coroutine executes the run so the UI can show live progress.
- The Coordinator creates a fixed task graph (`tasks`) and assigns each task to an agent by skill lookup (`agents`).
- The three planners run **in parallel** (async `gather`).
- The Referee scores proposals (deterministic rubric by default; optional OpenAI JSON judge).
- The HandoffWriter writes the final `handoff_markdown` artifact and a `council_transcript_markdown` artifact.

## Task + Message Format (MongoDB)

- `tasks` are the durable “work graph”:
  - `name`: `normalize_logs | plan_sleep_first | plan_errands_first | plan_admin_first | score_and_select | write_handoff`
  - `status`: `queued | running | completed | failed`
  - `attempt`, `max_attempts`
  - `inputs`: `{ context_pack_id, provider, model, timeout_s, archetype, ... }`
  - `outputs`: `{ proposal_id | decision_id | artifact_id, llm_used, provider, model, fallback, error, ... }`
- `messages` are the durable “debate log”:
  - `sender`: `Coordinator` or agent name
  - `task_id`: links each message to the task that produced it
  - `content`: Markdown (proposal blocks, rubric summary, status notes)

## Context Packs (Token-Limited Sharing)

- One `context_pack` is created per run and persisted (`context_packs`).
- It includes `summary`, `compiled_text`, and stable `log_refs` (with `log_id`), plus:
  - `token_budget` (configured)
  - `token_estimate` (cheap chars/token approximation)
  - `included_log_ids[]` (exact logs used for agent prompts)
- Trimming strategy (MVP):
  - Must-keep (most recent): `deadline`, `inventory`, then `task`
  - Fill remaining budget by recency across all logs

## Rubric Scoring + Selection

- Deterministic rubric (default):
  - 5 dimensions, score 1–5 each: `commitments_fit`, `deadline_risk`, `energy_match`, `time_window`, `inventory_risk`
  - Winner = total score, tie-breakers: `deadline_risk`, then `inventory_risk`, then `energy_match`
- Optional OpenAI referee (if `OPENAI_API_KEY` is set and `COUNCIL_MODE in {openai,multi}`):
  - LLM returns strict JSON scores + winner
  - Coordinator regenerates the handoff from the chosen proposal
  - If the judge fails, it falls back to deterministic scoring

## Artifacts + Exports

Artifacts are persisted in MongoDB (`artifacts`), versioned per shift:

- `handoff_markdown` → `GET /shifts/{shift_id}/runs/{run_id}/handoff.md`
- `council_transcript_markdown` → `GET /shifts/{shift_id}/runs/{run_id}/council.md`

The council export is generated from `messages` (plus `events` and the `context_pack`) so it is replayable and auditable.

## Failure Handling

- Planner failures:
  - Retries up to `1 + AGENT_MAX_RETRIES` with backoff (LLM calls only)
  - Falls back to deterministic heuristic proposals
- Referee failure:
  - Falls back to deterministic rubric decision
- Run still completes whenever possible and writes artifacts for judges to inspect.


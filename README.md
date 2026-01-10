# BabyHandoff

“It takes a village.” BabyHandoff is a minimalist web app for exhausted parents: quick notes in, a rubric-scored Decision Card out (Sleep-first vs Errands-first vs Admin-first), plus a clean Markdown handoff export. MongoDB Atlas is the durable context + audit engine.

## Local Setup
1. Create a virtualenv: `python3 -m venv .venv && source .venv/bin/activate`
2. Install deps: `python -m pip install -r requirements.txt`
3. Configure env: `cp docs/env.example .env` (or `cp .env.example .env`) and set `MONGODB_URI` (plus optional `OPENAI_API_KEY`)
   - Atlas note: add your current IP under **Project → Database & Network Access** or the app won’t connect.
   - Create a DB user under **Project → Database Access** and use those credentials in the URI.
   - Use the Atlas “Drivers” URI ending with `/` (this app selects the DB via `MONGODB_DB`; if you include a DB path, add `authSource=admin`).
   - Optional LLM:
     - `COUNCIL_MODE=openai` → set `OPENAI_API_KEY` + `OPENAI_MODEL=gpt-5.2`
        - Referee: if `COUNCIL_MODE=openai` and `OPENAI_API_KEY` is set, the rubric is also scored by an OpenAI JSON judge (falls back to deterministic scoring if it fails).
     - `COUNCIL_MODE=anthropic` → set `ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL`
     - `COUNCIL_MODE=gemini` → set `GEMINI_API_KEY` + `GEMINI_MODEL`
     - `COUNCIL_MODE=multi` → set any of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (best demo: all 3)
        - Referee: if `OPENAI_API_KEY` is set, the rubric is also scored by an OpenAI JSON judge (falls back to deterministic scoring if it fails).
4. Run: `PYTHONPYCACHEPREFIX=.pycache uvicorn server.main:app --reload` (helps on macOS if you hit `PermissionError` writing caches)
5. Open: `http://127.0.0.1:8000`
6. If you’re redirected to `http://127.0.0.1:8000/setup`, your Atlas connection isn’t configured/working yet (check DB user, IP allowlist, and `MONGODB_URI`).

## Demo Flow
- Click `Demo shift` (or create a shift and click `Load sample shift`)
- Click `Generate plan`
- Review the Decision Card + evidence links
- Export the Markdown handoff (and optionally `council.md` for the debate log)
- Optional: change `Energy` and re-run for a v1→v2 diff

More implementation detail: `docs/ORCHESTRATION.md`

## Safety
BabyHandoff is a planning tool and does not provide medical advice.

## Secrets
- Never commit `.env`.
- If an API key was pasted into chat/logs, revoke it and rotate immediately.

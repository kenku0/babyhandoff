from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from server.db import Mongo, connect_mongo
from server.repo.agents import AgentsRepo
from server.repo.artifacts import ArtifactsRepo
from server.repo.context_packs import ContextPacksRepo
from server.repo.decisions import DecisionsRepo
from server.repo.events import EventsRepo
from server.repo.logs import LogsRepo
from server.repo.messages import MessagesRepo
from server.repo.proposals import ProposalsRepo
from server.repo.shift_state import ShiftStateRepo
from server.repo.shifts import ShiftsRepo
from server.repo.runs import RunsRepo
from server.repo.tasks import TasksRepo
from server.repo.indexes import ensure_indexes
from server.services.council import (
    build_council_transcript_markdown_from_messages,
    decision_to_markdown,
    proposal_to_markdown,
    run_council,
    unified_diff,
)
from server.services.council_openai import generate_proposal_openai
from server.services.change_streams import ChangeStreamWorker
from server.services.context_packs import build_context_pack
from server.services.markdown_render import render_markdown_safe
from server.settings import Settings, load_settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _recompute_shift_state(
    mongo: Mongo,
    *,
    shift_id: str,
    emit_event: bool,
    message: str,
) -> None:
    logs = await LogsRepo(mongo.db).list_for_shift(shift_id, limit=200)
    from server.agents.risk_radar import run_risk_radar

    radar = run_risk_radar(logs)
    await ShiftStateRepo(mongo.db).upsert(
        shift_id,
        {
            "energy_inferred": radar.energy_inferred,
            "deadline_risk": radar.deadline_risk,
            "inventory_risk": radar.inventory_risk,
            "suggested_next_actions": radar.suggested_next_actions,
        },
    )
    if emit_event:
        await EventsRepo(mongo.db).add(
            shift_id=shift_id,
            event_type="agent_wakeup",
            agent="RiskRadar",
            message=message,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    settings = load_settings()

    app.state.settings = settings
    app.state.mongo = None
    app.state.mongo_ok = False
    app.state.mongo_error = None
    app.state.change_streams_enabled = False
    app.state.indexes_ok = False
    app.state.indexes_error = None

    mongo: Mongo | None = None
    stop_event: asyncio.Event | None = None
    worker_task: asyncio.Task | None = None

    try:
        mongo = await connect_mongo(settings)
        app.state.mongo = mongo
        app.state.mongo_ok = True
    except Exception as e:
        app.state.mongo = None
        app.state.mongo_ok = False
        app.state.mongo_error = str(e)

    if mongo is not None:
        try:
            await AgentsRepo(mongo.db).ensure_defaults()
        except Exception as e:
            app.state.mongo = None
            app.state.mongo_ok = False
            app.state.mongo_error = f"MongoDB connected, but app bootstrap failed: {e}"
            mongo.client.close()
            mongo = None

    if mongo is not None:
        try:
            await ensure_indexes(mongo.db)
            app.state.indexes_ok = True
        except Exception as e:
            app.state.indexes_ok = False
            app.state.indexes_error = str(e)

        stop_event = asyncio.Event()
        worker = ChangeStreamWorker(mongo.db)
        try:
            ready_event = asyncio.Event()
            worker_task = asyncio.create_task(worker.run(stop_event, ready_event=ready_event))
            app.state._stop_event = stop_event
            app.state._worker_task = worker_task
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=1.0)
                app.state.change_streams_enabled = True
            except asyncio.TimeoutError:
                app.state.change_streams_enabled = False
        except Exception:
            app.state.change_streams_enabled = False

    try:
        yield
    finally:
        if stop_event is not None:
            stop_event.set()
        if worker_task is not None:
            worker_task.cancel()
        if mongo is not None:
            mongo.client.close()


app = FastAPI(title="BabyHandoff", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")

SAMPLE_SHIFT_LOGS: list[tuple[str, str]] = [
    ("note", "Slept ~3–4 hours broken; feeling low energy."),
    ("deadline", "Submit daycare deposit form by tomorrow 5pm."),
    ("task", "Pick up diapers + wipes today."),
    ("inventory", "Wipes < 1 day; diaper cream almost out."),
    ("note", "Part-time class assignment due in 2 days; need 60–90 min focus block."),
    ("task", "Groceries: protein + easy snacks (15 min list)."),
]


def _mongo(request: Request) -> Mongo:
    mongo = getattr(request.app.state, "mongo", None)
    if mongo is None:
        raise RuntimeError("MongoDB is not connected.")
    return mongo


def _settings(request: Request) -> Settings:
    return request.app.state.settings


@app.middleware("http")
async def require_mongo(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static") or path in {"/setup", "/health", "/favicon.ico"}:
        return await call_next(request)
    if not getattr(request.app.state, "mongo_ok", False):
        return RedirectResponse(url="/setup", status_code=303)
    return await call_next(request)


@app.get("/setup", response_class=HTMLResponse)
async def setup(request: Request):
    return templates.TemplateResponse(
        request,
        "setup.html",
        {
            "mongo_ok": getattr(request.app.state, "mongo_ok", False),
            "mongo_error": getattr(request.app.state, "mongo_error", None),
            "indexes_ok": getattr(request.app.state, "indexes_ok", False),
            "indexes_error": getattr(request.app.state, "indexes_error", None),
            "change_streams_enabled": getattr(request.app.state, "change_streams_enabled", False),
            "council_mode": _settings(request).council_mode,
        },
    )


async def _check_change_streams(mongo: Mongo) -> dict:
    pipeline = [{"$match": {"operationType": "insert"}}]
    try:
        async with mongo.db["logs"].watch(pipeline):
            return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/health")
async def health(request: Request):
    settings = _settings(request)
    mongo = getattr(request.app.state, "mongo", None)
    mongo_ok = bool(getattr(request.app.state, "mongo_ok", False))
    mongo_error = getattr(request.app.state, "mongo_error", None)

    ping_ok = False
    ping_error = None
    change_streams: dict | None = None

    if mongo_ok and mongo is not None:
        try:
            await mongo.client.admin.command("ping")
            ping_ok = True
        except Exception as e:
            ping_ok = False
            ping_error = str(e)
        change_streams = await _check_change_streams(mongo)

    return JSONResponse(
        {
            "ok": bool(mongo_ok and ping_ok),
            "mongo": {
                "configured": bool(settings.mongodb_uri),
                "db": settings.mongodb_db,
                "connected": mongo_ok,
                "ping_ok": ping_ok,
                "error": mongo_error or ping_error,
            },
            "indexes": {
                "ok": bool(getattr(request.app.state, "indexes_ok", False)),
                "error": getattr(request.app.state, "indexes_error", None),
            },
            "change_streams": {
                "worker_enabled": bool(getattr(request.app.state, "change_streams_enabled", False)),
                "supported": (change_streams or {}).get("ok") if change_streams is not None else None,
                "error": (change_streams or {}).get("error") if change_streams is not None else None,
            },
            "council": {
                "mode": settings.council_mode,
                "openai_model": settings.openai_model,
                "openai_key_configured": bool(settings.openai_api_key),
            },
        }
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    mongo = _mongo(request)
    shifts_repo = ShiftsRepo(mongo.db)
    shifts = await shifts_repo.list_recent(limit=20)
    return templates.TemplateResponse(
        request,
        "home.html",
        {"shifts": shifts, "settings": _settings(request)},
    )

@app.post("/demo")
async def demo_shift(request: Request):
    mongo = _mongo(request)
    logs_repo = LogsRepo(mongo.db)
    shifts_repo = ShiftsRepo(mongo.db)

    shift_id = await shifts_repo.create_shift()
    for t, txt in SAMPLE_SHIFT_LOGS:
        await logs_repo.add(shift_id, t, txt)
    await shifts_repo.touch(shift_id)
    await _recompute_shift_state(
        mongo,
        shift_id=shift_id,
        emit_event=not getattr(request.app.state, "change_streams_enabled", False),
        message="Demo shift inserted; watchers recomputed shift state. (fallback)",
    )
    return RedirectResponse(url=f"/shifts/{shift_id}", status_code=303)


@app.post("/shifts/new")
async def new_shift(request: Request):
    mongo = _mongo(request)
    shift_id = await ShiftsRepo(mongo.db).create_shift()
    return RedirectResponse(url=f"/shifts/{shift_id}", status_code=303)


@app.get("/shifts/{shift_id}", response_class=HTMLResponse)
async def shift_view(request: Request, shift_id: str):
    mongo = _mongo(request)
    shifts_repo = ShiftsRepo(mongo.db)
    logs_repo = LogsRepo(mongo.db)
    state_repo = ShiftStateRepo(mongo.db)
    runs_repo = RunsRepo(mongo.db)

    shift = await shifts_repo.get(shift_id)
    if not shift:
        return RedirectResponse(url="/", status_code=303)

    logs = await logs_repo.list_for_shift(shift_id)
    state = await state_repo.get(shift_id)
    runs = await runs_repo.list_for_shift(shift_id, limit=8)
    return templates.TemplateResponse(
        request,
        "shift.html",
        {
            "shift": shift,
            "logs": logs,
            "state": state,
            "runs": runs,
            "settings": _settings(request),
        },
    )

@app.get("/shifts/{shift_id}/risk", response_class=HTMLResponse)
async def risk_fragment(request: Request, shift_id: str):
    mongo = _mongo(request)
    state = await ShiftStateRepo(mongo.db).get(shift_id)
    events = await EventsRepo(mongo.db).list_for_shift(shift_id, limit=8)
    # Filter to watcher events for the fragment
    watcher_events = [e for e in events if e.get("type") == "agent_wakeup"][:6]
    return templates.TemplateResponse(
        request,
        "risk_fragment.html",
        {"state": state, "events": watcher_events},
    )


async def _execute_council_run(
    mongo: Mongo,
    *,
    settings: Settings,
    shift_id: str,
    run_id: str,
    energy_override: str | None,
    task_ids: dict[str, str],
    task_agents: dict[str, str],
) -> None:
    events = EventsRepo(mongo.db)
    tasks = TasksRepo(mongo.db)
    messages = MessagesRepo(mongo.db)
    runs = RunsRepo(mongo.db)
    artifacts_repo = ArtifactsRepo(mongo.db)

    context_pack_id: str | None = None
    try:
        logs = await LogsRepo(mongo.db).list_for_shift(shift_id, limit=200)
        context_pack = build_context_pack(logs=logs, energy_override=energy_override)
        context_pack_id = await ContextPacksRepo(mongo.db).create(
            shift_id=shift_id,
            run_id=run_id,
            token_budget=settings.context_pack_token_budget,
            pack=context_pack,
        )
        context_logs = logs[-40:] if len(logs) > 40 else logs

        await tasks.mark_running(
            task_ids["normalize_logs"],
            attempt=1,
            inputs={
                "log_count": len(logs),
                "context_pack_id": context_pack_id,
                "context_logs": len(context_logs),
            },
        )
        await tasks.mark_completed(
            task_ids["normalize_logs"],
            outputs={
                "log_count": len(logs),
                "context_pack_id": context_pack_id,
                "context_logs": len(context_logs),
            },
        )

        await messages.add(
            shift_id=shift_id,
            run_id=run_id,
            task_id=None,
            sender="Coordinator",
            role="assistant",
            content="\n".join(
                [
                    "Council run started.",
                    "",
                    f"- Council mode: `{settings.council_mode}`",
                    f"- Energy override: `{energy_override or 'auto'}`",
                    f"- Logs loaded: `{len(logs)}`",
                    f"- Context pack: `{context_pack_id}` (logs: `{len(context_logs)}`)",
                    "",
                    "### Context pack summary",
                    context_pack.get("summary") or "",
                    "",
                    "### Context pack notes",
                    context_pack.get("compiled_text") or "_No notes._",
                ]
            ),
            meta={
                "council_mode": settings.council_mode,
                "energy_override": energy_override,
                "context_pack_id": context_pack_id,
                "token_budget": settings.context_pack_token_budget,
            },
        )

        proposal_overrides: dict[str, dict] = {}
        planner_meta: dict[str, dict] = {}

        use_openai = settings.council_mode == "openai" and bool(settings.openai_api_key)
        planner_max_attempts = max(1, 1 + max(0, settings.agent_max_retries)) if use_openai else 1

        if use_openai:
            await events.add(
                shift_id=shift_id,
                event_type="llm_mode",
                agent="Coordinator",
                message=f"COUNCIL_MODE=openai (model={settings.openai_model}).",
                run_id=run_id,
            )
            planner_tasks = {
                "sleep_first": "plan_sleep_first",
                "errands_first": "plan_errands_first",
                "admin_first": "plan_admin_first",
            }
            for archetype, task_name in planner_tasks.items():
                await tasks.mark_running(
                    task_ids[task_name],
                    attempt=1,
                    inputs={
                        "council_mode": settings.council_mode,
                        "context_pack_id": context_pack_id,
                        "context_logs": len(context_logs),
                        "model": settings.openai_model,
                        "energy_override": energy_override,
                        "timeout_s": settings.agent_timeout_seconds,
                        "archetype": archetype,
                    },
                )

            async def _planner_call(archetype: str) -> None:
                task_name = planner_tasks[archetype]
                attempts = 0
                last_error = ""
                for attempt in range(1, planner_max_attempts + 1):
                    attempts = attempt
                    await events.add(
                        shift_id=shift_id,
                        event_type="llm_attempt",
                        agent=task_agents.get(task_name),
                        message=f"Attempt {attempt}/{planner_max_attempts} for {archetype.replace('_', '-')}.",
                        run_id=run_id,
                    )
                    try:
                        llm = await asyncio.to_thread(
                            generate_proposal_openai,
                            api_key=settings.openai_api_key or "",
                            model=settings.openai_model,
                            archetype=archetype,  # type: ignore[arg-type]
                            logs=context_logs,
                            energy=energy_override,
                            timeout_s=settings.agent_timeout_seconds,
                        )
                        proposal_overrides[archetype] = llm
                        planner_meta[archetype] = {
                            "llm_used": True,
                            "attempts": attempt,
                            "max_attempts": planner_max_attempts,
                            "model": settings.openai_model,
                        }
                        await events.add(
                            shift_id=shift_id,
                            event_type="llm_ok",
                            agent=task_agents.get(task_name),
                            message=f"LLM proposal generated for {archetype.replace('_', '-')}.",
                            run_id=run_id,
                        )
                        return
                    except Exception as e:
                        last_error = str(e)
                        if attempt < planner_max_attempts:
                            await asyncio.sleep(min(2.0, 0.4 * (2 ** (attempt - 1))))
                            continue
                planner_meta[archetype] = {
                    "llm_used": False,
                    "attempts": attempts,
                    "max_attempts": planner_max_attempts,
                    "error": last_error,
                    "fallback": True,
                }
                await events.add(
                    shift_id=shift_id,
                    event_type="llm_fallback",
                    agent=task_agents.get(task_name),
                    message=f"LLM unavailable for {archetype.replace('_', '-')}; using heuristic proposal.",
                    run_id=run_id,
                    meta={"error": last_error} if last_error else None,
                )

            await asyncio.gather(*[_planner_call(a) for a in planner_tasks.keys()])
        else:
            await events.add(
                shift_id=shift_id,
                event_type="llm_mode",
                agent="Coordinator",
                message="COUNCIL_MODE=heuristic.",
                run_id=run_id,
            )
            for task_name in ["plan_sleep_first", "plan_errands_first", "plan_admin_first"]:
                await tasks.mark_running(
                    task_ids[task_name],
                    attempt=1,
                    inputs={
                        "council_mode": settings.council_mode,
                        "context_pack_id": context_pack_id,
                        "context_logs": len(context_logs),
                        "energy_override": energy_override,
                    },
                )
            for archetype in ["sleep_first", "errands_first", "admin_first"]:
                planner_meta[archetype] = {"llm_used": False, "attempts": 0, "max_attempts": 1}

        output = run_council(
            shift_id=shift_id,
            logs=logs,
            energy_override=energy_override,
            proposal_overrides=proposal_overrides or None,
        )

        proposals_repo = ProposalsRepo(mongo.db)
        proposal_ids: dict[str, str] = {}
        for p in output.proposals:
            pid = await proposals_repo.add(run_id, p)
            archetype = str(p.get("archetype") or "")
            if archetype:
                proposal_ids[archetype] = pid

        for archetype, task_name in {
            "sleep_first": "plan_sleep_first",
            "errands_first": "plan_errands_first",
            "admin_first": "plan_admin_first",
        }.items():
            await tasks.mark_completed(
                task_ids[task_name],
                outputs={
                    "proposal_id": proposal_ids.get(archetype),
                    "context_pack_id": context_pack_id,
                    **planner_meta.get(archetype, {}),
                },
            )

        await tasks.mark_running(
            task_ids["score_and_select"],
            attempt=1,
            inputs={"proposal_ids": proposal_ids, "rubric": "v1"},
        )
        decision_id = await DecisionsRepo(mongo.db).upsert_for_run(run_id, output.decision)
        await tasks.mark_completed(task_ids["score_and_select"], outputs={"decision_id": decision_id})

        prev = await artifacts_repo.latest_for_shift(shift_id, kind="handoff_markdown")
        prev_id = str(prev["_id"]) if prev else None
        diff = unified_diff(prev["markdown"], output.handoff_markdown) if prev else None
        await tasks.mark_running(task_ids["write_handoff"], attempt=1, inputs={"kind": "handoff_markdown"})
        handoff_artifact_id = await artifacts_repo.create(
            shift_id=shift_id,
            run_id=run_id,
            kind="handoff_markdown",
            markdown=output.handoff_markdown,
            diff=diff,
            prev_artifact_id=prev_id,
        )
        await tasks.mark_completed(
            task_ids["write_handoff"],
            outputs={"artifact_id": handoff_artifact_id, "kind": "handoff_markdown"},
        )

        proposals_by_archetype = {str(p.get("archetype") or ""): p for p in output.proposals}
        sleep = proposals_by_archetype.get("sleep_first")
        errands = proposals_by_archetype.get("errands_first")
        admin = proposals_by_archetype.get("admin_first")

        if sleep:
            await messages.add(
                shift_id=shift_id,
                run_id=run_id,
                task_id=task_ids.get("plan_sleep_first"),
                sender=task_agents.get("plan_sleep_first", "SleepPlanner"),
                role="assistant",
                content=proposal_to_markdown(sleep),
                meta={
                    "council_mode": settings.council_mode,
                    "context_pack_id": context_pack_id,
                    "context_logs": len(context_logs),
                    **planner_meta.get("sleep_first", {}),
                },
            )
        if errands:
            await messages.add(
                shift_id=shift_id,
                run_id=run_id,
                task_id=task_ids.get("plan_errands_first"),
                sender=task_agents.get("plan_errands_first", "ErrandsPlanner"),
                role="assistant",
                content=proposal_to_markdown(errands),
                meta={
                    "council_mode": settings.council_mode,
                    "context_pack_id": context_pack_id,
                    "context_logs": len(context_logs),
                    **planner_meta.get("errands_first", {}),
                },
            )
        if admin:
            await messages.add(
                shift_id=shift_id,
                run_id=run_id,
                task_id=task_ids.get("plan_admin_first"),
                sender=task_agents.get("plan_admin_first", "AdminPlanner"),
                role="assistant",
                content=proposal_to_markdown(admin),
                meta={
                    "council_mode": settings.council_mode,
                    "context_pack_id": context_pack_id,
                    "context_logs": len(context_logs),
                    **planner_meta.get("admin_first", {}),
                },
            )

        await messages.add(
            shift_id=shift_id,
            run_id=run_id,
            task_id=task_ids.get("score_and_select"),
            sender=task_agents.get("score_and_select", "Referee"),
            role="assistant",
            content=decision_to_markdown(output.decision),
            meta={"decision_id": decision_id, "context_pack_id": context_pack_id},
        )

        await messages.add(
            shift_id=shift_id,
            run_id=run_id,
            task_id=task_ids.get("write_handoff"),
            sender=task_agents.get("write_handoff", "HandoffWriter"),
            role="assistant",
            content="\n".join(
                [
                    "Handoff written to MongoDB artifact.",
                    "",
                    f"- artifact_id: `{handoff_artifact_id}`",
                    f"- download: `/shifts/{shift_id}/runs/{run_id}/handoff.md`",
                ]
            ),
            meta={"artifact_id": handoff_artifact_id, "context_pack_id": context_pack_id},
        )

        transcript_messages = await messages.list_for_run(run_id, limit=400)
        council_md = build_council_transcript_markdown_from_messages(
            shift_id=shift_id,
            run_id=run_id,
            messages=transcript_messages,
        )
        prev_c = await artifacts_repo.latest_for_shift(shift_id, kind="council_transcript_markdown")
        prev_c_id = str(prev_c["_id"]) if prev_c else None
        diff_c = unified_diff(prev_c["markdown"], council_md) if prev_c else None
        await artifacts_repo.create(
            shift_id=shift_id,
            run_id=run_id,
            kind="council_transcript_markdown",
            markdown=council_md,
            diff=diff_c,
            prev_artifact_id=prev_c_id,
        )

        await runs.set_status(run_id, "completed")
        await events.add(
            shift_id=shift_id,
            event_type="council_done",
            agent="Coordinator",
            message="Council run completed.",
            run_id=run_id,
        )
    except Exception as e:
        error = {"type": type(e).__name__, "message": str(e)}
        try:
            await runs.set_status(run_id, "failed")
        except Exception:
            pass
        try:
            await events.add(
                shift_id=shift_id,
                event_type="council_failed",
                agent="Coordinator",
                message="Council run failed.",
                run_id=run_id,
                meta=error,
            )
        except Exception:
            pass
        try:
            for t in await tasks.list_for_run(run_id):
                status = str(t.get("status") or "").lower()
                if status not in {"completed", "failed"}:
                    await tasks.mark_failed(t["_id"], error=error)
        except Exception:
            pass
        try:
            await messages.add(
                shift_id=shift_id,
                run_id=run_id,
                task_id=None,
                sender="Coordinator",
                role="assistant",
                content="\n".join(
                    [
                        "Council run failed.",
                        "",
                        f"- error: `{error['type']}: {error['message']}`",
                    ]
                ),
                meta={"error": error, "context_pack_id": context_pack_id},
            )
        except Exception:
            pass
        try:
            transcript_messages = await messages.list_for_run(run_id, limit=400)
            council_md = build_council_transcript_markdown_from_messages(
                shift_id=shift_id,
                run_id=run_id,
                messages=transcript_messages,
            )
            prev_c = await artifacts_repo.latest_for_shift(shift_id, kind="council_transcript_markdown")
            prev_c_id = str(prev_c["_id"]) if prev_c else None
            diff_c = unified_diff(prev_c["markdown"], council_md) if prev_c else None
            await artifacts_repo.create(
                shift_id=shift_id,
                run_id=run_id,
                kind="council_transcript_markdown",
                markdown=council_md,
                diff=diff_c,
                prev_artifact_id=prev_c_id,
            )
        except Exception:
            pass


@app.post("/shifts/{shift_id}/logs")
async def add_log(
    request: Request,
    shift_id: str,
    log_type: Literal["note", "task", "deadline", "inventory"] = Form(...),
    text: str = Form(...),
):
    mongo = _mongo(request)
    await LogsRepo(mongo.db).add(shift_id, log_type, text.strip())
    await ShiftsRepo(mongo.db).touch(shift_id)
    await _recompute_shift_state(
        mongo,
        shift_id=shift_id,
        emit_event=not getattr(request.app.state, "change_streams_enabled", False),
        message="New log inserted; watchers recomputed shift state. (fallback)",
    )
    return RedirectResponse(url=f"/shifts/{shift_id}", status_code=303)


@app.post("/shifts/{shift_id}/sample")
async def load_sample_shift(request: Request, shift_id: str):
    mongo = _mongo(request)
    logs_repo = LogsRepo(mongo.db)
    shifts_repo = ShiftsRepo(mongo.db)

    for t, txt in SAMPLE_SHIFT_LOGS:
        await logs_repo.add(shift_id, t, txt)
    await shifts_repo.touch(shift_id)
    await _recompute_shift_state(
        mongo,
        shift_id=shift_id,
        emit_event=not getattr(request.app.state, "change_streams_enabled", False),
        message="Sample shift inserted; watchers recomputed shift state. (fallback)",
    )
    return RedirectResponse(url=f"/shifts/{shift_id}", status_code=303)


@app.post("/shifts/{shift_id}/runs")
async def create_run(
    request: Request,
    shift_id: str,
    energy: str = Form(""),
):
    mongo = _mongo(request)
    settings = _settings(request)
    events = EventsRepo(mongo.db)
    agents = AgentsRepo(mongo.db)
    tasks = TasksRepo(mongo.db)
    shift = await ShiftsRepo(mongo.db).get(shift_id)
    if not shift:
        return RedirectResponse(url="/", status_code=303)

    run_id = await RunsRepo(mongo.db).create(shift_id, energy_override=energy or None)
    await events.add(
        shift_id=shift_id,
        event_type="council_start",
        agent="Coordinator",
        message="Council run started.",
        run_id=run_id,
        meta={"energy_override": energy or None},
    )
    await events.add(
        shift_id=shift_id,
        event_type="delegation",
        agent="Coordinator",
        message="Delegating tasks by skill.",
        run_id=run_id,
    )

    use_openai = settings.council_mode == "openai" and bool(settings.openai_api_key)
    planner_max_attempts = max(1, 1 + max(0, settings.agent_max_retries)) if use_openai else 1

    task_defs = [
        ("normalize_logs", "Normalizer", "normalize_logs"),
        ("plan_sleep_first", "SleepPlanner", "plan_sleep_first"),
        ("plan_errands_first", "ErrandsPlanner", "plan_errands_first"),
        ("plan_admin_first", "AdminPlanner", "plan_admin_first"),
        ("score_and_select", "Referee", "score_rubric"),
        ("write_handoff", "HandoffWriter", "write_markdown"),
    ]
    task_ids: dict[str, str] = {}
    task_agents: dict[str, str] = {}
    for name, fallback_agent, skill in task_defs:
        agent_doc = await agents.find_by_skill(skill)
        agent_name = agent_doc["name"] if agent_doc else fallback_agent
        max_attempts = planner_max_attempts if name.startswith("plan_") else 1
        task_id = await tasks.create(
            shift_id=shift_id,
            run_id=run_id,
            name=name,
            agent_name=agent_name,
            status="queued",
            max_attempts=max_attempts,
            meta={"skill": skill},
        )
        task_ids[name] = task_id
        task_agents[name] = agent_name
        await events.add(
            shift_id=shift_id,
            event_type="task_assigned",
            agent="Coordinator",
            message=f"Assigned {name} → {agent_name} (skill: {skill})",
            run_id=run_id,
        )

    asyncio.create_task(
        _execute_council_run(
            mongo,
            settings=settings,
            shift_id=shift_id,
            run_id=run_id,
            energy_override=energy or None,
            task_ids=task_ids,
            task_agents=task_agents,
        )
    )

    return RedirectResponse(url=f"/shifts/{shift_id}/runs/{run_id}", status_code=303)


@app.get("/shifts/{shift_id}/runs/{run_id}", response_class=HTMLResponse)
async def run_view(request: Request, shift_id: str, run_id: str):
    mongo = _mongo(request)
    run = await RunsRepo(mongo.db).get(run_id)
    if not run:
        return RedirectResponse(url=f"/shifts/{shift_id}", status_code=303)
    run_shift_id = str(run.get("shift_id") or "")
    if run_shift_id and run_shift_id != shift_id:
        return RedirectResponse(url=f"/shifts/{run_shift_id}/runs/{run_id}", status_code=303)
    tasks = await TasksRepo(mongo.db).list_for_run(run_id)
    proposals = await ProposalsRepo(mongo.db).list_for_run(run_id)
    decision = await DecisionsRepo(mongo.db).get_for_run(run_id)
    context_pack = await ContextPacksRepo(mongo.db).get_for_run(run_id)
    context_pack_html = render_markdown_safe(context_pack.get("compiled_text", "")) if context_pack else ""
    artifact = await ArtifactsRepo(mongo.db).get_for_run(run_id, kind="handoff_markdown")
    html = render_markdown_safe(artifact["markdown"]) if artifact else ""
    council_artifact = await ArtifactsRepo(mongo.db).get_for_run(run_id, kind="council_transcript_markdown")
    council_html = render_markdown_safe(council_artifact["markdown"]) if council_artifact else ""
    live_council_html = ""
    live_council_count = 0
    if not council_artifact:
        run_messages = await MessagesRepo(mongo.db).list_for_run(run_id, limit=200)
        live_council_count = len(run_messages)
        if run_messages:
            live_council_md = build_council_transcript_markdown_from_messages(
                shift_id=shift_id,
                run_id=run_id,
                messages=run_messages,
            )
            live_council_html = render_markdown_safe(live_council_md)
    run_events = [
        e
        for e in await EventsRepo(mongo.db).list_for_shift(shift_id, limit=120)
        if e.get("run_id") == run_id
    ]
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "shift_id": shift_id,
            "run": run,
            "tasks": tasks,
            "proposals": proposals,
            "decision": decision,
            "context_pack": context_pack,
            "context_pack_html": context_pack_html,
            "artifact": artifact,
            "artifact_html": html,
            "council_artifact": council_artifact,
            "council_html": council_html,
            "live_council_html": live_council_html,
            "live_council_count": live_council_count,
            "events": run_events,
        },
    )


@app.get("/shifts/{shift_id}/runs/{run_id}/handoff.md")
async def download_handoff(request: Request, shift_id: str, run_id: str):
    mongo = _mongo(request)
    artifact = await ArtifactsRepo(mongo.db).get_for_run(run_id, kind="handoff_markdown")
    if not artifact:
        return RedirectResponse(url=f"/shifts/{shift_id}", status_code=303)
    markdown_text = artifact["markdown"]
    filename = f"babyhandoff_shift_{shift_id}_run_{run_id}.md"
    return Response(
        content=markdown_text,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/shifts/{shift_id}/runs/{run_id}/council.md")
async def download_council(request: Request, shift_id: str, run_id: str):
    mongo = _mongo(request)
    artifact = await ArtifactsRepo(mongo.db).get_for_run(run_id, kind="council_transcript_markdown")
    if artifact:
        markdown_text = artifact["markdown"]
    else:
        run_messages = await MessagesRepo(mongo.db).list_for_run(run_id, limit=400)
        if not run_messages:
            return RedirectResponse(url=f"/shifts/{shift_id}/runs/{run_id}", status_code=303)
        markdown_text = build_council_transcript_markdown_from_messages(
            shift_id=shift_id,
            run_id=run_id,
            messages=run_messages,
        )
    filename = f"babyhandoff_council_shift_{shift_id}_run_{run_id}.md"
    return Response(
        content=markdown_text,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

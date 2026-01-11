from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from random import Random

from bson import ObjectId

from server.db import Mongo
from server.repo.logs import LogsRepo
from server.repo.shifts import ShiftsRepo


@dataclass(frozen=True)
class DemoShiftTemplate:
    title: str
    logs: list[tuple[str, str]]

def _demo_extra_logs() -> list[tuple[str, str]]:
    # Keep these non-medical and broadly relatable for new parents.
    return [
        ("task", "Refill diaper caddy (diapers/wipes/cream) and put it back in the same spot."),
        ("task", "Set out next outfit + spare onesie (save a future scramble)."),
        ("task", "Pack go-bag basics: wipes, diapers, changing pad, snacks, charger."),
        ("task", "Reset the kitchen: one load of bottles/dishes, then stop."),
        ("task", "Make a 3-item snack plan (protein + easy grab)."),
        ("task", "Charge essentials (phone/monitor/portable battery)."),
        ("task", "Do a 10-minute tidy pass: only surfaces you’ll use tonight."),
        ("deadline", "Send one short logistics message (daycare / family) to unblock tomorrow."),
        ("deadline", "Pay/submit the one bill/form that will wake you up at 2am if you forget."),
        ("inventory", "Easy snacks running low (bars/nuts/yogurt)."),
        ("inventory", "Paper towels / trash bags low."),
        ("inventory", "Coffee/tea low; tomorrow morning risk."),
        ("note", "Constraint: protect one quiet block; phone on Do Not Disturb."),
        ("note", "Energy likely dips later; keep plans reversible."),
        ("note", "If it turns chaotic: default to essentials + one admin, nothing else."),
    ]


DEMO_TEMPLATES: list[DemoShiftTemplate] = [
    DemoShiftTemplate(
        title="Night handoff — low energy + daycare paperwork",
        logs=[
            ("note", "Slept ~3–4 hours broken; feeling low energy."),
            ("deadline", "Submit daycare deposit form by tomorrow 5pm (needs bank login)."),
            ("inventory", "Wipes < 1 day; diaper cream almost out."),
            ("task", "Pick up diapers + wipes today (one quick stop)."),
            ("task", "Prep a 15-min grocery list (protein + easy snacks)."),
        ],
    ),
    DemoShiftTemplate(
        title="Morning sprint — errands window before nap",
        logs=[
            ("note", "Baby likely naps around 10:30; errands need to fit before/after."),
            ("task", "Pharmacy pickup (10 min) + diapers if nearby."),
            ("inventory", "Formula at ~2 days."),
            ("note", "Energy medium; can do one focused admin task."),
        ],
    ),
    DemoShiftTemplate(
        title="Admin-heavy — forms + scheduling",
        logs=[
            ("deadline", "Insurance form due end of week."),
            ("task", "Call pediatrician office to confirm appointment time."),
            ("task", "Update shared calendar with next 2 weeks of commitments."),
            ("note", "Try to protect one recovery block today (30–45 min)."),
        ],
    ),
    DemoShiftTemplate(
        title="Weekend reset — supplies + batch prep",
        logs=[
            ("inventory", "Diapers running low; wipes okay."),
            ("task", "Batch prep snacks (10–15 min) + refill water bottles."),
            ("task", "Laundry: baby basics only (one load)."),
            ("note", "Energy high this morning, likely dips after lunch."),
        ],
    ),
    DemoShiftTemplate(
        title="Daycare day — dropoff + pickup timing",
        logs=[
            ("note", "Dropoff 8:30am, pickup 4:30pm."),
            ("task", "Pack daycare bag: extra outfit, wipes, cream."),
            ("deadline", "Send daycare deposit receipt email today."),
            ("note", "Goal: 60–90 min focus block during daycare."),
        ],
    ),
    DemoShiftTemplate(
        title="Solo parent shift — keep it simple",
        logs=[
            ("note", "Solo until evening; minimize cognitive load."),
            ("task", "Top 1 admin: pay bill / confirm appointment."),
            ("inventory", "Diaper cream almost out."),
            ("note", "If baby is fussy, default to rest + essentials only."),
        ],
    ),
    DemoShiftTemplate(
        title="Travel day — pack list + buffer time",
        logs=[
            ("task", "Pack: diapers, wipes, extra clothes, snacks, charger."),
            ("note", "Hard constraint: leave house by 2pm."),
            ("inventory", "Snacks low; quick store stop possible."),
        ],
    ),
    DemoShiftTemplate(
        title="Low sleep follow-up — prioritize recovery",
        logs=[
            ("note", "Very low sleep; protect a 90-min recovery window."),
            ("task", "Cancel/defers non-urgent errands."),
            ("deadline", "One must-do: submit form / quick email."),
            ("inventory", "Wipes < 1 day."),
        ],
    ),
]


async def reset_demo_data(mongo: Mongo) -> None:
    """
    Clear shift-related collections (keeps `agents` intact) for a clean demo slate.
    """
    db = mongo.db
    for name in [
        "shifts",
        "logs",
        "runs",
        "tasks",
        "messages",
        "events",
        "context_packs",
        "proposals",
        "decisions",
        "artifacts",
        "shift_state",
    ]:
        await db[name].delete_many({})


async def seed_demo_month(
    mongo: Mongo,
    *,
    days: int = 30,
    shift_count: int = 10,
    seed: int = 42,
) -> list[str]:
    """
    Create a visually varied set of demo shifts across the last `days` days.
    Returns created shift_ids (most recent last).
    """
    rng = Random(seed)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=max(1, days) - 1)

    shifts_repo = ShiftsRepo(mongo.db)
    logs_repo = LogsRepo(mongo.db)

    created_shift_ids: list[str] = []
    templates = (DEMO_TEMPLATES or [])[:]
    if not templates:
        return []

    # Space shifts across the month, but keep some clustering near "recent".
    offsets: list[int] = []
    if shift_count <= 1:
        offsets = [days - 1]
    else:
        step = max(1, (days - 1) // (shift_count - 1))
        offsets = [min(days - 1, i * step) for i in range(shift_count)]
        offsets[-1] = days - 1

    for i, off in enumerate(offsets):
        tpl = templates[i % len(templates)]
        day = start + timedelta(days=off)
        # Add a deterministic time-of-day for nicer "Started Today 10:45 PM" variation.
        hour = [8, 10, 14, 17, 20, 22][i % 6]
        created_at = day.replace(hour=hour, minute=(5 * (i % 12)) % 60, second=0, microsecond=0)

        # Stagger log timestamps within the shift window.
        log_base = created_at + timedelta(minutes=3)
        target_logs = 10

        selected_logs: list[tuple[str, str]] = tpl.logs[:]
        extras = _demo_extra_logs()
        rng.shuffle(extras)
        seen = set((t, txt) for (t, txt) in selected_logs)
        for t, txt in extras:
            if len(selected_logs) >= target_logs:
                break
            key = (t, txt)
            if key in seen:
                continue
            selected_logs.append((t, txt))
            seen.add(key)
        if len(selected_logs) > target_logs:
            selected_logs = selected_logs[:target_logs]

        # Spread entries across the 12h window for a denser, more realistic timeline.
        # Keep ordering stable; time offsets create the visual variability.
        minute_offsets = sorted(rng.randint(0, (12 * 60) - 10) for _ in range(len(selected_logs)))

        last_log_at = log_base
        shift_id = await shifts_repo.create_shift(
            title=tpl.title,
            created_at=created_at,
            updated_at=created_at,
            extra={"demo": True, "demo_scenario": tpl.title},
        )
        for j, (t, txt) in enumerate(selected_logs):
            last_log_at = log_base + timedelta(minutes=minute_offsets[j])
            await logs_repo.add(shift_id, t, txt, created_at=last_log_at)

        # Make the shift look "recently worked on" relative to its own day.
        await mongo.db["shifts"].update_one(
            {"_id": ObjectId(shift_id)},
            {"$set": {"updated_at": last_log_at}},
        )
        created_shift_ids.append(shift_id)

    return created_shift_ids

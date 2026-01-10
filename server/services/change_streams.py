from __future__ import annotations

import asyncio
from dataclasses import dataclass

from motor.motor_asyncio import AsyncIOMotorDatabase

from server.agents.risk_radar import run_risk_radar
from server.repo.events import EventsRepo
from server.repo.logs import LogsRepo
from server.repo.shift_state import ShiftStateRepo


@dataclass(frozen=True)
class ChangeStreamWorker:
    db: AsyncIOMotorDatabase

    async def run(self, stop_event: asyncio.Event, *, ready_event: asyncio.Event | None = None) -> None:
        logs_repo = LogsRepo(self.db)
        state_repo = ShiftStateRepo(self.db)
        events_repo = EventsRepo(self.db)

        pipeline = [{"$match": {"operationType": "insert"}}]
        while not stop_event.is_set():
            try:
                async with self.db["logs"].watch(pipeline) as stream:
                    if ready_event is not None and not ready_event.is_set():
                        ready_event.set()
                    while not stop_event.is_set():
                        try:
                            change = await stream.try_next()
                            if not change:
                                await asyncio.sleep(0.1)
                                continue
                            full_doc = change.get("fullDocument") or {}
                            shift_id = full_doc.get("shift_id")
                            if not shift_id:
                                continue
                            await events_repo.add(
                                shift_id=shift_id,
                                event_type="agent_wakeup",
                                agent="RiskRadar",
                                message="New log inserted; watchers recomputed shift state.",
                                meta={"log_id": str(full_doc.get("_id")), "log_type": full_doc.get("type")},
                            )
                            logs = await logs_repo.list_for_shift(shift_id, limit=200)
                            radar = run_risk_radar(logs)
                            await state_repo.upsert(
                                shift_id,
                                {
                                    "energy_inferred": radar.energy_inferred,
                                    "deadline_risk": radar.deadline_risk,
                                    "inventory_risk": radar.inventory_risk,
                                    "suggested_next_actions": radar.suggested_next_actions,
                                },
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            # MVP: swallow and keep worker alive; surface errors in UI later.
                            await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(1.0)

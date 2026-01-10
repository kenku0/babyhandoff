from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EventsRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["events"]

    async def add(
        self,
        *,
        shift_id: str,
        event_type: str,
        agent: str | None = None,
        message: str | None = None,
        run_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        doc = {
            "shift_id": shift_id,
            "type": event_type,
            "agent": agent,
            "message": message,
            "run_id": run_id,
            "meta": meta or {},
            "created_at": _now(),
        }
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def list_for_shift(self, shift_id: str, limit: int = 50) -> list[dict]:
        cursor = self._col.find(
            {"shift_id": shift_id},
            sort=[("created_at", -1)],
            limit=limit,
        )
        docs = [doc async for doc in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

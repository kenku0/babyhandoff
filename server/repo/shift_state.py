from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ShiftStateRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["shift_state"]

    async def upsert(self, shift_id: str, state: dict) -> str:
        state_doc = {**state, "shift_id": shift_id, "updated_at": _now()}
        result = await self._col.update_one(
            {"shift_id": shift_id},
            {"$set": state_doc, "$setOnInsert": {"created_at": _now()}},
            upsert=True,
        )
        if result.upserted_id is not None:
            return str(result.upserted_id)
        found = await self._col.find_one({"shift_id": shift_id}, projection={"_id": 1})
        if not found:
            raise RuntimeError("Failed to upsert shift_state")
        return str(found["_id"])

    async def get(self, shift_id: str) -> dict | None:
        return await self._col.find_one({"shift_id": shift_id})


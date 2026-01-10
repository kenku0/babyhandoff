from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ShiftsRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["shifts"]

    async def create_shift(self) -> str:
        now = _now()
        result = await self._col.insert_one({"created_at": now, "updated_at": now})
        return str(result.inserted_id)

    async def touch(self, shift_id: str) -> None:
        await self._col.update_one(
            {"_id": ObjectId(shift_id)},
            {"$set": {"updated_at": _now()}},
        )

    async def list_recent(self, limit: int = 20) -> list[dict]:
        cursor = self._col.find({}, sort=[("updated_at", -1)], limit=limit)
        docs = [doc async for doc in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

    async def get(self, shift_id: str) -> dict | None:
        doc = await self._col.find_one({"_id": ObjectId(shift_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

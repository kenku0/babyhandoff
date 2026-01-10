from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunsRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["runs"]

    async def create(self, shift_id: str, energy_override: str | None) -> str:
        doc = {
            "shift_id": shift_id,
            "status": "running",
            "energy_override": energy_override,
            "created_at": _now(),
            "updated_at": _now(),
        }
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def set_status(self, run_id: str, status: str) -> None:
        await self._col.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": status, "updated_at": _now()}},
        )

    async def get(self, run_id: str) -> dict | None:
        doc = await self._col.find_one({"_id": ObjectId(run_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def list_for_shift(self, shift_id: str, limit: int = 20) -> list[dict]:
        cursor = self._col.find(
            {"shift_id": shift_id},
            sort=[("created_at", -1)],
            limit=limit,
        )
        docs = [doc async for doc in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

from __future__ import annotations

from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DecisionsRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["decisions"]

    async def upsert_for_run(self, run_id: str, decision: dict) -> str:
        doc = {**decision, "run_id": run_id, "created_at": _now()}
        result = await self._col.update_one(
            {"run_id": run_id},
            {"$set": doc},
            upsert=True,
        )
        if result.upserted_id is not None:
            return str(result.upserted_id)
        found = await self._col.find_one({"run_id": run_id}, projection={"_id": 1})
        if not found:
            raise RuntimeError("Failed to upsert decision")
        return str(found["_id"])

    async def get_for_run(self, run_id: str) -> dict | None:
        doc = await self._col.find_one({"run_id": run_id})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

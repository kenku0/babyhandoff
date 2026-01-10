from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProposalsRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["proposals"]

    async def add(self, run_id: str, proposal: dict) -> str:
        doc = {**proposal, "run_id": run_id, "created_at": _now()}
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def list_for_run(self, run_id: str) -> list[dict]:
        cursor = self._col.find({"run_id": run_id}, sort=[("created_at", 1)])
        docs = [doc async for doc in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

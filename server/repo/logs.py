from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


LogType = Literal["note", "task", "deadline", "inventory"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LogsRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["logs"]

    async def add(self, shift_id: str, log_type: LogType, text: str) -> str:
        doc = {
            "shift_id": shift_id,
            "type": log_type,
            "text": text,
            "created_at": _now(),
        }
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def list_for_shift(self, shift_id: str, limit: int = 200) -> list[dict]:
        cursor = self._col.find(
            {"shift_id": shift_id},
            sort=[("created_at", 1)],
            limit=limit,
        )
        docs = [doc async for doc in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

    async def get(self, log_id: str) -> dict | None:
        doc = await self._col.find_one({"_id": ObjectId(log_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

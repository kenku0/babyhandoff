from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MessagesRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["messages"]

    async def add(
        self,
        *,
        shift_id: str,
        run_id: str,
        task_id: str | None,
        sender: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> str:
        doc = {
            "shift_id": shift_id,
            "run_id": run_id,
            "task_id": task_id,
            "sender": sender,
            "role": role,
            "content": content,
            "meta": meta or {},
            "created_at": _now(),
        }
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def list_for_run(self, run_id: str, limit: int = 400) -> list[dict]:
        cursor = self._col.find({"run_id": run_id}, sort=[("created_at", 1)], limit=limit)
        docs = [doc async for doc in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs


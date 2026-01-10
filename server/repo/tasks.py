from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TasksRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["tasks"]

    async def create(
        self,
        *,
        shift_id: str,
        run_id: str,
        name: str,
        agent_name: str,
        status: str = "queued",
        attempt: int = 0,
        max_attempts: int = 1,
        inputs: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        now = _now()
        doc = {
            "shift_id": shift_id,
            "run_id": run_id,
            "name": name,
            "agent_name": agent_name,
            "status": status,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "inputs": inputs or {},
            "outputs": {},
            "error": None,
            "started_at": now if status == "running" else None,
            "finished_at": None,
            "meta": meta or {},
            "created_at": now,
            "updated_at": now,
        }
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def set_status(self, task_id: str, status: str) -> None:
        await self._col.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": {"status": status, "updated_at": _now()}},
        )

    async def mark_running(self, task_id: str, *, attempt: int, inputs: dict[str, Any] | None = None) -> None:
        update: dict[str, Any] = {"status": "running", "attempt": attempt, "started_at": _now(), "updated_at": _now()}
        if inputs is not None:
            update["inputs"] = inputs
        await self._col.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": update},
        )

    async def mark_completed(self, task_id: str, *, outputs: dict[str, Any] | None = None) -> None:
        update: dict[str, Any] = {"status": "completed", "finished_at": _now(), "updated_at": _now()}
        if outputs is not None:
            update["outputs"] = outputs
        await self._col.update_one({"_id": ObjectId(task_id)}, {"$set": update})

    async def mark_failed(self, task_id: str, *, error: dict[str, Any], outputs: dict[str, Any] | None = None) -> None:
        update: dict[str, Any] = {"status": "failed", "error": error, "finished_at": _now(), "updated_at": _now()}
        if outputs is not None:
            update["outputs"] = outputs
        await self._col.update_one({"_id": ObjectId(task_id)}, {"$set": update})

    async def list_for_run(self, run_id: str) -> list[dict]:
        cursor = self._col.find({"run_id": run_id}, sort=[("created_at", 1)])
        docs = [doc async for doc in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

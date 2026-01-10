from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ContextPacksRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["context_packs"]

    async def create(
        self,
        *,
        shift_id: str,
        run_id: str,
        token_budget: int,
        pack: dict[str, Any],
    ) -> str:
        doc = {
            "shift_id": shift_id,
            "run_id": run_id,
            "token_budget": token_budget,
            **pack,
            "created_at": _now(),
        }
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def get_for_run(self, run_id: str) -> dict | None:
        doc = await self._col.find_one({"run_id": run_id})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc


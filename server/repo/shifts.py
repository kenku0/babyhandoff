from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)

def _default_title(now: datetime) -> str:
    # Use UTC for stored title; UI will show local time via client-side formatting.
    return f"Shift — {now.strftime('%b %d')}"

def _with_window_fields(doc: dict) -> None:
    created_at = doc.get("created_at")
    if isinstance(created_at, datetime):
        doc["created_at_iso"] = created_at.isoformat()
        end_at = created_at + timedelta(hours=12)
        doc["window_end_iso"] = end_at.isoformat()
    updated_at = doc.get("updated_at")
    if isinstance(updated_at, datetime):
        doc["updated_at_iso"] = updated_at.isoformat()


class ShiftsRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["shifts"]

    async def create_shift(self, *, title: str | None = None) -> str:
        now = _now()
        result = await self._col.insert_one(
            {
                "title": (title or "").strip() or _default_title(now),
                "created_at": now,
                "updated_at": now,
            }
        )
        return str(result.inserted_id)

    async def touch(self, shift_id: str) -> None:
        await self._col.update_one(
            {"_id": ObjectId(shift_id)},
            {"$set": {"updated_at": _now()}},
        )

    async def rename(self, shift_id: str, title: str) -> None:
        clean = (title or "").strip()
        if not clean:
            return
        await self._col.update_one(
            {"_id": ObjectId(shift_id)},
            {"$set": {"title": clean, "updated_at": _now()}},
        )

    async def list_recent(self, limit: int = 20) -> list[dict]:
        cursor = self._col.find({}, sort=[("updated_at", -1)], limit=limit)
        docs = [doc async for doc in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
            if not (str(d.get("title") or "").strip()) and isinstance(d.get("created_at"), datetime):
                d["title"] = _default_title(d["created_at"])
            _with_window_fields(d)
        return docs

    async def get(self, shift_id: str) -> dict | None:
        doc = await self._col.find_one({"_id": ObjectId(shift_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
            if not (str(doc.get("title") or "").strip()) and isinstance(doc.get("created_at"), datetime):
                doc["title"] = _default_title(doc["created_at"])
            _with_window_fields(doc)
        return doc

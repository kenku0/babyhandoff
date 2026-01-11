from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)

def _default_title(now: datetime) -> str:
    # Use UTC for stored title; UI will show local time via client-side formatting.
    return f"Shift — {now.strftime('%b %d')}"

def _shorten(s: str, max_len: int = 48) -> str:
    txt = " ".join((s or "").strip().split())
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 1].rstrip() + "…"

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

    async def create_shift(
        self,
        *,
        title: str | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        now = created_at or _now()
        upd = updated_at or now
        clean_title = (title or "").strip()
        result = await self._col.insert_one(
            {
                "title": clean_title or _default_title(now),
                "title_user_set": bool(clean_title),
                "created_at": now,
                "updated_at": upd,
                **(extra or {}),
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
            {"$set": {"title": clean, "title_user_set": True, "updated_at": _now()}},
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

    async def maybe_autotitle(self, shift_id: str, *, log_type: str, text: str) -> str | None:
        """
        Auto-title a shift from its first high-signal log, but only if the user
        hasn't renamed it.
        """
        doc = await self._col.find_one({"_id": ObjectId(shift_id)}, projection={"title": 1, "created_at": 1, "title_user_set": 1})
        if not doc:
            return None
        if bool(doc.get("title_user_set")):
            return None
        created_at = doc.get("created_at")
        if not isinstance(created_at, datetime):
            return None
        current = str(doc.get("title") or "").strip()
        if current and current != _default_title(created_at):
            return None

        prefix = created_at.strftime("%b %d")
        t = (log_type or "").strip().lower()
        if t == "note":
            return None
        base = _shorten(text or "", 56)
        if not base:
            return None

        if t == "deadline":
            topic = f"Deadline — {base}"
        elif t == "inventory":
            topic = f"Supplies — {base}"
        elif t == "task":
            topic = f"Task — {base}"
        else:
            topic = base

        new_title = f"{prefix} · {topic}"
        await self._col.update_one(
            {"_id": ObjectId(shift_id)},
            {"$set": {"title": new_title, "updated_at": _now()}},
        )
        return new_title

    async def delete_cascade(self, shift_id: str, *, include_shift_doc: bool = True) -> dict[str, int]:
        """
        Deletes a shift and its related documents across collections.

        Notes:
        - Most collections store `shift_id` as a string (not ObjectId).
        - `proposals` and `decisions` are keyed by `run_id`, so we derive run_ids first.
        """
        try:
            oid = ObjectId(shift_id)
        except InvalidId:
            return {"shift": 0}

        db = self._col.database

        runs_col = db["runs"]
        run_ids: list[str] = []
        async for r in runs_col.find({"shift_id": shift_id}, projection={"_id": 1}):
            rid = str(r.get("_id") or "")
            if rid:
                run_ids.append(rid)

        counts: dict[str, int] = {}
        # Collections keyed directly by shift_id (string).
        for name in ["logs", "tasks", "messages", "events", "context_packs", "artifacts", "shift_state", "runs"]:
            res = await db[name].delete_many({"shift_id": shift_id})
            counts[name] = int(getattr(res, "deleted_count", 0) or 0)

        # Collections keyed by run_id (string).
        if run_ids:
            res_p = await db["proposals"].delete_many({"run_id": {"$in": run_ids}})
            res_d = await db["decisions"].delete_many({"run_id": {"$in": run_ids}})
            counts["proposals"] = int(getattr(res_p, "deleted_count", 0) or 0)
            counts["decisions"] = int(getattr(res_d, "deleted_count", 0) or 0)
        else:
            counts["proposals"] = 0
            counts["decisions"] = 0

        if include_shift_doc:
            res_s = await self._col.delete_one({"_id": oid})
            counts["shift"] = int(getattr(res_s, "deleted_count", 0) or 0)
        else:
            counts["shift"] = 0

        return counts

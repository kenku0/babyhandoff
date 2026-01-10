from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactsRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["artifacts"]

    async def latest_for_shift(self, shift_id: str, kind: str) -> dict | None:
        doc = await self._col.find_one(
            {"shift_id": shift_id, "kind": kind},
            sort=[("version", -1)],
        )
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def create(
        self,
        *,
        shift_id: str,
        run_id: str,
        kind: str,
        markdown: str,
        diff: str | None,
        prev_artifact_id: str | None,
    ) -> str:
        latest = await self.latest_for_shift(shift_id, kind)
        next_version = int(latest["version"]) + 1 if latest else 1
        doc = {
            "shift_id": shift_id,
            "run_id": run_id,
            "kind": kind,
            "version": next_version,
            "markdown": markdown,
            "diff": diff,
            "diff_from_artifact_id": prev_artifact_id,
            "created_at": _now(),
        }
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def get(self, artifact_id: str) -> dict | None:
        doc = await self._col.find_one({"_id": ObjectId(artifact_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def get_for_run(self, run_id: str, kind: str) -> dict | None:
        doc = await self._col.find_one({"run_id": run_id, "kind": kind})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

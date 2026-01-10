from __future__ import annotations

from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


DEFAULT_AGENTS = [
    {"name": "Coordinator", "skills": ["orchestrate", "delegate", "merge"], "kind": "system"},
    {"name": "Normalizer", "skills": ["normalize_logs"], "kind": "system"},
    {"name": "RiskRadar", "skills": ["triage", "infer_energy", "detect_risks"], "kind": "watcher"},
    {"name": "SleepPlanner", "skills": ["plan_sleep_first"], "kind": "planner"},
    {"name": "ErrandsPlanner", "skills": ["plan_errands_first"], "kind": "planner"},
    {"name": "AdminPlanner", "skills": ["plan_admin_first"], "kind": "planner"},
    {"name": "Referee", "skills": ["score_rubric", "select_plan"], "kind": "judge"},
    {"name": "HandoffWriter", "skills": ["write_markdown"], "kind": "writer"},
]


class AgentsRepo:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._col = db["agents"]

    async def ensure_defaults(self) -> None:
        now = _now()
        for agent in DEFAULT_AGENTS:
            await self._col.update_one(
                {"name": agent["name"]},
                {"$setOnInsert": {**agent, "created_at": now}},
                upsert=True,
            )

    async def find_by_skill(self, skill: str) -> dict | None:
        doc = await self._col.find_one({"skills": skill})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def list_all(self) -> list[dict]:
        cursor = self._col.find({}, sort=[("kind", 1), ("name", 1)])
        docs = [doc async for doc in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """
    Create the minimal indexes needed for the MVP.

    This is safe to call on every startup; MongoDB is idempotent for identical
    index definitions.
    """
    await db["agents"].create_index("name", unique=True, name="name_unique")

    await db["shifts"].create_index([("updated_at", -1)], name="updated_at_desc")

    await db["logs"].create_index([("shift_id", 1), ("created_at", 1)], name="shift_created_at_asc")

    await db["shift_state"].create_index("shift_id", unique=True, name="shift_id_unique")

    await db["runs"].create_index([("shift_id", 1), ("created_at", -1)], name="shift_created_at_desc")

    await db["tasks"].create_index([("run_id", 1), ("created_at", 1)], name="run_created_at_asc")

    await db["events"].create_index([("shift_id", 1), ("created_at", -1)], name="shift_created_at_desc")
    await db["events"].create_index([("run_id", 1), ("created_at", -1)], name="run_created_at_desc")

    await db["messages"].create_index([("run_id", 1), ("created_at", 1)], name="run_created_at_asc")

    await db["context_packs"].create_index("run_id", unique=True, name="run_id_unique")

    await db["proposals"].create_index([("run_id", 1), ("created_at", 1)], name="run_created_at_asc")

    await db["decisions"].create_index("run_id", unique=True, name="run_id_unique")

    await db["artifacts"].create_index([("run_id", 1), ("kind", 1)], name="run_kind")
    await db["artifacts"].create_index(
        [("shift_id", 1), ("kind", 1), ("version", -1)],
        name="shift_kind_version_desc",
    )

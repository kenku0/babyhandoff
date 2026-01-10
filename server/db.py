from __future__ import annotations

from dataclasses import dataclass

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError

from server.settings import Settings


@dataclass
class Mongo:
    client: AsyncIOMotorClient
    db: AsyncIOMotorDatabase


async def connect_mongo(settings: Settings) -> Mongo:
    if not settings.mongodb_uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Create a .env in the repo root with "
            "MONGODB_URI and (optionally) MONGODB_DB."
        )
    # Fail fast if Atlas IP allowlist / credentials are wrong.
    client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
    try:
        await client.admin.command("ping")
    except OperationFailure as e:
        client.close()
        raise RuntimeError(
            "MongoDB authentication failed. Check your MONGODB_URI username/password, "
            "URL-encode any special characters in the password, and ensure the Atlas DB user exists. "
            "Tip: use the Atlas “Drivers” connection string ending with `/` and set the database via MONGODB_DB "
            "(or add `authSource=admin` if your URI includes a database path)."
        ) from e
    except ServerSelectionTimeoutError as e:
        client.close()
        raise RuntimeError(
            "MongoDB connection timed out. If you're using Atlas, confirm your IP is in the "
            "Atlas Network Access allowlist and that the cluster is reachable."
        ) from e
    except ConfigurationError as e:
        client.close()
        raise RuntimeError(
            "MongoDB URI is invalid. Re-copy the connection string from Atlas and ensure it "
            "starts with mongodb+srv:// or mongodb:// and contains a database user."
        ) from e
    db = client[settings.mongodb_db]
    return Mongo(client=client, db=db)

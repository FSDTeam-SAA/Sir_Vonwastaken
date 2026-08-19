"""
database/mongodb.py

Single MongoDB connection point + small generic helpers used by every
other module (`get_db`, `find_one`, `find`, `insert_one`, `upsert`,
`update_one`). Centralising this avoids each module opening its own
MongoClient and keeps collection access consistent.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from config.settings import settings
from utils.logger import logger


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not set. Add it to your .env file before using the database.")
    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)
    return client


@lru_cache(maxsize=1)
def get_db() -> Database:
    return get_client()[settings.mongodb_db_name]


def ping() -> bool:
    """Health check used by the /health API endpoint."""
    try:
        get_client().admin.command("ping")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"MongoDB ping failed: {exc}")
        return False


def get_collection(name: str) -> Collection:
    return get_db()[name]


def insert_one(collection: str, doc: Dict) -> str:
    result = get_collection(collection).insert_one(doc)
    return str(result.inserted_id)


def insert_many(collection: str, docs: Sequence[Dict]) -> List[str]:
    if not docs:
        return []
    result = get_collection(collection).insert_many(list(docs))
    return [str(_id) for _id in result.inserted_ids]


def find_one(collection: str, query: Dict, sort: Optional[List[Tuple[str, int]]] = None) -> Optional[Dict]:
    return get_collection(collection).find_one(query, sort=sort)


def find(
    collection: str,
    query: Optional[Dict] = None,
    limit: int = 0,
    sort: Optional[List[Tuple[str, int]]] = None,
) -> List[Dict]:
    cursor = get_collection(collection).find(query or {})
    if sort:
        cursor = cursor.sort(sort)
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)


def update_one(collection: str, query: Dict, update: Dict, upsert: bool = False):
    return get_collection(collection).update_one(query, update, upsert=upsert)


def upsert(collection: str, query: Dict, doc: Dict) -> None:
    get_collection(collection).update_one(query, {"$set": doc}, upsert=True)


def count(collection: str, query: Optional[Dict] = None) -> int:
    return get_collection(collection).count_documents(query or {})


def strip_id(doc: Optional[Dict]) -> Optional[Dict]:
    """Removes the raw ObjectId `_id` field so a Mongo doc is safely JSON-serializable via FastAPI."""
    if doc is None:
        return None
    doc.pop("_id", None)
    return doc


def strip_ids(docs: List[Dict]) -> List[Dict]:
    for doc in docs:
        doc.pop("_id", None)
    return docs


def ensure_indexes() -> None:
    """
    Creates the indexes the app relies on for correctness/performance.
    Safe to call repeatedly (create_index is idempotent). Called once on
    FastAPI startup from main.py.
    """
    db = get_db()
    db.raw_content.create_index([("platform", 1), ("external_id", 1)], unique=True)
    db.processed_content.create_index([("platform", 1), ("external_id", 1)], unique=True)
    db.processed_content.create_index([("processed_at", -1)])
    db.creator_profiles.create_index("channel_id", unique=True)
    db.trend_candidates.create_index("content_id", unique=True)
    db.trend_candidates.create_index([("channel_id", 1), ("score", -1)])
    db.generated_content.create_index([("trend_id", 1), ("generated_at", -1)])
    db.emails.create_index("external_id", unique=True)
    db.email_drafts.create_index([("status", 1), ("created_at", -1)])
    db.embeddings.create_index([("namespace", 1), ("ref_id", 1)], unique=True)
    logger.info("MongoDB indexes ensured.")

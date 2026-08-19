"""
content_similarity_check/embedding_search.py

Module 3 — Similarity Analysis (embedding generation side).

Converts newly discovered/processed content into embeddings and stores
them in the vector store (database/vector_store.py), so
vector_search.py / similarity_engine.py can compare them against the
creator profile embedding.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from database.mongodb import find, get_collection
from database.vector_store import get_embedding as get_stored_embedding
from database.vector_store import store_embedding
from utils.llm_client import get_embedding
from utils.logger import logger

_NAMESPACE = "processed_content"


def embed_content_item(external_id: str, text: str, metadata: Optional[Dict] = None) -> List[float]:
    """Embeds one piece of processed content and stores the vector."""
    vector = get_embedding(text)
    store_embedding(_NAMESPACE, external_id, vector, metadata=metadata or {})
    return vector


def get_content_embedding(external_id: str) -> Optional[List[float]]:
    return get_stored_embedding(_NAMESPACE, external_id)


def embed_pending_content(limit: int = 200) -> int:
    """
    Module 3 batch runner: embeds every processed_content document that
    doesn't have a stored vector yet (tracked via an `embedded` flag on
    the processed_content doc itself, to avoid re-scanning the vector
    store on every call).
    """
    docs = find("processed_content", {"embedded": {"$ne": True}}, limit=limit)
    count = 0
    for doc in docs:
        text = doc.get("text_for_ai") or doc.get("title", "")
        if not text.strip():
            continue
        embed_content_item(
            doc["external_id"],
            text,
            metadata={"platform": doc.get("platform"), "title": doc.get("title"), "channel_id": doc.get("channel_id")},
        )
        get_collection("processed_content").update_one({"_id": doc["_id"]}, {"$set": {"embedded": True}})
        count += 1

    logger.info(f"Embedded {count} pending processed_content items.")
    return count

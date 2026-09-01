"""
content_similarity_check/embedding_search.py

Module 3 — Similarity Analysis (embedding generation side).

Converts newly discovered/processed content into embeddings and stores
them in the vector store (database/vector_store.py), so
vector_search.py / similarity_engine.py can compare them against the
creator profile embedding.

Supports two embedding backends:
  1. Local sentence-transformers (free, offline, fast)
  2. OpenAI embeddings API (higher quality, requires API key)
"""
from __future__ import annotations

from typing import Dict, List, Optional

from database.mongodb import find, get_collection
from database.vector_store import get_embedding as get_stored_embedding
from database.vector_store import search, store_embedding
from utils.logger import logger

_NAMESPACE = "processed_content"
_SENTENCE_TRANSFORMER = None


def _get_sentence_transformer():
    """Lazy-load sentence-transformers model."""
    global _SENTENCE_TRANSFORMER
    if _SENTENCE_TRANSFORMER is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model (first run may take a moment)...")
            _SENTENCE_TRANSFORMER = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to OpenAI embeddings")
            _SENTENCE_TRANSFORMER = None
    return _SENTENCE_TRANSFORMER


def embed_content_item(external_id: str, text: str, metadata: Optional[Dict] = None, use_local: bool = True) -> Optional[List[float]]:
    """
    Embeds one piece of processed content and stores the vector.
    
    Args:
        external_id: Content identifier
        text: Text to embed
        metadata: Optional metadata to store with embedding
        use_local: Use local sentence-transformers (True) or OpenAI (False)
    
    Returns:
        The embedding vector, or None on failure
    """
    try:
        if use_local:
            model = _get_sentence_transformer()
            if model:
                vector = model.encode(text, convert_to_tensor=False).tolist()
            else:
                # Fallback to OpenAI
                from utils.llm_client import get_embedding
                vector = get_embedding(text)
        else:
            from utils.llm_client import get_embedding
            vector = get_embedding(text)
        
        if vector:
            store_embedding(_NAMESPACE, external_id, vector, metadata=metadata or {})
            return vector
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error embedding content {external_id}: {exc}")
        return None


def get_content_embedding(external_id: str) -> Optional[List[float]]:
    """Retrieve stored embedding for content."""
    return get_stored_embedding(_NAMESPACE, external_id)


def embed_pending_content(limit: int = 200, use_local: bool = True) -> int:
    """
    Module 3 batch runner: embeds every processed_content document that
    doesn't have a stored vector yet (tracked via an `embedded` flag on
    the processed_content doc itself, to avoid re-scanning the vector
    store on every call).

    Only marks a document as `embedded` when a real vector was produced.
    If the embedding call fails (bad API key, network issue, etc.), the
    document is left unflagged so it's automatically retried on the next
    call instead of being silently skipped forever.
    
    Args:
        limit: Maximum number of documents to process
        use_local: Use local sentence-transformers (True) or OpenAI (False)
    
    Returns:
        Number of items successfully embedded
    """
    docs = find("processed_content", {"embedded": {"$ne": True}}, limit=limit)
    count = 0
    failures = 0
    for doc in docs:
        text = doc.get("text_for_ai") or doc.get("title", "")
        if not text.strip():
            continue
        vector = embed_content_item(
            doc["external_id"],
            text,
            metadata={"platform": doc.get("platform"), "title": doc.get("title"), "channel_id": doc.get("channel_id")},
            use_local=use_local,
        )
        if vector:
            get_collection("processed_content").update_one({"_id": doc["_id"]}, {"$set": {"embedded": True}})
            count += 1
        else:
            failures += 1

    if failures:
        logger.warning(f"{failures} items failed to embed (left unflagged for retry)")
    logger.info(f"Embedded {count} pending processed_content items (use_local={use_local}).")
    return count


def find_similar_content(
    content_external_id: str,
    top_k: int = 10,
    min_similarity: float = 0.5,
) -> List[tuple]:
    """
    Find content similar to a given piece of content.
    
    Args:
        content_external_id: ID of the reference content
        top_k: Number of results to return
        min_similarity: Minimum similarity score (0-1)
    
    Returns:
        List of (ref_id, similarity_score, metadata) tuples
    """
    query_embedding = get_stored_embedding(_NAMESPACE, content_external_id)
    if not query_embedding:
        logger.warning(f"No embedding found for content {content_external_id}")
        return []
    
    results = search(_NAMESPACE, query_embedding, top_k=top_k, exclude_ref_id=content_external_id)
    filtered = [(ref_id, score, metadata) for ref_id, score, metadata in results if score >= min_similarity]
    
    return filtered
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

import math
import numbers
from typing import Dict, List, Optional, Sequence, Tuple

from config.settings import settings
from database.mongodb import find, get_collection
from database.vector_store import get_embedding as get_stored_embedding
from database.vector_store import get_embedding_record as get_stored_embedding_record
from database.vector_store import search, store_embedding
from utils.logger import logger

_NAMESPACE = "processed_content"
_LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
_SENTENCE_TRANSFORMER = None
_SENTENCE_TRANSFORMER_UNAVAILABLE = False


def _get_sentence_transformer():
    """Lazy-load sentence-transformers model."""
    global _SENTENCE_TRANSFORMER, _SENTENCE_TRANSFORMER_UNAVAILABLE
    if _SENTENCE_TRANSFORMER is not None:
        return _SENTENCE_TRANSFORMER
    if _SENTENCE_TRANSFORMER_UNAVAILABLE:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformers model (first run may take a moment)...")
        _SENTENCE_TRANSFORMER = SentenceTransformer(_LOCAL_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 - use the configured API fallback
        _SENTENCE_TRANSFORMER_UNAVAILABLE = True
        logger.warning(
            "Could not load sentence-transformers ({}); falling back to OpenAI embeddings",
            exc,
        )
    return _SENTENCE_TRANSFORMER


def _normalise_vector(vector: object) -> Optional[List[float]]:
    """Return a finite one-dimensional vector, or None for invalid model output."""
    if not isinstance(vector, (list, tuple)) or not vector:
        return None
    try:
        normalised = [float(value) for value in vector]
    except (TypeError, ValueError):
        return None
    return normalised if all(math.isfinite(value) for value in normalised) else None


def _normalise_vectors(raw_vectors: object, expected_count: int) -> List[List[float]]:
    """Validate a batch response without silently misaligning texts and vectors."""
    if hasattr(raw_vectors, "tolist"):
        raw_vectors = raw_vectors.tolist()
    if not isinstance(raw_vectors, (list, tuple)):
        return []
    if expected_count == 1 and raw_vectors and isinstance(raw_vectors[0], numbers.Real):
        raw_vectors = [raw_vectors]
    if len(raw_vectors) != expected_count:
        return []

    vectors = [_normalise_vector(vector) for vector in raw_vectors]
    if not all(vectors):
        return []
    normalised_vectors = [vector for vector in vectors if vector is not None]
    if len({len(vector) for vector in normalised_vectors}) != 1:
        return []
    return normalised_vectors


def generate_embeddings(
    texts: Sequence[str],
    use_local: Optional[bool] = None,
) -> Tuple[List[List[float]], Optional[str]]:
    """Generate embeddings in the configured shared embedding space.

    Both creator profiles and collected content call this function.  Keeping
    the backend selection in one place prevents cosine similarity from
    comparing vectors produced by different models (for example 384-D local
    vectors against 1536-D OpenAI vectors).

    Returns a ``(vectors, embedding_space)`` tuple.  The space is recorded
    with each stored vector for diagnostics and safe future migrations.
    """
    clean_texts = [
        text.strip()
        for text in texts
        if isinstance(text, str) and text.strip()
    ]
    if not clean_texts:
        return [], None

    resolved_use_local = (
        settings.use_local_embeddings if use_local is None else use_local
    )
    if resolved_use_local:
        model = _get_sentence_transformer()
        if model is not None:
            try:
                vectors = _normalise_vectors(
                    model.encode(clean_texts, convert_to_tensor=False),
                    len(clean_texts),
                )
                if vectors:
                    return vectors, f"sentence-transformers:{_LOCAL_MODEL_NAME}"
                logger.error("Local embedding model returned invalid vectors; falling back to OpenAI")
            except Exception as exc:  # noqa: BLE001 - preserve the API fallback
                logger.exception("Local embedding generation failed: {}", exc)

    from utils.llm_client import get_embeddings_batch

    vectors = _normalise_vectors(get_embeddings_batch(clean_texts), len(clean_texts))
    if not vectors:
        logger.error(
            "OpenAI embedding generation returned no usable vectors for {} texts",
            len(clean_texts),
        )
        return [], None
    return vectors, f"openai:{settings.openai_embedding_model}"


def embed_content_item(
    external_id: str,
    text: str,
    metadata: Optional[Dict] = None,
    use_local: Optional[bool] = None,
) -> Optional[List[float]]:
    """
    Embeds one piece of processed content and stores the vector.
    
    Args:
        external_id: Content identifier
        text: Text to embed
        metadata: Optional metadata to store with embedding
        use_local: Use local sentence-transformers (True), OpenAI (False),
            or the configured default (None)
    
    Returns:
        The embedding vector, or None on failure
    """
    try:
        vectors, embedding_space = generate_embeddings([text], use_local=use_local)
        if vectors and embedding_space:
            vector = vectors[0]
            vector_metadata = dict(metadata or {})
            vector_metadata.update(
                {
                    "embedding_space": embedding_space,
                    "embedding_dimension": len(vector),
                }
            )
            store_embedding(_NAMESPACE, external_id, vector, metadata=vector_metadata)
            return vector
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error embedding content {external_id}: {exc}")
        return None


def get_content_embedding(external_id: str) -> Optional[List[float]]:
    """Retrieve stored embedding for content."""
    return get_stored_embedding(_NAMESPACE, external_id)


def get_content_embedding_record(external_id: str) -> Optional[Tuple[List[float], Dict]]:
    """Retrieve a content vector and its compatibility metadata atomically."""
    return get_stored_embedding_record(_NAMESPACE, external_id)


def embed_pending_content(
    limit: int = 200,
    use_local: Optional[bool] = None,
    force: bool = False,
) -> int:
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
        use_local: Use local sentence-transformers (True), OpenAI (False),
            or the configured default (None)
        force: Re-embed already embedded documents. Use after deliberately
            changing ``USE_LOCAL_EMBEDDINGS`` so legacy vectors migrate into
            the same space as a rebuilt creator profile.
    
    Returns:
        Number of items successfully embedded
    """
    query = {} if force else {"embedded": {"$ne": True}}
    docs = find("processed_content", query, limit=limit)
    count = 0
    failures = 0
    for doc in docs:
        external_id = doc.get("external_id")
        if not isinstance(external_id, str) or not external_id:
            logger.warning(
                "Skipping processed_content document without a valid external_id: {}",
                doc.get("_id"),
            )
            failures += 1
            continue

        text = doc.get("text_for_ai") or doc.get("title", "")
        if not isinstance(text, str):
            text = str(text or "")
        if not text.strip():
            continue
        vector = embed_content_item(
            external_id,
            text,
            metadata={
                "platform": doc.get("platform"),
                "title": doc.get("title"),
                "channel_id": doc.get("channel_id"),
            },
            use_local=use_local,
        )
        if vector:
            get_collection("processed_content").update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedded": True}},
            )
            count += 1
        else:
            failures += 1

    if failures:
        logger.warning(f"{failures} items failed to embed (left unflagged for retry)")
    logger.info(
        "Embedded {} processed_content items (force={}, use_local={}).",
        count,
        force,
        settings.use_local_embeddings if use_local is None else use_local,
    )
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
    query_record = get_stored_embedding_record(_NAMESPACE, content_external_id)
    if not query_record:
        logger.warning(f"No embedding found for content {content_external_id}")
        return []

    query_embedding, metadata = query_record
    results = search(
        _NAMESPACE,
        query_embedding,
        top_k=top_k,
        exclude_ref_id=content_external_id,
        embedding_space=metadata.get("embedding_space"),
    )
    filtered = [
        (ref_id, score, metadata)
        for ref_id, score, metadata in results
        if score >= min_similarity
    ]
    
    return filtered

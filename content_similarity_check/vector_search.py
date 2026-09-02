"""
content_similarity_check/vector_search.py

Module 3 — Similarity Analysis (search side).

Given a creator's profile embedding, finds which pieces of collected
content are most similar to that creator's established style. Wraps
database/vector_store.py's brute-force cosine search and
similarity_engine.check_similarity for single-pair checks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from content_similarity_check.similarity_engine import check_similarity
from creator_profile.build_creator_profile import get_creator_profile_embedding_record
from database.vector_store import search as vector_search
from utils.logger import logger


def find_content_similar_to_creator(channel_id: str, top_k: int = 20) -> List[Tuple[str, float, Dict]]:
    """
    Module 3: returns the top_k processed_content items (by external_id)
    most similar to the given creator's profile embedding, as
    (external_id, similarity_score, metadata) tuples.
    """
    profile_record = get_creator_profile_embedding_record(channel_id)
    if not profile_record:
        logger.warning(f"No creator profile embedding found for channel_id={channel_id}. Build the profile first.")
        return []

    profile_embedding, metadata = profile_record
    results = vector_search(
        "processed_content",
        profile_embedding,
        top_k=top_k,
        embedding_space=metadata.get("embedding_space"),
    )
    return results


def score_content_against_creator(
    content_embedding: List[float],
    channel_id: str,
    content_metadata: Optional[Dict] = None,
) -> Optional[float]:
    """Return pairwise similarity, or None when the vectors cannot be compared safely."""
    profile_record = get_creator_profile_embedding_record(channel_id)
    if not profile_record or not content_embedding:
        return None

    profile_embedding, profile_metadata = profile_record
    if len(content_embedding) != len(profile_embedding):
        logger.warning(
            "Cannot compare content and creator embeddings with dimensions {} and {}.",
            len(content_embedding),
            len(profile_embedding),
        )
        return None

    content_space = (content_metadata or {}).get("embedding_space")
    profile_space = profile_metadata.get("embedding_space")
    if content_space and profile_space and content_space != profile_space:
        logger.warning(
            "Cannot compare content embedding space '{}' with creator embedding space '{}'.",
            content_space,
            profile_space,
        )
        return None
    return check_similarity(content_embedding, profile_embedding)

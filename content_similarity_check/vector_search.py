"""
content_similarity_check/vector_search.py

Module 3 — Similarity Analysis (search side).

Given a creator's profile embedding, finds which pieces of collected
content are most similar to that creator's established style. Wraps
database/vector_store.py's brute-force cosine search and
similarity_engine.check_similarity for single-pair checks.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from content_similarity_check.similarity_engine import check_similarity
from creator_profile.build_creator_profile import get_creator_profile_embedding
from database.vector_store import search as vector_search
from utils.logger import logger


def find_content_similar_to_creator(channel_id: str, top_k: int = 20) -> List[Tuple[str, float, Dict]]:
    """
    Module 3: returns the top_k processed_content items (by external_id)
    most similar to the given creator's profile embedding, as
    (external_id, similarity_score, metadata) tuples.
    """
    profile_embedding = get_creator_profile_embedding(channel_id)
    if not profile_embedding:
        logger.warning(f"No creator profile embedding found for channel_id={channel_id}. Build the profile first.")
        return []

    results = vector_search("processed_content", profile_embedding, top_k=top_k)
    return results


def score_content_against_creator(content_embedding: List[float], channel_id: str) -> float:
    """Module 3: single-pair similarity score between one content embedding and a creator's profile."""
    profile_embedding = get_creator_profile_embedding(channel_id)
    if not profile_embedding or not content_embedding:
        return 0.0
    return check_similarity(content_embedding, profile_embedding)

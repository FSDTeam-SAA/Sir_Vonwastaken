"""Pairwise cosine similarity helpers."""
from __future__ import annotations

import math
from typing import Sequence


def check_similarity(
    new_content_embedding: Sequence[float],
    creator_profile_embedding: Sequence[float],
) -> float:
    """Return cosine similarity for two compatible finite vectors.

    Incompatible vectors are not partially compared: the former ``zip``
    implementation silently truncated 384-dimensional local embeddings when
    paired with 1536-dimensional OpenAI embeddings, producing meaningless
    ranking scores.
    """
    if not new_content_embedding or len(new_content_embedding) != len(creator_profile_embedding):
        return 0.0
    try:
        content = [float(value) for value in new_content_embedding]
        creator = [float(value) for value in creator_profile_embedding]
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(value) for value in content + creator):
        return 0.0

    dot_product = sum(a * b for a, b in zip(content, creator))
    magnitude_content = math.sqrt(sum(value**2 for value in content))
    magnitude_creator = math.sqrt(sum(value**2 for value in creator))
    if magnitude_content == 0 or magnitude_creator == 0:
        return 0.0

    score = dot_product / (magnitude_content * magnitude_creator)
    return max(min(score, 1.0), -1.0)

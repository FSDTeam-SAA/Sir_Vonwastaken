"""
database/vector_store.py

Lightweight vector store built on top of MongoDB (`embeddings` collection)
with brute-force cosine-similarity search done in Python/NumPy.

Note on scope: the technical proposal mentions Qdrant/pgVector as future
options for a production deployment at scale. Neither ships in this
project's requirements.txt today, and adding a new external vector DB
service wasn't something I wanted to silently introduce. This module
gives every other component (creator_profile, content_similarity_check)
a real, working `store_embedding` / `search` API now; swapping the
implementation for Qdrant later only requires changing this one file.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from database.mongodb import get_collection
from utils.logger import logger

_COLLECTION = "embeddings"


def store_embedding(namespace: str, ref_id: str, vector: List[float], metadata: Optional[Dict] = None) -> None:
    """
    Upserts an embedding. `namespace` groups vectors logically
    (e.g. "creator_profile", "processed_content"); `ref_id` is the id of
    the thing the vector represents (channel_id, content_id, ...).
    """
    if not vector:
        logger.warning(f"store_embedding called with empty vector for {namespace}/{ref_id}, skipping.")
        return
    get_collection(_COLLECTION).update_one(
        {"namespace": namespace, "ref_id": ref_id},
        {
            "$set": {
                "namespace": namespace,
                "ref_id": ref_id,
                "vector": vector,
                "metadata": metadata or {},
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )


def get_embedding(namespace: str, ref_id: str) -> Optional[List[float]]:
    doc = get_collection(_COLLECTION).find_one({"namespace": namespace, "ref_id": ref_id})
    return doc["vector"] if doc else None


def _cosine_sim_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * query_norm
    denom[denom == 0] = 1e-10
    return (matrix @ query) / denom


def search(namespace: str, query_vector: List[float], top_k: int = 10, exclude_ref_id: Optional[str] = None) -> List[Tuple[str, float, Dict]]:
    """
    Brute-force cosine similarity search over all vectors in `namespace`.
    Returns a list of (ref_id, score, metadata) sorted by score descending.
    Fine for the data volumes this project deals with (thousands, not
    millions, of candidate trends per run); revisit if that changes.
    """
    if not query_vector:
        return []
    docs = list(get_collection(_COLLECTION).find({"namespace": namespace}))
    if exclude_ref_id:
        docs = [d for d in docs if d["ref_id"] != exclude_ref_id]
    if not docs:
        return []

    matrix = np.array([d["vector"] for d in docs], dtype=float)
    query = np.array(query_vector, dtype=float)
    scores = _cosine_sim_matrix(query, matrix)

    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [(d["ref_id"], float(score), d.get("metadata", {})) for d, score in ranked]


def delete_embedding(namespace: str, ref_id: str) -> None:
    get_collection(_COLLECTION).delete_one({"namespace": namespace, "ref_id": ref_id})

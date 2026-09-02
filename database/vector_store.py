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


def _as_valid_vector(vector: object) -> Optional[np.ndarray]:
    """Coerce an embedding into a finite, one-dimensional float array."""
    if isinstance(vector, (str, bytes, dict)):
        return None
    try:
        array = np.asarray(vector, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array


def store_embedding(
    namespace: str,
    ref_id: str,
    vector: List[float],
    metadata: Optional[Dict] = None,
) -> None:
    """
    Upserts an embedding. `namespace` groups vectors logically
    (e.g. "creator_profile", "processed_content"); `ref_id` is the id of
    the thing the vector represents (channel_id, content_id, ...).
    """
    validated_vector = _as_valid_vector(vector)
    if validated_vector is None:
        logger.warning(
            "store_embedding called with an invalid vector for {}/{}; skipping.",
            namespace,
            ref_id,
        )
        return
    get_collection(_COLLECTION).update_one(
        {"namespace": namespace, "ref_id": ref_id},
        {
            "$set": {
                "namespace": namespace,
                "ref_id": ref_id,
                "vector": validated_vector.tolist(),
                "metadata": metadata or {},
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )


def get_embedding_record(namespace: str, ref_id: str) -> Optional[Tuple[List[float], Dict]]:
    """Return a validated vector and its metadata from one database read."""
    doc = get_collection(_COLLECTION).find_one({"namespace": namespace, "ref_id": ref_id})
    if not doc:
        return None
    vector = _as_valid_vector(doc.get("vector"))
    if vector is None:
        logger.warning("Ignoring invalid stored embedding for {}/{}.", namespace, ref_id)
        return None
    metadata = doc.get("metadata")
    return vector.tolist(), (metadata if isinstance(metadata, dict) else {})


def get_embedding(namespace: str, ref_id: str) -> Optional[List[float]]:
    record = get_embedding_record(namespace, ref_id)
    return record[0] if record else None


def get_embedding_metadata(namespace: str, ref_id: str) -> Dict:
    """Return stored embedding metadata, if any, without exposing Mongo internals."""
    record = get_embedding_record(namespace, ref_id)
    return record[1] if record else {}


def _cosine_sim_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * query_norm
    denom[denom == 0] = 1e-10
    return (matrix @ query) / denom


def search(
    namespace: str,
    query_vector: List[float],
    top_k: int = 10,
    exclude_ref_id: Optional[str] = None,
    embedding_space: Optional[str] = None,
) -> List[Tuple[str, float, Dict]]:
    """
    Brute-force cosine similarity search over all vectors in `namespace`.
    Returns a list of (ref_id, score, metadata) sorted by score descending.
    Fine for the data volumes this project deals with (thousands, not
    millions, of candidate trends per run); revisit if that changes.
    """
    query = _as_valid_vector(query_vector)
    if query is None:
        logger.warning(
            "Vector search skipped because the query vector for namespace {} is invalid.",
            namespace,
        )
        return []
    if top_k <= 0:
        return []
    docs = list(get_collection(_COLLECTION).find({"namespace": namespace}))
    if exclude_ref_id:
        docs = [d for d in docs if d.get("ref_id") != exclude_ref_id]
    if not docs:
        return []

    compatible_docs = []
    compatible_vectors = []
    invalid_count = 0
    dimension_mismatch_count = 0
    space_mismatch_count = 0
    for doc in docs:
        ref_id = doc.get("ref_id")
        if not isinstance(ref_id, str) or not ref_id:
            invalid_count += 1
            continue
        vector = _as_valid_vector(doc.get("vector"))
        if vector is None:
            invalid_count += 1
            continue
        if vector.size != query.size:
            dimension_mismatch_count += 1
            continue

        metadata = doc.get("metadata")
        candidate_space = metadata.get("embedding_space") if isinstance(metadata, dict) else None
        # Legacy vectors have no space metadata. Keep them when dimensions
        # match so deployments can recover without a destructive migration.
        if embedding_space and candidate_space and candidate_space != embedding_space:
            space_mismatch_count += 1
            continue

        compatible_docs.append(doc)
        compatible_vectors.append(vector)

    skipped_count = invalid_count + dimension_mismatch_count + space_mismatch_count
    if skipped_count:
        logger.warning(
            "Skipped {} incompatible embedding(s) in {} search (query_dim={}, invalid={}, "
            "dimension_mismatch={}, space_mismatch={}).",
            skipped_count,
            namespace,
            query.size,
            invalid_count,
            dimension_mismatch_count,
            space_mismatch_count,
        )
    if not compatible_docs:
        logger.warning(
            "No compatible embeddings found in {} for query_dim={} and embedding_space={}.",
            namespace,
            query.size,
            embedding_space or "<legacy/unspecified>",
        )
        return []

    matrix = np.vstack(compatible_vectors)
    scores = _cosine_sim_matrix(query, matrix)

    ranked = sorted(
        zip(compatible_docs, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_k]
    return [
        (
            doc["ref_id"],
            float(score),
            doc["metadata"] if isinstance(doc.get("metadata"), dict) else {},
        )
        for doc, score in ranked
    ]


def delete_embedding(namespace: str, ref_id: str) -> None:
    get_collection(_COLLECTION).delete_one({"namespace": namespace, "ref_id": ref_id})

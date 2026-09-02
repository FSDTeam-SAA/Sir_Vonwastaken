"""Regression tests for mixed embedding dimensions and model spaces."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from database import vector_store


class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def find(self, query):
        return [doc for doc in self._docs if doc.get("namespace") == query.get("namespace")]

    def find_one(self, query, projection=None):
        del projection
        return next(
            (
                doc
                for doc in self._docs
                if doc.get("namespace") == query.get("namespace") and doc.get("ref_id") == query.get("ref_id")
            ),
            None,
        )


class VectorStoreSimilarityTests(unittest.TestCase):
    def test_embedding_record_returns_vector_and_metadata_from_one_lookup(self) -> None:
        docs = [
            {
                "namespace": "creator_profile",
                "ref_id": "channel-1",
                "vector": [1.0, 0.0],
                "metadata": {
                    "embedding_space": "openai:test-model",
                    "embedding_dimension": 2,
                },
            }
        ]

        with patch.object(vector_store, "get_collection", return_value=_FakeCollection(docs)):
            record = vector_store.get_embedding_record("creator_profile", "channel-1")

        self.assertEqual(
            record,
            (
                [1.0, 0.0],
                {
                    "embedding_space": "openai:test-model",
                    "embedding_dimension": 2,
                },
            ),
        )

    def test_search_skips_mixed_dimensions_without_crashing(self) -> None:
        """A legacy 384-D vector must not break a search using a 1536-D profile."""
        docs = [
            {
                "namespace": "processed_content",
                "ref_id": "compatible",
                "vector": [1.0, 0.0, 0.0],
                "metadata": {"embedding_space": "openai:test-model"},
            },
            {
                "namespace": "processed_content",
                "ref_id": "wrong-dimension",
                "vector": [1.0, 0.0],
                "metadata": {"embedding_space": "openai:test-model"},
            },
        ]

        with patch.object(vector_store, "get_collection", return_value=_FakeCollection(docs)):
            results = vector_store.search(
                "processed_content",
                [1.0, 0.0, 0.0],
                top_k=10,
                embedding_space="openai:test-model",
            )

        self.assertEqual([ref_id for ref_id, _, _ in results], ["compatible"])

    def test_search_rejects_explicit_space_mismatch_but_keeps_legacy_vectors(self) -> None:
        """Model provenance is enforced when known, while untagged legacy data remains searchable."""
        docs = [
            {
                "namespace": "processed_content",
                "ref_id": "same-space",
                "vector": [1.0, 0.0, 0.0],
                "metadata": {"embedding_space": "openai:test-model"},
            },
            {
                "namespace": "processed_content",
                "ref_id": "other-space",
                "vector": [1.0, 0.0, 0.0],
                "metadata": {"embedding_space": "sentence-transformers:test-model"},
            },
            {
                "namespace": "processed_content",
                "ref_id": "legacy-untagged",
                "vector": [0.0, 1.0, 0.0],
                "metadata": {},
            },
        ]

        with patch.object(vector_store, "get_collection", return_value=_FakeCollection(docs)):
            results = vector_store.search(
                "processed_content",
                [1.0, 0.0, 0.0],
                top_k=10,
                embedding_space="openai:test-model",
            )

        self.assertEqual([ref_id for ref_id, _, _ in results], ["same-space", "legacy-untagged"])

    def test_mismatch_warning_is_rendered_with_loguru_placeholders(self) -> None:
        """Diagnostics should contain values, not literal printf-style placeholders."""
        docs = [
            {
                "namespace": "processed_content",
                "ref_id": "compatible",
                "vector": [1.0, 0.0, 0.0],
                "metadata": {"embedding_space": "openai:test-model"},
            },
            {
                "namespace": "processed_content",
                "ref_id": "wrong-dimension",
                "vector": [1.0, 0.0],
                "metadata": {"embedding_space": "openai:test-model"},
            },
        ]
        messages = []
        sink_id = vector_store.logger.add(lambda message: messages.append(str(message)), format="{message}")
        try:
            with patch.object(vector_store, "get_collection", return_value=_FakeCollection(docs)):
                vector_store.search(
                    "processed_content",
                    [1.0, 0.0, 0.0],
                    embedding_space="openai:test-model",
                )
        finally:
            vector_store.logger.remove(sink_id)

        rendered = "".join(messages)
        self.assertIn("Skipped 1 incompatible embedding(s)", rendered)
        self.assertNotIn("%d", rendered)
        self.assertNotIn("%s", rendered)


if __name__ == "__main__":
    unittest.main()

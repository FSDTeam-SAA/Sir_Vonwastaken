"""Regression tests for the shared creator/content embedding contract."""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import creator_profile.build_creator_profile as profile_builder
from content_similarity_check import embedding_search, vector_search


class EmbeddingBackendContractTests(unittest.TestCase):
    def test_explicit_creator_profile_records_shared_embedding_metadata(self) -> None:
        shared_generate = Mock(return_value=([[1.0, 0.0, 0.0]], "sentence-transformers:test-model"))

        with patch.object(profile_builder, "generate_embeddings", shared_generate, create=True), patch.object(
            profile_builder, "get_embedding", return_value=[9.0, 9.0, 9.0], create=True
        ), patch.object(profile_builder, "store_embedding") as store_embedding, patch.object(
            profile_builder, "upsert"
        ):
            result = profile_builder.build_creator_profile(
                {
                    "creator_id": "creator-1",
                    "titles": ["A title"],
                    "descriptions": [],
                    "transcripts": [],
                    "categories": [],
                    "topics": [],
                }
            )

        shared_generate.assert_called_once()
        self.assertEqual(result, [1.0, 0.0, 0.0])
        _, _, stored_vector = store_embedding.call_args.args
        metadata = store_embedding.call_args.kwargs["metadata"]
        self.assertEqual(stored_vector, [1.0, 0.0, 0.0])
        self.assertEqual(metadata["embedding_space"], "sentence-transformers:test-model")
        self.assertEqual(metadata["embedding_dimension"], 3)

    def test_channel_creator_profile_uses_the_same_backend_and_metadata(self) -> None:
        videos = [
            {
                "title": "First",
                "description": "Description one",
                "view_count": 100,
                "like_count": 10,
            },
            {
                "title": "Second",
                "description": "Description two",
                "view_count": 200,
                "like_count": 20,
            },
        ]
        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
        shared_generate = Mock(return_value=(vectors, "openai:test-model"))

        with patch.object(profile_builder, "get_channel_videos", return_value=videos), patch.object(
            profile_builder, "analyze_content", return_value={}
        ), patch.object(profile_builder, "generate_embeddings", shared_generate, create=True), patch.object(
            profile_builder, "get_embeddings_batch", return_value=vectors, create=True
        ), patch.object(profile_builder, "store_embedding") as store_embedding, patch.object(
            profile_builder, "upsert"
        ):
            result = profile_builder.build_creator_profile_from_channel("channel-1", max_videos=2)

        shared_generate.assert_called_once()
        self.assertEqual(result, [0.5, 0.5])
        _, _, stored_vector = store_embedding.call_args.args
        metadata = store_embedding.call_args.kwargs["metadata"]
        self.assertEqual(stored_vector, [0.5, 0.5])
        self.assertEqual(metadata["embedding_space"], "openai:test-model")
        self.assertEqual(metadata["embedding_dimension"], 2)

    def test_content_embedding_records_the_same_space_and_dimension_fields(self) -> None:
        with patch.object(
            embedding_search,
            "generate_embeddings",
            return_value=([[0.25, 0.75]], "openai:test-model"),
        ), patch.object(embedding_search, "store_embedding") as store_embedding:
            result = embedding_search.embed_content_item(
                "video-1",
                "Text to embed",
                metadata={"platform": "youtube"},
            )

        self.assertEqual(result, [0.25, 0.75])
        metadata = store_embedding.call_args.kwargs["metadata"]
        self.assertEqual(metadata["embedding_space"], "openai:test-model")
        self.assertEqual(metadata["embedding_dimension"], 2)

    def test_creator_search_propagates_profile_space_to_vector_store(self) -> None:
        profile_record = ([1.0, 0.0], {"embedding_space": "openai:test-model"})
        with patch.object(
            vector_search,
            "get_creator_profile_embedding_record",
            return_value=profile_record,
            create=True,
        ) as get_record, patch.object(
            vector_search, "get_creator_profile_embedding", return_value=profile_record[0], create=True
        ), patch.object(vector_search, "vector_search", return_value=[]) as search:
            result = vector_search.find_content_similar_to_creator("channel-1", top_k=7)

        self.assertEqual(result, [])
        get_record.assert_called_once_with("channel-1")
        search.assert_called_once_with(
            "processed_content",
            profile_record[0],
            top_k=7,
            embedding_space="openai:test-model",
        )


if __name__ == "__main__":
    unittest.main()

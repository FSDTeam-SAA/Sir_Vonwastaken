"""Regression tests for pairwise scoring and the similarity API boundary."""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import routes
from content_similarity_check import vector_search
from trend_ranking import ranking_engine


class SimilarityRankingTests(unittest.TestCase):
    def test_pairwise_dimension_mismatch_is_reported_as_unavailable(self) -> None:
        profile_record = ([1.0, 0.0, 0.0], {"embedding_space": "openai:test-model"})
        with patch.object(
            vector_search,
            "get_creator_profile_embedding_record",
            return_value=profile_record,
            create=True,
        ), patch.object(
            vector_search, "get_creator_profile_embedding", return_value=profile_record[0], create=True
        ):
            score = vector_search.score_content_against_creator([1.0, 0.0], "channel-1")

        self.assertIsNone(score)

    def test_pairwise_explicit_space_mismatch_is_reported_as_unavailable(self) -> None:
        profile_record = ([1.0, 0.0], {"embedding_space": "openai:test-model"})
        with patch.object(
            vector_search,
            "get_creator_profile_embedding_record",
            return_value=profile_record,
            create=True,
        ):
            score = vector_search.score_content_against_creator(
                [1.0, 0.0],
                "channel-1",
                content_metadata={"embedding_space": "sentence-transformers:test-model"},
            )

        self.assertIsNone(score)

    def test_unavailable_similarity_maps_to_zero_ranking_signal(self) -> None:
        content_record = ([1.0, 0.0], {"embedding_space": "openai:test-model"})
        with patch.object(
            ranking_engine,
            "get_content_embedding_record",
            return_value=content_record,
            create=True,
        ) as get_record, patch.object(
            ranking_engine, "get_content_embedding", return_value=content_record[0], create=True
        ), patch.object(ranking_engine, "score_content_against_creator", return_value=None):
            score = ranking_engine.similarity_to_creator_score(
                {"external_id": "video-1"},
                "channel-1",
            )

        get_record.assert_called_once_with("video-1")
        self.assertEqual(score, 0.0)


class SimilarityApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = FastAPI()
        app.include_router(routes.router, prefix="/api")
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_match_failure_is_logged_and_returned_as_json_500(self) -> None:
        fake_logger = Mock()
        with patch.object(
            routes,
            "find_content_similar_to_creator",
            side_effect=RuntimeError("simulated vector-search failure"),
        ), patch.object(routes, "logger", fake_logger):
            response = self.client.get("/api/similarity/channel-1/matches")

        self.assertEqual(response.status_code, 500)
        self.assertIsInstance(response.json().get("detail"), str)
        fake_logger.exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()

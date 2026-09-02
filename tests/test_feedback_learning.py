"""Regression tests for feedback compatibility and learned trend weights."""
from __future__ import annotations

import unittest
from typing import Dict, Optional
from unittest.mock import patch

from bson import ObjectId
from fastapi.encoders import jsonable_encoder

from config.settings import settings
from feedback_system import feedback_manager
from trend_ranking import ranking_engine


def _default_weights() -> Dict[str, float]:
    return {
        "growth": settings.trend_weight_growth,
        "engagement": settings.trend_weight_engagement,
        "freshness": settings.trend_weight_freshness,
        "similarity": settings.trend_weight_similarity,
        "cross_platform": settings.trend_weight_cross_platform,
    }


class FeedbackCompatibilityTests(unittest.TestCase):
    def test_pymongo_insert_mutation_is_json_safe(self) -> None:
        """PyMongo's in-place BSON ``_id`` must not leak into the API result."""
        candidate_id = ObjectId()
        inserted_id = ObjectId()

        def fake_insert_one(collection, doc):
            self.assertEqual(collection, "trend_feedback")
            # PyMongo mutates the exact mapping passed to insert_one.
            doc["_id"] = inserted_id
            return str(inserted_id)

        with patch.object(
            feedback_manager,
            "find_one",
            return_value={"_id": candidate_id, "content_id": "ozsKU77OWpE"},
        ), patch.object(feedback_manager, "insert_one", side_effect=fake_insert_one):
            result = feedback_manager.record_feedback(
                "ozsKU77OWpE", "creator-1", "accept", "Useful"
            )

        self.assertEqual(result["_id"], str(inserted_id))
        self.assertIsInstance(result["_id"], str)
        self.assertEqual(jsonable_encoder(result)["_id"], str(inserted_id))

    def test_valid_object_id_shaped_external_id_takes_precedence(self) -> None:
        """A 24-hex external ID is still a content ID when such a candidate exists."""
        external_id = "507f1f77bcf86cd799439011"
        candidate_id = ObjectId("64b64c90f0f0f0f0f0f0f0f0")
        lookups = []

        def fake_find_one(collection, query, sort=None):
            lookups.append((collection, query))
            if query == {"content_id": external_id}:
                return {"_id": candidate_id, "content_id": external_id}
            self.fail(f"External-ID lookup should have won, got fallback query {query}")

        with patch.object(feedback_manager, "find_one", side_effect=fake_find_one), patch.object(
            feedback_manager, "insert_one", return_value=str(ObjectId())
        ):
            result = feedback_manager.record_feedback(
                external_id, "creator-1", "accept"
            )

        self.assertEqual(result["trend_id"], external_id)
        self.assertEqual(result["trend_object_id"], str(candidate_id))
        self.assertEqual(
            lookups,
            [("trend_candidates", {"content_id": external_id})],
        )

    def test_history_matches_legacy_native_object_id(self) -> None:
        """History dual-reads rows whose old ``trend_id`` is BSON, not a string."""
        candidate_id = ObjectId()
        captured = {}

        def fake_find_one(collection, query, sort=None):
            self.assertEqual(collection, "trend_candidates")
            if query == {"content_id": str(candidate_id)}:
                return None
            self.assertEqual(query, {"_id": candidate_id})
            return {"_id": candidate_id, "content_id": "ozsKU77OWpE"}

        def fake_find(collection, query, limit=0, sort=None):
            self.assertEqual(collection, "trend_feedback")
            captured.update(query)
            return []

        with patch.object(feedback_manager, "find_one", side_effect=fake_find_one), patch.object(
            feedback_manager, "find", side_effect=fake_find
        ):
            feedback_manager.get_feedback_history(
                str(candidate_id), creator_id="creator-1"
            )

        self.assertIn({"trend_id": candidate_id}, captured["$or"])

    def test_summary_resolves_trend_object_id_without_trend_id(self) -> None:
        """Partially migrated rows can be resolved from ``trend_object_id`` alone."""
        candidate_id = ObjectId()
        feedback = [{"trend_object_id": str(candidate_id), "action": "accept"}]

        def fake_find_one(collection, query, sort=None):
            self.assertEqual(collection, "trend_candidates")
            self.assertEqual(query, {"_id": candidate_id})
            return {
                "_id": candidate_id,
                "content_id": "ozsKU77OWpE",
                "analysis": {"category": "Education"},
            }

        with patch.object(feedback_manager, "find", return_value=feedback), patch.object(
            feedback_manager, "find_one", side_effect=fake_find_one
        ):
            summary = feedback_manager.get_feedback_summary("creator-1")

        self.assertEqual(summary["accept_count"], 1)
        self.assertEqual(summary["top_accepted_categories"], [("Education", 1)])

    def test_summary_tolerates_malformed_analysis(self) -> None:
        """A legacy non-mapping analysis field must not crash feedback analytics."""
        feedback = [{"trend_id": "legacy-video", "action": "reject"}]
        candidate = {
            "_id": ObjectId(),
            "content_id": "legacy-video",
            "analysis": "legacy free-form analysis",
            "category": "Education",
        }

        with patch.object(feedback_manager, "find", return_value=feedback), patch.object(
            feedback_manager, "find_one", return_value=candidate
        ):
            summary = feedback_manager.get_feedback_summary("creator-1")

        self.assertEqual(summary["reject_count"], 1)
        self.assertEqual(summary["top_rejected_categories"], [("Education", 1)])


class FeedbackWeightLearningTests(unittest.TestCase):
    def test_no_feedback_preserves_existing_weights(self) -> None:
        """An empty period must not decay a creator's existing preferences."""
        existing = {
            "growth": 0.15,
            "engagement": 0.15,
            "freshness": 0.10,
            "similarity": 0.50,
            "cross_platform": 0.10,
        }

        with patch.object(
            feedback_manager,
            "get_feedback_summary",
            return_value={"total_feedback": 0, "accept_rate": 0.0},
        ), patch.object(
            feedback_manager, "get_personalized_weights", return_value=existing.copy()
        ), patch.object(feedback_manager, "upsert") as mocked_upsert:
            result = feedback_manager.update_recommendation_weights(
                "creator-1", alpha=0.5
            )

        self.assertEqual(result, existing)
        # Rewriting the same values is harmless, but persisting changed values is not.
        if mocked_upsert.called:
            stored_doc = mocked_upsert.call_args.args[2]
            self.assertEqual(stored_doc["weights"], existing)

    def test_ema_starts_from_stored_creator_weights(self) -> None:
        """Successive learning updates must use the prior personalized state."""
        existing = {
            "growth": 0.10,
            "engagement": 0.10,
            "freshness": 0.10,
            "similarity": 0.60,
            "cross_platform": 0.10,
        }
        summary = {"total_feedback": 8, "accept_rate": 0.75}
        alpha = 0.5
        expected_similarity = (
            existing["similarity"] * (1 - alpha)
            + (summary["accept_rate"] * existing["similarity"] * 2) * alpha
        )

        with patch.object(
            feedback_manager, "get_feedback_summary", return_value=summary
        ), patch.object(
            feedback_manager, "get_personalized_weights", return_value=existing.copy()
        ), patch.object(feedback_manager, "upsert") as mocked_upsert:
            result = feedback_manager.update_recommendation_weights(
                "creator-1", alpha=alpha
            )

        self.assertAlmostEqual(result["similarity"], expected_similarity)
        for key in ("growth", "engagement", "freshness", "cross_platform"):
            self.assertEqual(result[key], existing[key])
        stored_doc = mocked_upsert.call_args.args[2]
        self.assertEqual(stored_doc["weights"], result)


class PersonalizedRankingTests(unittest.TestCase):
    _SIGNALS = {
        "growth": 0.2,
        "engagement": 0.4,
        "freshness": 0.6,
        "similarity": 0.8,
        "cross_platform": 0.1,
    }

    def _rank_once(
        self, personalized: Optional[Dict[str, float]]
    ) -> tuple[Dict, bool]:
        doc = {
            "external_id": "video-1",
            "platform": "youtube",
            "title": "Example",
            "url": "https://example.test/video-1",
            "published_at": "",
            "analysis": {},
        }

        with patch.object(
            feedback_manager,
            "get_personalized_weights",
            return_value=personalized,
        ) as source_lookup, patch.object(
            ranking_engine,
            "get_personalized_weights",
            return_value=personalized,
            create=True,
        ) as local_lookup, patch.object(
            ranking_engine, "find", return_value=[doc]
        ), patch.object(
            ranking_engine,
            "growth_velocity_score",
            return_value=self._SIGNALS["growth"],
        ), patch.object(
            ranking_engine,
            "engagement_rate_score",
            return_value=self._SIGNALS["engagement"],
        ), patch.object(
            ranking_engine,
            "freshness_score",
            return_value=self._SIGNALS["freshness"],
        ), patch.object(
            ranking_engine,
            "similarity_to_creator_score",
            return_value=self._SIGNALS["similarity"],
        ), patch.object(
            ranking_engine,
            "cross_platform_score",
            return_value=self._SIGNALS["cross_platform"],
        ), patch.object(ranking_engine, "upsert"), patch.object(
            ranking_engine, "_notify_high_value_trend"
        ):
            result = ranking_engine.rank_trends(channel_id="creator-1", limit=1)

        return result[0], source_lookup.called or local_lookup.called

    def test_ranking_applies_personalized_weights(self) -> None:
        personalized = {
            "growth": 1.0,
            "engagement": 0.0,
            "freshness": 0.0,
            "similarity": 0.0,
            "cross_platform": 0.0,
        }

        candidate, lookup_called = self._rank_once(personalized)

        self.assertTrue(lookup_called)
        self.assertEqual(candidate["score"], self._SIGNALS["growth"])

    def test_ranking_falls_back_to_configured_weights(self) -> None:
        candidate, lookup_called = self._rank_once(None)
        expected = round(
            sum(
                self._SIGNALS[name] * weight
                for name, weight in _default_weights().items()
            ),
            4,
        )

        self.assertTrue(lookup_called)
        self.assertEqual(candidate["score"], expected)


if __name__ == "__main__":
    unittest.main()

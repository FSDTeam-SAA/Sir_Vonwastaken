"""Focused regression tests for feedback trend-ID resolution."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from bson import ObjectId

from feedback_system import feedback_manager


class FeedbackManagerTests(unittest.TestCase):
    def test_record_feedback_uses_external_content_id_and_returns_string_id(self) -> None:
        """External YouTube IDs must not be coerced to ObjectIds."""
        candidate_id = ObjectId()
        inserted_id = ObjectId()
        stored = {}

        def fake_find_one(collection, query, sort=None):
            self.assertEqual(collection, "trend_candidates")
            self.assertEqual(query, {"content_id": "ozsKU77OWpE"})
            return {"_id": candidate_id, "content_id": "ozsKU77OWpE"}

        def fake_insert_one(collection, doc):
            self.assertEqual(collection, "trend_feedback")
            stored.update(doc)
            return str(inserted_id)

        with patch.object(feedback_manager, "find_one", side_effect=fake_find_one), patch.object(
            feedback_manager, "insert_one", side_effect=fake_insert_one
        ):
            result = feedback_manager.record_feedback(
                "ozsKU77OWpE", "creator-1", "accept", "Useful example"
            )

        self.assertEqual(stored["trend_id"], "ozsKU77OWpE")
        self.assertEqual(stored["trend_object_id"], str(candidate_id))
        self.assertEqual(result["_id"], str(inserted_id))
        self.assertIsInstance(result["_id"], str)

    def test_summary_resolves_external_id_and_joins_processed_category(self) -> None:
        """Summary accepts normal video IDs and retrieves category metadata."""
        candidate = {"_id": ObjectId(), "content_id": "ozsKU77OWpE"}
        feedback = [{"trend_id": "ozsKU77OWpE", "action": "accept"}]

        def fake_find_one(collection, query, sort=None):
            if collection == "trend_candidates":
                self.assertEqual(query, {"content_id": "ozsKU77OWpE"})
                return candidate
            if collection == "processed_content":
                self.assertEqual(query, {"external_id": "ozsKU77OWpE"})
                return {"analysis": {"category": "Technology"}}
            self.fail(f"Unexpected collection lookup: {collection}")

        with patch.object(feedback_manager, "find", return_value=feedback), patch.object(
            feedback_manager, "find_one", side_effect=fake_find_one
        ):
            summary = feedback_manager.get_feedback_summary("creator-1")

        self.assertEqual(summary["total_feedback"], 1)
        self.assertEqual(summary["accept_count"], 1)
        self.assertEqual(summary["accept_rate"], 1.0)
        self.assertEqual(summary["top_accepted_categories"], [("Technology", 1)])

    def test_summary_still_supports_legacy_object_id_feedback(self) -> None:
        """Existing feedback stored with a trend-candidate ObjectId remains usable."""
        candidate_id = ObjectId()
        feedback = [{"trend_id": str(candidate_id), "action": "reject"}]

        def fake_find_one(collection, query, sort=None):
            if collection != "trend_candidates":
                self.fail(f"Unexpected collection lookup: {collection}")
            if query == {"content_id": str(candidate_id)}:
                return None
            self.assertEqual(query, {"_id": candidate_id})
            return {
                "_id": candidate_id,
                "content_id": "ozsKU77OWpE",
                "analysis": {"category": "Finance"},
            }

        with patch.object(feedback_manager, "find", return_value=feedback), patch.object(
            feedback_manager, "find_one", side_effect=fake_find_one
        ):
            summary = feedback_manager.get_feedback_summary("creator-1")

        self.assertEqual(summary["reject_count"], 1)
        self.assertEqual(summary["top_rejected_categories"], [("Finance", 1)])

    def test_summary_skips_invalid_actions_and_missing_candidates(self) -> None:
        """Malformed historical rows or deleted trends cannot crash analytics."""
        feedback = [
            {"trend_id": "missing-video", "action": "ignore"},
            {"trend_id": "missing-video", "action": "unexpected"},
        ]

        with patch.object(feedback_manager, "find", return_value=feedback), patch.object(
            feedback_manager, "find_one", return_value=None
        ):
            summary = feedback_manager.get_feedback_summary("creator-1")

        self.assertEqual(summary["total_feedback"], 1)
        self.assertEqual(summary["ignore_count"], 1)
        self.assertEqual(summary["top_accepted_categories"], [])
        self.assertEqual(summary["top_rejected_categories"], [])

    def test_history_filters_by_canonical_external_id_and_legacy_reference(self) -> None:
        """A history request with an old ObjectId can find migrated feedback."""
        candidate_id = ObjectId()
        captured = {}

        def fake_find_one(collection, query, sort=None):
            if query == {"content_id": str(candidate_id)}:
                return None
            self.assertEqual(query, {"_id": candidate_id})
            return {"_id": candidate_id, "content_id": "ozsKU77OWpE"}

        def fake_find(collection, query, limit=0, sort=None):
            captured.update(query)
            return []

        with patch.object(feedback_manager, "find_one", side_effect=fake_find_one), patch.object(
            feedback_manager, "find", side_effect=fake_find
        ):
            feedback_manager.get_feedback_history(str(candidate_id), "creator-1")

        self.assertEqual(captured["creator_id"], "creator-1")
        self.assertIn({"trend_id": str(candidate_id)}, captured["$or"])
        self.assertIn({"trend_id": "ozsKU77OWpE"}, captured["$or"])
        self.assertIn({"trend_object_id": str(candidate_id)}, captured["$or"])


if __name__ == "__main__":
    unittest.main()

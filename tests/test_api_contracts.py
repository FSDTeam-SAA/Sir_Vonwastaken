"""Regression tests for API contracts that previously caused Swagger-test confusion."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bson import ObjectId
from fastapi.testclient import TestClient

from AI_generator import generate_content as content_generator
from email_assistant import draft_replies
from email_assistant import wait_for_approval
from main import app


class ContentRegenerationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # Constructing TestClient without a context manager avoids running the
        # app's scheduler/Mongo startup hooks in these isolated route tests.
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_regeneration_accepts_each_documented_field(self) -> None:
        supported_fields = ("titles", "hooks", "outline", "script", "thumbnails")

        for field in supported_fields:
            with self.subTest(field=field), patch(
                "api.routes.regenerate_field", return_value={field: "generated"}
            ) as regenerate:
                response = self.client.post(
                    "/api/content/regenerate/video-123",
                    json={"field": field, "channel_id": "creator-123"},
                )

                self.assertEqual(response.status_code, 200, response.text)
                regenerate.assert_called_once_with(
                    "video-123", field, channel_id="creator-123"
                )

    def test_regeneration_rejects_undocumented_or_missing_fields(self) -> None:
        for payload in ({"field": "title"}, {"field": "script_draft"}, {}):
            with self.subTest(payload=payload):
                response = self.client.post(
                    "/api/content/regenerate/video-123", json=payload
                )

                self.assertEqual(response.status_code, 422, response.text)

    def test_missing_generated_content_is_not_found(self) -> None:
        with patch(
            "api.routes.regenerate_field",
            side_effect=ValueError(
                "No generated content found for content_id=missing-video."
            ),
        ):
            response = self.client.post(
                "/api/content/regenerate/missing-video", json={"field": "titles"}
            )

        self.assertEqual(response.status_code, 404, response.text)
        self.assertIn("No generated content found", response.json()["detail"])

    def test_title_regeneration_falls_back_to_stored_channel(self) -> None:
        generated_id = ObjectId()
        generated_collection = MagicMock()
        generated_collection.find_one.return_value = {
            "_id": generated_id,
            "trend_id": "video-123",
            "channel_id": "stored-creator",
            "trend_title": "A useful trend",
            "titles": ["Existing title"],
        }
        fake_db = SimpleNamespace(generated_content=generated_collection)

        with patch.object(content_generator, "get_db", return_value=fake_db), patch.object(
            content_generator, "gen_titles", return_value=["New title"]
        ) as generate_titles:
            result = content_generator.regenerate_field("video-123", "titles")

        self.assertEqual(result, {"titles": ["New title"]})
        generate_titles.assert_called_once_with(
            "A useful trend", channel_id="stored-creator"
        )
        generated_collection.update_one.assert_called_once()


class DraftApprovalContractTests(unittest.TestCase):
    def test_failed_gmail_draft_is_not_stored_as_pending(self) -> None:
        email_doc = {
            "external_id": "email-1",
            "subject": "Partnership",
            "body": "Hello",
            "from": "brand@example.com",
            "summary": {"summary": "Offer"},
        }

        with patch.object(draft_replies, "find_one", return_value=email_doc), patch.object(
            draft_replies, "generate_reply_text", return_value="Thanks for reaching out."
        ), patch.object(draft_replies, "create_draft", return_value=None), patch.object(
            draft_replies, "insert_one"
        ) as insert_one:
            with self.assertRaisesRegex(RuntimeError, "no pending approval record"):
                draft_replies.create_draft_reply("email-1")

        insert_one.assert_not_called()

    def test_get_draft_converts_a_returned_mongo_id(self) -> None:
        draft_object_id = ObjectId()
        expected = {"_id": draft_object_id, "status": "pending_approval"}

        with patch.object(wait_for_approval, "find_one", return_value=expected) as find_one:
            result = wait_for_approval.get_draft(str(draft_object_id))

        self.assertIs(result, expected)
        find_one.assert_called_once_with(
            "email_drafts", {"_id": draft_object_id}
        )

    def test_get_draft_rejects_non_object_id_formats(self) -> None:
        for draft_id in ("1", "gmail-draft-id", "ozsKU77OWpE"):
            with self.subTest(draft_id=draft_id), self.assertRaisesRegex(
                ValueError, "Invalid draft_id format"
            ):
                wait_for_approval.get_draft(draft_id)

    def test_reject_draft_refuses_a_non_pending_draft(self) -> None:
        draft = {"_id": ObjectId(), "status": "sent"}

        with patch.object(wait_for_approval, "get_draft", return_value=draft), patch.object(
            wait_for_approval, "get_collection"
        ) as get_collection:
            with self.assertRaisesRegex(ValueError, "not pending approval"):
                wait_for_approval.reject_draft(str(draft["_id"]), reason="No fit")

        get_collection.assert_not_called()

    def test_reject_draft_updates_a_pending_draft(self) -> None:
        draft = {"_id": ObjectId(), "status": "pending_approval"}
        collection = MagicMock()

        with patch.object(wait_for_approval, "get_draft", return_value=draft), patch.object(
            wait_for_approval, "get_collection", return_value=collection
        ):
            result = wait_for_approval.reject_draft(
                str(draft["_id"]), reason="Not aligned"
            )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["rejection_reason"], "Not aligned")
        collection.update_one.assert_called_once()


class NotificationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_notify_rejects_unknown_or_empty_channel_lists(self) -> None:
        for channels in (["carrier_pigeon"], []):
            with self.subTest(channels=channels):
                response = self.client.post(
                    "/api/notify/test",
                    json={"title": "Test", "message": "Message", "channels": channels},
                )

                self.assertEqual(response.status_code, 422, response.text)

    def test_notify_returns_delivery_result_for_each_requested_channel(self) -> None:
        with patch("api.routes.desktop_notifications.send", return_value=True) as desktop, patch(
            "api.routes.discord.send", return_value=False
        ) as discord, patch("api.routes.telegram.send", return_value=True) as telegram, patch(
            "api.routes.email_notify.send", return_value=False
        ) as email, patch("api.routes.slack.send", return_value=True) as slack:
            response = self.client.post(
                "/api/notify/test",
                json={
                    "title": "Contract test",
                    "message": "Delivery check",
                    "channels": ["desktop", "discord", "telegram", "email", "slack"],
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {
                "desktop": True,
                "discord": False,
                "telegram": True,
                "email": False,
                "slack": True,
            },
        )
        desktop.assert_called_once_with("Contract test", "Delivery check")
        discord.assert_called_once_with("Contract test", "Delivery check")
        telegram.assert_called_once_with("Contract test\nDelivery check")
        email.assert_called_once_with("Contract test", "Delivery check")
        slack.assert_called_once_with("Contract test", "Delivery check")


if __name__ == "__main__":
    unittest.main()

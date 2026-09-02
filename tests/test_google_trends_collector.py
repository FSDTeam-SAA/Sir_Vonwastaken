"""Regression tests for Google Trends collection and its API contract."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import routes
from data_collectors import data_processor
from data_collectors import google_trends_collector as collector


class GoogleTrendsCollectorTests(unittest.TestCase):
    def test_trending_now_rss_is_parsed_and_stored(self) -> None:
        response = MagicMock()
        response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
            <rss xmlns:ht="https://trends.google.com/trending/rss">
              <channel>
                <item>
                  <title>KATSEYE</title>
                  <ht:approx_traffic>20K+</ht:approx_traffic>
                  <pubDate>Wed, 02 Sep 2026 00:10:00 -0700</pubDate>
                </item>
              </channel>
            </rss>
        """

        with patch.object(collector.requests, "get", return_value=response) as get, patch.object(
            collector, "_store_raw"
        ) as store:
            result = collector.get_trending_searches("us")

        response.raise_for_status.assert_called_once_with()
        get.assert_called_once_with(
            collector._TRENDING_RSS_URL,
            params={"geo": "US"},
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ContentTrendAssistant/1.0)"
            },
            timeout=15,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["keyword"], "KATSEYE")
        self.assertEqual(result[0]["geo"], "US")
        self.assertEqual(result[0]["rank"], 0)
        self.assertEqual(result[0]["approx_traffic"], "20K+")
        self.assertEqual(result[0]["published_at"], "2026-09-02T00:10:00-07:00")
        store.assert_called_once_with(result[0])

    def test_trending_now_failure_is_reported_without_crashing(self) -> None:
        errors = []
        with patch.object(
            collector.requests,
            "get",
            side_effect=requests.RequestException("network unavailable"),
        ), patch.object(collector, "_store_raw") as store:
            result = collector.get_trending_searches("US", errors=errors)

        self.assertEqual(result, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("Trending Now RSS failed", errors[0])
        store.assert_not_called()

    def test_empty_trending_now_feed_is_a_partial_failure(self) -> None:
        response = MagicMock()
        response.content = b"<rss><channel /></rss>"
        errors = []

        with patch.object(collector.requests, "get", return_value=response), patch.object(
            collector, "_store_raw"
        ):
            result = collector.get_trending_searches("US", errors=errors)

        self.assertEqual(result, [])
        self.assertEqual(errors, ["Trending Now RSS returned no items for geo=US."])

    def test_malformed_trending_now_feed_is_reported(self) -> None:
        response = MagicMock()
        response.content = b"<rss><channel>"
        errors = []

        with patch.object(collector.requests, "get", return_value=response), patch.object(
            collector, "_store_raw"
        ) as store:
            result = collector.get_trending_searches("US", errors=errors)

        self.assertEqual(result, [])
        self.assertEqual(errors, ["Trending Now RSS failed for geo=US (ParseError)."])
        store.assert_not_called()

    def test_related_queries_preserve_each_success_and_are_paced(self) -> None:
        keywords = ["KATSEYE", "K-pop quiz", "guess the song"]
        client = MagicMock()
        client.related_queries.side_effect = [
            {keyword: {"top": pd.DataFrame([{"query": f"{keyword} songs", "value": 100}]), "rising": None}}
            for keyword in keywords
        ]

        with patch.object(collector, "get_trends_client", return_value=client), patch.object(
            collector, "_store_raw"
        ) as store, patch.object(collector.time, "sleep") as sleep:
            result = collector.get_related_queries_batch(keywords)

        self.assertEqual(list(result), keywords)
        self.assertEqual(client.build_payload.call_count, 3)
        self.assertEqual(client.related_queries.call_count, 3)
        self.assertEqual(store.call_count, 3)
        self.assertEqual(
            sleep.call_args_list,
            [
                call(collector._GOOGLE_REQUEST_DELAY_SECONDS),
                call(collector._GOOGLE_REQUEST_DELAY_SECONDS),
            ],
        )

    def test_related_queries_retry_once_after_429(self) -> None:
        rate_limit = RuntimeError("too many requests")
        rate_limit.response = SimpleNamespace(status_code=429)
        client = MagicMock()
        client.related_queries.side_effect = [
            rate_limit,
            {"KATSEYE": {"top": None, "rising": None}},
        ]

        with patch.object(collector, "get_trends_client", return_value=client), patch.object(
            collector, "_store_raw"
        ), patch.object(collector.time, "sleep") as sleep:
            result = collector.get_related_queries_batch(["KATSEYE"])

        self.assertIn("KATSEYE", result)
        self.assertEqual(client.related_queries.call_count, 2)
        sleep.assert_called_once_with(collector._RATE_LIMIT_RETRY_SECONDS)

    def test_repeated_429_stops_remaining_related_queries(self) -> None:
        first_rate_limit = RuntimeError("too many requests")
        first_rate_limit.response = SimpleNamespace(status_code=429)
        second_rate_limit = RuntimeError("still rate limited")
        second_rate_limit.response = SimpleNamespace(status_code=429)
        client = MagicMock()
        client.related_queries.side_effect = [first_rate_limit, second_rate_limit]
        errors = []

        with patch.object(collector, "get_trends_client", return_value=client), patch.object(
            collector, "_store_raw"
        ), patch.object(collector.time, "sleep") as sleep:
            result = collector.get_related_queries_batch(
                ["KATSEYE", "K-pop quiz"], errors=errors
            )

        self.assertEqual(result, {})
        self.assertEqual(client.related_queries.call_count, 2)
        sleep.assert_called_once_with(collector._RATE_LIMIT_RETRY_SECONDS)
        self.assertEqual(len(errors), 2)
        self.assertIn("Related queries failed", errors[0])
        self.assertIn("skipped after rate limit", errors[1])

    def test_non_rate_limit_related_failure_is_not_retried(self) -> None:
        client = MagicMock()
        client.related_queries.side_effect = RuntimeError("invalid response")
        errors = []

        with patch.object(collector, "get_trends_client", return_value=client), patch.object(
            collector, "_store_raw"
        ), patch.object(collector.time, "sleep") as sleep:
            result = collector.get_related_queries_batch(["KATSEYE"], errors=errors)

        self.assertEqual(result, {})
        self.assertEqual(client.related_queries.call_count, 1)
        self.assertEqual(
            errors,
            ["Related queries failed for 'KATSEYE' (RuntimeError)."],
        )
        sleep.assert_not_called()

    def test_interest_over_time_retries_once_after_429(self) -> None:
        rate_limit = RuntimeError("too many requests")
        rate_limit.response = SimpleNamespace(status_code=429)
        frame = pd.DataFrame(
            {"KATSEYE": [25]},
            index=[pd.Timestamp("2026-09-02T00:00:00Z")],
        )
        client = MagicMock()
        client.interest_over_time.side_effect = [rate_limit, frame]

        with patch.object(collector, "_store_raw"), patch.object(
            collector.time, "sleep"
        ) as sleep:
            result = collector.get_interest_over_time(
                ["KATSEYE"], client=client
            )

        self.assertEqual(result["KATSEYE"][0]["value"], 25)
        self.assertEqual(client.interest_over_time.call_count, 2)
        sleep.assert_called_once_with(collector._RATE_LIMIT_RETRY_SECONDS)

    def test_missing_related_keyword_is_not_counted_as_tracked(self) -> None:
        client = MagicMock()
        client.related_queries.return_value = {}
        errors = []

        with patch.object(collector, "get_trends_client", return_value=client), patch.object(
            collector, "_store_raw"
        ) as store, patch.object(collector.time, "sleep"):
            result = collector.get_related_queries_batch(["KATSEYE"], errors=errors)

        self.assertEqual(result, {})
        self.assertEqual(errors, ["Related queries response omitted 'KATSEYE'."])
        store.assert_not_called()

    def test_full_collection_reports_partial_counts_and_errors(self) -> None:
        def fail_trending(*, errors):
            errors.append("rss failed")
            return []

        with patch.object(
            collector.settings,
            "google_trends_keywords",
            ["KATSEYE", "K-pop quiz"],
        ), patch.object(
            collector, "get_trending_searches", side_effect=fail_trending
        ), patch.object(
            collector,
            "get_interest_over_time",
            return_value={"KATSEYE": [{"value": 80}]},
        ), patch.object(
            collector,
            "get_related_queries_batch",
            return_value={"KATSEYE": {"top": [], "rising": []}},
        ), patch.object(
            collector, "get_trends_client", return_value=MagicMock()
        ):
            result = collector.run_full_collection()

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["trending_searches"], 0)
        self.assertEqual(result["keywords_requested"], 2)
        self.assertEqual(result["keywords_tracked"], 1)
        self.assertEqual(result["related_queries_tracked"], 1)
        self.assertIn("rss failed", result["errors"])
        self.assertTrue(
            any("No interest-over-time data" in error for error in result["errors"])
        )
        self.assertTrue(
            any("No related-query data" in error for error in result["errors"])
        )

    def test_client_initialization_failure_returns_partial_result(self) -> None:
        with patch.object(
            collector.settings,
            "google_trends_keywords",
            ["KATSEYE"],
        ), patch.object(
            collector, "get_trending_searches", return_value=[]
        ), patch.object(
            collector,
            "get_trends_client",
            side_effect=requests.RequestException("proxy unavailable"),
        ):
            result = collector.run_full_collection()

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["keywords_requested"], 1)
        self.assertEqual(result["keywords_tracked"], 0)
        self.assertEqual(result["related_queries_tracked"], 0)
        self.assertTrue(
            any("client initialization failed" in error for error in result["errors"])
        )


class GoogleTrendsApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = FastAPI()
        app.include_router(routes.router, prefix="/api")
        cls.app = app
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_collection_response_has_documented_json_schema(self) -> None:
        operation = self.app.openapi()["paths"]["/api/collect/google-trends"]["post"]
        response_schema = operation["responses"]["200"]["content"][
            "application/json"
        ]["schema"]

        self.assertEqual(
            response_schema["$ref"],
            "#/components/schemas/GoogleTrendsCollectionResponse",
        )

    def test_collection_endpoint_returns_detailed_counts(self) -> None:
        expected = {
            "status": "ok",
            "trending_searches": 10,
            "keywords_requested": 2,
            "keywords_tracked": 2,
            "related_queries_tracked": 2,
            "errors": [],
        }
        with patch.object(
            routes.google_trends_collector,
            "run_full_collection",
            return_value=expected,
        ):
            response = self.client.post("/api/collect/google-trends")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), expected)


class GoogleTrendsProcessingTests(unittest.TestCase):
    def test_normalizer_preserves_rss_publication_time(self) -> None:
        result = data_processor._normalize_google_trends(
            {
                "external_id": "US:2026-09-02:KATSEYE",
                "keyword": "KATSEYE",
                "published_at": "2026-09-02T00:10:00-07:00",
                "collected_at": datetime(2026, 9, 2, tzinfo=timezone.utc),
                "rank": 0,
            }
        )

        self.assertEqual(result["published_at"], "2026-09-02T00:10:00-07:00")

    def test_processor_excludes_interest_and_related_analytics_records(self) -> None:
        raw_records = [
            {
                "_id": "trend-id",
                "platform": "google_trends",
                "external_id": "US:2026-09-02:KATSEYE",
                "record_kind": "trending_search",
                "keyword": "KATSEYE",
                "rank": 0,
            },
            {
                "_id": "interest-id",
                "platform": "google_trends",
                "external_id": "interest:KATSEYE:now 7-d",
                "record_kind": "interest_over_time",
                "keyword": "KATSEYE",
                "series": [],
            },
            {
                "_id": "related-id",
                "platform": "google_trends",
                "external_id": "related:KATSEYE",
                "record_kind": "related_queries",
                "keyword": "KATSEYE",
                "top": [],
                "rising": [],
            },
        ]
        raw_collection = MagicMock()

        with patch.object(data_processor, "find", return_value=raw_records), patch.object(
            data_processor, "upsert"
        ) as upsert, patch.object(
            data_processor, "get_collection", return_value=raw_collection
        ):
            result = data_processor.process_platform("google_trends")

        self.assertEqual(result, {"seen": 3, "filtered_out": 2, "processed": 1})
        upsert.assert_called_once()
        self.assertEqual(raw_collection.update_one.call_count, 3)


if __name__ == "__main__":
    unittest.main()

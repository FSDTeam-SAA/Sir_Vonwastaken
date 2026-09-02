"""Regression tests for scheduled embedding and personalized ranking configuration."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from scheduler import background_tasks


class _FakeScheduler:
    def __init__(self) -> None:
        self.running = False
        self.jobs = []

    def add_job(self, func, trigger, **kwargs) -> None:
        self.jobs.append((func, trigger, kwargs))

    def get_jobs(self):
        return self.jobs

    def start(self) -> None:
        self.running = True


class SchedulerContractTests(unittest.TestCase):
    def test_embedding_and_ranking_jobs_follow_configured_creator_backend(self) -> None:
        scheduler = _FakeScheduler()
        with patch.object(background_tasks, "get_scheduler", return_value=scheduler), patch.object(
            background_tasks.settings, "youtube_channel_id", "creator-1"
        ):
            background_tasks.start_scheduler()

        jobs_by_id = {job_options["id"]: (func, job_options) for func, _, job_options in scheduler.jobs}
        embed_func, embed_options = jobs_by_id["embed_content"]
        rank_func, rank_options = jobs_by_id["rank_trends"]

        self.assertIs(embed_func, background_tasks.embed_pending_content)
        self.assertEqual(embed_options["kwargs"], {"limit": 100})
        self.assertNotIn("use_local", embed_options["kwargs"])

        self.assertIs(rank_func, background_tasks.rank_trends)
        self.assertEqual(
            rank_options["kwargs"],
            {"limit": 100, "channel_id": "creator-1"},
        )


if __name__ == "__main__":
    unittest.main()

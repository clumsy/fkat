# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch

from fkat.pytorch.schedule import (
    Every,
    Fixed,
    Never,
)
from fkat.pytorch.schedule.mlflow import HasTag


class MLflowTagPresentsTest(unittest.TestCase):
    """Test cases for the MLFlow HasTag schedule."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock trainer
        self.trainer = MagicMock()
        # Create a mock CallbackLogger that will be created inside MLflowTagPresents
        self.cb_logger_patcher = patch("fkat.pytorch.schedule.mlflow.CallbackLogger")
        self.mock_cb_logger_class = self.cb_logger_patcher.start()
        self.mock_cb_logger = MagicMock()
        self.mock_cb_logger_class.return_value = self.mock_cb_logger
        # Default mock tags to return
        self.mock_cb_logger.tags.return_value = {"enable_flops": "true", "other_tag": "value"}

    def tearDown(self):
        """Clean up after each test."""
        self.cb_logger_patcher.stop()

    def test_init(self):
        """Test the initialization of the schedule."""
        # Test with Every schedule
        trigger_schedule = Every(n_batches=5)
        schedule = HasTag(tag="enable_flops", schedule=trigger_schedule)
        assert schedule._tag == "enable_flops"
        assert schedule._schedule == trigger_schedule

        # Test with Fixed schedule
        trigger_schedule = Fixed(warmup_steps=10, active_steps=20)
        schedule = HasTag(tag="enable_flops", schedule=trigger_schedule)
        assert schedule._tag == "enable_flops"
        assert schedule._schedule == trigger_schedule

    def test_check_with_tag_present_and_every_schedule(self):
        """Test schedule activation when tag is present and Every schedule is triggered."""
        # Create an Every schedule that triggers on specific batch indices
        schedule = Every(n_batches=5)
        schedule = HasTag(tag="enable_flops", schedule=schedule)

        # Should return True when trigger schedule is True and tag is present
        assert schedule.check(stage="train", batch_idx=0, trainer=self.trainer)
        assert schedule.check(stage="train", batch_idx=5, trainer=self.trainer)
        assert schedule.check(stage="train", batch_idx=10, trainer=self.trainer)

        # Should return False when trigger schedule is False
        assert not schedule.check(stage="train", batch_idx=1, trainer=self.trainer)
        assert not schedule.check(stage="train", batch_idx=4, trainer=self.trainer)
        assert not schedule.check(stage="train", batch_idx=6, trainer=self.trainer)

    def test_check_with_tag_present_and_fixed_schedule(self):
        """Test schedule activation when tag is present and Fixed schedule is triggered."""
        # Create a Fixed schedule for a specific range of steps
        trigger_schedule = Fixed(warmup_steps=5, active_steps=10)
        schedule = HasTag(tag="enable_flops", schedule=trigger_schedule)

        # Should return False for steps before warmup
        assert not schedule.check(stage="train", step=0, trainer=self.trainer)
        assert not schedule.check(stage="train", step=4, trainer=self.trainer)

        # Should return True for steps within active period and tag is present
        assert schedule.check(stage="train", step=5, trainer=self.trainer)
        assert schedule.check(stage="train", step=10, trainer=self.trainer)
        assert schedule.check(stage="train", step=14, trainer=self.trainer)

        # Should return False for steps after active period
        assert not schedule.check(stage="train", step=15, trainer=self.trainer)
        assert not schedule.check(stage="train", step=20, trainer=self.trainer)

    def test_check_with_never_schedule(self):
        """Test schedule with Never trigger schedule."""
        trigger_schedule = Never()
        schedule = HasTag(tag="enable_flops", schedule=trigger_schedule)

        # Should always return False because Never schedule never triggers
        assert not schedule.check(stage="train", batch_idx=0, trainer=self.trainer)
        assert not schedule.check(stage="train", step=5, trainer=self.trainer)
        assert not schedule.check(stage="train", batch_idx=10, step=10, trainer=self.trainer)

    def test_check_with_tag_missing(self):
        """Test schedule deactivation when tag is missing."""
        # Set the mock to return no matching tag
        self.mock_cb_logger.tags.return_value = {"other_tag": "value"}

        trigger_schedule = Every(n_batches=5)
        schedule = HasTag(tag="enable_flops", schedule=trigger_schedule)

        # Should return False even when trigger schedule would be True
        assert not schedule.check(stage="train", batch_idx=0, trainer=self.trainer)
        assert not schedule.check(stage="train", batch_idx=5, trainer=self.trainer)
        assert not schedule.check(stage="train", batch_idx=10, trainer=self.trainer)

    def test_check_without_trainer(self):
        """Test schedule deactivation when trainer is not provided."""
        trigger_schedule = Every(n_batches=5)
        schedule = HasTag(tag="enable_flops", schedule=trigger_schedule)

        # Should return False when trainer is None, even for matching batches
        assert not schedule.check(stage="train", batch_idx=0, trainer=None)
        assert not schedule.check(stage="train", batch_idx=5, trainer=None)
        assert not schedule.check(stage="train", batch_idx=10, trainer=None)

    def test_check_with_exception(self):
        """Test schedule behavior when an exception occurs during tag check."""
        # Make the cb_logger.tags() method raise an exception
        self.mock_cb_logger.tags.side_effect = Exception("Test exception")

        trigger_schedule = Every(n_batches=5)
        schedule = HasTag(tag="enable_flops", schedule=trigger_schedule)

        # Should return False and handle the exception gracefully
        with self.assertLogs(level="WARNING") as log:
            assert not schedule.check(stage="train", batch_idx=0, trainer=self.trainer)
            assert any("Error when checking if tag enable_flops exists" in msg for msg in log.output)

    def test_check_with_different_stages(self):
        """Test schedule behavior with different stages."""
        trigger_schedule = Every(n_batches=5)
        schedule = HasTag(tag="enable_flops", schedule=trigger_schedule)

        # Should work the same regardless of stage
        assert schedule.check(stage="train", batch_idx=0, trainer=self.trainer)
        assert schedule.check(stage="validate", batch_idx=0, trainer=self.trainer)
        assert schedule.check(stage="test", batch_idx=0, trainer=self.trainer)
        assert schedule.check(stage="predict", batch_idx=0, trainer=self.trainer)

        # Should still respect trigger schedule
        assert not schedule.check(stage="train", batch_idx=1, trainer=self.trainer)
        assert not schedule.check(stage="validate", batch_idx=1, trainer=self.trainer)

    def test_trigger_schedule_chain(self):
        """Test chaining MLflowTagPresents with another MLflowTagPresents as its trigger."""
        # Create a chain of schedules: Every(5) -> MLflowTagPresents("tag1") -> MLflowTagPresents("tag2")
        base_trigger = Every(n_batches=5)
        middle_trigger = HasTag(tag="tag1", schedule=base_trigger)
        final_schedule = HasTag(tag="tag2", schedule=middle_trigger)

        # Set up tags for the test
        self.mock_cb_logger.tags.return_value = {"tag1": "true", "tag2": "true"}

        # Both tags present, should follow base trigger pattern
        assert final_schedule.check(stage="train", batch_idx=0, trainer=self.trainer)
        assert not final_schedule.check(stage="train", batch_idx=1, trainer=self.trainer)
        assert final_schedule.check(stage="train", batch_idx=5, trainer=self.trainer)

        # Missing tag1, should always be False
        self.mock_cb_logger.tags.return_value = {"tag2": "true"}
        assert not final_schedule.check(stage="train", batch_idx=0, trainer=self.trainer)
        assert not final_schedule.check(stage="train", batch_idx=5, trainer=self.trainer)

        # Missing tag2, should always be False
        self.mock_cb_logger.tags.return_value = {"tag1": "true"}
        assert not final_schedule.check(stage="train", batch_idx=0, trainer=self.trainer)
        assert not final_schedule.check(stage="train", batch_idx=5, trainer=self.trainer)

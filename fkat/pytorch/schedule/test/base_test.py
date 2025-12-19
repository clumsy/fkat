# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import datetime as dt
import unittest
import operator
from unittest.mock import patch, MagicMock

import lightning as L

from fkat.pytorch.schedule import (
    Always,
    Every,
    Fixed,
    Never,
    Elapsed,
    GlobalRank,
    LocalRank,
    InvertedSchedule,
    CombinedSchedule,
)


class ScheduleTest(unittest.TestCase):
    def test_always_schedule(self):
        """Test that Always schedule always returns True."""
        train_stage = "train"
        schedule = Always()

        # Should always return True regardless of parameters
        assert schedule.check(stage=train_stage, batch_idx=0)
        assert schedule.check(stage=train_stage, batch_idx=5)
        assert schedule.check(stage=train_stage, step=0)
        assert schedule.check(stage=train_stage, step=100)

        # Test with different stages
        assert schedule.check(stage="validate", batch_idx=0)
        assert schedule.check(stage="test", batch_idx=0)
        assert schedule.check(stage="predict", batch_idx=0)

        # Test with no parameters
        assert schedule.check()

        # Test with None values
        assert schedule.check(stage=None, batch_idx=None, step=None, trainer=None)

    def test_fixed_length_schedule(self):
        # Test with custom setting (warmup_steps=5, active_steps=10)
        schedule = Fixed(warmup_steps=5, active_steps=10)
        train_stage = "train"

        assert not schedule.check(stage=train_stage, step=0)
        assert not schedule.check(stage=train_stage, step=4)
        assert schedule.check(stage=train_stage, step=5)
        assert schedule.check(stage=train_stage, step=14)
        assert not schedule.check(stage=train_stage, step=15)
        assert not schedule.check(stage=train_stage, step=20)

        # Test error case
        with pytest.raises(AssertionError):
            schedule.check(stage=train_stage, batch_idx=0, step=None)

    def test_every_n_batches_schedule(self):
        # Test with every_n_batches=5
        schedule = Every(n_batches=5)
        train_stage = "train"

        assert schedule.check(stage=train_stage, batch_idx=0)
        assert not schedule.check(stage=train_stage, batch_idx=1)
        assert not schedule.check(stage=train_stage, batch_idx=4)
        assert schedule.check(stage=train_stage, batch_idx=5)
        assert schedule.check(stage=train_stage, batch_idx=10)
        assert not schedule.check(stage=train_stage, batch_idx=11)

    def test_every_n_steps_schedule(self):
        # Test with every_n_steps=5
        schedule = Every(n_steps=5)
        train_stage = "train"

        assert schedule.check(stage=train_stage, step=0)
        assert not schedule.check(stage=train_stage, step=1)
        assert not schedule.check(stage=train_stage, step=4)
        assert schedule.check(stage=train_stage, step=5)
        assert schedule.check(stage=train_stage, step=10)
        assert not schedule.check(stage=train_stage, step=11)

    def test_every_with_stage_parameter(self):
        """Test Every schedule with stage parameter."""
        # Test with every_n_steps=5 and stage="train"
        schedule = Every(n_steps=5, stage="train")

        # Should return True for matching stage and step
        assert schedule.check(stage="train", step=0)
        assert schedule.check(stage="train", step=5)

        # Should return False for non-matching stage
        assert not schedule.check(stage="validation", step=0)
        assert not schedule.check(stage="validation", step=5)

        # Test with every_n_batches=5 and stage="validation"
        schedule = Every(n_batches=5, stage="validation")

        # Should return True for matching stage and batch_idx
        assert schedule.check(stage="validation", batch_idx=0)
        assert schedule.check(stage="validation", batch_idx=5)

        # Should return False for non-matching stage
        assert not schedule.check(stage="train", batch_idx=0)
        assert not schedule.check(stage="train", batch_idx=5)

    def test_every_n_batch_or_n_step_is_set(self):
        with pytest.raises(AssertionError):
            Every(n_batches=0, n_steps=0)

    def test_never_schedule(self):
        # Test with Never (should never log)
        train_stage = "train"
        schedule = Never()
        assert not schedule.check(stage=train_stage, batch_idx=0)
        assert not schedule.check(stage=train_stage, batch_idx=5)

    def test_inverted_schedule(self):
        """Test that InvertedSchedule inverts the result of the wrapped schedule."""
        train_stage = "train"

        # Test with Always schedule (should always return False when inverted)
        always_schedule = Always()
        inverted_always = InvertedSchedule(always_schedule)
        assert not inverted_always.check(stage=train_stage, batch_idx=0)
        assert not inverted_always.check(stage=train_stage, step=10)

        # Test with Never schedule (should always return True when inverted)
        never_schedule = Never()
        inverted_never = InvertedSchedule(never_schedule)
        assert inverted_never.check(stage=train_stage, batch_idx=0)
        assert inverted_never.check(stage=train_stage, step=10)

        # Test with Fixed schedule
        fixed_schedule = Fixed(warmup_steps=5, active_steps=10)
        inverted_fixed = InvertedSchedule(fixed_schedule)

        # Original returns False, inverted returns True
        assert not fixed_schedule.check(stage=train_stage, step=0)
        assert inverted_fixed.check(stage=train_stage, step=0)

        # Original returns True, inverted returns False
        assert fixed_schedule.check(stage=train_stage, step=10)
        assert not inverted_fixed.check(stage=train_stage, step=10)

    def test_combined_schedule_and(self):
        """Test CombinedSchedule with AND operator."""
        train_stage = "train"

        # Create test schedules
        always_schedule = Always()
        never_schedule = Never()
        fixed_schedule = Fixed(warmup_steps=5, active_steps=10)

        # Test Always AND Always (should always return True)
        combined_always_always = CombinedSchedule(operator.and_, (always_schedule, always_schedule))
        assert combined_always_always.check(stage=train_stage, step=0)
        assert combined_always_always.check(stage=train_stage, step=10)

        # Test Always AND Never (should always return False)
        combined_always_never = CombinedSchedule(operator.and_, (always_schedule, never_schedule))
        assert not combined_always_never.check(stage=train_stage, step=0)
        assert not combined_always_never.check(stage=train_stage, step=10)

        # Test Always AND Fixed
        combined_always_fixed = CombinedSchedule(operator.and_, (always_schedule, fixed_schedule))
        # Fixed returns False at step 0, so combined should be False
        assert not combined_always_fixed.check(stage=train_stage, step=0)
        # Fixed returns True at step 10, so combined should be True
        assert combined_always_fixed.check(stage=train_stage, step=10)
        # Fixed returns False at step 20, so combined should be False
        assert not combined_always_fixed.check(stage=train_stage, step=20)

    def test_combined_schedule_or(self):
        """Test CombinedSchedule with OR operator."""
        train_stage = "train"

        # Create test schedules
        always_schedule = Always()
        never_schedule = Never()
        fixed_schedule = Fixed(warmup_steps=5, active_steps=10)

        # Test Always OR Always (should always return True)
        combined_always_always = CombinedSchedule(operator.or_, (always_schedule, always_schedule))
        assert combined_always_always.check(stage=train_stage, step=0)
        assert combined_always_always.check(stage=train_stage, step=10)

        # Test Always OR Never (should always return True)
        combined_always_never = CombinedSchedule(operator.or_, (always_schedule, never_schedule))
        assert combined_always_never.check(stage=train_stage, step=0)
        assert combined_always_never.check(stage=train_stage, step=10)

        # Test Never OR Fixed
        combined_never_fixed = CombinedSchedule(operator.or_, (never_schedule, fixed_schedule))
        # Fixed returns False at step 0, so combined should be False
        assert not combined_never_fixed.check(stage=train_stage, step=0)
        # Fixed returns True at step 10, so combined should be True
        assert combined_never_fixed.check(stage=train_stage, step=10)
        # Fixed returns False at step 20, so combined should be False
        assert not combined_never_fixed.check(stage=train_stage, step=20)

    def test_combined_schedule_with_multiple_schedules(self):
        """Test CombinedSchedule with more than two schedules."""
        train_stage = "train"

        # Create test schedules
        always_schedule = Always()
        never_schedule = Never()
        fixed_schedule = Fixed(warmup_steps=5, active_steps=10)

        # Test AND with three schedules
        combined_and = CombinedSchedule(operator.and_, (always_schedule, fixed_schedule, always_schedule))
        # Fixed returns False at step 0, so combined should be False
        assert not combined_and.check(stage=train_stage, step=0)
        # Fixed returns True at step 10, so combined should be True
        assert combined_and.check(stage=train_stage, step=10)

        # Test OR with three schedules
        combined_or = CombinedSchedule(operator.or_, (never_schedule, fixed_schedule, never_schedule))
        # Fixed returns False at step 0, so combined should be False
        assert not combined_or.check(stage=train_stage, step=0)
        # Fixed returns True at step 10, so combined should be True
        assert combined_or.check(stage=train_stage, step=10)


class TestElapsed:
    def test_init(self):
        """Test initialization of Elapsed schedule."""
        interval = dt.timedelta(minutes=5)
        schedule = Elapsed(interval)
        assert schedule.interval == interval
        assert schedule.last_triggered is None

    @pytest.mark.parametrize(
        "current_time,last_triggered,interval,expected",
        [
            # Should trigger when enough time has elapsed
            ("2023-01-01T12:05:00+00:00", "2023-01-01T12:00:00+00:00", dt.timedelta(minutes=5), True),
            # Should not trigger when not enough time has elapsed
            ("2023-01-01T12:04:59+00:00", "2023-01-01T12:00:00+00:00", dt.timedelta(minutes=5), False),
            # Should trigger when exactly the interval has passed
            ("2023-01-01T12:05:00+00:00", "2023-01-01T12:00:00+00:00", dt.timedelta(minutes=5), True),
            # Should trigger when more than interval has passed
            ("2023-01-01T12:10:00+00:00", "2023-01-01T12:00:00+00:00", dt.timedelta(minutes=5), True),
        ],
    )
    def test_check(self, current_time: str, last_triggered: str, interval: dt.timedelta, expected: bool):
        """Test check method with various scenarios."""
        schedule = Elapsed(interval)
        if last_triggered:
            schedule.last_triggered = dt.datetime.fromisoformat(last_triggered)

        mock_now = dt.datetime.fromisoformat(current_time)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            result = schedule.check()
            assert result == expected

            # If triggered, verify last_triggered was updated
            if result:
                assert schedule.last_triggered == mock_now

    def test_check_first_call(self):
        """Test first check call when last_triggered is None."""
        schedule = Elapsed(dt.timedelta(minutes=5))
        mock_now = dt.datetime.fromisoformat("2023-01-01T12:00:00+00:00")

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            assert schedule.check() is True
            assert schedule.last_triggered == mock_now

    def test_check_ignores_parameters(self):
        """Test that check method ignores optional parameters."""
        schedule = Elapsed(dt.timedelta(minutes=5))
        mock_now = dt.datetime.fromisoformat("2023-01-01T12:00:00+00:00")

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            result = schedule.check(stage="train", batch_idx=1, step=100)
            assert result is True

    def test_multiple_checks(self):
        """Test multiple consecutive checks."""
        schedule = Elapsed(dt.timedelta(minutes=5))

        # First check
        mock_now = dt.datetime.fromisoformat("2023-01-01T12:00:00+00:00")
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            assert schedule.check() is True

        # Check before interval has elapsed
        mock_now = dt.datetime.fromisoformat("2023-01-01T12:04:59+00:00")
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            assert schedule.check() is False

        # Check after interval has elapsed
        mock_now = dt.datetime.fromisoformat("2023-01-01T12:05:00+00:00")
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            assert schedule.check() is True

    def test_timezone_awareness(self):
        """Test that timezone-aware datetimes are handled correctly."""
        schedule = Elapsed(dt.timedelta(minutes=5))
        mock_now = dt.datetime.fromisoformat("2023-01-01T12:00:00+00:00")

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            assert schedule.check() is True


class TestRankSchedules(unittest.TestCase):
    """Test suite for rank-based schedules."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_trainer = MagicMock(spec=L.Trainer)

    def test_global_rank_init(self):
        """Test initialization of GlobalRank schedule."""
        # Test with single rank
        schedule = GlobalRank(0)
        assert schedule.ranks == (0,)

        # Test with multiple ranks
        schedule = GlobalRank(0, 1, 2)
        assert schedule.ranks == (0, 1, 2)

    def test_local_rank_init(self):
        """Test initialization of LocalRank schedule."""
        # Test with single rank
        schedule = LocalRank(0)
        assert schedule.ranks == (0,)

        # Test with multiple ranks
        schedule = LocalRank(0, 1, 2)
        assert schedule.ranks == (0, 1, 2)

    def test_global_rank_check_with_trainer(self):
        """Test GlobalRank.check with a trainer instance."""
        schedule = GlobalRank(0, 2)

        # Test with rank in the specified ranks
        self.mock_trainer.global_rank = 0
        assert schedule.check(trainer=self.mock_trainer)

        self.mock_trainer.global_rank = 2
        assert schedule.check(trainer=self.mock_trainer)

        # Test with rank not in the specified ranks
        self.mock_trainer.global_rank = 1
        assert not schedule.check(trainer=self.mock_trainer)

        self.mock_trainer.global_rank = 3
        assert not schedule.check(trainer=self.mock_trainer)

    def test_local_rank_check_with_trainer(self):
        """Test LocalRank.check with a trainer instance."""
        schedule = LocalRank(0, 2)

        # Test with rank in the specified ranks
        self.mock_trainer.local_rank = 0
        assert schedule.check(trainer=self.mock_trainer)

        self.mock_trainer.local_rank = 2
        assert schedule.check(trainer=self.mock_trainer)

        # Test with rank not in the specified ranks
        self.mock_trainer.local_rank = 1
        assert not schedule.check(trainer=self.mock_trainer)

        self.mock_trainer.local_rank = 3
        assert not schedule.check(trainer=self.mock_trainer)

    def test_global_rank_check_without_trainer(self):
        """Test GlobalRank.check without a trainer instance."""
        schedule = GlobalRank(0)
        assert not schedule.check()
        assert not schedule.check(trainer=None)

    def test_local_rank_check_without_trainer(self):
        """Test LocalRank.check without a trainer instance."""
        schedule = LocalRank(0)
        assert not schedule.check()
        assert not schedule.check(trainer=None)

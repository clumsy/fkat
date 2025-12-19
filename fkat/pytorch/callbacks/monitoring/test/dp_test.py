# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest.mock import MagicMock, patch

from fkat.pytorch.callbacks.monitoring import dp
from fkat.pytorch.callbacks.monitoring.dp import (
    DpSyncMonitor,
    DistDpGroup,
    EnvDpGroup,
    MegatronDpGroup,
)
from fkat.pytorch.schedule import Always

_ = unittest.TestCase()


class TestDistDpGroup:
    def test_dp_group_info(self):
        with patch("torch.distributed.get_rank", return_value=6):
            strategy = DistDpGroup(dp_size=2)
            group_id, rank_in_group = strategy.dp_group_info()
            assert group_id == 3  # 6 // 2
            assert rank_in_group == 0  # 6 % 2


class TestEnvDpGroup:
    def test_dp_group_info_with_env_vars(self):
        with patch.dict(os.environ, {"RANK": "6", "WORLD_SIZE": "8"}):
            strategy = EnvDpGroup(dp_size=2)
            group_id, rank_in_group = strategy.dp_group_info()
            assert group_id == 3  # 6 // 2
            assert rank_in_group == 0  # 6 % 2

    def test_dp_group_info_no_env_vars(self):
        # Test default behavior when env vars are not set
        with patch.dict(os.environ, {}, clear=True):
            strategy = EnvDpGroup(dp_size=4)
            group_id, rank_in_group = strategy.dp_group_info()
            assert group_id == 0  # 0 // 4
            assert rank_in_group == 0  # 0 % 4


class TestMegatronDpGroup:
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_dp_group_info(self, mock_get_world_size, mock_get_rank):
        mock_group = MagicMock()
        # Set up mock to return different values for different calls
        mock_get_rank.side_effect = [2, 6]  # First call (with group) returns 2, second call (global) returns 6
        mock_get_world_size.return_value = 4  # group_size=4

        with patch.dict("sys.modules", {"megatron": MagicMock(), "megatron.core": MagicMock()}):
            with patch("megatron.core.parallel_state.get_data_parallel_group", return_value=mock_group):
                strategy = MegatronDpGroup()
                group_id, rank_in_group = strategy.dp_group_info()

        assert group_id == 1  # 6 // 4
        assert rank_in_group == 2


class TestDpSyncMonitor:
    @patch(f"{dp.__name__}.time.perf_counter")
    @patch(f"{dp.__name__}.CallbackLogger")
    def test_timing_measurement_with_dist_ddp(self, mock_cb_logger, mock_time):
        trainer = MagicMock()
        trainer.global_step = 0
        dp_group = DistDpGroup(dp_size=2)
        callback = DpSyncMonitor(dp_group=dp_group, schedule=Always())
        logger = mock_cb_logger.return_value
        callback.setup(trainer, MagicMock(), "train")

        # Mock timing
        start_time = 10.0
        end_time = 10.5
        mock_time.side_effect = [start_time, end_time]

        # Simulate batch processing
        callback.on_train_batch_start(trainer, MagicMock(), None, 0)
        assert callback.batch_start_time == start_time

        # Mock dp_group.dp_group_info to return group 1, rank 0
        with patch.object(dp_group, "dp_group_info", return_value=(1, 0)):
            callback.on_before_optimizer_step(trainer, MagicMock(), MagicMock())

        # Check timing calculation
        expected_time_s = end_time - start_time

        # Check logging was called with correct timing
        logger.log_batch.assert_called_once()
        args, kwargs = logger.log_batch.call_args
        assert kwargs["metrics"] == {"dp_sync/group1/sync_s": expected_time_s}

        # Check that batch_start_time was reset
        assert callback.batch_start_time is None

    @patch(f"{dp.__name__}.time.perf_counter")
    @patch(f"{dp.__name__}.CallbackLogger")
    def test_timing_measurement_with_fixed(self, mock_cb_logger, mock_time):
        trainer = MagicMock()
        trainer.global_step = 0
        dp_group = EnvDpGroup(dp_size=2)
        callback = DpSyncMonitor(dp_group=dp_group, schedule=Always())
        logger = mock_cb_logger.return_value
        callback.setup(trainer, MagicMock(), "train")

        # Mock timing
        start_time = 10.0
        end_time = 10.5
        mock_time.side_effect = [start_time, end_time]

        # Simulate batch processing
        callback.on_train_batch_start(trainer, MagicMock(), None, 0)
        assert callback.batch_start_time == start_time

        # Mock dp_group.dp_group_info to return group 1, rank 0
        with patch.object(dp_group, "dp_group_info", return_value=(1, 0)):
            callback.on_before_optimizer_step(trainer, MagicMock(), MagicMock())

        # Check timing calculation
        expected_time_s = end_time - start_time

        # Check logging was called with correct timing
        logger.log_batch.assert_called_once()
        args, kwargs = logger.log_batch.call_args
        assert kwargs["metrics"] == {"dp_sync/group1/sync_s": expected_time_s}

        # Check that batch_start_time was reset
        assert callback.batch_start_time is None

    @patch(f"{dp.__name__}.CallbackLogger")
    def test_only_dp_group_rank_zero_logs(self, mock_cb_logger):
        trainer = MagicMock()
        trainer.global_step = 0
        dp_group = DistDpGroup(dp_size=2)
        callback = DpSyncMonitor(dp_group=dp_group, schedule=Always())
        logger = mock_cb_logger.return_value
        callback.setup(trainer, MagicMock(), "train")

        sync_time_s = 0.5

        # Test non-rank-zero doesn't log
        with patch.object(dp_group, "dp_group_info", return_value=(0, 1)):  # group 0, rank 1
            callback._log_statistics(trainer, "train", 0, sync_time_s)
        logger.log_batch.assert_not_called()

        # Test rank-zero does log
        with patch.object(dp_group, "dp_group_info", return_value=(0, 0)):  # group 0, rank 0
            callback._log_statistics(trainer, "train", 0, sync_time_s)
        logger.log_batch.assert_called_once()

    @patch("torch.distributed.get_rank")
    def test_dist_ddp_strategy_rank_calculation(self, mock_get_rank):
        dp_group = DistDpGroup(dp_size=4)

        # Test rank 0 (0 % 4 == 0)
        mock_get_rank.return_value = 0
        group_id, rank_in_group = dp_group.dp_group_info()
        assert group_id == 0 and rank_in_group == 0

        # Test rank 4 (4 % 4 == 0)
        mock_get_rank.return_value = 4
        group_id, rank_in_group = dp_group.dp_group_info()
        assert group_id == 1 and rank_in_group == 0

        # Test rank 1 (1 % 4 != 0)
        mock_get_rank.return_value = 1
        group_id, rank_in_group = dp_group.dp_group_info()
        assert group_id == 0 and rank_in_group == 1

        # Test rank 3 (3 % 4 != 0)
        mock_get_rank.return_value = 3
        group_id, rank_in_group = dp_group.dp_group_info()
        assert group_id == 0 and rank_in_group == 3

    def test_fixed_strategy_rank_calculation(self):
        dp_group = EnvDpGroup(dp_size=4)

        # Test rank 0 (0 % 4 == 0)
        with patch.dict(os.environ, {"RANK": "0"}):
            group_id, rank_in_group = dp_group.dp_group_info()
            assert group_id == 0 and rank_in_group == 0

        # Test rank 4 (4 % 4 == 0)
        with patch.dict(os.environ, {"RANK": "4"}):
            group_id, rank_in_group = dp_group.dp_group_info()
            assert group_id == 1 and rank_in_group == 0

        # Test rank 1 (1 % 4 != 0)
        with patch.dict(os.environ, {"RANK": "1"}):
            group_id, rank_in_group = dp_group.dp_group_info()
            assert group_id == 0 and rank_in_group == 1

        # Test rank 3 (3 % 4 != 0)
        with patch.dict(os.environ, {"RANK": "3"}):
            group_id, rank_in_group = dp_group.dp_group_info()
            assert group_id == 0 and rank_in_group == 3

    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_megatron_strategy_rank_calculation(self, mock_get_world_size, mock_get_rank):
        dp_group = MegatronDpGroup()

        mock_group = MagicMock()
        mock_get_world_size.return_value = 4
        with patch.dict("sys.modules", {"megatron": MagicMock(), "megatron.core": MagicMock()}):
            with patch("megatron.core.parallel_state.get_data_parallel_group", return_value=mock_group):
                # Test rank 0 in DP group
                mock_get_rank.side_effect = [0, 8]  # First call (with group) returns 0, second call (global) returns 8
                group_id, rank_in_group = dp_group.dp_group_info()
                assert group_id == 2 and rank_in_group == 0  # 8 // 4 = 2, rank 0

                # Test non-rank 0 in DP group
                mock_get_rank.side_effect = [1, 9]  # First call (with group) returns 1, second call (global) returns 9
                group_id, rank_in_group = dp_group.dp_group_info()
                assert group_id == 2 and rank_in_group == 1  # 9 // 4 = 2, rank 1

    @patch(f"{dp.__name__}.CallbackLogger")
    def test_schedule_controls_logging(self, mock_cb_logger):
        trainer = MagicMock()
        trainer.global_step = 0

        # Create schedule that never logs
        schedule = MagicMock()
        schedule.check.return_value = False

        dp_group = DistDpGroup(dp_size=2)
        callback = DpSyncMonitor(dp_group=dp_group, schedule=schedule)
        logger = mock_cb_logger.return_value
        callback.setup(trainer, MagicMock(), "train")

        sync_time_s = 0.5

        with patch.object(dp_group, "dp_group_info", return_value=(0, 0)):  # group 0, rank 0
            callback._log_statistics(trainer, "train", 0, sync_time_s)

        # Should not log because schedule returned False
        logger.log_batch.assert_not_called()
        schedule.check.assert_called_once_with(stage="train", batch_idx=0, step=trainer.global_step, trainer=trainer)

    def test_batch_start_time_reset(self):
        """Test that batch_start_time is reset in on_before_optimizer_step."""
        dp_group = DistDpGroup(dp_size=2)
        callback = DpSyncMonitor(dp_group=dp_group)
        callback.batch_start_time = 10.0

        with patch.object(dp_group, "dp_group_info", return_value=(0, 1)):  # non-rank-zero, won't log
            callback.on_before_optimizer_step(MagicMock(), MagicMock(), MagicMock())

        assert callback.batch_start_time is None

    @patch(f"{dp.__name__}.time.perf_counter")
    def test_no_timing_without_batch_start(self, mock_time):
        dp_group = DistDpGroup(dp_size=2)
        callback = DpSyncMonitor(dp_group=dp_group)
        callback.batch_start_time = None

        with patch.object(dp_group, "dp_group_info", return_value=(0, 0)):  # rank 0, would log if timing existed
            callback.on_before_optimizer_step(MagicMock(), MagicMock(), MagicMock())

        # Should not call perf_counter if batch_start_time is None
        mock_time.assert_not_called()
        assert callback.batch_start_time is None

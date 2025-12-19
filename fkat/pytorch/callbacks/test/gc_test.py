# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch, MagicMock
from typing import Any

import lightning as L

from fkat.pytorch.schedule import Every
from fkat.pytorch.callbacks import gc
from fkat.pytorch.callbacks.gc import ManualGc
from fkat.pytorch.loggers import LightningLogger


class TestManualGc(unittest.TestCase):
    def setUp(self):
        self.trainer = MagicMock(spec=L.Trainer)
        self.trainer.global_step = 0
        self.module = MagicMock(spec=L.LightningModule)

    @patch("gc.disable")
    def test_setup_with_gc_enabled(self, mock_disable: Any):
        """Test setup doesn't disable GC when Never"""
        callback = ManualGc()
        callback.setup(self.trainer, self.module, "fit")
        mock_disable.assert_not_called()

    @patch("gc.disable")
    def test_setup_with_gc_disabled(self, mock_disable: Any):
        """Test setup disables GC when not Never"""
        callback = ManualGc(schedule=Every(n_batches=1))
        callback.setup(self.trainer, self.module, "fit")
        mock_disable.assert_called_once()

    @patch("gc.collect")
    def test_maybe_gc_no_collection(self, mock_collect: Any):
        """Test no GC collection when Never"""
        callback = ManualGc()
        callback.maybe_gc(self.trainer, "train", batch_idx=5)
        mock_collect.assert_not_called()

    @patch("gc.collect")
    def test_maybe_gc_collection_on_interval(self, mock_collect: Any):
        """Test GC collection occurs on specified interval"""
        callback = ManualGc(Every(n_batches=2))

        # Should collect
        callback.maybe_gc(self.trainer, "train", batch_idx=0)
        callback.maybe_gc(self.trainer, "train", batch_idx=2)
        callback.maybe_gc(self.trainer, "train", batch_idx=4)

        # Should not collect
        callback.maybe_gc(self.trainer, "train", batch_idx=1)
        callback.maybe_gc(self.trainer, "train", batch_idx=3)
        callback.maybe_gc(self.trainer, "train", batch_idx=5)

        assert mock_collect.call_count == 3

    @patch("gc.collect")
    def test_train_epoch_end(self, mock_collect: Any):
        """Test GC collection at end of training epoch"""
        callback = ManualGc(Every(n_batches=100))  # Large interval
        callback.on_train_epoch_end(self.trainer, self.module)
        mock_collect.assert_called_once()

    @patch("gc.collect")
    def test_train_batch_end(self, mock_collect: Any):
        """Test GC collection in training stage"""
        callback = ManualGc(Every(n_batches=2))

        # Should collect
        callback.on_train_batch_end(self.trainer, self.module, None, None, 0)
        callback.on_train_batch_end(self.trainer, self.module, None, None, 2)

        # Should not collect
        callback.on_train_batch_end(self.trainer, self.module, None, None, 1)

        assert mock_collect.call_count == 2

    @patch("gc.collect")
    def test_validation_epoch_end(self, mock_collect: Any):
        """Test GC collection at end of validation epoch"""
        callback = ManualGc(Every(n_batches=100))  # Large interval
        callback.on_validation_epoch_end(self.trainer, self.module)
        mock_collect.assert_called_once()

    @patch("gc.collect")
    def test_validation_batch_end(self, mock_collect: Any):
        """Test GC collection in validation stage"""
        callback = ManualGc(Every(n_batches=2))

        # Should collect
        callback.on_validation_batch_end(self.trainer, self.module, None, None, 0)
        callback.on_validation_batch_end(self.trainer, self.module, None, None, 2)

        # Should not collect
        callback.on_validation_batch_end(self.trainer, self.module, None, None, 1)

        assert mock_collect.call_count == 2

    @patch("gc.collect")
    def test_test_epoch_end(self, mock_collect: Any):
        """Test GC collection at end of test epoch"""
        callback = ManualGc(Every(n_batches=100))  # Large interval
        callback.on_test_epoch_end(self.trainer, self.module)
        mock_collect.assert_called_once()

    @patch("gc.collect")
    def test_test_batch_end(self, mock_collect: Any):
        """Test GC collection in test stage"""
        callback = ManualGc(Every(n_batches=2))

        # Should collect
        callback.on_test_batch_end(self.trainer, self.module, None, None, 0)
        callback.on_test_batch_end(self.trainer, self.module, None, None, 2)

        # Should not collect
        callback.on_test_batch_end(self.trainer, self.module, None, None, 1)

        assert mock_collect.call_count == 2

    @patch("gc.collect")
    def test_predict_epoch_end(self, mock_collect: Any):
        """Test GC collection at end of predict epoch"""
        callback = ManualGc(Every(n_batches=100))  # Large interval
        callback.on_predict_epoch_end(self.trainer, self.module)
        mock_collect.assert_called_once()

    @patch("gc.collect")
    def test_predict_batch_end(self, mock_collect: Any):
        """Test GC collection in predict stage"""
        callback = ManualGc(Every(n_batches=2))

        # Should collect
        callback.on_predict_batch_end(self.trainer, self.module, None, None, 0)
        callback.on_predict_batch_end(self.trainer, self.module, None, None, 2)

        # Should not collect
        callback.on_predict_batch_end(self.trainer, self.module, None, None, 1)

        assert mock_collect.call_count == 2

    def test_default_stats_parameters(self):
        """Test default values for stats parameters"""
        callback = ManualGc()

        # Check default values
        assert callback.stats == {
            "0": ["collections", "collected", "uncollected"],
            "1": ["collections", "collected", "uncollected"],
            "2": ["collections", "collected", "uncollected"],
        }

    def test_custom_stats_parameters(self):
        """Test custom values for stats parameters"""
        custom_stats = {"0": ["collections"], "1": ["collected"], "2": ["uncollected"]}
        callback = ManualGc(stats=custom_stats)

        # Check custom values
        assert callback.stats == custom_stats

    @patch(f"{gc.__name__}.CallbackLogger")
    def test_setup_initializes_logger(self, mock_logger_class: Any):
        """Test that setup initializes the callback logger"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        callback = ManualGc()
        callback.setup(self.trainer, self.module, "fit")

        # Check that logger was initialized
        mock_logger_class.assert_called_once_with(self.trainer)
        assert callback._cb_logger == mock_logger

    @patch("gc.collect")
    @patch(f"{gc.__name__}.perf_counter")
    @patch("gc.get_stats")
    def test_maybe_gc_with_stats_logging(self, mock_get_stats: Any, mock_perf_counter: Any, mock_collect: Any):
        """Test GC collection with stats logging"""
        # Setup mocks
        mock_perf_counter.side_effect = [10.0, 10.5]  # Start and end times
        mock_get_stats.return_value = [
            {"collections": 1, "collected": 10, "uncollected": 0},
            {"collections": 2, "collected": 20, "uncollected": 1},
            {"collections": 3, "collected": 30, "uncollected": 2},
        ]

        # Setup callback with mock logger
        callback = ManualGc(schedule=Every(n_batches=1))
        callback._cb_logger = MagicMock(spec=LightningLogger)

        # Call the method
        callback.maybe_gc(self.trainer, "train", batch_idx=0)

        # Verify GC was called
        mock_collect.assert_called_once()

        # Verify stats were logged
        callback._cb_logger.log_batch.assert_called_once()
        actual_call = callback._cb_logger.log_batch.call_args

        # Check that metrics contain the expected keys
        metrics = actual_call[1]["metrics"]
        assert f"gc/rank{self.trainer.global_rank}/time" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen0/collections" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen0/collected" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen0/uncollected" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen1/collections" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen1/collected" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen1/uncollected" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen2/collections" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen2/collected" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen2/uncollected" in metrics

        # Check values
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen0/collections"] == 1.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen0/collected"] == 10.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen0/uncollected"] == 0.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen1/collections"] == 2.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen1/collected"] == 20.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen1/uncollected"] == 1.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen2/collections"] == 3.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen2/collected"] == 30.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen2/uncollected"] == 2.0

        # Check other parameters
        assert actual_call[1]["step"] == 0

    @patch("gc.collect")
    @patch(f"{gc.__name__}.perf_counter")
    @patch("gc.get_stats")
    def test_maybe_gc_with_custom_stats(self, mock_get_stats: Any, mock_perf_counter: Any, mock_collect: Any):
        """Test GC collection with custom stats configuration"""
        # Setup mocks
        mock_perf_counter.side_effect = [10.0, 10.5]  # Start and end times
        mock_get_stats.return_value = [
            {"collections": 1, "collected": 10, "uncollected": 0},
            {"collections": 2, "collected": 20, "uncollected": 1},
            {"collections": 3, "collected": 30, "uncollected": 2},
        ]

        # Setup callback with custom stats and mock logger
        custom_stats = {"0": ["collections"], "1": ["collected"], "2": ["uncollected"]}
        callback = ManualGc(schedule=Every(n_batches=1), stats=custom_stats)
        callback._cb_logger = MagicMock(spec=LightningLogger)

        # Call the method
        callback.maybe_gc(self.trainer, "train", batch_idx=0)

        # Verify stats were logged according to custom configuration
        callback._cb_logger.log_batch.assert_called_once()
        actual_call = callback._cb_logger.log_batch.call_args
        metrics = actual_call[1]["metrics"]

        # Check that metrics contain only the expected keys
        assert f"gc/rank{self.trainer.global_rank}/time" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen0/collections" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen1/collected" in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen2/uncollected" in metrics

        # Check that metrics don't contain other keys
        assert f"gc/rank{self.trainer.global_rank}/gen0/collected" not in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen0/uncollected" not in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen1/collections" not in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen1/uncollected" not in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen2/collections" not in metrics
        assert f"gc/rank{self.trainer.global_rank}/gen2/collected" not in metrics

        # Check values
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen0/collections"] == 1.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen1/collected"] == 20.0
        assert metrics[f"gc/rank{self.trainer.global_rank}/gen2/uncollected"] == 2.0

    @patch("gc.collect")
    @patch(f"{gc.__name__}.perf_counter")
    @patch("gc.get_stats")
    def test_maybe_gc_with_different_schedules(self, mock_get_stats: Any, mock_perf_counter: Any, mock_collect: Any):
        """Test GC collection and stats logging with different schedules"""
        # This test is removed as the stats_schedule parameter is not in the implementation
        pass

    @patch("gc.collect")
    @patch(f"{gc.__name__}.perf_counter")
    @patch("gc.get_stats")
    def test_maybe_gc_no_logger(self, mock_get_stats: Any, mock_perf_counter: Any, mock_collect: Any):
        """Test GC collection without a logger"""
        # Setup callback without initializing the logger
        callback = ManualGc(schedule=Every(n_batches=1))

        # Call the method
        callback.maybe_gc(self.trainer, "train", batch_idx=0)

        # Verify GC was called but no errors occurred due to missing logger
        mock_collect.assert_called_once()

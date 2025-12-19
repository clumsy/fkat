# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch, MagicMock
from typing import Any

import lightning as L

from fkat.pytorch.schedule import Every, Fixed
from fkat.pytorch.callbacks.cuda.cache import EmptyCache


class TestEmptyCache(unittest.TestCase):
    def setUp(self):
        self.trainer = MagicMock(spec=L.Trainer)
        self.trainer.global_step = 0
        self.module = MagicMock(spec=L.LightningModule)

    @patch("torch.cuda.empty_cache")
    def test_maybe_empty_cache_no_execution(self, mock_empty_cache: Any):
        """Test no cache emptying when using Never schedule"""
        callback = EmptyCache()
        callback.maybe_empty_cache(self.trainer, "train", batch_idx=5)
        mock_empty_cache.assert_not_called()

    @patch("torch.cuda.empty_cache")
    def test_maybe_empty_cache_execution_on_interval(self, mock_empty_cache: Any):
        """Test cache emptying occurs on specified interval"""
        callback = EmptyCache(Every(n_batches=2))

        # Should empty cache
        callback.maybe_empty_cache(self.trainer, "train", batch_idx=0)
        callback.maybe_empty_cache(self.trainer, "train", batch_idx=2)
        callback.maybe_empty_cache(self.trainer, "train", batch_idx=4)

        # Should not empty cache
        callback.maybe_empty_cache(self.trainer, "train", batch_idx=1)
        callback.maybe_empty_cache(self.trainer, "train", batch_idx=3)
        callback.maybe_empty_cache(self.trainer, "train", batch_idx=5)

        assert mock_empty_cache.call_count == 3

    @patch("torch.cuda.empty_cache")
    def test_train_epoch_end(self, mock_empty_cache: Any):
        """Test cache emptying at end of training epoch"""
        # Use Fixed schedule that will always return True for step 0
        callback = EmptyCache(Fixed(warmup_steps=0, active_steps=1))
        callback.on_train_epoch_end(self.trainer, self.module)
        mock_empty_cache.assert_called_once()

    @patch("torch.cuda.empty_cache")
    def test_train_batch_end(self, mock_empty_cache: Any):
        """Test cache emptying in training stage"""
        callback = EmptyCache(Every(n_batches=2))

        # Should empty cache
        callback.on_train_batch_end(self.trainer, self.module, None, None, 0)
        callback.on_train_batch_end(self.trainer, self.module, None, None, 2)

        # Should not empty cache
        callback.on_train_batch_end(self.trainer, self.module, None, None, 1)

        assert mock_empty_cache.call_count == 2

    @patch("torch.cuda.empty_cache")
    def test_validation_epoch_end(self, mock_empty_cache: Any):
        """Test cache emptying at end of validation epoch"""
        # Use Fixed schedule that will always return True for step 0
        callback = EmptyCache(Fixed(warmup_steps=0, active_steps=1))
        callback.on_validation_epoch_end(self.trainer, self.module)
        mock_empty_cache.assert_called_once()

    @patch("torch.cuda.empty_cache")
    def test_validation_batch_end(self, mock_empty_cache: Any):
        """Test cache emptying in validation stage"""
        callback = EmptyCache(Every(n_batches=2))

        # Should empty cache
        callback.on_validation_batch_end(self.trainer, self.module, None, None, 0)
        callback.on_validation_batch_end(self.trainer, self.module, None, None, 2)

        # Should not empty cache
        callback.on_validation_batch_end(self.trainer, self.module, None, None, 1)

        assert mock_empty_cache.call_count == 2

    @patch("torch.cuda.empty_cache")
    def test_test_epoch_end(self, mock_empty_cache: Any):
        """Test cache emptying at end of test epoch"""
        # Use Fixed schedule that will always return True for step 0
        callback = EmptyCache(Fixed(warmup_steps=0, active_steps=1))
        callback.on_test_epoch_end(self.trainer, self.module)
        mock_empty_cache.assert_called_once()

    @patch("torch.cuda.empty_cache")
    def test_test_batch_end(self, mock_empty_cache: Any):
        """Test cache emptying in test stage"""
        callback = EmptyCache(Every(n_batches=2))

        # Should empty cache
        callback.on_test_batch_end(self.trainer, self.module, None, None, 0)
        callback.on_test_batch_end(self.trainer, self.module, None, None, 2)

        # Should not empty cache
        callback.on_test_batch_end(self.trainer, self.module, None, None, 1)

        assert mock_empty_cache.call_count == 2

    @patch("torch.cuda.empty_cache")
    def test_predict_epoch_end(self, mock_empty_cache: Any):
        """Test cache emptying at end of predict epoch"""
        # Use Fixed schedule that will always return True for step 0
        callback = EmptyCache(Fixed(warmup_steps=0, active_steps=1))
        callback.on_predict_epoch_end(self.trainer, self.module)
        mock_empty_cache.assert_called_once()

    @patch("torch.cuda.empty_cache")
    def test_predict_batch_end(self, mock_empty_cache: Any):
        """Test cache emptying in predict stage"""
        callback = EmptyCache(Every(n_batches=2))

        # Should empty cache
        callback.on_predict_batch_end(self.trainer, self.module, None, None, 0)
        callback.on_predict_batch_end(self.trainer, self.module, None, None, 2)

        # Should not empty cache
        callback.on_predict_batch_end(self.trainer, self.module, None, None, 1)

        assert mock_empty_cache.call_count == 2

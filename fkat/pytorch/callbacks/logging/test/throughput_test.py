# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import unittest
from unittest.mock import MagicMock, ANY, patch

from fkat.pytorch.callbacks.logging import throughput
from fkat.pytorch.callbacks.logging.throughput import Throughput

_ = unittest.TestCase()


class TestThroughput:
    def test_only_one_instance_running(self):
        callback = Throughput()
        trainer = MagicMock()
        trainer.callbacks = [
            callback,
            callback,
        ]
        with _.assertRaises(AssertionError):
            callback.setup(trainer, MagicMock(), "train")

    @patch(f"{throughput.__name__}.extract_batch_size")
    @patch(f"{throughput.__name__}.time")
    @patch(f"{throughput.__name__}.CallbackLogger")
    def test_logs_train_metrics(self, mock_cb_logger, mock_time, mock_extract_batch_size):
        trainer = MagicMock()
        trainer.global_step = 0
        trainer.callbacks = [(callback := Throughput(dp_ranks=42))]
        logger = mock_cb_logger.return_value
        callback.setup(trainer, MagicMock(), "train")
        assert callback.dp_ranks is not None

        batch, batch_size = MagicMock(), 42
        mock_extract_batch_size.assert_not_called()
        mock_extract_batch_size.return_value = batch_size
        mock_time.return_value = start = 24
        pl_module = MagicMock()
        callback.on_train_epoch_start(trainer, pl_module)
        callback.on_train_batch_start(trainer, MagicMock(), batch, 0)
        assert callback.step_start_time["train"]

        callback.on_before_zero_grad(MagicMock(), MagicMock(), MagicMock())
        mock_time.return_value = end = 42
        callback.on_train_batch_end(trainer, pl_module, None, batch, 0)
        step_time = end - start

        callback.on_train_end(trainer, pl_module)
        assert callback.total_time["train"] == step_time
        assert callback.total_samples["train"] == callback.total_samples["train"] == batch_size
        args, kwargs = logger.log_batch.call_args_list[0]
        assert kwargs["metrics"] == {
            "train/steps/step_time": step_time,
            "train/throughput/current_rank0": (cur_tput := batch_size / step_time),
            "train/throughput/current": callback.dp_ranks * cur_tput,
            "train/throughput/running_avg_rank0": cur_tput,
            "train/throughput/running_avg": callback.dp_ranks * cur_tput,
        }
        logger.reset_mock()
        batch_size = 52
        mock_extract_batch_size.return_value = batch_size
        trainer.global_step = 1

        callback.on_train_batch_start(trainer, MagicMock(), batch, 1)

    @pytest.mark.parametrize("stage", ["validation", "test", "predict"])
    @patch(f"{throughput.__name__}.extract_batch_size")
    @patch(f"{throughput.__name__}.time")
    @patch(f"{throughput.__name__}.CallbackLogger")
    def test_logs_metrics(self, mock_cb_logger, mock_time, mock_extract_batch_size, stage: str):
        trainer = MagicMock()
        trainer.callbacks = [(callback := Throughput())]
        logger = mock_cb_logger.return_value
        trainer.world_size = 1
        callback.setup(trainer, MagicMock(), stage)

        batch, batch_size = MagicMock(), 42
        mock_extract_batch_size.assert_not_called()
        mock_extract_batch_size.return_value = batch_size
        mock_time.return_value = start = 24
        getattr(callback, f"on_{stage}_epoch_start")(trainer, None)
        getattr(callback, f"on_{stage}_batch_start")(trainer, None, batch, 0, 0)
        mock_time.return_value = end = 42
        getattr(callback, f"on_{stage}_batch_end")(trainer, None, None, batch, 0, 0)
        step_time = end - start

        getattr(callback, f"on_{stage}_end")(trainer, None)
        assert callback.total_time[stage] == step_time
        assert callback.total_samples[stage] == batch_size
        args, kwargs = logger.log_batch.call_args_list[0]
        assert kwargs["metrics"] == {
            f"{stage}/throughput/current_rank0": (cur_tput := batch_size / step_time),
            f"{stage}/throughput/current": trainer.world_size * cur_tput,
            f"{stage}/throughput/running_avg_rank0": (cur_tput := batch_size / step_time),
            f"{stage}/throughput/running_avg": cur_tput,
        }

    @pytest.mark.parametrize("stage", ["validation", "test", "predict"])
    @patch(f"{throughput.__name__}.CallbackLogger")
    def test_logs_epochs(self, mock_cb_logger, stage: str):
        trainer = MagicMock()
        trainer.callbacks = [(callback := Throughput())]
        logger = mock_cb_logger.return_value
        callback.setup(trainer, MagicMock(), stage)

        getattr(callback, f"on_{stage}_epoch_start")(trainer, None)
        assert callback.epoch_start_time[stage]
        getattr(callback, f"on_{stage}_epoch_end")(trainer, None)
        getattr(callback, f"on_{stage}_end")(trainer, None)

        args, kwargs = logger.log_batch.call_args_list[0]
        assert kwargs["metrics"] == {
            f"{stage}/epochs/epoch_time": ANY,
        }

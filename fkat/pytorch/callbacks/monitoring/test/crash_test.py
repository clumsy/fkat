# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
from unittest.mock import Mock, patch

import pytest
import lightning as L

from fkat.pytorch.callbacks.monitoring.crash import CrashDetector, _monitor_process


class TestCrashDetector:
    def test_init(self):
        callback = CrashDetector()
        assert callback.error_tag == "error"
        assert callback.crash_info_tag == "crash_info"
        assert callback._cb_logger is None
        assert callback._processes == []
        assert callback._queue is None

    def test_init_custom_tags(self):
        callback = CrashDetector(error_tag="err", crash_info_tag="crash")
        assert callback.error_tag == "err"
        assert callback.crash_info_tag == "crash"

    @patch("fkat.pytorch.callbacks.monitoring.crash.CallbackLogger")
    @patch("fkat.pytorch.callbacks.monitoring.crash.multiprocessing.Process")
    @patch("fkat.pytorch.callbacks.monitoring.crash.multiprocessing.Queue")
    def test_setup(self, mock_queue_class, mock_process_class, mock_logger_class):
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        mock_queue = Mock()
        mock_queue_class.return_value = mock_queue
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        trainer = Mock(spec=L.Trainer)
        trainer.local_rank = 0
        trainer.global_rank = 0
        pl_module = Mock(spec=L.LightningModule)

        callback = CrashDetector()
        callback.setup(trainer, pl_module, "fit")

        mock_logger_class.assert_called_once_with(trainer)
        mock_queue_class.assert_called_once()
        mock_process.start.assert_called_once()
        assert callback._cb_logger == mock_logger
        assert callback._queue == mock_queue
        assert len(callback._processes) == 1

    @patch("fkat.pytorch.callbacks.monitoring.crash.CallbackLogger")
    def test_setup_non_zero_rank(self, mock_logger_class):
        trainer = Mock(spec=L.Trainer)
        trainer.local_rank = 1
        pl_module = Mock(spec=L.LightningModule)

        callback = CrashDetector()
        callback.setup(trainer, pl_module, "fit")

        mock_logger_class.assert_not_called()
        assert callback._cb_logger is None

    def test_on_exception(self):
        mock_logger = Mock()
        trainer = Mock(spec=L.Trainer)
        trainer.global_rank = 0
        trainer.loggers = []
        pl_module = Mock(spec=L.LightningModule)

        callback = CrashDetector()
        callback._cb_logger = mock_logger

        exception = ValueError("Test error")
        callback.on_exception(trainer, pl_module, exception)

        assert mock_logger.log_tag.call_count == 2
        calls = mock_logger.log_tag.call_args_list
        assert calls[0][0][0] == "error"
        assert "ValueError" in calls[0][0][1]
        assert calls[1][0][0] == "crash_info"
        assert "pid" in calls[1][0][1]
        assert "rank" in calls[1][0][1]
        assert "stacktrace" in calls[1][0][1]

    def test_teardown_with_crash_info(self):
        mock_logger = Mock()
        mock_queue = Mock()
        crash_info = {"pid": 123, "rank": 0, "exit_code": 1}
        mock_queue.empty.side_effect = [False, True]
        mock_queue.get_nowait.return_value = crash_info

        trainer = Mock(spec=L.Trainer)
        trainer.global_rank = 0
        pl_module = Mock(spec=L.LightningModule)

        callback = CrashDetector()
        callback._cb_logger = mock_logger
        callback._queue = mock_queue
        callback.teardown(trainer, pl_module, "fit")

        mock_logger.log_tag.assert_called_once()
        assert mock_logger.log_tag.call_args[0][0] == "crash_info"

    def test_terminate_monitors(self):
        mock_process1 = Mock()
        mock_process1.is_alive.return_value = True
        mock_process2 = Mock()
        mock_process2.is_alive.return_value = False

        callback = CrashDetector()
        callback._processes = [mock_process1, mock_process2]
        callback._terminate_monitors()

        mock_process1.kill.assert_called_once()
        mock_process2.kill.assert_not_called()
        assert callback._processes == []


class TestMonitorProcess:
    @pytest.mark.timeout(2)
    def test_monitor_process_error(self):
        """Test monitor process handles errors gracefully."""
        queue = multiprocessing.Queue()
        # Test with a PID that will immediately fail - should log error and return
        _monitor_process(queue, 999999, 0)
        # Queue should be empty since error was caught
        assert queue.empty()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch, call
import signal

import pytest
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder

from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.monitoring.shutdown import (
    GracefulShutdown,
    detect_shutdown_from_logger,
    start_shutdown_detection_process,
)

MODULE = "fkat.pytorch.callbacks.monitoring.shutdown"  # Update to match your actual module path


class TestGracefulShutdown(unittest.TestCase):
    def assert_tags(self, cb_logger, shutdown_tag: str = "shutdown"):
        cb_logger.tags.assert_called()
        cb_logger.log_tag.assert_called_with(shutdown_tag, "SHUTTING_DOWN")

    def mock_shutdown_tags(self, cb_logger, shutdown_tag: str = "shutdown", shutdown_info_tag: str = "shutdown_info"):
        tags_dict = {shutdown_tag: "True", shutdown_info_tag: "[{'Strategy': 'GRACEFUL'}]"}
        cb_logger.tags.return_value = tags_dict

    @patch(f"{MODULE}.start_shutdown_detection_process")
    @patch(f"{MODULE}.CallbackLogger")
    @patch("lightning.Trainer")
    def test_checks_on_train_batch_start(self, mock_trainer, mock_cb_logger, mock_detection):
        # Arrange
        mock_trainer.should_stop = False
        mock_trainer.global_step = 0
        cb_logger = MagicMock()
        mock_cb_logger.return_value = cb_logger
        self.mock_shutdown_tags(cb_logger)
        callback = GracefulShutdown()
        callback.setup(mock_trainer, MagicMock(), "train")

        # Act
        pl_module = MagicMock()
        callback.on_train_batch_start(mock_trainer, pl_module, None, 0)
        callback.on_train_batch_start(mock_trainer, pl_module, None, 1)

        # Assert
        self.assert_tags(cb_logger)
        assert mock_trainer.should_stop
        mock_detection.assert_called_once_with(cb_logger, "shutdown", mock_trainer)

    @patch(f"{MODULE}.start_shutdown_detection_process")
    @patch(f"{MODULE}.CallbackLogger")
    @patch("lightning.Trainer")
    def test_checks_on_test_batch_start(self, mock_trainer, mock_cb_logger, mock_detection):
        # Arrange
        mock_trainer.should_stop = False
        mock_trainer.global_step = 0
        cb_logger = MagicMock()
        mock_cb_logger.return_value = cb_logger
        self.mock_shutdown_tags(cb_logger)
        callback = GracefulShutdown()
        callback.setup(mock_trainer, MagicMock(), "train")

        # Act
        pl_module = MagicMock()
        callback.on_test_batch_start(mock_trainer, pl_module, None, 0)
        callback.on_test_batch_start(mock_trainer, pl_module, None, 1)

        # Assert
        self.assert_tags(cb_logger)
        assert mock_trainer.should_stop
        mock_detection.assert_called_once_with(cb_logger, "shutdown", mock_trainer)

    @patch(f"{MODULE}.start_shutdown_detection_process")
    @patch(f"{MODULE}.CallbackLogger")
    @patch("lightning.Trainer")
    def test_checks_on_validation_batch_start(self, mock_trainer, mock_cb_logger, mock_detection):
        # Arrange
        mock_trainer.should_stop = False
        mock_trainer.global_step = 0
        cb_logger = MagicMock()
        mock_cb_logger.return_value = cb_logger
        self.mock_shutdown_tags(cb_logger)
        callback = GracefulShutdown()
        callback.setup(mock_trainer, MagicMock(), "train")

        # Act
        pl_module = MagicMock()
        callback.on_validation_batch_start(mock_trainer, pl_module, None, 0)
        callback.on_validation_batch_start(mock_trainer, pl_module, None, 1)

        # Assert
        self.assert_tags(cb_logger)
        assert mock_trainer.should_stop
        mock_detection.assert_called_once_with(cb_logger, "shutdown", mock_trainer)

    @patch(f"{MODULE}.start_shutdown_detection_process")
    @patch(f"{MODULE}.CallbackLogger")
    @patch("lightning.Trainer")
    def test_checks_on_predict_batch_start(self, mock_trainer, mock_cb_logger, mock_detection):
        # Arrange
        mock_trainer.should_stop = False
        mock_trainer.global_step = 0
        cb_logger = MagicMock()
        mock_cb_logger.return_value = cb_logger
        self.mock_shutdown_tags(cb_logger)
        callback = GracefulShutdown()
        callback.setup(mock_trainer, MagicMock(), "train")

        # Act
        pl_module = MagicMock()
        callback.on_predict_batch_start(mock_trainer, pl_module, None, 0)
        callback.on_predict_batch_start(mock_trainer, pl_module, None, 1)

        # Assert
        self.assert_tags(cb_logger)
        assert mock_trainer.should_stop
        mock_detection.assert_called_once_with(cb_logger, "shutdown", mock_trainer)

    @patch(f"{MODULE}.CallbackLogger")
    @patch("lightning.Trainer")
    def test_checks_teardown(self, mock_trainer, mock_cb_logger):
        # Arrange
        mock_trainer.global_rank = 0
        cb_logger = MagicMock()
        mock_cb_logger.return_value = cb_logger
        self.mock_shutdown_tags(cb_logger)
        callback = GracefulShutdown()
        callback._terminate_monitor = MagicMock()  # type: ignore[assignment]
        callback.setup(mock_trainer, MagicMock(), "train")

        # Act
        callback.teardown(mock_trainer, MagicMock(), "train")

        # Assert
        cb_logger.log_tag.assert_called_with("shutdown", "JOB_FINISHED")
        callback._terminate_monitor.assert_called_once_with()  # type: ignore[attr-defined]

    def test_on_exception(self):
        # Arrange
        callback = GracefulShutdown()
        callback._terminate_monitor = MagicMock()  # type: ignore[assignment]  # type: ignore[assignment]

        # Act
        callback.on_exception(MagicMock(), MagicMock(), Exception())

        # Assert
        callback._terminate_monitor.assert_called_once_with()  # type: ignore[attr-defined]  # type: ignore[attr-defined]

    @patch("multiprocessing.Process")
    def test_terminate_monitor(self, mock_process):
        # Arrange
        callback = GracefulShutdown()
        callback._process = subprocess = MagicMock()
        subprocess.is_alive.return_value = True

        # Act
        callback._terminate_monitor()

        # Assert
        subprocess.is_alive.assert_called_once()
        subprocess.kill.assert_called_once()

    @patch(f"{MODULE}.CallbackLogger")
    @patch("lightning.Trainer")
    def test_tuning_detection(self, mock_trainer, mock_cb_logger):
        # Arrange
        mock_trainer.global_rank = 0
        mock_trainer.callbacks = [MagicMock(spec=LearningRateFinder)]
        cb_logger = MagicMock()
        mock_cb_logger.return_value = cb_logger
        callback = GracefulShutdown()
        callback._terminate_monitor = MagicMock()  # type: ignore[assignment]  # type: ignore[assignment]
        callback.setup(mock_trainer, MagicMock(), "train")

        # Act
        callback.teardown(mock_trainer, MagicMock(), "train")

        # Assert
        # Should not log job finished during tuning
        cb_logger.log_tag.assert_not_called()
        callback._terminate_monitor.assert_called_once_with()  # type: ignore[attr-defined]  # type: ignore

    @patch(f"{MODULE}.CallbackLogger")
    @patch("lightning.Trainer")
    def test_schedule_check(self, mock_trainer, mock_cb_logger):
        # Arrange
        mock_trainer.should_stop = False
        mock_trainer.global_step = 0
        cb_logger = MagicMock()
        mock_cb_logger.return_value = cb_logger
        self.mock_shutdown_tags(cb_logger)

        # Create a mock schedule that returns True after a certain batch index
        mock_schedule = MagicMock()
        mock_schedule.check.side_effect = lambda stage, batch_idx, step, trainer: batch_idx > 5

        callback = GracefulShutdown(schedule=mock_schedule)
        callback.setup(mock_trainer, MagicMock(), "train")

        # Act - Should not stop yet
        pl_module = MagicMock()
        callback.on_train_batch_start(mock_trainer, pl_module, None, 3)

        # Assert
        assert not mock_trainer.should_stop
        mock_schedule.check.assert_called_with(stage="train", batch_idx=3, step=0, trainer=mock_trainer)

        self.mock_shutdown_tags(cb_logger)

        # Act - Should stop now due to schedule
        callback.on_train_batch_start(mock_trainer, pl_module, None, 6)

        # Assert
        assert mock_trainer.should_stop

    @patch(f"{MODULE}.start_shutdown_detection_process")
    @patch(f"{MODULE}.CallbackLogger")
    @patch("lightning.Trainer")
    def test_custom_shutdown_tags(self, mock_trainer, mock_cb_logger, mock_detection):
        # Arrange
        mock_trainer.should_stop = False
        mock_trainer.global_step = 0
        cb_logger = MagicMock()
        mock_cb_logger.return_value = cb_logger

        custom_shutdown_tag = "custom_shutdown"
        custom_info_tag = "custom_info"

        # Set up the mock to return tags when called
        cb_logger.tags.return_value = {custom_shutdown_tag: "True", custom_info_tag: "[{'Strategy': 'GRACEFUL'}]"}

        callback = GracefulShutdown(shutdown_tag=custom_shutdown_tag, shutdown_info_tag=custom_info_tag)
        callback.setup(mock_trainer, MagicMock(), "train")

        # Act
        pl_module = MagicMock()
        callback.on_train_batch_start(mock_trainer, pl_module, None, 0)

        # Assert
        cb_logger.tags.assert_called_once()
        cb_logger.log_tag.assert_called_with(custom_shutdown_tag, "SHUTTING_DOWN")
        assert mock_trainer.should_stop
        mock_detection.assert_called_once_with(cb_logger, custom_shutdown_tag, mock_trainer)

    @patch("os.getpid")
    @patch("multiprocessing.Process")
    def test_start_shutdown_detection_process(self, mock_process_class, mock_getpid):
        # Arrange
        mock_logger = MagicMock(spec=LightningLogger)
        mock_trainer = MagicMock()
        mock_trainer.local_rank = 0
        mock_getpid.return_value = 12345

        # Create a mock process that doesn't actually start
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process

        # Act
        process = start_shutdown_detection_process(mock_logger, "shutdown", mock_trainer)

        # Assert
        mock_process_class.assert_called_once()
        assert process == mock_process
        mock_process.start.assert_called_once()

        # Check that the process was created with the right arguments
        args = mock_process_class.call_args[1]
        assert args["target"].__name__ == "detect_shutdown_from_logger"
        assert args["args"] == (mock_logger, "shutdown", 12345, 60)
        assert process.daemon  # type: ignore[possibly-unbound-attribute]

    @patch("os.getpid")
    @patch("multiprocessing.Process")
    def test_start_shutdown_detection_process_non_zero_rank(self, mock_process_class, mock_getpid):
        # Arrange
        mock_logger = MagicMock(spec=LightningLogger)
        mock_trainer = MagicMock()
        mock_trainer.local_rank = 1  # Non-zero rank

        # Act
        process = start_shutdown_detection_process(mock_logger, "shutdown", mock_trainer)

        # Assert
        mock_process_class.assert_not_called()
        assert process is None

    @patch("os.getenv")
    @patch("time.sleep")
    @patch("os.kill")
    @patch("random.uniform")
    def test_detect_shutdown_from_logger(self, mock_uniform, mock_kill, mock_sleep, mock_getenv):
        # Arrange
        mock_logger = MagicMock(spec=LightningLogger)
        mock_logger.tags.return_value = {"shutdown": "True"}
        mock_getenv.return_value = "30"  # 30 seconds interval
        mock_uniform.return_value = 5  # 5 seconds random delay
        pid = 12345

        # Make sleep raise an exception after the first call to break the infinite loop
        mock_sleep.side_effect = [None, SystemExit("Break the loop")]

        # Act - This will run until the exception is raised
        with pytest.raises(SystemExit):
            detect_shutdown_from_logger(mock_logger, "shutdown", pid, 60)

        # Assert
        mock_logger.tags.assert_called_once()
        mock_kill.assert_called_once_with(pid, signal.SIGABRT)

    @patch("os.getenv")
    @patch("time.sleep")
    @patch("os.kill")
    @patch("random.uniform")
    def test_detect_shutdown_from_logger_no_tag(self, mock_uniform, mock_kill, mock_sleep, mock_getenv):
        # Arrange
        mock_logger = MagicMock(spec=LightningLogger)
        mock_logger.tags.return_value = {}  # No shutdown tag
        mock_getenv.return_value = "30"  # 30 seconds interval
        mock_uniform.return_value = 5  # 5 seconds random delay
        pid = 12345

        # Set up sleep to raise an exception after the second call to break the infinite loop
        mock_sleep.side_effect = [None, SystemExit("Break the loop")]

        # Act
        with pytest.raises(SystemExit):
            detect_shutdown_from_logger(mock_logger, "shutdown", pid, 60)

        # Assert
        mock_logger.tags.assert_called_once()
        mock_kill.assert_not_called()  # Should not kill the process
        mock_sleep.assert_has_calls([call(5), call(25)])  # First random delay, then remaining time

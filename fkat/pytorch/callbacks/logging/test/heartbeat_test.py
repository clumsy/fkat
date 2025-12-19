# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock, patch
import datetime as dt

from fkat.pytorch.schedule import Elapsed
from fkat.pytorch.callbacks.logging import Heartbeat


class TestHeartbeat:
    @pytest.fixture
    def mock_schedule(self) -> MagicMock:
        mock = MagicMock()
        mock.check.return_value = True
        return mock

    @pytest.fixture
    def mock_trainer(self) -> MagicMock:
        mock = MagicMock()
        mock.global_step = 42
        return mock

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        mock = MagicMock()
        mock.tags.return_value = {"last_check_in_time": "2023-01-01T00:00:00+00:00", "last_check_in_step": "42"}
        return mock

    def test_init_default(self):
        """Test initialization with default parameters."""
        cb = Heartbeat()
        assert isinstance(cb.schedule, Elapsed)
        assert cb.last_check_in_time_tag == "last_check_in_time"
        assert cb.last_check_in_step_tag == "last_check_in_step"
        assert cb._cb_logger is None

    def test_init_custom(self, mock_schedule):
        """Test initialization with custom parameters."""
        cb = Heartbeat(
            schedule=mock_schedule,
            last_check_in_time_tag="time",
            last_check_in_step_tag="step",
        )
        assert cb.schedule == mock_schedule
        assert cb.last_check_in_time_tag == "time"
        assert cb.last_check_in_step_tag == "step"

    @patch("fkat.pytorch.callbacks.logging.heartbeat.CallbackLogger")
    def test_setup(
        self,
        mock_callback_logger,
        mock_schedule,
        mock_trainer,
        mock_logger,
    ):
        """Test setup method."""
        mock_callback_logger.return_value = mock_logger
        callback = Heartbeat(schedule=mock_schedule)

        callback.setup(mock_trainer, MagicMock(), "train")

        assert callback._cb_logger == mock_logger

    @patch("datetime.datetime")
    def test_publish_tags(self, mock_datetime, mock_schedule, mock_logger):
        """Test _publish_tags method."""
        mock_now = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
        mock_datetime.now.return_value = mock_now
        callback = Heartbeat(schedule=mock_schedule)

        callback._cb_logger = mock_logger
        mock_trainer = MagicMock()
        mock_trainer.global_step = 42
        callback._publish_tags("train", 0, mock_trainer)

        callback.schedule.check.assert_called_once_with(stage="train", batch_idx=0, step=42, trainer=mock_trainer)  # type: ignore[possibly-unbound-attribute]
        mock_logger.log_batch.assert_called_once_with(
            tags={"last_check_in_time": str(mock_now), "last_check_in_step": "42"}
        )

    @pytest.mark.parametrize(
        "method_name,stage",
        [
            ("on_train_batch_start", "train"),
            ("on_test_batch_start", "test"),
            ("on_validation_batch_start", "validation"),
            ("on_predict_batch_start", "predict"),
        ],
    )
    def test_batch_methods(self, method_name: str, stage: str, mock_schedule, mock_trainer, mock_logger):
        """Test all batch start methods."""
        callback = Heartbeat(schedule=mock_schedule)
        callback._cb_logger = mock_logger

        method = getattr(callback, method_name)
        method(mock_trainer, MagicMock(), MagicMock(), 0)

        callback.schedule.check.assert_called_once_with(  # type: ignore[possibly-unbound-attribute]
            stage=stage, batch_idx=0, step=mock_trainer.global_step, trainer=mock_trainer
        )

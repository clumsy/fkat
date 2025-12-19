# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
import unittest
from unittest.mock import MagicMock, patch

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.entities import Metric, RunTag
from mlflow.tracking import MlflowClient  # type: ignore[possibly-unbound-import]

from fkat.pytorch.callbacks import loggers
from fkat.pytorch.callbacks.loggers import (
    CallbackLogger,
    MLFlowCallbackLogger,
)


class TestMLFlowCallbackLogger(unittest.TestCase):
    @patch(f"{loggers.__name__}.broadcast_mlflow_run_id")
    def test_init(self, mock_broadcast):
        # Arrange
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_trainer.logger = (mock_logger := MagicMock(spec=MLFlowLogger))
        mock_logger._mlflow_client = MagicMock(spec=MlflowClient)
        mock_logger._run_id = "123"
        # Act
        MLFlowCallbackLogger(trainer=mock_trainer)
        # Assert
        mock_broadcast.assert_called_once_with(mock_logger, mock_trainer)

    @patch(f"{loggers.__name__}.broadcast_mlflow_run_id")
    def test_log_tag(self, mock_broadcast):
        # Arrange
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_trainer.logger = (mock_logger := MagicMock(spec=MLFlowLogger))
        mock_logger._mlflow_client = MagicMock(spec=MlflowClient)
        mock_logger._run_id = "123"
        # Act
        callback_logger = MLFlowCallbackLogger(trainer=mock_trainer)
        callback_logger.log_tag("k1", "v1")
        # Assert
        mock_logger._mlflow_client.set_tag.assert_called_once_with(run_id="123", key="k1", value="v1", synchronous=None)

    @patch(f"{loggers.__name__}.broadcast_mlflow_run_id")
    def test_log_batch(self, mock_broadcast):
        # Arrange
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_trainer.logger = (mock_logger := MagicMock(spec=MLFlowLogger))
        mock_logger._mlflow_client = MagicMock(spec=MlflowClient)
        mock_logger._run_id = "456"
        # Act
        callback_logger = MLFlowCallbackLogger(trainer=mock_trainer)
        callback_logger.log_batch(
            metrics={"m1": 1.234, "m2": 0.789},
            tags={"k0": "v0", "k1": "v1"},
            timestamp=1741825741,
            step=20,
        )
        # Assert
        mock_logger._mlflow_client.log_batch.assert_called_once_with(
            run_id="456",
            metrics=[
                Metric("m1", 1.234, 1741825741, 20),
                Metric("m2", 0.789, 1741825741, 20),
            ],
            params=[],
            tags=[RunTag("k0", "v0"), RunTag("k1", "v1")],
            synchronous=None,
        )

    @patch(f"{loggers.__name__}.broadcast_mlflow_run_id")
    def test_log_artifact(self, mock_broadcast):
        # Arrange
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_trainer.logger = (mock_logger := MagicMock(spec=MLFlowLogger))
        mock_logger._mlflow_client = MagicMock(spec=MlflowClient)
        mock_logger._run_id = "456"
        # Act
        callback_logger = MLFlowCallbackLogger(trainer=mock_trainer)
        callback_logger.log_artifact("/some/local/path", "prefix")
        # Assert
        callback_logger._client.log_artifact.assert_called_once_with(  # type: ignore[attr-defined]
            run_id="456", local_path="/some/local/path", artifact_path="prefix"
        )


class TestCallbackLogger(unittest.TestCase):
    @patch(f"{loggers.__name__}.MLFlowCallbackLogger")
    def setUp(self, mock_mlflow_logger):
        # Mock trainer and MLFlow logger
        self.trainer = MagicMock(spec=L.Trainer)
        self.mlflow_logger = MagicMock(spec=MLFlowLogger)
        self.trainer.logger = self.mlflow_logger

        self.callback_logger = CallbackLogger(self.trainer)

    def test_log_tag(self):
        # Arrange
        key = "test_key"
        value = "test_value"
        # Act
        self.callback_logger.log_tag(key=key, value=value)
        # Assert
        for logger in self.callback_logger.loggers:
            logger.log_tag.assert_called_once_with(key=key, value=value)  # type: ignore[attr-defined]

    def test_log_batch(self):
        # Arrange
        metrics = {"metric1": 1.0, "metric2": 2.0}
        tags = {"tag1": "value1", "tag2": "value2"}
        timestamp = int(time.time() * 1000)
        step = 1
        # Act
        self.callback_logger.log_batch(metrics=metrics, tags=tags, timestamp=timestamp, step=step)
        # Assert
        for logger in self.callback_logger.loggers:
            logger.log_batch.assert_called_once_with(metrics=metrics, tags=tags, timestamp=timestamp, step=step)  # type: ignore[attr-defined]

    def test_log_artifact(self):
        # Arrange
        local_path = "test_file.txt"
        artifact_path = "test_folder"
        # Act
        self.callback_logger.log_artifact(local_path=local_path, artifact_path=artifact_path)
        # Assert
        for logger in self.callback_logger.loggers:
            logger.log_artifact.assert_called_once_with(local_path=local_path, artifact_path=artifact_path)  # type: ignore[attr-defined]

    def tearDown(self):
        CallbackLogger._instance = None  # type: ignore[attr-defined]

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch, call

import lightning as L
from mlflow.tracking import MlflowClient  # type: ignore[possibly-unbound-import]

from fkat.pytorch import loggers
from fkat.pytorch.loggers import MLFlowLogger, CompositeLogger, LightningLogger


class TestMLFlowLogger(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.mock_client = MagicMock(spec=MlflowClient)
        self.run_id = "test-run-123"
        self.logger = MLFlowLogger(client=self.mock_client, run_id=self.run_id)

    @patch("fkat.pytorch.loggers.rank_zero_only")
    def test_log_tag(self, mock_rank_zero_only):
        # Arrange
        mock_rank_zero_only.return_value = lambda func: func  # Make decorator pass-through

        # Act
        self.logger.log_tag("test_key", "test_value")

        # Assert
        self.mock_client.set_tag.assert_called_once_with(
            run_id=self.run_id, key="test_key", value="test_value", synchronous=None
        )

    def test_tags(self):
        # Arrange
        mock_run = MagicMock()
        mock_run.data.tags = {"tag1": "value1", "tag2": "value2"}
        self.mock_client.get_run.return_value = mock_run

        # Act
        result = self.logger.tags()

        # Assert
        self.mock_client.get_run.assert_called_once_with(self.run_id)
        assert result == {"tag1": "value1", "tag2": "value2"}

    @patch("fkat.pytorch.loggers.rank_zero_only")
    def test_log_batch(self, mock_rank_zero_only):
        # Arrange
        mock_rank_zero_only.return_value = lambda func: func  # Make decorator pass-through
        metrics = {"accuracy": 0.95, "loss": 0.1}
        params = {"learning_rate": 0.001}
        tags = {"stage": "training"}
        timestamp = 1234567890
        step = 100

        # Act
        self.logger.log_batch(metrics=metrics, params=params, tags=tags, timestamp=timestamp, step=step)

        # Assert
        self.mock_client.log_batch.assert_called_once()
        call_args = self.mock_client.log_batch.call_args[1]

        assert call_args["run_id"] == self.run_id
        assert len(call_args["metrics"]) == 2
        assert len(call_args["params"]) == 1
        assert len(call_args["tags"]) == 1

        # Check metrics were properly converted to Metric objects
        metric_dict = {m.key: (m.value, m.timestamp, m.step) for m in call_args["metrics"]}
        assert metric_dict["accuracy"][0] == 0.95
        assert metric_dict["loss"][0] == 0.1
        assert metric_dict["accuracy"][1] == timestamp
        assert metric_dict["accuracy"][2] == step

    def test_log_artifact(self):
        # Arrange
        local_path = "/path/to/local/file.txt"
        artifact_path = "artifacts/files"

        # Act
        self.logger.log_artifact(local_path=local_path, artifact_path=artifact_path)

        # Assert
        self.mock_client.log_artifact.assert_called_once_with(
            run_id=self.run_id, local_path=local_path, artifact_path=artifact_path
        )

    @patch("fkat.pytorch.loggers.mlflow_logger")
    @patch("fkat.pytorch.loggers.assert_not_none")
    def test_init_with_trainer(self, mock_assert_not_none, mock_mlflow_logger):
        # Arrange
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_lightning_logger = MagicMock(spec=MLFlowLogger)
        mock_lightning_logger._mlflow_client = self.mock_client
        mock_lightning_logger._run_id = self.run_id
        mock_lightning_logger._log_batch_kwargs = {"synchronous": True}

        mock_mlflow_logger.return_value = mock_lightning_logger
        mock_assert_not_none.side_effect = lambda x, *args: x  # Return the input

        # Act
        logger = MLFlowLogger(trainer=mock_trainer)

        # Assert
        mock_mlflow_logger.assert_called_once_with(mock_trainer)
        assert logger._client == self.mock_client
        assert logger._run_id == self.run_id
        assert logger._synchronous


class TestCompositeLogger(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.mock_logger1 = MagicMock(spec=LightningLogger)
        self.mock_logger2 = MagicMock(spec=LightningLogger)
        self.loggers = [self.mock_logger1, self.mock_logger2]
        self.composite_logger = CompositeLogger(trainer=None, loggers=self.loggers)

    def test_log_tag(self):
        # Arrange
        key = "test_key"
        value = "test_value"

        # Act
        self.composite_logger.log_tag(key=key, value=value)

        # Assert
        self.mock_logger1.log_tag.assert_called_once_with(key=key, value=value)
        self.mock_logger2.log_tag.assert_called_once_with(key=key, value=value)

    def test_log_batch(self):
        # Arrange
        metrics = {"accuracy": 0.95}
        tags = {"stage": "validation"}
        timestamp = 1234567890
        step = 200

        # Act
        self.composite_logger.log_batch(metrics=metrics, tags=tags, timestamp=timestamp, step=step)

        # Assert
        expected_call = call(metrics=metrics, tags=tags, timestamp=timestamp, step=step)
        self.mock_logger1.log_batch.assert_has_calls([expected_call])
        self.mock_logger2.log_batch.assert_has_calls([expected_call])

    def test_log_artifact(self):
        # Arrange
        local_path = "/path/to/model.pt"
        artifact_path = "models"

        # Act
        self.composite_logger.log_artifact(local_path=local_path, artifact_path=artifact_path)

        # Assert
        expected_call = call(local_path=local_path, artifact_path=artifact_path)
        self.mock_logger1.log_artifact.assert_has_calls([expected_call])
        self.mock_logger2.log_artifact.assert_has_calls([expected_call])

    def test_tags(self):
        # Arrange
        self.mock_logger1.tags.return_value = {"tag1": "value1", "tag2": "value2"}
        self.mock_logger2.tags.return_value = {"tag2": "override", "tag3": "value3"}

        # Act
        result = self.composite_logger.tags()

        # Assert
        assert result == {"tag1": "value1", "tag2": "override", "tag3": "value3"}
        self.mock_logger1.tags.assert_called_once()
        self.mock_logger2.tags.assert_called_once()

    @patch(f"{loggers.__name__}.MLFlowLogger")
    def test_init_with_trainer(self, mock_mlflow_logger_cls):
        # Arrange
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_lightning_logger = MagicMock(spec=MLFlowLogger)
        mock_trainer.logger = [mock_lightning_logger]  # List of loggers

        # Make the mock a proper class for isinstance
        mock_mlflow_logger_cls.__bases__ = (object,)
        mock_lightning_logger.__class__ = mock_mlflow_logger_cls

        mock_mlflow_instance = MagicMock(spec=MLFlowLogger)
        mock_mlflow_logger_cls.return_value = mock_mlflow_instance

        # Act
        composite_logger = CompositeLogger(trainer=mock_trainer)

        # Assert
        mock_mlflow_logger_cls.assert_called_once_with(trainer=mock_trainer)
        assert len(composite_logger.loggers) == 1
        assert composite_logger.loggers[0] == mock_mlflow_instance

    @patch(f"{loggers.__name__}.MLFlowLogger")
    def test_init_with_single_logger(self, mock_mlflow_logger_cls):
        # Arrange
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_lightning_logger = MagicMock(spec=MLFlowLogger)
        mock_trainer.logger = mock_lightning_logger  # Single logger

        # Make the mock a proper class for isinstance
        mock_mlflow_logger_cls.__bases__ = (object,)
        mock_lightning_logger.__class__ = mock_mlflow_logger_cls

        mock_mlflow_instance = MagicMock(spec=MLFlowLogger)
        mock_mlflow_logger_cls.return_value = mock_mlflow_instance

        # Act
        composite_logger = CompositeLogger(trainer=mock_trainer)

        # Assert
        mock_mlflow_logger_cls.assert_called_once_with(trainer=mock_trainer)
        assert len(composite_logger.loggers) == 1
        assert composite_logger.loggers[0] == mock_mlflow_instance

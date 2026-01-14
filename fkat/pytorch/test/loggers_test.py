# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch, call

import lightning as L
from mlflow.tracking import MlflowClient  # type: ignore[possibly-unbound-import]
from lightning.pytorch.loggers import MLFlowLogger as _MLFlowLogger

from fkat.pytorch import loggers
from fkat.pytorch.loggers import (
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
    CompositeLogger,
    LightningLogger,
    _is_logger_type,
)


class TestLoggerTypeChecking(unittest.TestCase):
    def test_is_logger_type_mlflow_lightning(self):
        mock_logger = MagicMock()
        mock_logger.__class__.__name__ = "MLFlowLogger"
        mock_logger.__class__.__module__ = "lightning.pytorch.loggers.mlflow"
        assert _is_logger_type(mock_logger, "MLFlowLogger")

    def test_is_logger_type_mlflow_pytorch_lightning(self):
        mock_logger = MagicMock()
        mock_logger.__class__.__name__ = "MLFlowLogger"
        mock_logger.__class__.__module__ = "pytorch_lightning.loggers.mlflow"
        assert _is_logger_type(mock_logger, "MLFlowLogger")

    def test_is_logger_type_wrong_name(self):
        mock_logger = MagicMock()
        mock_logger.__class__.__name__ = "WandbLogger"
        mock_logger.__class__.__module__ = "lightning.pytorch.loggers.wandb"
        assert not _is_logger_type(mock_logger, "MLFlowLogger")

    def test_is_logger_type_wrong_module(self):
        mock_logger = MagicMock()
        mock_logger.__class__.__name__ = "MLFlowLogger"
        mock_logger.__class__.__module__ = "some.other.module"
        assert not _is_logger_type(mock_logger, "MLFlowLogger")


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

    def test_init_with_trainer(self):
        # Arrange
        mock_lightning_logger = MagicMock(spec=_MLFlowLogger)
        mock_lightning_logger._mlflow_client = self.mock_client
        mock_lightning_logger._run_id = self.run_id
        mock_lightning_logger._log_batch_kwargs = {"synchronous": True}

        # Act
        logger = MLFlowLogger(logger=mock_lightning_logger)

        # Assert
        assert logger._client == self.mock_client
        assert logger._run_id == self.run_id
        assert logger._synchronous


class TestTensorBoardLogger(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()
        self.mock_experiment = MagicMock()
        self.mock_logger.experiment = self.mock_experiment
        self.mock_logger.log_dir = "/tmp/logs"
        self.logger = TensorBoardLogger(logger=self.mock_logger)

    @patch("fkat.pytorch.loggers.rank_zero_only")
    def test_log_tag(self, mock_rank_zero_only):
        mock_rank_zero_only.return_value = lambda func: func
        self.logger.log_tag("key", "value")
        self.mock_experiment.add_text.assert_called_once_with("key", "value")

    @patch("fkat.pytorch.loggers.rank_zero_only")
    def test_log_batch(self, mock_rank_zero_only):
        mock_rank_zero_only.return_value = lambda func: func
        self.logger.log_batch(metrics={"loss": 0.5}, tags={"tag": "val"}, step=10)
        self.mock_experiment.add_scalar.assert_called_once_with("loss", 0.5, 10)
        self.mock_experiment.add_text.assert_called_once_with("tag", "val", 10)

    @patch("shutil.copy2")
    @patch("pathlib.Path")
    def test_log_artifact(self, mock_path, mock_copy):
        mock_dest = MagicMock()
        mock_path.return_value = mock_dest
        self.mock_logger.log_dir = "/tmp/logs"

        self.logger.log_artifact("/tmp/file.txt", "artifacts/file.txt")

        mock_copy.assert_called_once()


class TestWandbLogger(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()
        self.mock_experiment = MagicMock()
        self.mock_logger.experiment = self.mock_experiment
        self.mock_experiment.config = MagicMock()
        self.logger = WandbLogger(logger=self.mock_logger)

    @patch("fkat.pytorch.loggers.rank_zero_only")
    def test_log_tag(self, mock_rank_zero_only):
        mock_rank_zero_only.return_value = lambda func: func
        self.logger.log_tag("key", "value")
        self.mock_experiment.config.update.assert_called_once_with({"key": "value"})

    @patch("fkat.pytorch.loggers.rank_zero_only")
    def test_log_batch(self, mock_rank_zero_only):
        mock_rank_zero_only.return_value = lambda func: func
        self.logger.log_batch(metrics={"loss": 0.5}, tags={"tag": "val"}, step=10)
        self.mock_experiment.log.assert_called_once_with({"loss": 0.5, "tag": "val"}, step=10)

    def test_log_artifact(self):
        self.logger.log_artifact("/tmp/file.txt")
        self.mock_experiment.save.assert_called_once_with("/tmp/file.txt")

    def test_log_artifact_offline_replaces_symlink(self):
        self.mock_experiment.settings.mode = "offline"
        self.mock_experiment.settings.files_dir = "/wandb/files"

        with patch("pathlib.Path") as mock_path, patch("shutil.copy2") as mock_copy:
            # Setup mocks
            mock_src_inst = MagicMock()
            mock_src_inst.name = "file.txt"
            mock_dest_inst = MagicMock()
            mock_dest_inst.is_symlink.return_value = True

            # Path() is called twice: once for src, once for dest
            mock_path.return_value.absolute.return_value = mock_src_inst
            mock_path.return_value.__truediv__.return_value = mock_dest_inst

            self.logger.log_artifact("/tmp/file.txt")

            # Verify save was called
            self.mock_experiment.save.assert_called_once_with("/tmp/file.txt")
            # Verify symlink was removed and file copied
            assert mock_dest_inst.unlink.called or mock_copy.called


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
    def test_init_with_single_logger(self, mock_mlflow_logger_cls):
        # Arrange
        mock_trainer = MagicMock(spec=L.Trainer)

        # Create a mock that _is_logger_type will recognize
        mock_lightning_logger = MagicMock()
        type(mock_lightning_logger).__name__ = "MLFlowLogger"
        type(mock_lightning_logger).__module__ = "lightning.pytorch.loggers.mlflow"
        mock_trainer.loggers = [mock_lightning_logger]

        mock_mlflow_instance = MagicMock(spec=MLFlowLogger)
        mock_mlflow_logger_cls.return_value = mock_mlflow_instance

        # Act
        composite_logger = CompositeLogger(trainer=mock_trainer)

        # Assert
        assert len(composite_logger.loggers) == 1
        mock_mlflow_logger_cls.assert_called_once_with(logger=mock_lightning_logger)

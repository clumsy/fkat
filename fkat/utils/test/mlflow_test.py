# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import TestCase
from unittest.mock import MagicMock

from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger

from fkat.utils.mlflow import broadcast_mlflow_run_id, mlflow_logger


class MlflowUtilsTest(TestCase):
    def test_mlflow_logger(self):
        # Arrange
        mock_trainer = MagicMock(spec=Trainer)
        mock_mlflow_logger = MagicMock(spec=MLFlowLogger)
        mock_trainer.logger = [mock_mlflow_logger]
        # Act
        logger = mlflow_logger(mock_trainer)
        # Assert
        assert mock_mlflow_logger == logger

    def test_broadcast_mlflow_run_id(self):
        # Arrange
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.strategy.broadcast.return_value = "asdf123e"
        mock_logger = MagicMock(spec=MLFlowLogger)
        mock_logger.run_id = "asdf123e"
        # Act
        broadcast_mlflow_run_id(mock_logger, mock_trainer)
        # Assert
        mock_trainer.strategy.broadcast.assert_called_once_with("asdf123e", src=0)
        assert mock_logger._run_id == "asdf123e"

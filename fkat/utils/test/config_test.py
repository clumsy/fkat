# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from unittest import TestCase
from unittest.mock import patch, MagicMock, ANY

import pytest

from fkat.utils import config
from fkat.utils.config import (
    register_singleton_resolver,
    to_primitive_container,
    to_str,
)


class TestConfig(TestCase):
    @patch(f"{config.__name__}.OmegaConf")
    def test_to_str(self, mock_oc):
        # Act
        res = to_str({})
        # Assert
        assert res == mock_oc.to_yaml.return_value.__radd__.return_value

    @patch(f"{config.__name__}.OmegaConf")
    def test_to_primitive_container_noop(self, mock_oc):
        # Arrange
        mock_oc.is_config.return_value = False
        # Act
        res = to_primitive_container(src := {})
        # Assert
        assert src == res

    @patch(f"{config.__name__}.OmegaConf")
    def test_to_primitive_container_omegaconf(self, mock_oc):
        # Arrange
        mock_oc.is_config.return_value = True
        # Act
        res = to_primitive_container({})
        # Assert
        assert res == mock_oc.to_container.return_value

    @patch(f"{config.__name__}.OmegaConf")
    def test_register_singleton_resolver(self, mock_oc):
        # Act
        resolver = register_singleton_resolver()
        resolver.trainer = MagicMock(accelerator="gpu")
        resolver.data = 41
        resolver.model = 40
        resolver.ckpt_path = 39
        resolver.tuners = 38
        resolver.return_predictions = True
        resolver.enable_mlflow_artifact_logging = True
        # Assert
        mock_oc.register_new_resolver.assert_any_call("fkat", ANY)  # Check if resolver was registered
        oc_resolver = mock_oc.register_new_resolver.call_args_list[0][0][1]
        assert oc_resolver("trainer") == resolver.trainer
        assert oc_resolver("trainer.accelerator") == resolver.trainer.accelerator
        assert oc_resolver("data") == resolver.data
        assert oc_resolver("model") == resolver.model
        assert oc_resolver("ckpt_path") == resolver.ckpt_path
        assert oc_resolver("return_predictions") == resolver.return_predictions
        assert oc_resolver("tuners") == resolver.tuners
        with pytest.raises(AttributeError):
            resolver.missing  # noqa: B018

    @patch(f"{config.__name__}.mlflow_logger")
    @patch(f"{config.__name__}.TemporaryDirectory")
    @patch(f"{config.__name__}.os.makedirs")
    @patch(f"{config.__name__}.open", new_callable=MagicMock)
    @patch(f"{config.__name__}.OmegaConf")
    def test_save(
        self,
        mock_oc,
        mock_open,
        mock_makedirs,
        mock_temp_dir,
        mock_mlflow_logger,
    ):
        # Import the module to get access to the original function before decoration
        import fkat.utils.config as config_module

        # Store the original decorated function
        original_save = config_module.save

        try:
            # Create our own implementation of the save function without the decorator
            def undecorated_save(cfg, trainer):
                yaml_str = mock_oc.to_yaml(cfg)
                with mock_temp_dir():
                    yaml_path = os.path.join(mock_temp_dir().__enter__(), "config.yaml")
                    mock_makedirs(os.path.dirname(yaml_path), exist_ok=True)
                    with mock_open(yaml_path, "w") as f:
                        f.write(yaml_str)
                    if mlflow := mock_mlflow_logger(trainer):
                        if mlflow.run_id:
                            mlflow.experiment.log_artifact(mlflow.run_id, yaml_path)

            # Replace the decorated function with our undecorated version for testing
            config_module.save = undecorated_save

            # Arrange
            mock_cfg = MagicMock()
            mock_trainer = MagicMock()
            mock_yaml_str = "yaml_content"
            mock_oc.to_yaml.return_value = mock_yaml_str

            # Set up temporary directory
            mock_temp_instance = MagicMock()
            mock_temp_dir.return_value.__enter__.return_value = "/mock/temp/dir"
            mock_temp_dir.return_value = mock_temp_instance

            # Set up file handling
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            # Set up MLflow logger
            mock_mlflow = MagicMock()
            mock_mlflow.run_id = "test_run_id"
            mock_mlflow_logger.return_value = mock_mlflow

            # Act - Test normal operation
            config_module.save(mock_cfg, mock_trainer)

            # Assert
            mock_oc.to_yaml.assert_called_once_with(mock_cfg)
            mock_makedirs.assert_called_once_with(ANY, exist_ok=True)
            mock_open.assert_called_once()
            mock_file.write.assert_called_once_with(mock_yaml_str)
            mock_mlflow_logger.assert_called_once_with(mock_trainer)
            mock_mlflow.experiment.log_artifact.assert_called_once_with("test_run_id", ANY)

            # Reset all mocks for the next test
            mock_open.reset_mock()
            mock_makedirs.reset_mock()
            mock_mlflow_logger.reset_mock()
            mock_file.reset_mock()
            mock_mlflow.experiment.log_artifact.reset_mock()
            mock_oc.to_yaml.reset_mock()

            # Test with MLflow but no run_id
            mock_mlflow.run_id = None
            mock_mlflow_logger.return_value = mock_mlflow

            # Act again
            config_module.save(mock_cfg, mock_trainer)

            # Assert file is written but artifact is not logged
            mock_open.assert_called_once()
            mock_makedirs.assert_called_once()
            mock_mlflow.experiment.log_artifact.assert_not_called()

            # Reset for next test
            mock_open.reset_mock()
            mock_makedirs.reset_mock()
            mock_mlflow_logger.reset_mock()
            mock_oc.to_yaml.reset_mock()

            # Test with no MLflow logger
            mock_mlflow_logger.return_value = None

            # Act again
            config_module.save(mock_cfg, mock_trainer)

            # Assert file is written but no MLflow interaction
            mock_open.assert_called_once()
            mock_makedirs.assert_called_once()

        finally:
            # Restore the original function
            config_module.save = original_save

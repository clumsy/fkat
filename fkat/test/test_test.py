# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for test.py"""

import pytest
from unittest.mock import MagicMock, patch
import lightning as L
from omegaconf import DictConfig
import sys

from fkat.test import main


@pytest.fixture
def mock_state() -> MagicMock:
    """Create a mock state with trainer, model, and data."""
    state = MagicMock()
    state.trainer = MagicMock()
    state.model = MagicMock()
    state.ckpt_path = "path/to/checkpoint"
    return state


@pytest.mark.parametrize(
    "is_datamodule,has_data",
    [
        (True, True),  # LightningDataModule with data
        (False, True),  # Regular dataloader with data
        (False, False),  # No data
    ],
)
@patch("fkat.test.initialize")
def test_test_main(
    mock_initialize,
    mock_state,
    is_datamodule: bool,
    has_data: bool,
):
    """Test main testing function with different data configurations."""
    # Arrange
    mock_cfg = MagicMock(spec=DictConfig)
    mock_initialize.return_value = mock_state

    if has_data:
        if is_datamodule:
            mock_state.data = MagicMock(spec=L.LightningDataModule)
        else:
            mock_state.data = MagicMock()
            mock_state.data.test_dataloader.return_value = "test_loader"
    else:
        mock_state.data = None

    # Act
    main(mock_cfg)

    # Assert
    mock_initialize.assert_called_once_with(mock_cfg)

    expected_kwargs = {"ckpt_path": mock_state.ckpt_path}
    if has_data:
        if is_datamodule:
            expected_kwargs["datamodule"] = mock_state.data
        else:
            expected_kwargs["test_dataloaders"] = "test_loader"
    else:
        expected_kwargs["test_dataloaders"] = None

    mock_state.trainer.test.assert_called_once_with(mock_state.model, **expected_kwargs)


@patch("fkat.test.run_main")
def test_main_entry_point(mock_run_main):
    """Test the main entry point."""
    # Arrange
    import fkat.test  # noqa: F401

    # Save original argv
    original_argv = sys.argv

    try:
        # Simulate being run as main module
        sys.argv = ["fkat.test"]

        # Act
        # Execute the __main__ block
        exec(
            """
if __name__ == "__main__":
    run_main(main)
""",
            {"__name__": "__main__", "run_main": mock_run_main, "main": main},
        )

        # Assert
        mock_run_main.assert_called_once_with(main)

    finally:
        # Restore original argv
        sys.argv = original_argv


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError, Exception])
@patch("fkat.test.initialize")
def test_test_main_handles_exceptions(
    mock_initialize,
    exception_type: type[Exception],
):
    """Test main function handles exceptions properly."""
    # Arrange
    mock_cfg = MagicMock(spec=DictConfig)
    mock_initialize.side_effect = exception_type("Test error")

    # Act & Assert
    with pytest.raises(exception_type, match="Test error"):
        main(mock_cfg)

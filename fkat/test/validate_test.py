# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validate.py"""

import pytest
from unittest.mock import MagicMock, patch
import lightning as L
from omegaconf import DictConfig
import sys

from fkat.validate import main


@pytest.fixture
def mock_state() -> MagicMock:
    state = MagicMock()
    state.trainer = MagicMock()
    state.model = MagicMock()
    state.ckpt_path = "path/to/checkpoint"
    return state


@pytest.mark.parametrize(
    "is_datamodule,has_data",
    [
        (True, True),
        (False, True),
        (False, False),
    ],
)
@patch("fkat.validate.initialize")
def test_validate_main(
    mock_initialize,
    mock_state,
    is_datamodule: bool,
    has_data: bool,
):
    mock_cfg = MagicMock(spec=DictConfig)
    mock_initialize.return_value = mock_state

    if has_data:
        if is_datamodule:
            mock_state.data = MagicMock(spec=L.LightningDataModule)
        else:
            mock_state.data = MagicMock()
            mock_state.data.val_dataloader.return_value = "val_loader"
    else:
        mock_state.data = None

    main(mock_cfg)

    mock_initialize.assert_called_once_with(mock_cfg)

    expected_kwargs = {"ckpt_path": mock_state.ckpt_path}
    if has_data:
        if is_datamodule:
            expected_kwargs["datamodule"] = mock_state.data
        else:
            expected_kwargs["val_dataloaders"] = "val_loader"
    else:
        expected_kwargs["val_dataloaders"] = None

    mock_state.trainer.validate.assert_called_once_with(mock_state.model, **expected_kwargs)


@patch("fkat.validate.run_main")
def test_main_entry_point(mock_run_main):
    import fkat.validate  # noqa: F401

    original_argv = sys.argv
    try:
        sys.argv = ["fkat.validate"]
        exec(
            """
if __name__ == "__main__":
    run_main(main)
""",
            {"__name__": "__main__", "run_main": mock_run_main, "main": main},
        )
        mock_run_main.assert_called_once_with(main)
    finally:
        sys.argv = original_argv


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError, Exception])
@patch("fkat.validate.initialize")
def test_validate_main_handles_exceptions(
    mock_initialize,
    exception_type: type[Exception],
):
    mock_cfg = MagicMock(spec=DictConfig)
    mock_initialize.side_effect = exception_type("Test error")

    with pytest.raises(exception_type, match="Test error"):
        main(mock_cfg)

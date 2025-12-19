# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import patch, MagicMock
import datetime as dt


from fkat.pytorch.callbacks.debugging import OptimizerSnapshot

MODULE = "fkat.pytorch.callbacks.debugging.optimizer"


@pytest.mark.parametrize("global_rank", [0, 1, 2])
@patch(f"{MODULE}.dt.datetime")
@patch(f"{MODULE}.fsspec.open")
@patch(f"{MODULE}.torch.save")
def test_optimizer_snapshot_saves_optimizers(
    mock_torch_save,
    mock_fsspec_open,
    mock_datetime,
    global_rank,
):
    # Arrange
    batch_idx = 10
    output_path_prefix = "/tmp/optimizer_snapshot_"
    timestamp = "2025-06-01T12-00-00Z"

    # Mock datetime.now to return a fixed timestamp
    mock_now = MagicMock()
    mock_datetime.now.return_value = mock_now
    mock_now.strftime.return_value = timestamp

    # Create mock objects directly without using spec
    mock_trainer = MagicMock()
    mock_trainer.global_rank = global_rank
    mock_trainer.global_step = batch_idx

    mock_module = MagicMock()

    # Create mock optimizers
    mock_opt1 = MagicMock()
    mock_opt2 = MagicMock()
    mock_trainer.optimizers = [mock_opt1, mock_opt2]

    # Create a mock schedule that returns True for the test
    schedule = MagicMock()
    schedule.check.return_value = True

    # Mock the file context manager
    mock_file_context = MagicMock()
    mock_fsspec_open.return_value.__enter__.return_value = mock_file_context

    # Create the callback
    optimizer_snapshot = OptimizerSnapshot(output_path_prefix=output_path_prefix, schedule=schedule)

    # Act
    optimizer_snapshot.on_train_batch_start(mock_trainer, mock_module, None, batch_idx)

    # Assert
    schedule.check.assert_called_once_with(trainer=mock_trainer, stage="train", batch_idx=batch_idx, step=batch_idx)

    # Verify datetime.now was called with UTC timezone
    mock_datetime.now.assert_called_once_with(dt.timezone.utc)

    # Check that fsspec.open was called for each optimizer with the correct paths
    assert mock_fsspec_open.call_count == 2
    mock_fsspec_open.assert_any_call(f"{output_path_prefix}rank{global_rank}_opt0_{timestamp}.pt", "wb", makedirs=True)
    mock_fsspec_open.assert_any_call(f"{output_path_prefix}rank{global_rank}_opt1_{timestamp}.pt", "wb", makedirs=True)

    # Check that torch.save was called for each optimizer
    assert mock_torch_save.call_count == 2
    mock_torch_save.assert_any_call(mock_opt1, mock_file_context)
    mock_torch_save.assert_any_call(mock_opt2, mock_file_context)


@patch(f"{MODULE}.dt.datetime")
@patch(f"{MODULE}.fsspec.open")
def test_optimizer_snapshot_respects_schedule(
    mock_fsspec_open,
    mock_datetime,
):
    # Arrange
    batch_idx = 10
    output_path_prefix = "/tmp/optimizer_snapshot_"

    # Create mock objects directly without using spec
    mock_trainer = MagicMock()
    mock_trainer.global_rank = 0
    mock_trainer.global_step = batch_idx
    mock_trainer.optimizers = [MagicMock()]

    mock_module = MagicMock()

    # Create a mock schedule that returns False for the test
    schedule = MagicMock()
    schedule.check.return_value = False

    # Create the callback
    optimizer_snapshot = OptimizerSnapshot(output_path_prefix=output_path_prefix, schedule=schedule)

    # Act
    optimizer_snapshot.on_train_batch_start(mock_trainer, mock_module, None, batch_idx)

    # Assert
    schedule.check.assert_called_once_with(trainer=mock_trainer, stage="train", batch_idx=batch_idx, step=batch_idx)

    # Check that datetime.now was not called
    mock_datetime.now.assert_not_called()

    # Check that fsspec.open was not called
    mock_fsspec_open.assert_not_called()


@patch(f"{MODULE}.dt.datetime")
@patch(f"{MODULE}.fsspec.open")
def test_optimizer_snapshot_with_no_optimizers(
    mock_fsspec_open,
    mock_datetime,
):
    # Arrange
    batch_idx = 10
    output_path_prefix = "/tmp/optimizer_snapshot_"
    timestamp = "2025-06-01T12-00-00Z"

    # Mock datetime.now to return a fixed timestamp
    mock_now = MagicMock()
    mock_datetime.now.return_value = mock_now
    mock_now.strftime.return_value = timestamp

    # Create mock objects directly without using spec
    mock_trainer = MagicMock()
    mock_trainer.global_rank = 0
    mock_trainer.global_step = batch_idx
    mock_trainer.optimizers = []  # No optimizers

    mock_module = MagicMock()

    # Create a mock schedule that returns True for the test
    schedule = MagicMock()
    schedule.check.return_value = True

    # Create the callback
    optimizer_snapshot = OptimizerSnapshot(output_path_prefix=output_path_prefix, schedule=schedule)

    # Act
    optimizer_snapshot.on_train_batch_start(mock_trainer, mock_module, None, batch_idx)

    # Assert
    schedule.check.assert_called_once_with(trainer=mock_trainer, stage="train", batch_idx=batch_idx, step=batch_idx)

    # Verify datetime.now was called with UTC timezone
    mock_datetime.now.assert_called_once_with(dt.timezone.utc)

    # Check that fsspec.open was not called (no optimizers to save)
    mock_fsspec_open.assert_not_called()


@patch(f"{MODULE}.Never")
@patch(f"{MODULE}.dt.datetime")
@patch(f"{MODULE}.fsspec.open")
def test_optimizer_snapshot_default_schedule(
    mock_fsspec_open,
    mock_datetime,
    mock_never,
):
    # Arrange
    batch_idx = 10
    output_path_prefix = "/tmp/optimizer_snapshot_"

    # Create mock objects directly without using spec
    mock_trainer = MagicMock()
    mock_trainer.global_rank = 0
    mock_trainer.global_step = batch_idx
    mock_trainer.optimizers = [MagicMock()]

    mock_module = MagicMock()

    # Create a mock Never schedule
    mock_never_instance = MagicMock()
    mock_never.return_value = mock_never_instance
    mock_never_instance.check.return_value = False

    # Create the callback with default schedule (Never)
    optimizer_snapshot = OptimizerSnapshot(output_path_prefix=output_path_prefix)

    # Act
    optimizer_snapshot.on_train_batch_start(mock_trainer, mock_module, None, batch_idx)

    # Assert
    mock_never_instance.check.assert_called_once_with(
        trainer=mock_trainer, stage="train", batch_idx=batch_idx, step=batch_idx
    )

    # Check that datetime.now was not called
    mock_datetime.now.assert_not_called()

    # Check that fsspec.open was not called
    mock_fsspec_open.assert_not_called()

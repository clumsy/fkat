# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import patch, MagicMock, mock_open, ANY
from typing import Any

import torch

from fkat.pytorch.schedule import Schedule, Never
from fkat.pytorch.callbacks.profiling import PyTorch

MODULE = "fkat.pytorch.callbacks.profiling.torch"


@pytest.mark.parametrize("stage", ["train", "validation", "predict", "test"])
@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("ranks", [None, [0], [1]])
@patch(f"{MODULE}.ThreadPoolExecutor")
@patch(f"{MODULE}.CallbackLogger")
@patch(f"{MODULE}.shutil")
@patch(f"{MODULE}.gzip")
@patch(f"{MODULE}.os")
@patch(f"{MODULE}.get_rank")
@patch(f"{MODULE}.torch.profiler.profile")
@patch(f"{MODULE}.torch.profiler.ExecutionTraceObserver")
@patch(f"{MODULE}.L.LightningModule")
@patch(f"{MODULE}.L.Trainer")
@patch(f"{MODULE}.signal")
@patch(f"{MODULE}.atexit")
def test_pytorch_profiler_lifecycle(
    mock_atexit: Any,
    mock_signal: Any,
    mock_trainer: Any,
    mock_module: Any,
    mock_observer: Any,
    mock_profile: Any,
    mock_get_rank: Any,
    mock_os: Any,
    mock_gzip: Any,
    mock_shutil: Any,
    mock_logger: Any,
    mock_tpe: Any,
    ranks: list[int] | None,
    compress: bool,
    stage: str,
):
    """Test the complete lifecycle of PyTorch profiler callback."""
    # Arrange
    batch_idx = 10
    mock_get_rank.return_value = 0

    class MockExecutor:
        def submit(self, fn: Any, *args: Any, **kwargs: Any):
            fn(*args, **kwargs)

        def shutdown(self):
            pass

    mock_tpe.return_value = MockExecutor()
    schedule = MagicMock(spec=Schedule)
    schedule.check.return_value = True

    # Create profiler instance
    profiler = PyTorch(ranks=ranks, compress=compress, schedule=schedule)

    # Act - setup
    profiler.setup(mock_trainer, mock_module, stage)

    # Assert profiler was started
    if ranks is None or 0 in ranks:
        mock_observer.return_value.register_callback.assert_called_once()
        mock_profile.return_value.start.assert_called_once()
    else:
        mock_observer.return_value.register_callback.assert_not_called()
        mock_profile.return_value.start.assert_not_called()

    # Act - batch end
    getattr(profiler, f"on_{stage}_batch_end")(mock_trainer, mock_module, None, None, batch_idx)

    # Assert profiler step was called if rank is being profiled
    if ranks is None or 0 in ranks:
        mock_profile.return_value.step.assert_called_once()
    else:
        mock_profile.return_value.step.assert_not_called()

    # Act - teardown
    profiler.teardown(mock_trainer, mock_module, stage)

    # Assert profiler was stopped
    if ranks is None or 0 in ranks:
        mock_profile.return_value.stop.assert_called_once()
        mock_observer.return_value.unregister_callback.assert_called_once()


@pytest.mark.parametrize("compress", [True, False])
@patch("builtins.open", mock_open(read_data="test data"))
@patch(f"{MODULE}.ThreadPoolExecutor")
@patch(f"{MODULE}.CallbackLogger")
@patch(f"{MODULE}.shutil")
@patch(f"{MODULE}.gzip.open")
@patch(f"{MODULE}.os")
@patch(f"{MODULE}.get_rank")
@patch(f"{MODULE}.torch.profiler.profile")
@patch(f"{MODULE}.torch.profiler.ExecutionTraceObserver")
def test_pytorch_profiler_publish(
    mock_observer: Any,
    mock_profile: Any,
    mock_get_rank: Any,
    mock_os: Any,
    mock_gzip_open: Any,
    mock_shutil: Any,
    mock_logger: Any,
    mock_tpe: Any,
    compress: bool,
):
    """Test the publish functionality of PyTorch profiler."""
    # Arrange
    mock_get_rank.return_value = 0

    class MockExecutor:
        def submit(self, fn: Any, *args: Any, **kwargs: Any):
            fn(*args, **kwargs)

        def shutdown(self):
            pass

    mock_tpe.return_value = MockExecutor()
    mock_os.path.dirname.return_value = "/tmp"
    mock_os.path.basename.return_value = "rank0.json"

    # Create profiler instance
    profiler = PyTorch(compress=compress)
    profiler._cb_logger = MagicMock()
    profiler.stage = "train"
    profiler.batch_idx = "10"

    # Act - call publish method
    profiler._publish(mock_profile.return_value)

    # Assert
    mock_profile.return_value.export_chrome_trace.assert_called_once()
    mock_os.makedirs.assert_called_once_with(ANY, exist_ok=True)
    mock_shutil.move.assert_called_once()

    # Check compression behavior
    if compress:
        mock_gzip_open.assert_called_once()
        profiler._cb_logger.log_artifact.assert_called_once_with(ANY, "pt_profiler/train/10")
    else:
        mock_gzip_open.assert_not_called()
        profiler._cb_logger.log_artifact.assert_called_once_with(ANY, "pt_profiler/train/10")


@patch(f"{MODULE}.ThreadPoolExecutor")
@patch(f"{MODULE}.tempfile.mkdtemp")
@patch(f"{MODULE}.os.path.join")
@patch(f"{MODULE}.get_rank")
@patch(f"{MODULE}.torch.profiler.profile")
@patch(f"{MODULE}.torch.profiler.ExecutionTraceObserver")
def test_pytorch_profiler_initialization(
    mock_observer: Any,
    mock_profile: Any,
    mock_get_rank: Any,
    mock_join: Any,
    mock_mkdtemp: Any,
    mock_tpe: Any,
):
    """Test initialization with different parameters."""
    # Arrange
    mock_get_rank.return_value = 0
    mock_mkdtemp.return_value = "/tmp/profiler"
    mock_join.return_value = "/tmp/profiler/rank0.json"

    # Act - default initialization
    PyTorch()

    # Assert
    mock_profile.assert_called_with(
        schedule=ANY,
        on_trace_ready=ANY,
        execution_trace_observer=mock_observer.return_value,
    )

    # Act - with custom parameters
    custom_kwargs = {
        "activities": [torch.profiler.ProfilerActivity.CPU],
        "record_shapes": True,
    }
    PyTorch(ranks=[0], output_path_prefix="/custom/path", schedule=Never(), compress=False, **custom_kwargs)

    # Assert
    mock_profile.assert_called_with(
        schedule=ANY,
        on_trace_ready=ANY,
        execution_trace_observer=mock_observer.return_value,
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    )


@pytest.mark.parametrize("stage", ["train", "validation", "predict", "test"])
@patch(f"{MODULE}.ThreadPoolExecutor")
@patch(f"{MODULE}.get_rank")
@patch(f"{MODULE}.torch.profiler.profile")
@patch(f"{MODULE}.torch.profiler.ExecutionTraceObserver")
@patch(f"{MODULE}.L.LightningModule")
@patch(f"{MODULE}.L.Trainer")
def test_pytorch_profiler_batch_methods(
    mock_trainer: Any,
    mock_module: Any,
    mock_observer: Any,
    mock_profile: Any,
    mock_get_rank: Any,
    mock_tpe: Any,
    stage: str,
):
    """Test that batch methods correctly call _on_batch_end."""
    # Arrange
    mock_get_rank.return_value = 0

    class MockExecutor:
        def submit(self, fn: Any, *args: Any, **kwargs: Any):
            pass

        def shutdown(self):
            pass

    mock_tpe.return_value = MockExecutor()

    # Create profiler with mocked _on_batch_end
    profiler = PyTorch()
    profiler._on_batch_end = MagicMock()  # type: ignore[assignment]

    # Act - call batch end method
    batch_idx = 5
    getattr(profiler, f"on_{stage}_batch_end")(mock_trainer, mock_module, None, None, batch_idx)

    # Assert
    profiler._on_batch_end.assert_called_once_with(mock_trainer, stage, batch_idx)  # type: ignore[attr-defined]


@patch(f"{MODULE}.ThreadPoolExecutor")
@patch(f"{MODULE}.get_rank")
@patch(f"{MODULE}.torch.profiler.profile")
@patch(f"{MODULE}.torch.profiler.ExecutionTraceObserver")
@patch(f"{MODULE}.L.LightningModule")
@patch(f"{MODULE}.L.Trainer")
@patch(f"{MODULE}.signal")
@patch(f"{MODULE}.atexit")
def test_pytorch_profiler_exception_handling(
    mock_atexit: Any,
    mock_signal: Any,
    mock_trainer: Any,
    mock_module: Any,
    mock_observer: Any,
    mock_profile: Any,
    mock_get_rank: Any,
    mock_tpe: Any,
):
    """Test exception handling in the profiler."""
    # Arrange
    mock_get_rank.return_value = 0

    class MockExecutor:
        def submit(self, fn: Any, *args: Any, **kwargs: Any):
            pass

        def shutdown(self):
            pass

    mock_tpe.return_value = MockExecutor()

    # Create profiler with mocked _terminate
    profiler = PyTorch()
    profiler._terminate = MagicMock()  # type: ignore[assignment]

    # Act - call on_exception
    exception = Exception("Test exception")
    profiler.on_exception(mock_trainer, mock_module, exception)

    # Assert
    profiler._terminate.assert_called_once()  # type: ignore[attr-defined]

    # Verify signal handlers were registered
    # We can't check the exact function since we mocked it after registration
    assert mock_signal.signal.call_count >= 2
    assert mock_atexit.register.call_count >= 1

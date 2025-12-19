# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import patch, mock_open, MagicMock

from fkat.pytorch.schedule import Schedule
from fkat.pytorch.callbacks.profiling import Memray

MODULE = "fkat.pytorch.callbacks.profiling.memray"


@pytest.mark.parametrize("stage", ["train", "validation", "predict", "test"])
@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("flamegraph", [True, False])
@patch("builtins.open", mock_open(read_data=""))
@patch("builtins.__import__")
@patch(f"{MODULE}.ThreadPoolExecutor")
@patch(f"{MODULE}.CallbackLogger")
@patch(f"{MODULE}.gzip")
@patch(f"{MODULE}.shutil")
@patch(f"{MODULE}.Path")
@patch(f"{MODULE}.os")
@patch(f"{MODULE}.tempfile")
@patch(f"{MODULE}.L.LightningModule")
@patch(f"{MODULE}.L.Trainer")
@patch(f"{MODULE}.signal")
@patch(f"{MODULE}.atexit")
def test_memray_traces(
    mock_atexit,
    mock_signal,
    mock_trainer,
    mock_module,
    mock_tempfile,
    mock_os,
    mock_path,
    mock_shutil,
    mock_gzip,
    mock_logger,
    mock_tpe,
    mock_import,
    flamegraph,
    compress,
    stage,
):
    # Arrange
    batch_idx = 10

    class MockExecutor:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

    mock_tpe.return_value = MockExecutor()
    schedule = MagicMock(spec=Schedule)
    schedule.check.return_value = True
    memray = Memray(flamegraph=flamegraph, compress=compress, schedule=schedule)

    # Act - setup and start tracing
    memray.setup(mock_trainer, mock_module, stage)
    getattr(memray, f"on_{stage}_start")(mock_trainer, mock_module)

    # Assert tracker was initialized
    mock_memray = mock_import.return_value.Tracker.return_value
    mock_memray.__enter__.assert_called_once()

    # Act - finish tracing on batch end
    mock_os.listdir.return_value = ["some_path"]
    getattr(memray, f"on_{stage}_batch_end")(mock_trainer, mock_module, None, None, batch_idx - 1)

    # Assert
    output_file = mock_os.path.join.return_value
    mock_memray.__exit__.assert_called_once_with(None, None, None)

    if flamegraph:
        mock_flamegraph = mock_import.return_value.FlamegraphCommand.return_value
        mock_flamegraph.write_report.assert_called_once_with(
            mock_path(mock_os.path.join.return_value), mock_path(output_file + ".html"), True, -1, False
        )

    if compress:
        memray._cb_logger.log_artifact.assert_called_once_with(output_file + ".gz", f"memray/{stage}/{batch_idx}")  # type: ignore
    else:
        memray._cb_logger.log_artifact.assert_called_once_with(output_file, f"memray/{stage}/{batch_idx}")  # type: ignore

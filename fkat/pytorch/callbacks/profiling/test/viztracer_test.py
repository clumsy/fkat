# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import patch, mock_open, MagicMock

from fkat.pytorch.schedule import Schedule
from fkat.pytorch.callbacks.profiling import VizTracer

MODULE = "fkat.pytorch.callbacks.profiling.viztracer"


@pytest.mark.parametrize("stage", ["train", "validation", "predict", "test"])
@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("patch", [True, False])
@patch("builtins.open", mock_open(read_data=""))
@patch("builtins.__import__")
@patch(f"{MODULE}.ThreadPoolExecutor")
@patch(f"{MODULE}.CallbackLogger")
@patch(f"{MODULE}.json")
@patch(f"{MODULE}.gzip")
@patch(f"{MODULE}.os")
@patch(f"{MODULE}.L.LightningModule")
@patch(f"{MODULE}.L.Trainer")
@patch(f"{MODULE}.signal")
@patch(f"{MODULE}.atexit")
def test_viztracer_traces(
    mock_atexit,
    mock_signal,
    mock_trainer,
    mock_module,
    mock_os,
    mock_gzip,
    mock_json,
    mock_logger,
    mock_tpe,
    mock_import,
    patch,
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
    viztracer = VizTracer(schedule=schedule, compress=compress, patch=patch)

    # Act - start tracing
    viztracer.setup(mock_trainer, mock_module, stage)
    # Assert
    mock_viztracer = mock_import.return_value.VizTracer.return_value
    mock_viztracer.start.assert_called_once()

    # Act - finish tracing
    getattr(viztracer, f"on_{stage}_batch_end")(mock_trainer, mock_module, None, None, batch_idx - 1)
    # Assert
    mock_viztracer.stop.assert_called_once()
    output_file = mock_os.path.join.return_value + ".json"
    mock_viztracer.save.assert_called_once_with(output_file=output_file, verbose=0)
    if compress:
        mock_vcompressor = mock_import.return_value.VCompressor.return_value
        compressed_file = mock_os.path.splitext.return_value[0] + ".cvf"
        mock_vcompressor.compress.assert_called_once_with(mock_json.load.return_value, compressed_file)
        viztracer._cb_logger.log_artifact.assert_called_once_with(compressed_file, f"viztracer/{stage}/{batch_idx}")  # type: ignore[attr-defined]
    else:
        viztracer._cb_logger.log_artifact.assert_called_once_with(output_file + ".gz", f"viztracer/{stage}/{batch_idx}")  # type: ignore[attr-defined]
    install_all_hooks = mock_import.return_value.install_all_hooks
    if patch:
        install_all_hooks.assert_called_once()
    else:
        install_all_hooks.assert_not_called()

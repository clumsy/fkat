# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch, MagicMock, mock_open
from typing import Any

import os
import lightning as L


class TestNsys(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = MagicMock(spec=L.Trainer)
        self.trainer.global_step = 10
        self.pl_module = MagicMock(spec=L.LightningModule)
        self.batch = MagicMock()
        self.outputs = MagicMock()

    @patch("fkat.pytorch.callbacks.cuda.nsys.get_rank")
    @patch("fkat.pytorch.callbacks.cuda.nsys.exec_with_nsys")
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.environ", {"NSYS_OUTPUT": "/tmp/test.nsys-rep"})
    def test_init_with_output_file(self, mock_exec: MagicMock, mock_get_rank: MagicMock) -> None:
        # Arrange
        mock_get_rank.return_value = 0

        # Import here to ensure all mocks are applied
        from fkat.pytorch.callbacks.cuda.nsys import Nsys

        # Act
        with patch("fkat.pytorch.callbacks.cuda.nsys.signal"), patch("fkat.pytorch.callbacks.cuda.nsys.atexit"):
            callback = Nsys()

        # Assert
        mock_exec.assert_not_called()
        self.assertEqual(callback.output_file, "/tmp/test.nsys-rep")

    @patch("fkat.pytorch.callbacks.cuda.nsys.get_rank")
    @patch("fkat.pytorch.callbacks.cuda.nsys.torch.cuda.cudart")
    @patch("fkat.pytorch.callbacks.cuda.nsys.torch.autograd.profiler.emit_nvtx")
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.environ", {"NSYS_OUTPUT": "/tmp/test.nsys-rep"})
    def test_maybe_trace_start(
        self, mock_emit_nvtx: MagicMock, mock_cudart: MagicMock, mock_get_rank: MagicMock
    ) -> None:
        # Arrange
        mock_get_rank.return_value = 0
        mock_emit_nvtx_instance = MagicMock()
        mock_emit_nvtx.return_value = mock_emit_nvtx_instance

        # Import here to ensure all mocks are applied
        from fkat.pytorch.callbacks.cuda.nsys import Nsys

        with patch("fkat.pytorch.callbacks.cuda.nsys.signal"), patch("fkat.pytorch.callbacks.cuda.nsys.atexit"):
            callback = Nsys()

        # Create a custom schedule that always returns True
        class MockSchedule:
            def check(self, **kwargs: Any) -> bool:
                return True

        callback.schedule = MockSchedule()

        # Act
        callback._maybe_trace(self.trainer, "train", 1)

        # Assert
        mock_cudart.return_value.cudaProfilerStart.assert_called_once()
        mock_emit_nvtx.assert_called_once()
        mock_emit_nvtx_instance.__enter__.assert_called_once()

    @patch("fkat.pytorch.callbacks.cuda.nsys.get_rank")
    @patch("fkat.pytorch.callbacks.cuda.nsys.torch.cuda.cudart")
    @patch("fkat.pytorch.callbacks.cuda.nsys.torch.autograd.profiler.emit_nvtx")
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.environ", {"NSYS_OUTPUT": "/tmp/test.nsys-rep"})
    def test_maybe_trace_stop(
        self, mock_emit_nvtx: MagicMock, mock_cudart: MagicMock, mock_get_rank: MagicMock
    ) -> None:
        # Arrange
        mock_get_rank.return_value = 0
        mock_emit_nvtx_instance = MagicMock()
        mock_emit_nvtx.return_value = mock_emit_nvtx_instance

        # Import here to ensure all mocks are applied
        from fkat.pytorch.callbacks.cuda.nsys import Nsys

        with patch("fkat.pytorch.callbacks.cuda.nsys.signal"), patch("fkat.pytorch.callbacks.cuda.nsys.atexit"):
            callback = Nsys()
        callback._enabled = True

        # Act
        callback._stop()

        # Assert
        mock_cudart.return_value.cudaProfilerStop.assert_called_once()
        mock_emit_nvtx_instance.__exit__.assert_called_once_with(None, None, None)
        self.assertFalse(callback._enabled)

    @patch("fkat.pytorch.callbacks.cuda.nsys.get_rank")
    @patch("fkat.pytorch.callbacks.cuda.nsys.CallbackLogger")
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.environ", {"NSYS_OUTPUT": "/tmp/test.nsys-rep"})
    def test_setup(self, mock_callback_logger: MagicMock, mock_get_rank: MagicMock) -> None:
        # Arrange
        mock_get_rank.return_value = 0

        # Import here to ensure all mocks are applied
        from fkat.pytorch.callbacks.cuda.nsys import Nsys

        with patch("fkat.pytorch.callbacks.cuda.nsys.signal"), patch("fkat.pytorch.callbacks.cuda.nsys.atexit"):
            callback = Nsys()
        stage = "fit"

        # Act
        with patch.object(callback, "_maybe_trace") as mock_maybe_trace:
            callback.setup(self.trainer, self.pl_module, stage)

        # Assert
        mock_callback_logger.assert_called_once_with(self.trainer)
        self.assertEqual(callback.stage, stage)
        mock_maybe_trace.assert_called_once_with(stage=stage)

    @patch("fkat.pytorch.callbacks.cuda.nsys.get_rank")
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.environ", {"NSYS_OUTPUT": "/tmp/test.nsys-rep"})
    def test_on_train_batch_end(self, mock_get_rank: MagicMock) -> None:
        # Arrange
        mock_get_rank.return_value = 0

        # Import here to ensure all mocks are applied
        from fkat.pytorch.callbacks.cuda.nsys import Nsys

        with patch("fkat.pytorch.callbacks.cuda.nsys.signal"), patch("fkat.pytorch.callbacks.cuda.nsys.atexit"):
            callback = Nsys()
        batch_idx = 5

        # Act
        with patch.object(callback, "_maybe_trace") as mock_maybe_trace:
            callback.on_train_batch_end(self.trainer, self.pl_module, self.outputs, self.batch, batch_idx)

        # Assert
        mock_maybe_trace.assert_called_once_with(self.trainer, "train", batch_idx + 1)

    @patch("fkat.pytorch.callbacks.cuda.nsys.get_rank")
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.environ", {"NSYS_OUTPUT": "/tmp/test.nsys-rep"})
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.makedirs")
    @patch("fkat.pytorch.callbacks.cuda.nsys.shutil")
    @patch("fkat.pytorch.callbacks.cuda.nsys.gzip.open")
    def test_publish(
        self, mock_gzip_open: MagicMock, mock_shutil: MagicMock, mock_makedirs: MagicMock, mock_get_rank: MagicMock
    ) -> None:
        # Arrange
        mock_get_rank.return_value = 0

        # Import here to ensure all mocks are applied
        from fkat.pytorch.callbacks.cuda.nsys import Nsys

        with patch("fkat.pytorch.callbacks.cuda.nsys.signal"), patch("fkat.pytorch.callbacks.cuda.nsys.atexit"):
            callback = Nsys()
        callback.output_file = "/tmp/test.nsys-rep"
        callback._cb_logger = MagicMock()
        mock_file = MagicMock()
        mock_gzip_file = MagicMock()
        mock_gzip_open.return_value.__enter__.return_value = mock_gzip_file

        # Act
        with patch("builtins.open", mock_open()) as mock_file_open:
            mock_file_open.return_value.__enter__.return_value = mock_file
            callback._publish()

        # Assert
        mock_makedirs.assert_called_once_with(os.path.dirname("/tmp/test.nsys-rep"), exist_ok=True)
        mock_file_open.assert_called_once_with("/tmp/test.nsys-rep", "rb")
        mock_gzip_open.assert_called_once_with("/tmp/test.nsys-rep.gz", "wb")
        callback._cb_logger.log_artifact.assert_called_once_with("/tmp/test.nsys-rep.gz", "nsys")

    @patch("fkat.pytorch.callbacks.cuda.nsys.get_rank")
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.environ", {"NSYS_OUTPUT": "/tmp/test.nsys-rep"})
    def test_teardown(self, mock_get_rank: MagicMock) -> None:
        # Arrange
        mock_get_rank.return_value = 0

        # Import here to ensure all mocks are applied
        from fkat.pytorch.callbacks.cuda.nsys import Nsys

        with patch("fkat.pytorch.callbacks.cuda.nsys.signal"), patch("fkat.pytorch.callbacks.cuda.nsys.atexit"):
            callback = Nsys()

        # Act
        with patch.object(callback, "_terminate") as mock_terminate:
            callback.teardown(self.trainer, self.pl_module, "fit")

        # Assert
        mock_terminate.assert_called_once()

    @patch("fkat.pytorch.callbacks.cuda.nsys.get_rank")
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.environ", {"NSYS_OUTPUT": "/tmp/test.nsys-rep"})
    def test_on_validation_batch_end(self, mock_get_rank: MagicMock) -> None:
        # Arrange
        mock_get_rank.return_value = 0

        # Import here to ensure all mocks are applied
        from fkat.pytorch.callbacks.cuda.nsys import Nsys

        with patch("fkat.pytorch.callbacks.cuda.nsys.signal"), patch("fkat.pytorch.callbacks.cuda.nsys.atexit"):
            callback = Nsys()
        batch_idx = 3

        # Act
        with patch.object(callback, "_maybe_trace") as mock_maybe_trace:
            callback.on_validation_batch_end(self.trainer, self.pl_module, self.outputs, self.batch, batch_idx)

        # Assert
        mock_maybe_trace.assert_called_once_with(self.trainer, "validation", batch_idx + 1)

    @patch("fkat.pytorch.callbacks.cuda.nsys.get_rank")
    @patch("fkat.pytorch.callbacks.cuda.nsys.os.environ", {"NSYS_OUTPUT": "/tmp/test.nsys-rep"})
    def test_on_exception(self, mock_get_rank: MagicMock) -> None:
        # Arrange
        mock_get_rank.return_value = 0

        # Import here to ensure all mocks are applied
        from fkat.pytorch.callbacks.cuda.nsys import Nsys

        with patch("fkat.pytorch.callbacks.cuda.nsys.signal"), patch("fkat.pytorch.callbacks.cuda.nsys.atexit"):
            callback = Nsys()
        exception = Exception("Test exception")

        # Act
        with patch.object(callback, "_terminate") as mock_terminate:
            callback.on_exception(self.trainer, self.pl_module, exception)

        # Assert
        mock_terminate.assert_called_once()

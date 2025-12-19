# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch, MagicMock

import lightning as L
import torch

mock_mark = MagicMock()


@patch("fkat.pytorch.callbacks.cuda.nvtx.nvtx")
class TestNvtx(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = MagicMock(spec=L.Trainer)
        self.pl_module = MagicMock(spec=L.LightningModule)
        self.batch = MagicMock()
        self.outputs = MagicMock()
        self.optimizer = MagicMock(spec=torch.optim.Optimizer)
        self.loss = MagicMock(spec=torch.Tensor)
        self.checkpoint = {}
        self.exception = Exception("Test exception")

    def test_domain_from_stage(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Domain

        # Test various stage conversions
        self.assertEqual(Domain.from_stage("fit"), Domain.TRAIN)
        self.assertEqual(Domain.from_stage("train"), Domain.TRAIN)
        self.assertEqual(Domain.from_stage("validation"), Domain.VALIDATION)
        self.assertEqual(Domain.from_stage("test"), Domain.TEST)
        self.assertEqual(Domain.from_stage("predict"), Domain.PREDICT)
        self.assertEqual(Domain.from_stage("tune"), Domain.TUNE)

        # Test invalid stage
        with self.assertRaises(NotImplementedError):
            Domain.from_stage("invalid_stage")

    def test_init(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()

        # Act
        Nvtx()

        # Assert
        mock_nvtx.mark.assert_called_once()
        args, kwargs = mock_nvtx.mark.call_args
        self.assertEqual(kwargs["domain"], Domain.INIT)

    def test_setup(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        stage = "fit"
        mock_nvtx.mark.reset_mock()

        # Act
        callback.setup(self.trainer, self.pl_module, stage)

        # Assert
        mock_nvtx.mark.assert_called_with(f"setup(stage={stage})", domain=Domain.TRAIN)

    def test_on_train_batch_start(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        batch_idx = 5
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_train_batch_start(self.trainer, self.pl_module, self.batch, batch_idx)

        # Assert
        mock_nvtx.mark.assert_called_with(f"on_train_batch_start(batch_idx={batch_idx})", domain=Domain.TRAIN)

    def test_on_validation_epoch_end(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_validation_epoch_end(self.trainer, self.pl_module)

        # Assert
        mock_nvtx.mark.assert_called_with("on_validation_epoch_end()", domain=Domain.VALIDATION)

    def test_on_exception(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_exception(self.trainer, self.pl_module, self.exception)

        # Assert
        mock_nvtx.mark.assert_called_with(f"on_exception({type(self.exception)})", domain=Domain.ERROR)

    def test_on_save_checkpoint(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_save_checkpoint(self.trainer, self.pl_module, self.checkpoint)

        # Assert
        mock_nvtx.mark.assert_called_with("on_save_checkpoint()", domain=Domain.CHECKPOINT)

    def test_on_train_epoch_start(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_train_epoch_start(self.trainer, self.pl_module)

        # Assert
        mock_nvtx.mark.assert_called_with("on_train_epoch_start()", domain=Domain.TRAIN)

    def test_on_before_zero_grad(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_before_zero_grad(self.trainer, self.pl_module, self.optimizer)

        # Assert
        mock_nvtx.mark.assert_called_with("on_before_zero_grad()", domain=Domain.TRAIN)

    def test_on_before_backward(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_before_backward(self.trainer, self.pl_module, self.loss)

        # Assert
        mock_nvtx.mark.assert_called_with("on_before_backward()", domain=Domain.TRAIN)

    def test_on_after_backward(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_after_backward(self.trainer, self.pl_module)

        # Assert
        mock_nvtx.mark.assert_called_with("on_after_backward()", domain=Domain.TRAIN)

    def test_on_before_optimizer_step(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_before_optimizer_step(self.trainer, self.pl_module, self.optimizer)

        # Assert
        mock_nvtx.mark.assert_called_with("on_before_optimizer_step()", domain=Domain.TRAIN)

    def test_on_train_batch_end(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        batch_idx = 5
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_train_batch_end(self.trainer, self.pl_module, self.outputs, self.batch, batch_idx)

        # Assert
        mock_nvtx.mark.assert_called_with(f"on_train_batch_end(batch_idx={batch_idx})", domain=Domain.TRAIN)

    def test_on_train_epoch_end(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_train_epoch_end(self.trainer, self.pl_module)

        # Assert
        mock_nvtx.mark.assert_called_with("on_train_epoch_end()", domain=Domain.TRAIN)

    def test_on_train_end(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_train_end(self.trainer, self.pl_module)

        # Assert
        mock_nvtx.mark.assert_called_with("on_train_end()", domain=Domain.TRAIN)

    def test_on_test_start(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_test_start(self.trainer, self.pl_module)

        # Assert
        mock_nvtx.mark.assert_called_with("on_test_start()", domain=Domain.TEST)

    def test_on_predict_start(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_predict_start(self.trainer, self.pl_module)

        # Assert
        mock_nvtx.mark.assert_called_with("on_predict_start()", domain=Domain.PREDICT)

    def test_state_dict_and_load(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act - state_dict
        result = callback.state_dict()

        # Assert - state_dict
        mock_nvtx.mark.assert_called_with("state_dict()", domain=Domain.CHECKPOINT)
        self.assertEqual(result, {})

        # Reset mock
        mock_nvtx.mark.reset_mock()

        # Act - load_state_dict
        callback.load_state_dict({})

        # Assert - load_state_dict
        mock_nvtx.mark.assert_called_with("load_state_dict()", domain=Domain.CHECKPOINT)

    def test_on_load_checkpoint(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()
        mock_nvtx.mark.reset_mock()

        # Act
        callback.on_load_checkpoint(self.trainer, self.pl_module, self.checkpoint)

        # Assert
        mock_nvtx.mark.assert_called_with("on_load_checkpoint()", domain=Domain.CHECKPOINT)

    def test_sanity_check_methods(self, mock_nvtx: MagicMock) -> None:
        # Import here to ensure the mock is applied
        from fkat.pytorch.callbacks.cuda.nvtx import Nvtx, Domain

        # Arrange
        mock_nvtx.mark = MagicMock()
        callback = Nvtx()

        # Test on_sanity_check_start
        mock_nvtx.mark.reset_mock()
        callback.on_sanity_check_start(self.trainer, self.pl_module)
        mock_nvtx.mark.assert_called_with("on_validation_start()", domain=Domain.VALIDATION)

        # Test on_sanity_check_end
        mock_nvtx.mark.reset_mock()
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        mock_nvtx.mark.assert_called_with("on_sanity_check_start()", domain=Domain.VALIDATION)

        # Test on_validation_start
        mock_nvtx.mark.reset_mock()
        callback.on_validation_start(self.trainer, self.pl_module)
        mock_nvtx.mark.assert_called_with("on_sanity_check_end()", domain=Domain.VALIDATION)

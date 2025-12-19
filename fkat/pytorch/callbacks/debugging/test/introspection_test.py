# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from fkat.pytorch.callbacks.debugging.introspection import _process_collection, _ensure_hashable, process_item

import unittest
from unittest.mock import mock_open, call, MagicMock, patch
from collections import namedtuple

import torch
import numpy as np

from fkat.pytorch.schedule import Fixed
from fkat.pytorch.callbacks.debugging.introspection import Introspection, _batch_checksums
from fkat.pytorch.callbacks.loggers import CallbackLogger

MODULE = "fkat.pytorch.callbacks.debugging.introspection"


class TestIntrospection(unittest.TestCase):
    @patch(f"{MODULE}._batch_checksums")
    @patch("builtins.open", new_callable=mock_open, read_data="dummy_content")
    @patch(f"{MODULE}.tempfile.TemporaryDirectory")
    @patch(f"{MODULE}.CallbackLogger")
    @patch(f"{MODULE}.L.LightningModule")
    @patch(f"{MODULE}.L.Trainer")
    @patch(f"{MODULE}.distributions")
    @patch(f"{MODULE}.hashlib")
    @patch(f"{MODULE}.torch")
    @patch(f"{MODULE}.yaml")
    @patch(f"{MODULE}.os")
    @patch(f"{MODULE}.get_checksums")
    def test_introspection(
        self,
        mock_get_checksums,
        mock_os,
        mock_yaml,
        mock_torch,
        mock_hashlib,
        mock_distributions,
        mock_trainer,
        mock_module,
        mock_callback_logger,
        mock_temp_dir,
        mock_open_func,
        mock_batch_checksums,
    ):
        # Arrange
        mock_schedule = MagicMock(spec=Fixed)
        callback = Introspection(
            checksums={"params", "buffers", "grads", "rngs", "batch", "optimizers"},
            tensor_stats={"shape", "dtype", "hash"},
            env_vars=True,
            pip_freeze=True,
            schedule=mock_schedule,
        )
        mock_trainer.global_rank = 2
        mock_callback_logger.return_value = (mock_cb_logger := MagicMock(spec=CallbackLogger))
        mock_temp_dir.return_value.__enter__.return_value = "test_dir"
        mock_temp_dir.return_value.__exit__.return_value = None
        mock_module.named_parameters.return_value = [(str(i), MagicMock(spec=torch.nn.Parameter)) for i in range(3)]
        mock_module.named_buffers.return_value = [(str(i), MagicMock(spec=torch.Tensor)) for i in range(2)]
        mock_os.environ = envs = {"some_var": "some_val"}
        mock_os.path.join.side_effect = [
            "test_dir/env_vars.yaml",
            "test_dir/pip_freeze.yaml",
            "test_dir/rank2.yaml",
        ]
        mock_distributions.return_value = (
            pkgs := [
                namedtuple("distribution", ["metadata", "version"])(metadata={"Name": "some_pkg"}, version="some_ver"),
            ]
        )
        # Configure mock_schedule to return True when check is called
        mock_schedule.check.return_value = True

        # Mock batch data and batch_checksums result
        mock_batch = MagicMock()
        mock_batch_checksums.return_value = {
            "[]": {
                "0": "checksum1",
                "1": "checksum2",
            },
            "__all_batch__": "batch_checksum",
        }

        # Mock get_checksums to return a predefined result
        mock_get_checksums.return_value = {
            "batch": {
                "[]": {
                    "0": "checksum1",
                    "1": "checksum2",
                },
                "__all_batch__": "batch_checksum",
            },
            "parameters": {
                "__all_data__": "data_checksum",
                "__all_grads__": "grads_checksum",
                "0": {"data": "2×3|fp32|abc123", "grad": "2×3|fp32|def456"},
                "1": {"data": "4×5|fp32|ghi789", "grad": "4×5|fp32|jkl012"},
                "2": {"data": "6×7|fp32|mno345", "grad": "6×7|fp32|pqr678"},
            },
            "buffers": {
                "__all_buffers__": "buffers_checksum",
                "0": "2×2|fp32|buffer0",
                "1": "3×3|fp32|buffer1",
            },
            "optimizers": [
                {
                    "__type__": "Adam",
                    "defaults": {"lr": 0.001, "betas": [0.9, 0.999]},
                    "param_groups": [{"params": ["param0", "param1"], "lr": 0.001}],
                    "state": {"param0": {"step": 10, "exp_avg": "exp_avg_hash"}},
                }
            ],
            "rngs": {
                "torch": "torch_rng",
                "torch.cuda": "cuda_rng",
                "numpy": "numpy_rng",
                "python": "python_rng",
            },
        }

        # Act
        callback.setup(mock_trainer, mock_module, "some_stage")
        mock_trainer.global_step = 100
        # Test the on_train_batch_start method with batch data
        callback.on_train_batch_start(mock_trainer, mock_module, mock_batch, 42)
        callback.on_before_optimizer_step(mock_trainer, mock_module, mock_torch.optim.Optimizer())
        callback.on_train_batch_end(mock_trainer, mock_module, None, mock_batch, 42)

        # Assert
        mock_os.path.join.assert_has_calls(
            [
                call("test_dir", "env_vars.yaml"),
                call("test_dir", "pip_freeze.yaml"),
                call("test_dir", "rank2.yaml"),
            ]
        )
        mock_cb_logger.log_artifact.assert_has_calls(
            [
                call("test_dir/env_vars.yaml", "introspection"),
                call("test_dir/pip_freeze.yaml", "introspection"),
                call("test_dir/rank2.yaml", "introspection/some_stage/step=100"),
            ]
        )

        # Verify that open was called with UTF-8 encoding
        # Note: mock_open includes context manager calls (__enter__ and __exit__)
        expected_calls = [
            call("test_dir/env_vars.yaml", "w", encoding="utf-8"),
            call().__enter__(),
            call().__exit__(None, None, None),
            call("test_dir/pip_freeze.yaml", "w", encoding="utf-8"),
            call().__enter__(),
            call().__exit__(None, None, None),
            call("test_dir/rank2.yaml", "w", encoding="utf-8"),
            call().__enter__(),
            call().__exit__(None, None, None),
        ]
        mock_open_func.assert_has_calls(expected_calls)

        # Verify that schedule.check was called with the correct parameters including batch_idx and trainer
        batch_idx = 42
        mock_schedule.check.assert_called_once_with(stage="train", batch_idx=batch_idx, step=100, trainer=mock_trainer)

        # Verify that _batch_checksums was called with the mock_batch and tensor_stats
        mock_batch_checksums.assert_called_once_with(mock_batch, batch_idx, callback.tensor_stats)

        # Verify that get_checksums was called with the right parameters
        mock_get_checksums.assert_called_once_with(
            mock_trainer,
            mock_module,
            callback.gradients,
            callback.checksums,
            callback.tensor_stats,
        )

        # Verify yaml.dump was called instead of json.dumps
        assert mock_yaml.dump.call_count == 3
        mock_yaml.dump.assert_has_calls(
            [
                call(
                    envs,
                    mock_open_func(),
                    sort_keys=False,
                    indent=2,
                    default_flow_style=False,
                    allow_unicode=True,
                    width=10**6,
                ),
                call(
                    {p.metadata["Name"]: p.version for p in pkgs},
                    mock_open_func(),
                    sort_keys=False,
                    indent=2,
                    default_flow_style=False,
                    allow_unicode=True,
                    width=10**6,
                ),
                call(
                    mock_get_checksums.return_value,
                    mock_open_func(),
                    sort_keys=False,
                    indent=2,
                    default_flow_style=False,
                    allow_unicode=True,
                    width=10**6,
                ),
            ]
        )

    @patch(f"{MODULE}._batch_checksums")
    @patch(f"{MODULE}.get_checksums")
    @patch(f"{MODULE}.CallbackLogger")
    @patch(f"{MODULE}.L.Trainer")
    @patch(f"{MODULE}.L.LightningModule")
    def test_batch_checksum_disabled(
        self,
        mock_module,
        mock_trainer,
        mock_callback_logger,
        mock_get_checksums,
        mock_batch_checksums,
    ):
        # Arrange
        mock_schedule = MagicMock(spec=Fixed)
        callback = Introspection(checksums=set(), schedule=mock_schedule)  # No checksums enabled
        mock_batch = MagicMock()

        # Configure mocks
        mock_schedule.check.return_value = True

        # Act
        callback.setup(mock_trainer, mock_module, "train")
        callback.on_train_batch_start(mock_trainer, mock_module, mock_batch, 0)
        callback.on_train_batch_end(mock_trainer, mock_module, None, mock_batch, 0)

        # Assert
        mock_schedule.check.assert_called_once()
        mock_batch_checksums.assert_not_called()

    @patch(f"{MODULE}._batch_checksums")
    @patch(f"{MODULE}.get_checksums")
    @patch(f"{MODULE}.CallbackLogger")
    @patch(f"{MODULE}.L.Trainer")
    @patch(f"{MODULE}.L.LightningModule")
    def test_batch_checksum_enabled_but_not_scheduled(
        self,
        mock_module,
        mock_trainer,
        mock_callback_logger,
        mock_get_checksums,
        mock_batch_checksums,
    ):
        # Arrange
        mock_schedule = MagicMock(spec=Fixed)
        callback = Introspection(checksums={"batch"}, schedule=mock_schedule)  # Only batch checksum enabled
        mock_batch = MagicMock()

        # Configure mocks
        mock_schedule.check.return_value = False  # Schedule returns False

        # Act
        callback.setup(mock_trainer, mock_module, "train")
        callback.on_train_batch_start(mock_trainer, mock_module, mock_batch, 0)
        callback.on_train_batch_end(mock_trainer, mock_module, None, mock_batch, 0)

        # Assert
        mock_schedule.check.assert_called_once()
        mock_batch_checksums.assert_not_called()

    @patch(f"{MODULE}._batch_checksums")
    @patch(f"{MODULE}.get_checksums")
    @patch(f"{MODULE}.CallbackLogger")
    @patch(f"{MODULE}.L.Trainer")
    @patch(f"{MODULE}.L.LightningModule")
    def test_batch_checksum_enabled_and_scheduled(
        self,
        mock_module,
        mock_trainer,
        mock_callback_logger,
        mock_get_checksums,
        mock_batch_checksums,
    ):
        # Arrange
        mock_schedule = MagicMock(spec=Fixed)
        callback = Introspection(checksums={"batch"}, schedule=mock_schedule)  # Only batch checksum enabled
        mock_batch = MagicMock()

        # Configure mocks
        mock_schedule.check.return_value = True
        mock_batch_checksums.return_value = {"test": "checksum"}
        mock_get_checksums.return_value = {"checksums": "data"}

        # Act
        callback.setup(mock_trainer, mock_module, "train")
        callback.on_train_batch_start(mock_trainer, mock_module, mock_batch, 0)
        callback.on_train_batch_end(mock_trainer, mock_module, None, mock_batch, 0)

        # Assert
        batch_idx = 0
        mock_schedule.check.assert_called_once_with(
            stage="train", batch_idx=batch_idx, step=mock_trainer.global_step, trainer=mock_trainer
        )
        mock_batch_checksums.assert_called_once_with(mock_batch, batch_idx, callback.tensor_stats)

    @patch(f"{MODULE}.tensor_checksum")
    @patch(f"{MODULE}.hashlib.md5")
    def test_batch_checksums_tensor(self, mock_md5, mock_tensor_checksum):
        # Arrange
        mock_tensor_checksum.return_value.hexdigest.return_value = "tensor_checksum"
        mock_tensor_checksum.return_value.digest.return_value = b"tensor_digest"

        # Create a mock tensor with the necessary attributes for _format
        tensor_batch = MagicMock(spec=torch.Tensor)
        tensor_batch.dim.return_value = 2
        tensor_batch.shape = (2, 3)
        tensor_batch.dtype = torch.float32
        tensor_stats = {"hash"}  # Default tensor stats

        # Act
        with patch(f"{MODULE}._format", return_value="formatted_tensor"):
            result = _batch_checksums(tensor_batch, 42, tensor_stats)

        # Assert
        mock_tensor_checksum.assert_called_once_with(tensor_batch)
        # The actual implementation converts the bytes to hex, so we'll just check it exists
        assert "__batch_idx__" in result
        assert "__all_batch__" in result

    @patch(f"{MODULE}.tensor_checksum")
    @patch(f"{MODULE}.hashlib.md5")
    def test_batch_checksums_list(self, mock_md5, mock_tensor_checksum):
        # Arrange
        mock_tensor_checksum.return_value.hexdigest.return_value = "tensor_checksum"
        mock_tensor_checksum.return_value.digest.return_value = b"tensor_digest"
        # Create a mock for the hex method
        hex_mock = MagicMock(return_value="md5_digest_hex")
        mock_md5.return_value.digest.return_value.hex = hex_mock

        # Create mock tensors with necessary attributes
        tensor1 = MagicMock(spec=torch.Tensor)
        tensor1.dim.return_value = 2
        tensor1.shape = (2, 3)
        tensor1.dtype = torch.float32

        tensor2 = MagicMock(spec=torch.Tensor)
        tensor2.dim.return_value = 1
        tensor2.shape = (5,)
        tensor2.dtype = torch.float32

        list_batch = [tensor1, tensor2]
        tensor_stats = {"shape", "dtype"}  # Custom tensor stats

        # Act
        with patch(f"{MODULE}._format", return_value="formatted_tensor"):
            result = _batch_checksums(list_batch, 42, tensor_stats)

        # Assert
        assert mock_tensor_checksum.call_count == 2
        # With the new implementation, the list is stored in __batch__ key
        assert isinstance(result["__batch__"], list)
        assert len(result["__batch__"]) == 2
        assert "__batch_idx__" in result
        assert "__all_batch__" in result

    @patch(f"{MODULE}.tensor_checksum")
    @patch(f"{MODULE}.hashlib.md5")
    def test_batch_checksums_dict(self, mock_md5, mock_tensor_checksum):
        # Arrange
        mock_tensor_checksum.return_value.hexdigest.return_value = "tensor_checksum"
        mock_tensor_checksum.return_value.digest.return_value = b"tensor_digest"
        # Create a mock for the hex method
        hex_mock = MagicMock(return_value="md5_digest_hex")
        mock_md5.return_value.digest.return_value.hex = hex_mock

        # Create mock tensors with necessary attributes
        tensor1 = MagicMock(spec=torch.Tensor)
        tensor1.dim.return_value = 2
        tensor1.shape = (2, 3)
        tensor1.dtype = torch.float32

        tensor2 = MagicMock(spec=torch.Tensor)
        tensor2.dim.return_value = 1
        tensor2.shape = (5,)
        tensor2.dtype = torch.float32

        dict_batch = {"input": tensor1, "target": tensor2}
        tensor_stats = {"hash", "mean", "std"}  # Custom tensor stats

        # Act
        with patch(f"{MODULE}._format", return_value="formatted_tensor"):
            result = _batch_checksums(dict_batch, 42, tensor_stats)

        # Assert
        assert mock_tensor_checksum.call_count == 2
        # With the new implementation, the dict is stored in __batch__ key
        assert isinstance(result["__batch__"], dict)
        assert "input" in result["__batch__"]
        assert "target" in result["__batch__"]
        assert "__batch_idx__" in result
        assert "__all_batch__" in result

    @patch(f"{MODULE}.tensor_checksum")
    @patch(f"{MODULE}.hashlib.md5")
    def test_batch_checksums_nested_list(self, mock_md5, mock_tensor_checksum):
        # Arrange
        mock_tensor_checksum.return_value.hexdigest.return_value = "tensor_checksum"
        mock_tensor_checksum.return_value.digest.return_value = b"tensor_digest"
        # Create a mock for the hex method
        hex_mock = MagicMock(return_value="md5_digest_hex")
        mock_md5.return_value.digest.return_value.hex = hex_mock

        # Create mock tensors with necessary attributes
        tensor1 = MagicMock(spec=torch.Tensor)
        tensor1.dim.return_value = 2
        tensor1.shape = (2, 3)
        tensor1.dtype = torch.float32

        tensor2 = MagicMock(spec=torch.Tensor)
        tensor2.dim.return_value = 1
        tensor2.shape = (5,)
        tensor2.dtype = torch.float32

        tensor3 = MagicMock(spec=torch.Tensor)
        tensor3.dim.return_value = 3
        tensor3.shape = (2, 3, 4)
        tensor3.dtype = torch.float32

        nested_list_batch = [[tensor1, tensor2], tensor3]
        tensor_stats = {"hash"}

        # Act
        with patch(f"{MODULE}._format", return_value="formatted_tensor"):
            result = _batch_checksums(nested_list_batch, 42, tensor_stats)

        # Assert
        assert mock_tensor_checksum.call_count == 3
        # With the new implementation, the nested list is stored in __batch__ key
        assert isinstance(result["__batch__"], list)
        assert len(result["__batch__"]) == 2
        assert isinstance(result["__batch__"][0], list)
        assert len(result["__batch__"][0]) == 2
        assert "__batch_idx__" in result
        assert "__all_batch__" in result

    @patch(f"{MODULE}.numpy_checksum")
    @patch(f"{MODULE}.hashlib.md5")
    def test_batch_checksums_numpy_array(self, mock_md5, mock_numpy_checksum):
        # Arrange
        mock_numpy_checksum.return_value.hexdigest.return_value = "numpy_checksum"
        mock_numpy_checksum.return_value.digest.return_value = b"numpy_digest"
        # Create a mock for the hex method
        hex_mock = MagicMock(return_value="md5_digest_hex")
        mock_md5.return_value.digest.return_value.hex = hex_mock

        # Create a mock numpy array with necessary attributes for _format
        numpy_array = MagicMock(spec=np.ndarray)
        numpy_array.ndim = 2
        numpy_array.shape = (3, 4)
        numpy_array.dtype = np.float32

        # Create a dictionary with numpy arrays
        dict_batch = {"input": numpy_array}
        tensor_stats = {"hash"}

        # Act
        with patch(f"{MODULE}._format", return_value="formatted_ndarray"):
            result = _batch_checksums(dict_batch, 42, tensor_stats)

        # Assert
        mock_numpy_checksum.assert_called_once_with(numpy_array)
        # With the new implementation, the dict is stored in __batch__ key
        assert isinstance(result["__batch__"], dict)
        assert "input" in result["__batch__"]
        assert "__batch_idx__" in result
        assert "__all_batch__" in result

    @patch(f"{MODULE}.tensor_checksum")
    @patch(f"{MODULE}.hashlib.md5")
    def test_deeply_nested_structure(self, mock_md5, mock_tensor_checksum):
        # Arrange
        mock_tensor_checksum.return_value.hexdigest.return_value = "tensor_checksum"
        mock_tensor_checksum.return_value.digest.return_value = b"tensor_digest"
        # Create a mock for the hex method
        hex_mock = MagicMock(return_value="md5_digest_hex")
        mock_md5.return_value.digest.return_value.hex = hex_mock

        # Create mock tensors with necessary attributes
        tensor1 = MagicMock(spec=torch.Tensor)
        tensor1.dim.return_value = 2
        tensor1.shape = (2, 3)
        tensor1.dtype = torch.float32

        tensor2 = MagicMock(spec=torch.Tensor)
        tensor2.dim.return_value = 1
        tensor2.shape = (5,)
        tensor2.dtype = torch.float32

        tensor3 = MagicMock(spec=torch.Tensor)
        tensor3.dim.return_value = 3
        tensor3.shape = (2, 3, 4)
        tensor3.dtype = torch.float32

        tensor4 = MagicMock(spec=torch.Tensor)
        tensor4.dim.return_value = 0
        tensor4.shape = ()
        tensor4.dtype = torch.float32
        tensor4.item.return_value = 42.0

        # Create a deeply nested structure
        # Structure: {'level1': [tensor1, {'level2': [tensor2, [tensor3, {'level3': tensor4}]]}]}
        nested_batch = {"level1": [tensor1, {"level2": [tensor2, [tensor3, {"level3": tensor4}]]}]}
        tensor_stats = {"shape", "dtype", "hash"}

        # Act
        with patch(f"{MODULE}._format", return_value="formatted_tensor"):
            result = _batch_checksums(nested_batch, 42, tensor_stats)

        # Assert
        # Verify all tensors were processed
        assert mock_tensor_checksum.call_count == 4

        # Check that the structure was properly traversed
        assert "__batch__" in result
        assert "__batch_idx__" in result
        assert "__all_batch__" in result
        assert isinstance(result["__batch__"], dict)
        assert "level1" in result["__batch__"]

    @patch(f"{MODULE}.tensor_checksum")
    @patch(f"{MODULE}.numpy_checksum")
    @patch(f"{MODULE}.hashlib.md5")
    def test_mixed_types(self, mock_md5, mock_numpy_checksum, mock_tensor_checksum):
        # Arrange
        mock_tensor_checksum.return_value.hexdigest.return_value = "tensor_checksum"
        mock_tensor_checksum.return_value.digest.return_value = b"tensor_digest"
        mock_numpy_checksum.return_value.hexdigest.return_value = "numpy_checksum"
        mock_numpy_checksum.return_value.digest.return_value = b"numpy_digest"
        mock_md5.return_value.hexdigest.return_value = "md5_hexdigest"
        # Create a mock for the hex method
        hex_mock = MagicMock(return_value="md5_digest_hex")
        mock_md5.return_value.digest.return_value.hex = hex_mock

        # Create mock objects with necessary attributes
        tensor = MagicMock(spec=torch.Tensor)
        tensor.dim.return_value = 2
        tensor.shape = (2, 3)
        tensor.dtype = torch.float32

        numpy_array = MagicMock(spec=np.ndarray)
        numpy_array.ndim = 2
        numpy_array.shape = (3, 4)
        numpy_array.dtype = np.float32

        # Create a structure with mixed types
        mixed_batch = [
            tensor,
            numpy_array,
            {"tensor_key": tensor, "numpy_key": numpy_array, "nested": [tensor, numpy_array]},
        ]
        tensor_stats = {"hash", "shape"}

        # Act
        with patch(f"{MODULE}._format", return_value="formatted_data"):
            result = _batch_checksums(mixed_batch, 42, tensor_stats)

        # Assert
        # Verify both tensor and numpy checksums were called
        assert mock_tensor_checksum.called
        assert mock_numpy_checksum.called

        # Check that the structure was properly traversed
        assert "__batch__" in result
        assert "__batch_idx__" in result
        assert "__all_batch__" in result
        assert isinstance(result["__batch__"], list)
        assert len(result["__batch__"]) == 3

    @patch(f"{MODULE}.tensor_checksum")
    @patch(f"{MODULE}.hashlib.md5")
    def test_empty_structures(self, mock_md5, mock_tensor_checksum):
        # Arrange
        mock_md5.return_value.hexdigest.return_value = "md5_hexdigest"
        # Create a mock for the hex method
        hex_mock = MagicMock(return_value="md5_digest_hex")
        mock_md5.return_value.digest.return_value.hex = hex_mock

        # Create empty structures
        empty_list = []
        empty_dict = {}
        empty_tuple = ()
        tensor_stats = {"hash"}

        # Act
        result_list = _batch_checksums(empty_list, 42, tensor_stats)
        result_dict = _batch_checksums(empty_dict, 42, tensor_stats)
        result_tuple = _batch_checksums(empty_tuple, 42, tensor_stats)

        # Assert
        # Verify tensor_checksum was not called
        mock_tensor_checksum.assert_not_called()

        # Check that the empty structures were handled properly
        assert "__batch__" in result_list
        assert "__batch__" in result_dict
        assert "__batch__" in result_tuple
        assert "__batch_idx__" in result_list
        assert "__batch_idx__" in result_dict
        assert "__batch_idx__" in result_tuple
        assert result_list["__batch__"] == []
        assert result_dict["__batch__"] == {}
        assert result_tuple["__batch__"] == []

    @patch(f"{MODULE}.tensor_checksum")
    @patch(f"{MODULE}.hashlib.md5")
    def test_gradient_capture_via_hooks(self, mock_md5, mock_tensor_checksum):
        """Test that gradients are properly captured via hooks."""
        # Create a real tensor with requires_grad=True to test hooks
        real_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # Create a mock for the gradient
        mock_grad = torch.tensor([0.1, 0.2, 0.3])

        # Set up tensor_checksum mock
        mock_tensor_checksum.return_value.hexdigest.return_value = "grad_checksum"
        mock_tensor_checksum.return_value.digest.return_value = b"grad_digest"

        # Import the actual function to test
        from fkat.pytorch.callbacks.debugging.introspection import _params_checksums

        # Create a simple model with our tensor as parameter
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(real_tensor)

            def forward(self, x):
                return self.param * x

        model = SimpleModel()

        # Manually set the gradient
        model.param.grad = mock_grad

        # Create gradients dictionary as expected by _params_checksums
        gradients = {"param": (mock_tensor_checksum.return_value, "3×1|fp32|grad_checksum")}
        tensor_stats = {"shape", "dtype", "hash"}

        # Call the function that should capture gradients
        result = _params_checksums(
            model, params_checksum=True, grads_checksum=True, gradients=gradients, tensor_stats=tensor_stats
        )

        # Check that the checksums were properly stored
        assert "param" in result
        assert "data" in result["param"]
        assert "grad" in result["param"]
        assert result["param"]["grad"] == "3×1|fp32|grad_checksum"

        # Check that the overall checksums were computed
        assert "__all_data__" in result
        assert "__all_grads__" in result

    def test_gradient_hook_with_backward_pass(self):
        """Test that gradients are properly captured during an actual backward pass."""

        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Create input and target
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[3.0], [7.0]])

        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)

        # Create a dictionary to store gradients
        captured_grads = {}

        # Register hooks on parameters
        hooks = []
        for name, param in model.named_parameters():

            def get_hook(param_name):
                def hook(grad):
                    captured_grads[param_name] = grad.detach().clone()
                    return grad

                return hook

            hooks.append(param.register_hook(get_hook(name)))

        # Backward pass
        loss.backward()

        # Import the function to test
        from fkat.pytorch.callbacks.debugging.introspection import _params_checksums, tensor_checksum, _format

        # Create gradients dictionary as expected by _params_checksums
        gradients = {}
        tensor_stats = {"shape", "dtype", "hash"}
        for name, grad in captured_grads.items():
            cks = tensor_checksum(grad)
            gradients[name] = (cks, _format(grad, cks.hexdigest(), tensor_stats))

        # Get checksums
        result = _params_checksums(
            model, params_checksum=True, grads_checksum=True, gradients=gradients, tensor_stats=tensor_stats
        )

        # Verify that gradients were captured
        assert all(param.grad is not None for param in model.parameters())

        # Verify that our hook captured the same gradients
        for name, param in model.named_parameters():
            assert name in captured_grads
            assert torch.allclose(param.grad, captured_grads[name])  # type: ignore[arg-type]

        # Verify that checksums were computed for all parameters
        for name, _ in model.named_parameters():
            assert name in result
            assert "data" in result[name]
            assert "grad" in result[name]

        # Clean up hooks
        for hook in hooks:
            hook.remove()

    @patch(f"{MODULE}.get_checksums")
    @patch(f"{MODULE}.CallbackLogger")
    @patch(f"{MODULE}.L.Trainer")
    @patch(f"{MODULE}.L.LightningModule")
    def test_introspection_with_hooks(
        self,
        mock_module,
        mock_trainer,
        mock_callback_logger,
        mock_get_checksums,
    ):
        """Test that the Introspection callback properly registers hooks and captures gradients."""
        # Arrange
        mock_schedule = MagicMock(spec=Fixed)
        callback = Introspection(checksums={"params", "grads"}, schedule=mock_schedule)

        # Configure mocks
        mock_schedule.check.return_value = True
        mock_get_checksums.return_value = {"test": "data"}

        # Act - simulate the training flow
        callback.setup(mock_trainer, mock_module, "train")
        callback.on_train_start(mock_trainer, mock_module)

        # Assert
        # Verify that hooks were registered
        assert callback._hooks_registered

        # Verify that the hook registration function was called
        mock_module.named_parameters.assert_called_once()

        # Clean up
        callback.on_train_end(mock_trainer, mock_module)

    @patch(f"{MODULE}._optimizers_checksums")
    @patch(f"{MODULE}.get_checksums")
    @patch(f"{MODULE}.CallbackLogger")
    @patch(f"{MODULE}.L.Trainer")
    @patch(f"{MODULE}.L.LightningModule")
    def test_optimizers_checksum(
        self,
        mock_module,
        mock_trainer,
        mock_callback_logger,
        mock_get_checksums,
        mock_optimizers_checksums,
    ):
        """Test that the optimizers_checksum feature properly captures optimizer state."""
        # Arrange
        mock_schedule = MagicMock(spec=Fixed)
        callback = Introspection(
            checksums={"optimizers"},  # Only enable optimizers checksum
            schedule=mock_schedule,
        )

        # Mock optimizer data
        mock_optimizer = MagicMock()
        mock_trainer.optimizers = [mock_optimizer]

        # Configure mocks
        mock_schedule.check.return_value = True
        mock_optimizers_checksums.return_value = [
            {
                "__type__": "Adam",
                "defaults": {"lr": 0.001, "betas": [0.9, 0.999]},
                "param_groups": [{"params": ["param0", "param1"], "lr": 0.001}],
                "state": {"param0": {"step": 10, "exp_avg": "exp_avg_hash"}},
            }
        ]
        mock_get_checksums.return_value = {"optimizers": mock_optimizers_checksums.return_value}

        # Act - simulate the training flow
        callback.setup(mock_trainer, mock_module, "train")
        mock_trainer.global_step = 100
        callback.on_train_batch_start(mock_trainer, mock_module, MagicMock(), 0)
        callback.on_before_optimizer_step(mock_trainer, mock_module, mock_optimizer)
        callback.on_train_batch_end(mock_trainer, mock_module, None, MagicMock(), 0)

        # Assert
        # Verify that get_checksums was called with the right parameters
        mock_get_checksums.assert_called_once()
        args, kwargs = mock_get_checksums.call_args

        # Check that the checksums set contains "optimizers"
        assert args[3] == {"optimizers"}  # args[3] should be the checksums set

    @patch("builtins.open", new_callable=mock_open)
    @patch(f"{MODULE}.tempfile.TemporaryDirectory")
    @patch(f"{MODULE}.yaml.dump")
    @patch(f"{MODULE}.CallbackLogger")
    def test_publish_with_unicode_characters(
        self,
        mock_callback_logger,
        mock_yaml_dump,
        mock_temp_dir,
        mock_open_func,
    ):
        """Test that _publish method properly handles Unicode characters by using UTF-8 encoding."""
        # Arrange
        callback = Introspection()
        mock_temp_dir.return_value.__enter__.return_value = "test_dir"
        mock_temp_dir.return_value.__exit__.return_value = None

        # Create test data with Unicode characters (multiplication sign ×)
        test_data = {
            "tensor_shape": "2×3×4",  # Contains Unicode multiplication sign
            "parameter_name": "weight_α",  # Contains Greek letter alpha
            "description": "Test with émojis 🚀",  # Contains accented character and emoji
        }

        # Mock the callback logger
        mock_cb_logger = MagicMock(spec=CallbackLogger)
        callback._cb_logger = mock_cb_logger

        # Act
        callback._publish("test_file.yaml", "test/path", test_data)

        # Assert
        # Verify that open was called with UTF-8 encoding
        mock_open_func.assert_called_once_with("test_dir/test_file.yaml", "w", encoding="utf-8")

        # Verify that yaml.dump was called with the test data
        mock_yaml_dump.assert_called_once_with(
            test_data,
            mock_open_func(),
            sort_keys=False,
            indent=2,
            default_flow_style=False,
            allow_unicode=True,
            width=10**6,
        )

        # Verify that the artifact was logged
        mock_cb_logger.log_artifact.assert_called_once_with("test_dir/test_file.yaml", "test/path")


# ProxyDict class for testing optimizer state
class ProxyDict:
    """
    A dictionary-like object that proxies to a list of dictionaries.

    e.g., ProxyDict([{'a': 1}, {'b': 2}]) behaves like:
    {
        (0, 'a'): 1,
        (1, 'b'): 2,
    }
    We use tuples as keys to avoid ambiguity with the keys of the inner dicts.
    """

    def __init__(self, inner_dicts: list[dict]):
        self._inner_dicts = inner_dicts

    def __getitem__(self, key: tuple[int, str]):
        idx, inner_key = key
        return self._inner_dicts[idx].get(inner_key)

    def __setitem__(self, key: tuple[int, str], value):
        idx, inner_key = key
        self._inner_dicts[idx][inner_key] = value

    def __len__(self) -> int:
        return sum([len(inner_dict) for inner_dict in self._inner_dicts])

    def __iter__(self):
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key in inner_dict:
                yield (idx, inner_key)

    def items(self):
        """Return generator over underlying items."""
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key, value in inner_dict.items():
                yield (idx, inner_key), value


class TestProxyDictOptimizer(unittest.TestCase):
    """Tests for ProxyDict optimizer state handling."""

    @patch(f"{MODULE}.process_item")
    def test_optimizers_checksums_with_proxy_dict(self, mock_process_item):
        """Test that _optimizers_checksums properly handles ProxyDict state."""
        from unittest.mock import ANY

        # Create a mock optimizer with ProxyDict state
        mock_optimizer = MagicMock()
        mock_optimizer.__class__.__name__ = "MockOptimizer"

        # Create a ProxyDict instance for the optimizer state
        inner_dicts = [
            {"exp_avg": torch.tensor([0.1, 0.2]), "exp_avg_sq": torch.tensor([0.01, 0.04])},
            {"exp_avg": torch.tensor([0.3, 0.4]), "exp_avg_sq": torch.tensor([0.09, 0.16])},
        ]
        proxy_dict_state = ProxyDict(inner_dicts)

        # Set up the optimizer with the ProxyDict state
        mock_optimizer.defaults = {"lr": 0.001, "betas": (0.9, 0.999)}
        mock_optimizer.param_groups = [{"params": ["param0"], "lr": 0.001}, {"params": ["param1"], "lr": 0.002}]
        mock_optimizer.state = proxy_dict_state

        # Set up the trainer with the mock optimizer
        mock_trainer = MagicMock()
        mock_trainer.optimizers = [mock_optimizer]
        mock_module = MagicMock()
        tensor_stats = {"hash", "shape"}

        # Mock the process_item function to return a digest
        mock_process_item.return_value = b"mock_digest"

        # Import the function to test
        from fkat.pytorch.callbacks.debugging.introspection import _optimizers_checksums

        # Call the function
        result = _optimizers_checksums(mock_trainer, mock_module, tensor_stats)

        # Verify that process_item was called for each component of the optimizer
        mock_process_item.assert_any_call(mock_optimizer.defaults, tensor_stats, "defaults", ANY)
        mock_process_item.assert_any_call(mock_optimizer.param_groups, tensor_stats, "param_groups", ANY)
        mock_process_item.assert_any_call(mock_optimizer.state, tensor_stats, "state", ANY)

        # Verify the structure of the result
        assert len(result) == 1  # One optimizer
        assert result[0]["__type__"] == "MockOptimizer"

        # Verify that process_item was called the expected number of times
        # (once for defaults, once for param_groups, once for state)
        assert mock_process_item.call_count == 3


def test_ensure_hashable():
    """Test the _ensure_hashable function with various input types."""
    # Test with already hashable types
    assert _ensure_hashable(1) == 1
    assert _ensure_hashable("test") == "test"
    assert _ensure_hashable((1, 2, 3)) == (1, 2, 3)
    assert _ensure_hashable(None) is None

    # Test with non-hashable types
    assert _ensure_hashable([1, 2, 3]) == (1, 2, 3)
    assert _ensure_hashable({1, 2, 3}) == frozenset({1, 2, 3})

    # Test with custom non-hashable object
    class NonHashable:
        def __str__(self):
            return "non-hashable"

    non_hashable = NonHashable()
    # The current implementation returns the object itself if it's not a list or set
    assert _ensure_hashable(non_hashable) is non_hashable


class CustomDictWithNonHashableKeys:
    """A custom dictionary-like class that can have non-hashable keys in its items() method."""

    def __init__(self):
        self._items = []

    def add_item(self, key, value):
        self._items.append((key, value))

    def items(self):
        return self._items


def test_process_collection_with_nonhashable_keys():
    """Test processing a dictionary-like object with non-hashable keys."""
    # Create a custom dict with a list as a key
    test_dict = CustomDictWithNonHashableKeys()
    test_dict.add_item([1, 2, 3], "value1")
    test_dict.add_item("normal_key", "value2")

    result_dict = {}
    tensor_stats = {"hash"}
    digest = _process_collection(test_dict, tensor_stats, "test_path", result_dict)

    # Verify the result contains both keys (converted to hashable form)
    assert "test_path" in result_dict
    processed_dict = result_dict["test_path"]

    # The list key should be converted to a tuple
    assert (1, 2, 3) in processed_dict
    assert processed_dict[(1, 2, 3)] == "value1"
    assert "normal_key" in processed_dict
    assert processed_dict["normal_key"] == "value2"

    # Verify digest is not None
    assert digest is not None


def test_process_collection_with_nested_nonhashable_keys():
    """Test processing a dictionary-like object with nested non-hashable keys."""
    # Create a custom dict with nested non-hashable keys
    test_dict = CustomDictWithNonHashableKeys()
    test_dict.add_item([1, 2], "list_value")
    test_dict.add_item([1, 2, 3], "set_value")  # Changed from set to list for simplicity

    result_dict = {}
    tensor_stats = {"hash"}
    digest = _process_collection(test_dict, tensor_stats, "nested_path", result_dict)

    # Verify the result contains the processed keys
    assert "nested_path" in result_dict
    processed_dict = result_dict["nested_path"]

    # Check that keys were properly converted to hashable types
    assert (1, 2) in processed_dict
    assert (1, 2, 3) in processed_dict

    # Verify digest is not None
    assert digest is not None


def test_process_item_with_nonhashable_keys():
    """Test the process_item function with non-hashable dictionary keys."""
    # Create a complex structure with non-hashable keys using our custom dict
    complex_dict = CustomDictWithNonHashableKeys()
    complex_dict.add_item("normal_key", "normal_value")

    # Create a tensor with necessary attributes
    tensor = torch.tensor([1.0, 2.0, 3.0])

    # Create a numpy array with necessary attributes
    numpy_array = np.array([4.0, 5.0, 6.0])

    complex_dict.add_item([1, 2, 3], tensor)
    complex_dict.add_item([4, 5, 6], numpy_array)

    result_dict = {}
    tensor_stats = {"hash", "shape"}

    # Mock _format to avoid attribute errors
    with patch(f"{MODULE}._format", return_value="formatted_data"):
        digest = process_item(complex_dict, tensor_stats, "complex", result_dict)

    # Verify the result contains the processed structure
    assert "complex" in result_dict
    processed = result_dict["complex"]

    # Check that keys were properly converted and values processed
    assert "normal_key" in processed
    assert processed["normal_key"] == "normal_value"

    # The list key should be converted to a tuple
    assert (1, 2, 3) in processed
    # The tensor should be converted to a hash
    assert isinstance(processed[(1, 2, 3)], str)

    # Check the second list key
    assert (4, 5, 6) in processed
    # The numpy array should be converted to a hash
    assert isinstance(processed[(4, 5, 6)], str)

    # Verify digest is not None
    assert digest is not None


def test_process_collection_with_items_returning_nonhashable():
    """Test processing an object with .items() method that returns non-hashable keys."""

    class CustomDict:
        def __init__(self):
            self.data = {"key1": "value1", "key2": "value2"}

        def items(self):
            # Return a list as one of the keys
            return [([1, 2, 3], "list_value"), ("normal_key", "normal_value")]

    custom_dict = CustomDict()
    result_dict = {}
    tensor_stats = {"hash"}
    digest = _process_collection(custom_dict, tensor_stats, "custom_path", result_dict)

    # Verify the result contains the processed structure
    assert "custom_path" in result_dict
    processed = result_dict["custom_path"]

    # Check that the list key was properly converted to a tuple
    assert (1, 2, 3) in processed
    assert processed[(1, 2, 3)] == "list_value"
    assert "normal_key" in processed
    assert processed["normal_key"] == "normal_value"

    # Verify digest is not None
    assert digest is not None

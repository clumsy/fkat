# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from unittest.mock import call

import lightning as L

from fkat.pytorch.callbacks.profiling.flops import GPTModel, Flops
from fkat.pytorch.schedule import Every

MODULE = "fkat.pytorch.callbacks.profiling.flops"


class MockTensor:
    def __init__(self, initial_value=0):
        self._value = initial_value  # Mimic tensor storage

    def fill_(self, value):
        """Mimic PyTorch's fill_() by updating the internal value."""
        self._value = value
        return self  # Mimic in-place operation

    def item(self):
        """Mimic PyTorch's item() by returning the stored value."""
        return self._value


@pytest.mark.parametrize("depth", [None, 6])
@pytest.mark.parametrize("stage", ["train", "validation", "predict", "test"])
@patch(f"{MODULE}.Trace")
@patch(f"{MODULE}.torch")
@patch(f"{MODULE}.L.LightningModule")
@patch(f"{MODULE}.L.Trainer")
@patch(f"{MODULE}.CallbackLogger")
def test_profiles_flops(mock_cb_logger_cls, mock_trainer, mock_module, mock_torch, mock_trace, stage, depth):
    # Arrange
    mock_cb_logger = MagicMock()
    mock_cb_logger_cls.return_value = mock_cb_logger

    mock_trainer.max_steps = 100
    mock_trainer.precision = "16"
    mock_trainer.global_rank = 0
    mock_trainer.num_nodes = 1
    mock_trainer.num_devices = 8
    mock_trainer.sanity_checking = False
    mock_trainer.logger = [MagicMock()]

    mock_module.device.type = "cuda"
    mock_torch.cuda.get_device_name.return_value = "A100"
    mock_torch.tensor.side_effect = [MockTensor(), MockTensor(), MockTensor()]

    mock_ftdm_instance = MagicMock()
    mock_ftdm_instance.get_batch_flop.return_value = 50
    mock_ftdm_instance.get_tracked_operations_count.return_value = 10
    mock_ftdm_instance.get_untracked_operations_count.return_value = 20
    mock_trace.return_value = mock_ftdm_instance

    schedule = Every(n_batches=10)
    profiler = Flops(schedule=schedule, depth=depth)
    profiler.setup(mock_trainer, mock_module, stage)

    # Act Start profiling
    batch_idx = 10  # Should trigger logging
    getattr(profiler, f"on_{stage}_batch_start")(mock_trainer, mock_module, None, batch_idx)

    # Assert
    assert profiler.trace_flops_recipe is mock_ftdm_instance
    assert profiler.start_time
    mock_ftdm_instance.__enter__.assert_called_once()

    # Act Finish profiling
    getattr(profiler, f"on_{stage}_batch_end")(mock_trainer, mock_module, None, None, batch_idx)

    # Assert
    mock_trainer.strategy.reduce.assert_has_calls(
        [
            call(profiler.batch_flops, reduce_op="sum"),
            call(profiler.operations_tracked, reduce_op="sum"),
            call(profiler.operations_untracked, reduce_op="sum"),
        ]
    )
    mock_ftdm_instance.__exit__.assert_called_once_with(None, None, None)

    assert profiler.batch_flops.item() == 50
    assert profiler.operations_tracked.item() == 10
    assert profiler.operations_untracked.item() == 20

    mock_cb_logger.log_batch.assert_called_once()
    args, kwargs = mock_cb_logger.log_batch.call_args

    metrics = kwargs["metrics"]
    assert "mfu" in metrics
    assert "actual_batches_per_sec" in metrics
    assert "max_batches_per_sec" in metrics
    assert "batch_flops" in metrics
    assert "total_flops" in metrics
    assert "batch_flops_tracked_operations" in metrics
    assert "batch_flops_untracked_operations" in metrics


@pytest.mark.parametrize("depth", [None])
@pytest.mark.parametrize("stage", ["train"])
@patch(f"{MODULE}.Trace")
@patch(f"{MODULE}.torch")
@patch(f"{MODULE}.L.LightningModule")
@patch(f"{MODULE}.L.Trainer")
@patch(f"{MODULE}.CallbackLogger")
def test_profiles_flops_with_formula_based_mfu(
    mock_cb_logger_cls, mock_trainer, mock_module, mock_torch, mock_trace, stage, depth
):
    # Arrange
    mock_cb_logger = MagicMock()
    mock_cb_logger_cls.return_value = mock_cb_logger

    mock_trainer.max_steps = 100
    mock_trainer.precision = "16"
    mock_trainer.global_rank = 0
    mock_trainer.num_nodes = 1
    mock_trainer.num_devices = 8
    mock_trainer.sanity_checking = False
    mock_trainer.logger = [MagicMock()]
    mock_module.device.type = "cuda"
    mock_torch.cuda.get_device_name.return_value = "A100"
    mock_torch.tensor.side_effect = [MockTensor(), MockTensor(), MockTensor()]

    # Simulate FLOP counter mode
    mock_ftdm_instance = MagicMock()
    mock_ftdm_instance.get_batch_flop.return_value = 100
    mock_ftdm_instance.get_tracked_operations_count.return_value = 10
    mock_ftdm_instance.get_untracked_operations_count.return_value = 20
    mock_trace.return_value = mock_ftdm_instance

    # Provide a valid config to enable formula-based flop calculation
    mock_module.cfg = {
        "global_batch_size": 1,
        "encoder_seq_length": 128,
        "num_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "ffn_hidden_size": 256,
        "padded_vocab_size": 50257,
        "num_moe_layers": 0,
        "moe_ffn_hidden_size": 0,
        "num_experts": None,
        "moe_router_topk": 1,
    }

    schedule = Every(n_batches=10)
    profiler = Flops(schedule=schedule, depth=depth)
    profiler.setup(mock_trainer, mock_module, stage)

    # Act: Start profiling
    batch_idx = 10
    getattr(profiler, f"on_{stage}_batch_start")(mock_trainer, mock_module, None, batch_idx)
    assert profiler.trace_flops_recipe is mock_ftdm_instance
    assert profiler.start_time
    mock_ftdm_instance.__enter__.assert_called_once()

    getattr(profiler, f"on_{stage}_batch_end")(mock_trainer, mock_module, None, None, batch_idx)

    mock_trainer.strategy.reduce.assert_has_calls(
        [
            call(profiler.batch_flops, reduce_op="sum"),
            call(profiler.operations_tracked, reduce_op="sum"),
            call(profiler.operations_untracked, reduce_op="sum"),
        ]
    )
    mock_ftdm_instance.__exit__.assert_called_once_with(None, None, None)

    # Assert metrics logged to MLflow
    mock_cb_logger.log_batch.assert_called_once()
    _, kwargs = mock_cb_logger.log_batch.call_args
    metrics_dict = kwargs["metrics"]

    # Base MFU metrics
    assert "mfu" in metrics_dict
    assert "actual_batches_per_sec" in metrics_dict
    assert "max_batches_per_sec" in metrics_dict
    assert "batch_flops" in metrics_dict
    assert "total_flops" in metrics_dict
    assert "batch_flops_tracked_operations" in metrics_dict
    assert "batch_flops_untracked_operations" in metrics_dict
    assert "mfu_from_formula" in metrics_dict
    assert "batch_flops_from_formula" in metrics_dict
    assert isinstance(metrics_dict["mfu_from_formula"], float)
    assert isinstance(metrics_dict["batch_flops_from_formula"], int)


@pytest.fixture
def mock_pl_module():
    # Mocking a LightningModule with a cfg attribute
    cfg = {
        "global_batch_size": 4,
        "encoder_seq_length": 1024,
        "num_layers": 12,
        "kv_channels": None,  # Will be computed inside the method
        "num_attention_heads": 12,
        "num_query_groups": 6,
        "hidden_size": 768,
        "ffn_hidden_size": 3072,
        "group_query_attention": False,
        "swiglu": True,
        "padded_vocab_size": 50257,
        "num_moe_layers": 2,
        "moe_ffn_hidden_size": 8192,
        "num_experts": 8,
        "moe_router_topk": 2,
        "consider_activation_recompute": True,
    }
    return SimpleNamespace(cfg=cfg)


def test_get_batch_flop_returns_positive_integer(mock_pl_module):
    flop_recipe = GPTModel()
    flop_count = flop_recipe.get_batch_flop(mock_pl_module)

    assert isinstance(flop_count, int), "FLOP count should be an integer"
    assert flop_count > 0, "FLOP count should be a positive number"


@pytest.mark.parametrize(
    "missing_key",
    [
        "global_batch_size",
        "encoder_seq_length",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "ffn_hidden_size",
    ],
)
def test_get_batch_flop_raises_key_error_on_missing_config_key(missing_key):
    """Test that missing mandatory config keys raise KeyError with expected message."""
    cfg = {
        "global_batch_size": 4,
        "encoder_seq_length": 1024,
        "num_layers": 12,
        "kv_channels": None,
        "num_attention_heads": 12,
        "num_query_groups": 6,
        "hidden_size": 768,
        "ffn_hidden_size": 3072,
        "group_query_attention": False,
        "swiglu": True,
        "padded_vocab_size": 50257,
        "num_moe_layers": 2,
        "moe_ffn_hidden_size": 8192,
        "num_experts": 8,
        "moe_router_topk": 2,
        "consider_activation_recompute": True,
    }

    # Remove one mandatory key
    cfg.pop(missing_key)

    mock_pl_module = MagicMock(spec=L.LightningModule)
    mock_pl_module.cfg = cfg
    flop_recipe = GPTModel()

    with pytest.raises(KeyError) as exc_info:
        flop_recipe.get_batch_flop(mock_pl_module)

    assert missing_key in str(exc_info.value), f"Expected missing key '{missing_key}' in error message"

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, mock_open, patch, call

import torch

from fkat.pytorch.schedule import Every, Never
from fkat.pytorch.callbacks.cuda import MemoryObserver
from fkat.pytorch.callbacks.cuda import memory
from fkat.pytorch.callbacks.cuda.memory import _artifact_path
from fkat.pytorch.loggers import LightningLogger


class TestArtifactPath(unittest.TestCase):
    @patch("os.makedirs")
    @patch("fkat.utils.datetime")
    def test_artifact_path_creates_correct_structure(self, mock_datetime, mock_makedirs):
        # Arrange
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025-06-18_20-00-00-"
        mock_now.microsecond = 0
        mock_datetime.now.return_value = mock_now
        root_dir = "/tmp/test"
        rank = 1
        file_type = "snapshot"
        ext = "pickle"

        # Act
        base_dir, file_path = _artifact_path(root_dir, rank, file_type, ext)

        # Assert
        expected_base_dir = "/tmp/test/torch.cuda.memory"
        expected_file_path = "/tmp/test/torch.cuda.memory/rank1/snapshot/rank1_2025-06-18_20-00-00-000.pickle"

        assert base_dir == expected_base_dir
        assert file_path == expected_file_path
        mock_makedirs.assert_called_once_with("/tmp/test/torch.cuda.memory/rank1/snapshot", exist_ok=True)

    @patch("os.makedirs")
    @patch(f"{memory.__name__}.safe_timestamp")
    def test_artifact_path_different_parameters(self, mock_safe_timestamp, mock_makedirs):
        # Arrange
        mock_safe_timestamp.return_value = "2025-06-18_15-30-45-123"
        root_dir = "/var/logs"
        rank = 0
        file_type = "flamegraph"
        ext = "svg"

        # Act
        base_dir, file_path = _artifact_path(root_dir, rank, file_type, ext)

        # Assert
        expected_base_dir = "/var/logs/torch.cuda.memory"
        expected_file_path = "/var/logs/torch.cuda.memory/rank0/flamegraph/rank0_2025-06-18_15-30-45-123.svg"

        assert base_dir == expected_base_dir
        assert file_path == expected_file_path
        mock_makedirs.assert_called_once_with("/var/logs/torch.cuda.memory/rank0/flamegraph", exist_ok=True)


class TestMemoryObserver(unittest.TestCase):
    @patch("torch.cuda")
    @patch(f"{memory.__name__}._reset_recording")
    def test_constructor_default_values(self, mock_reset_recording, mock_cuda):
        # Arrange & Act
        mock_cuda.is_available.return_value = True
        callback = MemoryObserver()

        # Assert
        assert callback.flamegraph  # Default is True now
        assert not callback.reset_memory_history
        assert not callback.snapshot_pickle
        assert not callback.tensor_cycles
        assert isinstance(callback.schedule, Never)
        assert callback.oom
        assert callback.kwargs == {}
        mock_reset_recording.assert_called_once_with({})

    @patch("torch.cuda")
    @patch(f"{memory.__name__}._reset_recording")
    def test_constructor_custom_values(self, mock_reset_recording, mock_cuda):
        # Arrange & Act
        mock_cuda.is_available.return_value = True
        schedule = Every(n_batches=5)
        callback = MemoryObserver(
            flamegraph=False,
            reset_memory_history=True,
            snapshot_pickle=True,
            tensor_cycles=True,
            schedule=schedule,
            oom=False,
            enabled=True,
            context=10,
        )

        # Assert
        assert not callback.flamegraph
        assert callback.reset_memory_history
        assert callback.snapshot_pickle
        assert callback.tensor_cycles
        assert callback.schedule == schedule
        assert not callback.oom
        assert callback.kwargs == {"enabled": True, "context": 10}
        mock_reset_recording.assert_called_once_with({"enabled": True, "context": 10})

    @patch("torch.cuda")
    @patch(f"{memory.__name__}._reset_recording")
    def test_constructor_calls_reset_recording(self, mock_reset_recording, mock_cuda):
        # Arrange
        mock_cuda.is_available.return_value = True

        # Act
        MemoryObserver(enabled=True, context=5)

        # Assert
        mock_reset_recording.assert_called_once_with({"enabled": True, "context": 5})

    @patch("torch.cuda")
    @patch(f"{memory.__name__}._reset_recording")
    def test_constructor_calls_reset_recording_when_cuda_unavailable(self, mock_reset_recording, mock_cuda):
        # Arrange
        mock_cuda.is_available.return_value = False

        # Act
        MemoryObserver()

        # Assert
        mock_reset_recording.assert_called_once_with({})

    @patch(f"{memory.__name__}.CallbackLogger")
    @patch(f"{memory.__name__}._detect_tensor_cycles")
    @patch("torch._C")
    @patch("torch.cuda")
    @patch("lightning.Trainer")
    @patch(f"{memory.__name__}._reset_recording")
    def test_setup_with_tensor_cycles_enabled(
        self,
        mock_reset_recording,
        mock_trainer,
        mock_cuda,
        mock_torch_c,
        mock_detect_tensor_cycles,
        mock_callback_logger,
    ):
        # Arrange
        mock_cuda.is_available.return_value = True
        callback = MemoryObserver(tensor_cycles=True)
        mock_trainer.global_rank = 0

        # Act
        result = callback.setup(mock_trainer, MagicMock(), "fit")

        # Assert
        mock_callback_logger.assert_called_once_with(mock_trainer)
        mock_detect_tensor_cycles.assert_called_once()
        mock_torch_c._cuda_attach_out_of_memory_observer.assert_called_once()  # OOM is enabled by default
        # The setup method returns None
        assert result is None

    @patch(f"{memory.__name__}.CallbackLogger")
    @patch(f"{memory.__name__}._detect_tensor_cycles")
    @patch("torch._C")
    @patch("torch.cuda")
    @patch("lightning.Trainer")
    @patch(f"{memory.__name__}._reset_recording")
    def test_tensor_cycles_observer_with_cuda_tensors(
        self,
        mock_reset_recording,
        mock_trainer,
        mock_cuda,
        mock_torch_c,
        mock_detect_tensor_cycles,
        mock_callback_logger,
    ):
        # Arrange
        mock_cuda.is_available.return_value = True
        callback = MemoryObserver(tensor_cycles=True)
        mock_trainer.global_rank = 1

        # Act
        callback.setup(mock_trainer, MagicMock(), "fit")

        # Assert
        mock_callback_logger.assert_called_once_with(mock_trainer)
        mock_detect_tensor_cycles.assert_called_once_with(callback._cb_logger, 1)

    @patch(f"{memory.__name__}.CallbackLogger")
    @patch(f"{memory.__name__}._detect_tensor_cycles")
    @patch("torch._C")
    @patch("torch.cuda")
    @patch("lightning.Trainer")
    @patch(f"{memory.__name__}._reset_recording")
    def test_tensor_cycles_observer_without_cuda_tensors(
        self,
        mock_reset_recording,
        mock_trainer,
        mock_cuda,
        mock_torch_c,
        mock_detect_tensor_cycles,
        mock_callback_logger,
    ):
        # Arrange
        mock_cuda.is_available.return_value = True
        callback = MemoryObserver(tensor_cycles=True)
        mock_trainer.global_rank = 0

        # Act
        callback.setup(mock_trainer, MagicMock(), "fit")

        # Assert
        mock_callback_logger.assert_called_once_with(mock_trainer)
        mock_detect_tensor_cycles.assert_called_once_with(callback._cb_logger, 0)

    @patch(f"{memory.__name__}.CallbackLogger")
    @patch(f"{memory.__name__}._detect_tensor_cycles")
    @patch("torch._C")
    @patch("torch.cuda")
    @patch("lightning.Trainer")
    @patch(f"{memory.__name__}._reset_recording")
    def test_tensor_cycles_observer_with_empty_garbage(
        self,
        mock_reset_recording,
        mock_trainer,
        mock_cuda,
        mock_torch_c,
        mock_detect_tensor_cycles,
        mock_callback_logger,
    ):
        # Arrange
        mock_cuda.is_available.return_value = True
        callback = MemoryObserver(tensor_cycles=True)
        mock_trainer.global_rank = 0

        # Act
        callback.setup(mock_trainer, MagicMock(), "fit")

        # Assert
        mock_callback_logger.assert_called_once_with(mock_trainer)
        mock_detect_tensor_cycles.assert_called_once_with(callback._cb_logger, 0)

    @patch("builtins.open", new_callable=mock_open, read_data="dummy_content")
    @patch(f"{memory.__name__}.memory._snapshot")
    @patch("torch.cuda")
    @patch("os.makedirs")
    @patch("tempfile.TemporaryDirectory")
    @patch(f"{memory.__name__}._reset_recording")
    def test_dump_memory_snapshot_with_new_artifact_path_structure(
        self,
        mock_reset_recording,
        mock_temp_dir,
        mock_makedirs,
        mock_cuda,
        mock_snapshot,
        mock_open,
    ):
        # Arrange
        callback = MemoryObserver(snapshot_pickle=True)
        callback._cb_logger = MagicMock(spec=LightningLogger)
        mock_snapshot.return_value = {"test": "data"}
        mock_cuda._memory_viz.memory.return_value = "<svg>test flamegraph</svg>"

        # Mock the temporary directory context manager
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"

        # Act
        with (
            patch("pickle.dump") as mock_pickle_dump,
            patch("fkat.pytorch.callbacks.cuda.memory.datetime") as mock_datetime,
        ):
            mock_datetime.now.return_value.isoformat.return_value = "2025-06-18T20:00:00"
            callback.dump_memory_snapshot(2)

        # Assert
        mock_pickle_dump.assert_called_once()
        callback._cb_logger.log_artifact.assert_called_once_with("/tmp/test_dir/torch.cuda.memory")

        # Verify the correct directory structure was created
        expected_calls = [
            call("/tmp/test_dir/torch.cuda.memory/rank2/snapshot", exist_ok=True),
            call("/tmp/test_dir/torch.cuda.memory/rank2/flamegraph", exist_ok=True),
        ]
        mock_makedirs.assert_has_calls(expected_calls, any_order=True)
        # Verify _reset_recording was called during dump
        mock_reset_recording.assert_called()

    @patch("builtins.open", new_callable=mock_open, read_data="dummy_content")
    @patch(f"{memory.__name__}.memory._snapshot")
    @patch("torch.cuda")
    @patch("os.makedirs")
    @patch("tempfile.TemporaryDirectory")
    @patch(f"{memory.__name__}._reset_recording")
    def test_dump_memory_snapshot_without_pickle(
        self,
        mock_reset_recording,
        mock_temp_dir,
        mock_makedirs,
        mock_cuda,
        mock_snapshot,
        mock_open,
    ):
        # Arrange
        callback = MemoryObserver(snapshot_pickle=False)
        callback._cb_logger = MagicMock(spec=LightningLogger)
        mock_snapshot.return_value = {"test": "data"}
        mock_cuda._memory_viz.memory.return_value = "<svg>test flamegraph</svg>"

        # Mock the temporary directory context manager
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"

        # Act
        with patch("pickle.dump") as mock_pickle_dump:
            callback.dump_memory_snapshot(0)

        # Assert
        mock_pickle_dump.assert_not_called()
        callback._cb_logger.log_artifact.assert_called_once()
        # Only one file should be opened (for SVG)
        assert mock_open.call_count == 1

    @patch("builtins.open", new_callable=mock_open, read_data="dummy_content")
    @patch(f"{memory.__name__}.memory._snapshot")
    @patch("torch.cuda")
    @patch("os.makedirs")
    @patch("tempfile.TemporaryDirectory")
    @patch(f"{memory.__name__}._reset_recording")
    def test_dump_memory_snapshot_with_reset_memory_history(
        self,
        mock_reset_recording,
        mock_temp_dir,
        mock_makedirs,
        mock_cuda,
        mock_snapshot,
        mock_open,
    ):
        # Arrange
        callback = MemoryObserver(reset_memory_history=True, flamegraph=True)
        callback._cb_logger = MagicMock(spec=LightningLogger)
        mock_snapshot.return_value = {"test": "data"}
        mock_cuda._memory_viz.memory.return_value = "<svg>test flamegraph</svg>"

        # Mock the temporary directory context manager
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"

        # Act
        callback.dump_memory_snapshot(0)

        # Assert
        # _reset_recording should be called twice: once in constructor, once in dump_memory_snapshot
        assert mock_reset_recording.call_count == 2
        callback._cb_logger.log_artifact.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data="dummy_content")
    @patch(f"{memory.__name__}.memory._snapshot")
    @patch("torch.cuda")
    @patch("os.makedirs")
    @patch("tempfile.TemporaryDirectory")
    @patch(f"{memory.__name__}._reset_recording")
    def test_dump_memory_snapshot_without_flamegraph(
        self,
        mock_reset_recording,
        mock_temp_dir,
        mock_makedirs,
        mock_cuda,
        mock_snapshot,
        mock_open,
    ):
        # Arrange
        callback = MemoryObserver(flamegraph=False, snapshot_pickle=True)
        callback._cb_logger = MagicMock(spec=LightningLogger)
        mock_snapshot.return_value = {"test": "data"}

        # Mock the temporary directory context manager
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"

        # Act
        with patch("pickle.dump") as mock_pickle_dump:
            callback.dump_memory_snapshot(0)

        # Assert
        mock_pickle_dump.assert_called_once()
        callback._cb_logger.log_artifact.assert_called_once()
        # Only one file should be opened (for pickle)
        assert mock_open.call_count == 1

    @patch("torch.cuda")
    @patch(f"{memory.__name__}.memory")
    @patch(f"{memory.__name__}._reset_recording")
    def test_dump_memory_snapshot_missing_snapshot(
        self,
        mock_reset_recording,
        mock_memory,
        mock_cuda,
    ):
        # Arrange
        callback = MemoryObserver()
        callback._cb_logger = MagicMock()
        delattr(mock_memory, "_snapshot")
        # Act
        with patch("logging.Logger.warning") as mock_warning:
            callback.dump_memory_snapshot(0)
        # Assert
        mock_warning.assert_called_once()
        assert "Failed to capture memory snapshot" in mock_warning.call_args[0][0]

    @patch("builtins.open", new_callable=mock_open, read_data="dummy_content")
    @patch("torch.cuda")
    @patch("os.makedirs")
    @patch("tempfile.TemporaryDirectory")
    @patch(f"{memory.__name__}.memory")
    @patch(f"{memory.__name__}._reset_recording")
    def test_dump_memory_snapshot_missing_memory_viz(
        self,
        mock_reset_recording,
        mock_memory,
        mock_temp_dir,
        mock_makedirs,
        mock_cuda,
        mock_open,
    ):
        # Arrange
        callback = MemoryObserver()
        callback._cb_logger = MagicMock(spec=LightningLogger)
        mock_memory._snapshot.return_value = {"test": "data"}
        delattr(mock_cuda, "_memory_viz")

        # Mock the temporary directory context manager
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"

        # Act
        with patch("logging.Logger.warning") as mock_warning:
            callback.dump_memory_snapshot(0)

        # Assert
        mock_warning.assert_called_once()
        assert "Failed to create flamegraph" in mock_warning.call_args[0][0]
        # When _memory_viz is missing and no pickle is generated, log_artifact is not called
        callback._cb_logger.log_artifact.assert_not_called()

    @patch(f"{memory.__name__}.CallbackLogger")
    @patch("torch._C")
    @patch("torch.cuda")
    @patch("lightning.Trainer")
    @patch(f"{memory.__name__}._reset_recording")
    def test_setup_when_cuda_not_available(
        self,
        mock_reset_recording,
        mock_trainer,
        mock_cuda,
        mock_torch_c,
        mock_callback_logger,
    ):
        # Arrange
        mock_cuda.is_available.return_value = False
        callback = MemoryObserver()

        # Act
        with patch("logging.Logger.warning") as mock_warning:
            callback.setup(mock_trainer, MagicMock(), "fit")

        # Assert
        mock_warning.assert_called_once()
        assert "No CUDA device is available" in mock_warning.call_args[0][0]
        mock_torch_c._cuda_attach_out_of_memory_observer.assert_not_called()
        mock_callback_logger.assert_not_called()

    @patch(f"{memory.__name__}.CallbackLogger")
    @patch("torch._C")
    @patch("torch.cuda")
    @patch("lightning.Trainer")
    @patch(f"{memory.__name__}._reset_recording")
    def test_oom_disabled(
        self,
        mock_reset_recording,
        mock_trainer,
        mock_cuda,
        mock_torch_c,
        mock_callback_logger,
    ):
        # Arrange
        mock_cuda.is_available.return_value = True
        callback = MemoryObserver(oom=False)

        # Act
        callback.setup(mock_trainer, MagicMock(), "fit")

        # Assert
        mock_callback_logger.assert_called_once_with(mock_trainer)
        mock_torch_c._cuda_attach_out_of_memory_observer.assert_not_called()

    @patch(f"{memory.__name__}.CallbackLogger")
    @patch("torch._C")
    @patch("torch.cuda")
    @patch("lightning.Trainer")
    @patch(f"{memory.__name__}._reset_recording")
    def test_on_train_batch_start(
        self,
        mock_reset_recording,
        mock_trainer,
        mock_cuda,
        mock_torch_c,
        mock_callback_logger,
    ):
        # Arrange
        mock_cuda.is_available.return_value = True
        callback = MemoryObserver(schedule=Every(n_batches=2))
        mock_trainer.global_step = 0
        mock_trainer.global_rank = 0
        # Mock maybe_dump_memory_snapshot method
        with patch.object(callback, "maybe_dump_memory_snapshot") as mock_method:
            # Act
            callback.setup(mock_trainer, MagicMock(), "fit")
            callback.on_train_batch_start(mock_trainer, MagicMock(), None, 1)
            mock_method.assert_called_once_with(mock_trainer, stage="train", batch_idx=1)

    @patch("torch._C")
    @patch("torch.cuda")
    @patch("lightning.Trainer")
    @patch(f"{memory.__name__}._reset_recording")
    def test_maybe_dump_memory_snapshot_schedule_check_true(
        self,
        mock_reset_recording,
        mock_trainer,
        mock_cuda,
        mock_torch_c,
    ):
        # Arrange
        mock_cuda.is_available.return_value = True
        mock_torch_c._cuda_memorySnapshot.return_value = {"segments": [], "device_traces": []}
        schedule = MagicMock()
        schedule.check.return_value = True
        callback = MemoryObserver(schedule=schedule)
        callback._cb_logger = MagicMock(spec=LightningLogger)
        mock_trainer.global_step = 10
        mock_trainer.global_rank = 0

        # Act
        with patch("fkat.pytorch.callbacks.cuda.memory.memory._snapshot") as mock_snapshot:
            mock_snapshot.return_value = {"test": "data"}
            with patch.object(callback, "dump_memory_snapshot") as mock_dump:
                callback.maybe_dump_memory_snapshot(mock_trainer, stage="train", batch_idx=5)

        # Assert
        schedule.check.assert_called_once_with(stage="train", batch_idx=5, step=10, trainer=mock_trainer)
        mock_dump.assert_called_once_with(0)

    @patch("torch._C")
    @patch("torch.cuda")
    @patch("lightning.Trainer")
    @patch(f"{memory.__name__}._reset_recording")
    def test_maybe_dump_memory_snapshot_schedule_check_false(
        self,
        mock_reset_recording,
        mock_trainer,
        mock_cuda,
        mock_torch_c,
    ):
        # Arrange
        mock_cuda.is_available.return_value = True
        schedule = MagicMock()
        schedule.check.return_value = False
        callback = MemoryObserver(schedule=schedule)
        mock_trainer.global_step = 10
        mock_trainer.global_rank = 0

        # Act
        with patch.object(callback, "dump_memory_snapshot") as mock_dump:
            callback.maybe_dump_memory_snapshot(mock_trainer, stage="train", batch_idx=5)

        # Assert
        schedule.check.assert_called_once_with(stage="train", batch_idx=5, step=10, trainer=mock_trainer)
        mock_dump.assert_not_called()


class TestDetectTensorCycles(unittest.TestCase):
    @patch("torch.utils.viz._cycles.observe_garbage")
    @patch("torch.utils.viz._cycles.to_html")
    @patch("torch.utils.viz._cycles.create_graph")
    @patch("builtins.open", new_callable=mock_open)
    @patch("tempfile.TemporaryDirectory")
    @patch("torch.cuda.is_available")
    def test_detect_tensor_cycles_with_cuda_tensors(
        self,
        mock_cuda_available,
        mock_temp_dir,
        mock_open,
        mock_create_graph,
        mock_to_html,
        mock_observe_garbage,
    ):
        # Arrange
        from fkat.pytorch.callbacks.cuda.memory import _detect_tensor_cycles
        from fkat.pytorch.callbacks.loggers import CallbackLogger

        cb_logger = MagicMock(spec=CallbackLogger)
        rank = 1

        # Mock CUDA as available
        mock_cuda_available.return_value = True

        # Mock the temporary directory
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"

        # Mock the cycle detection
        mock_create_graph.return_value = "test_graph"
        mock_to_html.return_value = b"<html>test cycles</html>"

        # Act
        _detect_tensor_cycles(cb_logger, rank)

        # Get the observer function that was passed to observe_garbage
        observer_func = mock_observe_garbage.call_args[0][0]

        # Create a mock CUDA tensor that will pass the isinstance and device checks
        mock_cuda_tensor = MagicMock()
        # Make it pass isinstance(obj, torch.Tensor) check
        mock_cuda_tensor.__class__ = torch.Tensor
        # Make it have a CUDA device
        mock_cuda_tensor.device.type = "cuda"
        # Make it not be a FakeTensor
        mock_cuda_tensor.__class__.__name__ = "Tensor"

        test_garbage = [mock_cuda_tensor, "non_tensor_object"]

        with patch("logging.Logger.warning") as mock_warning:
            observer_func(test_garbage)

        # Assert
        mock_warning.assert_called_once_with("Reference cycle includes a CUDA Tensor")
        mock_create_graph.assert_called_once_with(test_garbage)
        mock_to_html.assert_called_once_with("test_graph")
        cb_logger.log_artifact.assert_called_once()

    @patch("torch.utils.viz._cycles.observe_garbage")
    def test_detect_tensor_cycles_without_cuda_tensors(
        self,
        mock_observe_garbage,
    ):
        # Arrange
        from fkat.pytorch.callbacks.cuda.memory import _detect_tensor_cycles
        from fkat.pytorch.callbacks.loggers import CallbackLogger

        cb_logger = MagicMock(spec=CallbackLogger)
        rank = 0

        # Act
        _detect_tensor_cycles(cb_logger, rank)

        # Get the observer function
        observer_func = mock_observe_garbage.call_args[0][0]

        # Create a real CPU tensor for testing
        cpu_tensor = torch.tensor([1.0, 2.0])  # This will be on CPU by default

        # Simulate garbage without CUDA tensors
        test_garbage = [cpu_tensor, "non_tensor_object"]

        with patch("logging.Logger.debug") as mock_debug:
            observer_func(test_garbage)

        # Assert
        mock_debug.assert_called_once_with("No CUDA Tensors found in garbage")
        cb_logger.log_artifact.assert_not_called()

    @patch("torch.utils.viz._cycles.observe_garbage")
    def test_detect_tensor_cycles_with_empty_garbage(
        self,
        mock_observe_garbage,
    ):
        # Arrange
        from fkat.pytorch.callbacks.cuda.memory import _detect_tensor_cycles
        from fkat.pytorch.callbacks.loggers import CallbackLogger

        cb_logger = MagicMock(spec=CallbackLogger)
        rank = 0

        # Act
        _detect_tensor_cycles(cb_logger, rank)

        # Get the observer function
        observer_func = mock_observe_garbage.call_args[0][0]

        # Simulate empty garbage
        observer_func([])

        # Assert - should return early and not call log_artifact
        cb_logger.log_artifact.assert_not_called()

    @patch("torch.utils.viz._cycles.observe_garbage")
    def test_detect_tensor_cycles_with_sharded_tensor_spec(
        self,
        mock_observe_garbage,
    ):
        """Test tensor_cycles with ShardedTensor spec (no is_cuda method)."""
        from fkat.pytorch.callbacks.cuda.memory import _detect_tensor_cycles
        from fkat.pytorch.callbacks.loggers import CallbackLogger

        cb_logger = MagicMock(spec=CallbackLogger)
        rank = 0

        try:
            from torch.distributed._shard.sharded_tensor import ShardedTensor
            from torch.distributed._shard.sharding_spec import ChunkShardingSpec

            # Create mock ShardedTensor with real ShardedTensor spec
            # This ensures it doesn't have is_cuda method that regular tensors have
            mock_sharded_tensor = MagicMock(spec=ShardedTensor)
            mock_sharded_tensor.__class__ = ShardedTensor

            # Add real sharding spec
            spec = ChunkShardingSpec(dim=0, placements=["rank:0/cpu"])
            mock_sharded_tensor._sharding_spec = spec

            # Test the tensor cycle detection
            _detect_tensor_cycles(cb_logger, rank)
            observer_func = mock_observe_garbage.call_args[0][0]

            # Test with garbage containing ShardedTensor (which has no is_cuda method)
            test_garbage = [mock_sharded_tensor, "other_object"]

            with patch("logging.Logger.debug") as mock_debug:
                observer_func(test_garbage)

            # Should handle ShardedTensor gracefully without calling is_cuda
            # Your fix should catch the exception and return False
            mock_debug.assert_called_with("No CUDA Tensors found in garbage")
            cb_logger.log_artifact.assert_not_called()

        except ImportError:
            self.skipTest("ShardedTensor not available")

    @patch("torch.utils.viz._cycles.observe_garbage")
    @patch("torch.utils.viz._cycles.to_html")
    @patch("torch.utils.viz._cycles.create_graph")
    @patch("builtins.open", new_callable=mock_open)
    @patch("tempfile.TemporaryDirectory")
    def test_detect_tensor_cycles_with_cuda_sharded_tensor_spec(
        self,
        mock_temp_dir,
        mock_open,
        mock_create_graph,
        mock_to_html,
        mock_observe_garbage,
    ):
        """Test tensor_cycles detects CUDA ShardedTensor and generates artifacts."""
        from fkat.pytorch.callbacks.cuda.memory import _detect_tensor_cycles
        from fkat.pytorch.callbacks.loggers import CallbackLogger

        cb_logger = MagicMock(spec=CallbackLogger)
        rank = 0

        # Mock the temporary directory and artifacts
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/cuda_test_dir"
        mock_create_graph.return_value = "cuda_sharded_tensor_graph"
        mock_to_html.return_value = b"<html>CUDA ShardedTensor cycles detected</html>"

        try:
            from torch.distributed._shard.sharded_tensor import ShardedTensor
            from torch.distributed._shard.sharding_spec import ChunkShardingSpec

            # Create mock CUDA ShardedTensor with real ShardedTensor spec
            mock_cuda_sharded_tensor = MagicMock(spec=ShardedTensor)
            mock_cuda_sharded_tensor.__class__ = ShardedTensor

            # Add real sharding spec for CUDA device
            spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:0"])
            mock_cuda_sharded_tensor._sharding_spec = spec

            # Mock device property to simulate CUDA ShardedTensor
            mock_cuda_sharded_tensor.device.type = "cuda"

            # Test the tensor cycle detection
            _detect_tensor_cycles(cb_logger, rank)
            observer_func = mock_observe_garbage.call_args[0][0]

            # Test with garbage containing CUDA ShardedTensor
            test_garbage = [mock_cuda_sharded_tensor, "other_object"]

            with patch("logging.Logger.warning") as mock_warning:
                observer_func(test_garbage)

            # Should detect CUDA ShardedTensor and generate artifacts
            mock_warning.assert_called_with("Reference cycle includes a CUDA Tensor")
            mock_create_graph.assert_called_once_with(test_garbage)
            mock_to_html.assert_called_once_with("cuda_sharded_tensor_graph")

            # Verify file operations
            mock_open.assert_called_once()
            file_handle = mock_open.return_value.__enter__.return_value
            file_handle.write.assert_called_once_with(b"<html>CUDA ShardedTensor cycles detected</html>")

            # Verify artifact logging
            expected_base_dir = "/tmp/cuda_test_dir/torch.cuda.memory"
            cb_logger.log_artifact.assert_called_once_with(expected_base_dir)

        except ImportError:
            self.skipTest("ShardedTensor not available")

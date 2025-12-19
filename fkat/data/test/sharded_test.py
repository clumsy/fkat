# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from lightning.pytorch.profilers import Profiler
from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch, call
from pathlib import Path

import pytest
import signal
from multiprocessing.synchronize import Event
from concurrent.futures import ThreadPoolExecutor

import torch

from fkat.data.sharded import (
    DataLoaderFactory,
    DataLoaderIterGenerator,
    DistributedDataParallelShardSampler,
    FsShardSampler,
    ShardedDataLoader,
    ShardSampler,
    ShuffledShardSampler,
    initialize,
)

MODULE = "fkat.data.sharded"


class TestInitialize(TestCase):
    def setUp(self):
        self.shutdown = MagicMock(spec=Event)
        self.seed = 42
        self.dp_rank = 2

    @patch(f"{MODULE}.signal.signal")
    @patch(f"{MODULE}.os.getpid")
    @patch(f"{MODULE}.logger")
    def test_basic_initialization(self, mock_logger, mock_getpid, mock_signal):
        # Arrange
        mock_pid = 12345
        mock_getpid.return_value = mock_pid
        expected_seed = self.seed + self.dp_rank
        # Act
        initialize(self.seed, self.dp_rank, self.shutdown)
        # Assert
        mock_signal.assert_called_once_with(signal.SIGINT, signal.SIG_IGN)
        mock_logger.debug.assert_has_calls(
            [call(f"shard worker init {mock_pid} ..."), call(f"shard worker init {mock_pid} complete")]
        )
        mock_logger.info.assert_called_once_with(f"RNG seed is set with {expected_seed}")

    @patch(f"{MODULE}.signal.signal")
    @patch(f"{MODULE}.os.getpid")
    @patch(f"{MODULE}.profile_until_exit")
    def test_initialization_with_profiler(
        self,
        mock_profile,
        mock_getpid,
        mock_signal,
    ):
        # Arrange
        mock_pid = 12345
        mock_getpid.return_value = mock_pid
        mock_profiler = MagicMock(spec=Profiler)
        # Act
        initialize(self.seed, self.dp_rank, self.shutdown, profiler=mock_profiler)
        # Assert
        mock_profile.assert_called_once_with(
            mock_profiler, action=f"ShardedDataLoader[worker_pid={mock_pid}]", filename_suffix=f"_{mock_pid}"
        )

    @patch(f"{MODULE}.signal.signal")
    def test_signal_handler_error(self, mock_signal):
        # Arrange
        mock_signal.side_effect = ValueError()
        # Act & Assert (should not raise exception)
        initialize(self.seed, self.dp_rank, self.shutdown)


class TestShardSampler(TestCase):
    def setUp(self):
        # Create a ShardSampler object for testing
        self.sampler = ShardSampler()

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.get_backend")
    @patch("torch.distributed.is_initialized")
    def test_state_dict(
        self,
        mock_is_initialized,
        mock_get_backend,
        mock_all_gather,
        mock_get_world_size,
    ):
        # Setup
        mock_is_initialized.return_value = True
        mock_get_backend.return_value = "gloo"
        mock_get_world_size.return_value = 2

        # Mock the gathered tensor
        rank_data_shard_index = torch.tensor(1, dtype=torch.int)
        all_rank_indices = [rank_data_shard_index, torch.tensor(2, dtype=torch.int)]

        # Mocking all_gather to simulate gathering from all ranks
        def mock_all_gather_side_effect(output: list[torch.Tensor], input: torch.Tensor):
            # Replace the contents of the output list with the gathered tensors
            output[:] = all_rank_indices

        mock_all_gather.side_effect = mock_all_gather_side_effect

        # Create the expected state dictionary
        expected_state_dict = {"all_rank_indices": all_rank_indices}

        # Get the actual state dictionary from the sampler
        actual_state_dict = self.sampler.state_dict()

        # Compare the expected and actual state dictionaries
        assert actual_state_dict == expected_state_dict

    @patch("torch.distributed.is_initialized")
    def test_state_dict_distributed_not_initialized(self, mock_is_initialized):
        # Setup
        mock_is_initialized.return_value = False

        # Assert that a RuntimeError is raised when calling state_dict
        with pytest.raises(RuntimeError) as context:
            self.sampler.state_dict()

        # Check if the correct error message is raised
        assert str(context.value) == "torch.distributed is not initialized."

        # Ensure that `is_initialized` was called
        mock_is_initialized.assert_called_once()

    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_backend")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.is_initialized")
    def test_load_state_dict(
        self,
        mock_is_initialized,
        mock_get_world_size,
        mock_get_backend,
        mock_get_rank,
    ):
        # Setup
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 2
        mock_get_backend.return_value = "gloo"
        mock_get_rank.return_value = 1

        # Create a sample state dictionary to load
        all_rank_indices = [torch.tensor(1), torch.tensor(5)]
        test_state_dict = {"all_rank_indices": all_rank_indices}

        # Load the sample state dictionary into the sampler
        self.sampler.load_state_dict(test_state_dict)

        # Check if the sampler's state matches the loaded state dictionary
        assert self.sampler.all_rank_indices == [1, 5]
        assert self.sampler.index == 5  # rank 1 corresponds to index 5
        assert not self.sampler.reset

    @patch("torch.distributed.is_initialized")
    def test_load_state_dict_distributed_not_initialized(self, mock_is_initialized):
        # Setup
        mock_is_initialized.return_value = False

        # Create a sample state dictionary to load
        all_rank_indices = [torch.tensor(1), torch.tensor(5)]
        test_state_dict = {"all_rank_indices": all_rank_indices}

        # Assert that a RuntimeError is raised when calling state_dict
        with pytest.raises(RuntimeError) as context:
            self.sampler.load_state_dict(test_state_dict)

        # Check if the correct error message is raised
        assert str(context.value) == "torch.distributed is not initialized."

        # Ensure that `is_initialized` was called
        mock_is_initialized.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    def test_load_state_dict_different_world_size(
        self,
        mock_get_world_size,
        mock_is_initialized,
    ):
        # Setup
        mock_get_world_size.return_value = 4
        mock_is_initialized.return_value = True

        # Create a sample state dictionary to load
        all_rank_indices = [torch.tensor(1), torch.tensor(5)]
        test_state_dict = {"all_rank_indices": all_rank_indices}

        # Assert that a ValueError is raised when calling load_state_dict
        with pytest.raises(ValueError) as context:
            self.sampler.load_state_dict(test_state_dict)

        # Check if the correct error message is raised
        expected_error_message = (
            "Inconsistent distributed training configuration: the loaded state_dict contains "
            "checkpoint data for 2 ranks, but the current world size is 4. "
            "Ensure that you are resuming from a checkpoint with the same distributed setup "
            "(number of nodes and devices)."
        )

        # Check if the correct error message is raised
        assert str(context.value) == expected_error_message

        # Ensure that `get_world_size` and `is_initialized` were called
        mock_get_world_size.assert_called_once()
        mock_is_initialized.assert_called_once()


class DistributedDataParallelShardSamplerTest(TestCase):
    def test_iter(self):
        # Arrange
        all_shards = [f"shard{i}" for i in range(7)]
        sampler = MagicMock(spec=ShardSampler)
        sampler.__iter__.side_effect = lambda: iter(all_shards)
        ddp_sampler = DistributedDataParallelShardSampler(sampler, (dp_size := 3), (dp_rank := 2))
        # Act
        shards = list(ddp_sampler)
        # Assert
        assert shards == all_shards[dp_rank * (dp_size - 1) : dp_rank * dp_size]  # truncation+offset

    def test_list_shard_read_list_iter(self):
        # Arrange
        all_shards = [[f"shard{i}"] for i in range(7)]
        sampler = MagicMock(spec=ShardSampler)
        sampler.__iter__.side_effect = lambda: iter(all_shards)
        ddp_sampler = DistributedDataParallelShardSampler(sampler, (dp_size := 3), (dp_rank := 2), num_uri_merge=3)
        # Act
        shards = list(ddp_sampler)
        # Assert
        expected = [[shard[0] for shard in all_shards[dp_rank * (dp_size - 1) : dp_rank * dp_size]]]
        assert shards == expected  # truncation+offset

    def test_list_shard_iter(self):
        # Arrange
        all_shards = [f"shard{i}" for i in range(7)]
        sampler = MagicMock(spec=ShardSampler)
        sampler.__iter__.side_effect = lambda: iter(all_shards)
        ddp_sampler = DistributedDataParallelShardSampler(sampler, (dp_size := 3), (dp_rank := 2), num_uri_merge=2)
        # Act
        shards = list(ddp_sampler)
        # Assert
        assert shards == [all_shards[dp_rank * (dp_size - 1) : dp_rank * dp_size]]  # truncation+offset

    def test_not_drop_last(self):
        # Arrange
        num_shards = 10
        all_shards = [f"shard{i}" for i in range(num_shards)]
        sampler = MagicMock(spec=ShardSampler)
        sampler.__iter__.side_effect = lambda: iter(all_shards)
        dp_size = 3
        total_shards = []
        for dp_rank in range(dp_size):
            ddp_sampler = DistributedDataParallelShardSampler(sampler, dp_size, dp_rank, drop_last=False)
            # Act
            total_shards.extend(list(ddp_sampler))

        # Assert
        assert total_shards == all_shards  # not truncate


class ShuffledShardSamplerTest(TestCase):
    @patch(f"{MODULE}.random")
    def test_iter(self, mock_random):
        # Arrange
        all_shards = [f"shard{i}" for i in range(7)]
        sampler = MagicMock(spec=ShardSampler)
        sampler.__iter__.side_effect = lambda: iter(all_shards)
        shuffled_sampler = ShuffledShardSampler(sampler)
        # Act
        shards = list(shuffled_sampler)
        # Assert
        mock_random.shuffle.assert_called_once_with(list(range(len(all_shards))))
        assert shards == all_shards


class FsShardSamplerTest(TestCase):
    @patch(f"{MODULE}.FileSelector")
    @patch(f"{MODULE}.FileSystem")
    def test_iter(self, mock_fs, mock_selector):
        # Arrange
        all_shards = [f"shard{i}" for i in range(7)]
        fs, path = MagicMock(type_name="s4"), "some/path"
        mock_fs.from_uri.return_value = fs, path
        fs.get_file_info.return_value = [
            *[MagicMock(path=f"{name}.json") for name in all_shards],
            MagicMock(path="wrong.jsonl"),
        ]
        uri, glob = "some://uri", "*.json"
        fs_sampler = FsShardSampler(uri, glob)
        # Act
        shards = list(fs_sampler)
        # Assert
        mock_fs.from_uri.assert_called_once_with(uri)
        mock_selector.assert_called_once_with(path, recursive=True)
        fs.get_file_info.assert_called_once_with(mock_selector.return_value)
        assert shards == [f"{fs.type_name}://{f.path}" for f in fs.get_file_info(path) if f.path.endswith("json")]

    @patch(f"{MODULE}.FileSelector")
    @patch(f"{MODULE}.FileSystem")
    def test_iter_merge_uri(self, mock_fs, mock_selector):
        # Arrange
        all_shards = [f"shard{i}" for i in range(7)]
        fs, path = MagicMock(type_name="s4"), "some/path"
        mock_fs.from_uri.return_value = fs, path
        fs.get_file_info.return_value = [
            *[MagicMock(path=f"{name}.json") for name in all_shards],
            MagicMock(path="wrong.jsonl"),
        ]
        uri, glob = "some://uri", "*.json"
        fs_sampler = FsShardSampler(uri, glob, num_uri_merge=2)
        # Act
        shards = list(fs_sampler)
        # Assert
        mock_fs.from_uri.assert_called_once_with(uri)
        mock_selector.assert_called_once_with(path, recursive=True)
        fs.get_file_info.assert_called_once_with(mock_selector.return_value)
        assert len(shards) == 4
        total_target_shards = [f"{fs.type_name}://{f.path}" for f in fs.get_file_info(path) if f.path.endswith("json")]
        assert shards[0] == total_target_shards[0:2]
        assert shards[1] == total_target_shards[2:4]
        assert shards[2] == total_target_shards[4:6]
        assert shards[3] == total_target_shards[6:8]


class DataLoaderFactoryTest(TestCase):
    def setUp(self):
        self.dataset = MagicMock()
        self.dataset_generator = MagicMock(return_value=self.dataset)
        self.sampler = MagicMock()
        self.sampler_generator = MagicMock(return_value=self.sampler)
        self.batch_sampler = MagicMock()
        self.batch_sampler_generator = MagicMock(return_value=self.batch_sampler)
        self.dataloader = MagicMock()
        self.dataloader_generator = MagicMock(return_value=self.dataloader)

    def test_factory(self):
        # Arrange
        factory = DataLoaderFactory(
            self.dataset_generator, self.sampler_generator, self.batch_sampler_generator, self.dataloader_generator
        )
        # Act
        res = factory(shard := "42")
        # Assert
        assert res == self.dataloader
        self.dataset_generator.assert_called_once_with(shard)
        self.sampler_generator.assert_called_once_with(self.dataset)
        self.batch_sampler_generator.assert_called_once_with(self.sampler)
        self.dataloader_generator.assert_called_once_with(
            self.dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=self.batch_sampler
        )


class DataLoaderIterGeneratorTest(TestCase):
    @patch(f"{MODULE}.shm")
    @patch(f"{MODULE}._shutdown")
    def test_iter(self, mock_shutdown, mock_shm):
        # Arrange
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        dataloader_iter = MagicMock()
        dataloader.__iter__.side_effect = lambda: dataloader_iter
        num_microbatch_prefetches = 17
        generator = DataLoaderIterGenerator(dataloader_factory, num_microbatch_prefetches)
        # Act
        path = Path("some/path")
        generator(shard := "42", path)
        # Assert
        dataloader_factory.assert_called_once_with(shard)
        mock_shm.save_iter.assert_called_once_with(
            dataloader_iter, path=path, max_items=num_microbatch_prefetches, should_stop=ANY
        )
        should_stop = mock_shm.save_iter.call_args_list[0][1]["should_stop"]
        should_stop()
        mock_shutdown.is_set.assert_called_once()


class SharedDataLoaderTest(TestCase):
    @patch(f"{MODULE}.shm")
    def test_iter_reset(self, mock_shm):
        # Arrange
        shard_sampler = MagicMock(spec=ShardSampler)
        shards = [1]
        shard_sampler.__iter__.side_effect = lambda: iter(shards)
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        microbatches = [1, 2, 3]
        dataloader.__iter__.side_effect = lambda: iter(microbatches)

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return MagicMock()

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShardedDataLoader(
                42, shard_sampler, dataloader_factory, num_shard_prefetches=0, num_microbatch_prefetches=17, dp_rank=1
            )
            # Act
            assert not hasattr(dataloader, "shard_sampler_iter")
            it = iter(dataloader)
            # Assert
            shard_sampler_iter = dataloader.shard_sampler_iter
            assert len(dataloader.data_jobs) == 0
            # Act
            next(it)
            iter(dataloader)
            # Assert
            assert len(dataloader.data_jobs) == 0
            assert shard_sampler_iter != dataloader.shard_sampler_iter  # a new one

    @patch("multiprocessing.pool.ApplyResult.successful")
    @patch("multiprocessing.pool.ApplyResult.get")
    @patch("multiprocessing.pool.ApplyResult.ready")
    @patch("fkat.utils.shm.saved")
    @patch("fkat.utils.shm.load")
    def test_wait_callback(
        self,
        shm_load,
        shm_saved,
        apply_result_ready,
        apply_result_get,
        apply_result_successful,
    ):
        apply_result_get.side_effect = [None, None, None, None]
        apply_result_ready.side_effect = [False, False, True, True]
        apply_result_successful.side_effect = [True, True, True, True]
        shard_sampler = MagicMock(spec=ShardSampler)
        shards = [1]
        shard_sampler.__iter__.side_effect = lambda: iter(shards)
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        microbatches = [1, 2, 3]
        fut = MagicMock()
        fut.result.side_effect = [None, *microbatches[:2], None, None]

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return fut

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShardedDataLoader(
                42, shard_sampler, dataloader_factory, num_shard_prefetches=0, num_microbatch_prefetches=1, dp_rank=1
            )
            # Act
            assert not hasattr(dataloader, "shard_sampler_iter")
            it = iter(dataloader)
            # Assert
            shard_sampler_iter = dataloader.shard_sampler_iter
            assert len(dataloader.data_jobs) == 0
            # Act
            assert microbatches[0] == next(it)
            assert microbatches[1] == next(it)
            iter(dataloader)
            # Assert
            assert len(dataloader.data_jobs) == 0
            assert shard_sampler_iter != dataloader.shard_sampler_iter  # a new one

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.shm")
    def test_teardown(self, mock_shm, mock_pool, mock_profile):
        # Arrange
        shard_sampler = MagicMock(spec=ShardSampler)
        shards = [1]
        shard_sampler.__iter__.side_effect = lambda: iter(shards)
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        microbatches = [1, 2, 3]
        batches_iter = iter(microbatches)
        dataloader.__iter__.side_effect = lambda: iter(microbatches)
        profiler = MagicMock(spec=Profiler)

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                fut = MagicMock()
                fut.result.return_value = next(batches_iter, None)
                return fut

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShardedDataLoader(
                (seed := 42),
                shard_sampler,
                dataloader_factory,
                num_shard_prefetches=(num_shard_prefetches := 1),
                num_microbatch_prefetches=17,
                dp_rank=(dp_rank := 1),
                profiler=profiler,
            )
            # Act
            iter(dataloader)
            dataloader.prefetch_shards(1)
            # Assert
            assert dataloader.writing_pool is not None
            mock_pool.assert_called_once_with(
                num_shard_prefetches, initializer=initialize, initargs=(seed, dp_rank, dataloader.shutdown, profiler)
            )
            # Act
            dataloader.teardown()
            # Assert
            assert len(dataloader.data_jobs) == 0
            mock_pool.return_value.close.assert_called_once_with()
            mock_pool.return_value.join.assert_called_once_with()

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.shm")
    def test_del(self, mock_shm, mock_pool, mock_profile):
        # Arrange
        shard_sampler = MagicMock(spec=ShardSampler)
        shards = [1]
        shard_sampler.__iter__.side_effect = lambda: iter(shards)
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        microbatches = [1, 2, 3]
        dataloader.__iter__.side_effect = lambda: iter(microbatches)
        profiler = MagicMock(spec=Profiler)

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return MagicMock()

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShardedDataLoader(
                (seed := 42),
                shard_sampler,
                dataloader_factory,
                num_shard_prefetches=(num_shard_prefetches := 1),
                num_microbatch_prefetches=17,
                dp_rank=(dp_rank := 1),
                profiler=profiler,
            )
            # Act
            iter(dataloader)
            dataloader.prefetch_shards(1)
            # Assert
            assert dataloader.writing_pool is not None
            mock_pool.assert_called_once_with(
                num_shard_prefetches, initializer=initialize, initargs=(seed, dp_rank, dataloader.shutdown, profiler)
            )
            # Act
            dataloader.__del__()
            # Assert
            assert len(dataloader.data_jobs) == 0
            mock_pool.return_value.close.assert_called_once_with()
            mock_pool.return_value.join.assert_called_once_with()

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.move_data_to_device")
    @patch(f"{MODULE}.shm")
    def test_on_exception(
        self,
        mock_shm,
        mock_move,
        mock_pool,
        mock_profile,
    ):
        # Arrange
        shard_sampler = MagicMock(spec=ShardSampler)
        shards = [1]
        shard_sampler.__iter__.side_effect = lambda: iter(shards)
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        microbatches = [1, 2, 3]
        dataloader.__iter__.side_effect = lambda: iter(microbatches)
        profiler = MagicMock(spec=Profiler)
        device = MagicMock(spec=torch.device)

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return MagicMock()

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShardedDataLoader(
                (seed := 42),
                shard_sampler,
                dataloader_factory,
                num_shard_prefetches=(num_shard_prefetches := 1),
                num_microbatch_prefetches=17,
                dp_rank=(dp_rank := 1),
                profiler=profiler,
                device=device,
            )
            # Act
            next(iter(dataloader))
            dataloader.prefetch_shards(1)
            # Assert
            assert all(args == call(ANY, device) for args in mock_move.call_args_list)
            assert dataloader.writing_pool is not None
            mock_pool.assert_called_once_with(
                num_shard_prefetches, initializer=initialize, initargs=(seed, dp_rank, dataloader.shutdown, profiler)
            )
            # Act
            dataloader.on_exception(AssertionError())
            # Assert
            assert len(dataloader.data_jobs) == 0
            mock_pool.return_value.close.assert_called_once_with()
            mock_pool.return_value.join.assert_called_once_with()

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.shm")
    def test_iter_async(self, mock_shm, mock_pool, mock_profile):
        # Arrange
        shard_sampler = MagicMock(spec=ShardSampler)
        shards = [1]
        shard_sampler.__iter__.side_effect = lambda: iter(shards)
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        microbatches = [1, 2, 3]
        fut = MagicMock()
        fut.result.side_effect = [None, None, *microbatches, StopIteration(), None, None]
        profiler = MagicMock(spec=Profiler)

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return fut

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShardedDataLoader(
                42,
                shard_sampler,
                dataloader_factory,
                num_shard_prefetches=2,
                num_microbatch_prefetches=17,
                dp_rank=1,
                profiler=profiler,
            )
            # Act
            it = iter(dataloader)
            # Assert
            assert it == dataloader
            assert microbatches == list(it)

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.shm")
    def test_state_dict(self, mock_shm, mock_pool, mock_profile):
        # Arrange
        shard_sampler = MagicMock(spec=ShardSampler)
        shards = [1]
        shard_sampler.__iter__.side_effect = lambda: iter(shards)
        shard_sampler.state_dict.return_value = {
            "all_rank_indices": [torch.tensor(5, dtype=torch.int), torch.tensor(3, dtype=torch.int)]
        }
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        profiler = MagicMock(spec=Profiler, filename="test")

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return MagicMock()

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShardedDataLoader(
                seed=42,
                shard_sampler=shard_sampler,
                dataloader_factory=dataloader_factory,
                num_shard_prefetches=2,
                num_microbatch_prefetches=17,
                dp_rank=1,
                profiler=profiler,
            )
            # Act
            state_dict = dataloader.state_dict()
            expected_state_dict = {
                "all_rank_indices": [torch.tensor(2, dtype=torch.int), torch.tensor(0, dtype=torch.int)]
            }
            # Assert
            assert state_dict == expected_state_dict

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.shm")
    def test_load_state_dict(self, mock_shm, mock_pool, mock_profile):
        # Arrange
        shard_sampler_state_dict = {
            "all_rank_indices": [torch.tensor(0, dtype=torch.int), torch.tensor(1, dtype=torch.int)]
        }
        shard_sampler = MagicMock()
        shard_sampler.load_state_dict.return_value = None
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader = MagicMock()
        dataloader_factory.return_value = (dataloader := MagicMock())
        profiler = MagicMock(spec=Profiler, filename="test")

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return MagicMock()

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShardedDataLoader(
                seed=42,
                shard_sampler=shard_sampler,
                dataloader_factory=dataloader_factory,
                num_shard_prefetches=0,
                num_microbatch_prefetches=17,
                dp_rank=1,
                profiler=profiler,
            )
            # Act
            state_dict = {
                "all_rank_indices": [torch.tensor(0), torch.tensor(1)],
            }
            dataloader.load_state_dict(state_dict)
            # Assert
            shard_sampler.load_state_dict.assert_called_once_with(shard_sampler_state_dict)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from lightning.pytorch.profilers import Profiler
from unittest import TestCase
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import ANY, MagicMock, patch, call
from typing import Any
from pathlib import Path

import torch
import signal
from multiprocessing import Event
from torch.utils.data import BatchSampler, Dataset, Sampler

from fkat.data.shm import DataLoaderFactory, DataLoaderIterGenerator, ShmDataLoader, initialize

MODULE = "fkat.data.shm"


class TestInitialize(TestCase):
    def setUp(self):
        self.shutdown = MagicMock(spec=Event)
        self.seed = 42
        self.dp_rank = 2

    @patch(f"{MODULE}.signal.signal")
    @patch(f"{MODULE}.os.getpid")
    @patch(f"{MODULE}.logger")
    def test_basic_initialization(self, mock_logger: Any, mock_getpid: Any, mock_signal: Any):
        # Arrange
        mock_pid = 12345
        mock_getpid.return_value = mock_pid
        expected_seed = self.seed + self.dp_rank
        # Act
        initialize(self.seed, self.dp_rank, self.shutdown)
        # Assert
        mock_signal.assert_called_once_with(signal.SIGINT, signal.SIG_IGN)
        mock_logger.debug.assert_has_calls(
            [call(f"worker init {mock_pid} ..."), call(f"worker init {mock_pid} complete")]
        )
        mock_logger.info.assert_called_once_with(f"RNG seed is set with {expected_seed}")

    @patch(f"{MODULE}.signal.signal")
    @patch(f"{MODULE}.os.getpid")
    @patch(f"{MODULE}.profile_until_exit")
    def test_initialization_with_profiler(self, mock_profile: Any, mock_getpid: Any, mock_signal: Any):
        # Arrange
        mock_pid = 12345
        mock_getpid.return_value = mock_pid
        mock_profiler = MagicMock(spec=Profiler)
        # Act
        initialize(self.seed, self.dp_rank, self.shutdown, profiler=mock_profiler)
        # Assert
        mock_profile.assert_called_once_with(
            mock_profiler, action=f"ShmDataLoader[worker_pid={mock_pid}]", filename_suffix=f"_{mock_pid}"
        )

    @patch(f"{MODULE}.signal.signal")
    def test_signal_handler_error(self, mock_signal: Any):
        # Arrange
        mock_signal.side_effect = ValueError()
        # Act & Assert (should not raise exception)
        initialize(self.seed, self.dp_rank, self.shutdown)


class DataLoaderFactoryTest(TestCase):
    def test_factory(self):
        # Arrange
        dataset_generator = MagicMock()
        dataset_generator.return_value = (dataset := MagicMock(spec=Dataset))
        sampler_generator = MagicMock()
        sampler_generator.return_value = (sampler := MagicMock(spec=Sampler))
        batch_sampler_generator = MagicMock()
        batch_sampler_generator.return_value = (batch_sampler := MagicMock(spec=BatchSampler))
        dataloader_generator = MagicMock()
        dataloader_generator.return_value = (dataloader := MagicMock())
        factory = DataLoaderFactory(dataset_generator, sampler_generator, batch_sampler_generator, dataloader_generator)
        # Act
        res = factory()
        # Assert
        assert res == dataloader
        dataset_generator.assert_called_once_with()
        sampler_generator.assert_called_once_with(dataset)
        batch_sampler_generator.assert_called_once_with(sampler)
        dataloader_generator.assert_called_once_with(
            dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=batch_sampler
        )


class DataLoaderIterGeneratorTest(TestCase):
    @patch(f"{MODULE}.shm")
    @patch(f"{MODULE}._shutdown")
    def test_iter(self, mock_shutdown: Any, mock_shm: Any):
        # Arrange
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        dataloader_iter = MagicMock()
        dataloader.__iter__.side_effect = lambda: dataloader_iter
        num_microbatch_prefetches = 17
        generator = DataLoaderIterGenerator(dataloader_factory, num_microbatch_prefetches)
        # Act
        path = Path("some/path")
        generator(path)
        # Assert
        dataloader_factory.assert_called_once()
        mock_shm.save_iter.assert_called_once_with(
            dataloader_iter, path=path, max_items=num_microbatch_prefetches, should_stop=ANY
        )
        should_stop = mock_shm.save_iter.call_args_list[0][1]["should_stop"]
        should_stop()
        mock_shutdown.is_set.assert_called_once()


class ShmDataLoaderTest(TestCase):
    @patch(f"{MODULE}.shm")
    def test_iter_reset(self, mock_shm: Any):
        # Arrange
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        microbatches = [1, 2, 3]
        dataloader.__iter__.side_effect = lambda: iter(microbatches)

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return MagicMock()

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShmDataLoader(42, dataloader_factory, num_microbatch_prefetches=17, dp_rank=1)
            # Act
            it = iter(dataloader)
            # Assert
            assert len(dataloader.data_jobs) == 0
            # Act
            next(it)
            iter(dataloader)
            # Assert
            assert len(dataloader.data_jobs) == 0

    @patch("multiprocessing.pool.ApplyResult.successful")
    @patch("multiprocessing.pool.ApplyResult.get")
    @patch("multiprocessing.pool.ApplyResult.ready")
    @patch("fkat.utils.shm.saved")
    @patch("fkat.utils.shm.load")
    def test_wait_callback(
        self,
        shm_load: Any,
        shm_saved: Any,
        apply_result_ready: Any,
        apply_result_get: Any,
        apply_result_successful: Any,
    ):
        shm_saved.side_effect = [False, True, False, False, True]
        shm_load.side_effect = [1, 2, 3, 4]
        apply_result_get.side_effect = [None, None, None, None]
        apply_result_ready.side_effect = [False, False, True, True]
        apply_result_successful.side_effect = [True, True, True, True]
        # Arrange
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        microbatches = [1, 2, 3]
        dataloader.__iter__.side_effect = lambda: iter(microbatches)

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return MagicMock()

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShmDataLoader(42, dataloader_factory, num_microbatch_prefetches=17, dp_rank=1)
            # Act
            it = iter(dataloader)
            # Assert
            assert len(dataloader.data_jobs) == 0
            # Act
            next(it)
            iter(dataloader)
            # Assert
            assert len(dataloader.data_jobs) == 0

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.shm")
    def test_teardown(self, mock_shm: Any, mock_pool: Any, _):
        # Arrange
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
            dataloader = ShmDataLoader(
                (seed := 42),
                dataloader_factory,
                num_microbatch_prefetches=17,
                dp_rank=(dp_rank := 1),
                profiler=profiler,
            )
            # Act
            iter(dataloader)
            dataloader.prefetch()
            # Assert
            assert dataloader.writing_pool is not None
            mock_pool.assert_called_once_with(
                1, initializer=initialize, initargs=(seed, dp_rank, dataloader.shutdown, profiler)
            )
            assert len(dataloader.data_jobs) == 1
            # Act
            dataloader.teardown()
            # Assert
            assert len(dataloader.data_jobs) == 0
            mock_pool.return_value.close.assert_called_once_with()

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.shm")
    def test_del(self, mock_shm: Any, mock_pool: Any, _):
        # Arrange
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
            dataloader = ShmDataLoader(
                (seed := 42),
                dataloader_factory,
                num_microbatch_prefetches=17,
                dp_rank=(dp_rank := 1),
                profiler=profiler,
            )
            # Act
            iter(dataloader)
            dataloader.prefetch()
            # Assert
            assert dataloader.writing_pool is not None
            mock_pool.assert_called_once_with(
                1, initializer=initialize, initargs=(seed, dp_rank, dataloader.shutdown, profiler)
            )
            assert len(dataloader.data_jobs) == 1
            # Act
            dataloader.__del__()
            # Assert
            assert len(dataloader.data_jobs) == 0
            mock_pool.return_value.close.assert_called_once_with()

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.move_data_to_device")
    @patch(f"{MODULE}.shm")
    def test_on_exception(self, mock_shm: Any, mock_move: Any, mock_pool: Any, _):
        # Arrange
        dataloader_factory = MagicMock(spec=DataLoaderFactory)
        dataloader_factory.return_value = (dataloader := MagicMock())
        profiler = MagicMock(spec=Profiler)
        device = MagicMock(spec=torch.device)

        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                return MagicMock()

        with patch(f"{MODULE}.ThreadPoolExecutor", MockExecutor):
            dataloader = ShmDataLoader(
                (seed := 42),
                dataloader_factory,
                num_microbatch_prefetches=17,
                dp_rank=(dp_rank := 1),
                profiler=profiler,
                device=device,
            )
            # Act
            next(iter(dataloader))
            dataloader.prefetch()
            # Assert
            assert all(args == call(ANY, device) for args in mock_move.call_args_list)
            assert dataloader.writing_pool is not None
            mock_pool.assert_called_once_with(
                1, initializer=initialize, initargs=(seed, dp_rank, dataloader.shutdown, profiler)
            )
            assert len(dataloader.data_jobs) == 1
            # Act
            dataloader.on_exception(AssertionError())
            # Assert
            assert len(dataloader.data_jobs) == 0
            mock_pool.return_value.close.assert_called_once_with()

    @patch(f"{MODULE}.profile_until_exit")
    @patch(f"{MODULE}.NoDaemonPool")
    @patch(f"{MODULE}.shm")
    def test_iter_async(self, mock_shm: Any, mock_pool: Any, _):
        # Arrange
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
            dataloader = ShmDataLoader(
                42,
                dataloader_factory,
                num_microbatch_prefetches=17,
                dp_rank=1,
                profiler=profiler,
            )
            # Act
            it = iter(dataloader)
            # Assert
            assert it == dataloader
            assert microbatches == list(it)

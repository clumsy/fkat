# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import atexit
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event
from concurrent.futures import ThreadPoolExecutor, Future

# mp.pool is not eagerly imported, needs an explicit import
import os
import random
import shutil
import signal
from collections import deque
from pathlib import Path
from typing import Any, Generic, TypeVar
from collections.abc import Callable, Iterable, Iterator

import numpy as np
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.utilities import move_data_to_device
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from fkat.utils import shm
from fkat.utils.pool import ThreadPool
from fkat.utils.profiler import profile_until_exit

logger = logging.getLogger(__name__)

_shutdown: Event | None = None

DEFAULT_SHUTDOWN_TIMEOUT = 60  # time for workers to gracefully shutdown


def initialize(
    seed: int,
    dp_rank: int,
    shutdown: Event,
    profiler: Profiler | None = None,
) -> None:
    # signal handlers are inherited, we used shudown flag to gracefully terminate child processes
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        pass  # won't work from non-MainThread
    # this allows the worker function to access `shutdown` even though it is
    # not passed as an argument to the function.
    global _shutdown
    _shutdown = shutdown
    pid = os.getpid()
    logger.debug(f"worker init {pid} ...")
    if profiler:
        action = f"ShmDataLoader[worker_pid={pid}]"
        profile_until_exit(profiler, action=action, filename_suffix=f"_{pid}")

    # Set RNG seed ensure TP rank within same DP group load and iterate
    # the same data in the same order with consistent RNG states
    rng_seed = seed + dp_rank
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    logger.info(f"RNG seed is set with {rng_seed}")
    logger.debug(f"worker init {pid} complete")


T_co = TypeVar("T_co", covariant=True)


class DataLoaderFactory(Generic[T_co]):
    """Factory class for creating DataLoaders.

    Args:
        dataset_generator (Callable): A function that generates a dataset.
        sampler_generator (Optional[Callable]): An optional function that generates a sampler for the dataset.
        batch_sampler_generator (Optional[Callable]): An optional function that generates a batch sampler.
        dataloader_generator (Optional[Callable]): An optional function that generates a DataLoader instance.
    """

    def __init__(
        self,
        dataset_generator: Callable[[], Dataset[T_co]],
        sampler_generator: Callable[[Dataset[T_co]], Sampler[T_co]] | None = None,
        batch_sampler_generator: Callable[[Sampler[Any] | Dataset[T_co]], Iterable[list[Any]]] | None = None,
        dataloader_generator: Callable[[Any], Iterable[list[T_co]]] | None = None,
    ) -> None:
        # Assert that either sampler_generator or batch_sampler_generator is provided
        assert sampler_generator or batch_sampler_generator, (
            "either sampler_generator or batch_sampler_generation must be provided"
        )

        # Initialize instance variables
        self.dataset_generator = dataset_generator
        self.sampler_generator = sampler_generator
        self.batch_sampler_generator = batch_sampler_generator
        self.dataloader_generator = dataloader_generator or DataLoader

    def __call__(self) -> Iterable[list[T_co]]:
        """Generates a DataLoader.

        Returns:
            Iterable[List[T_co]]: An iterable of batches of data.
        """
        # Generate dataset using dataset_generator
        dataset = self.dataset_generator()

        # Generate sampler if sampler_generator is provided
        sampler = self.sampler_generator(dataset) if self.sampler_generator else None

        # Generate batch sampler if batch_sampler_generator is provided
        if self.batch_sampler_generator:
            batch_sampler = self.batch_sampler_generator(sampler if sampler else dataset)
            sampler = None  # mutually exclusive

        # Generate DataLoader instance using dataloader_generator
        dataloader = self.dataloader_generator(  # type: ignore[call-arg]
            dataset, batch_size=1, shuffle=None, sampler=sampler, batch_sampler=batch_sampler
        )
        return dataloader


class DataLoaderIterGenerator(Generic[T_co]):
    """Generates and saves an iterator over DataLoaders.

    Args:
        dataloader_factory (DataLoaderFactory): An instance of DataLoaderFactory responsible for generating DataLoaders.
        num_microbatches_prefetch (int, optional): The number of microbatches to prefetch. Defaults to -1.
    """

    def __init__(
        self,
        dataloader_factory: DataLoaderFactory[T_co],
        num_microbatch_prefetches: int = -1,
    ) -> None:
        """Initializes the DataLoaderIterGenerator.

        Args:
            dataloader_factory (DataLoaderFactory): DataLoaders provider.
            num_microbatches_prefetch (int, optional): The number of microbatches to prefetch.
                Defaults to -1.
        """
        self.dataloader_factory = dataloader_factory
        self.num_microbatch_prefetches = num_microbatch_prefetches

    def __call__(self, path: Path) -> None:
        """Generates and saves an iterator over DataLoaders.

        Args:
            path (Path): The path where the iterator will be saved.
        """
        # Log debug message indicating the start of the process
        logger.debug("generate ...")

        # Generate a DataLoader
        dataloader = self.dataloader_factory()

        # Create an iterator over the DataLoader
        dataloader_iter = iter(dataloader)

        # Access global variable _shutdown
        global _shutdown

        # Save the iterator using shm.save_iter
        shm.save_iter(
            dataloader_iter,
            path=path,
            max_items=self.num_microbatch_prefetches,
            should_stop=lambda: _shutdown is not None and _shutdown.is_set(),
        )

        # Log debug message indicating the completion of the process
        logger.debug("generate complete")


# Sub-class multiprocessing.Process to make sure it's not started in daemon mode by the Pool
class NoDaemonProcess(mp.Process):
    @property
    def daemon(self) -> bool:
        return False

    @daemon.setter
    def daemon(self, value: bool) -> None:
        pass


class NoDaemonContext(type(mp.get_context())):  # type: ignore[misc]
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonPool(mp.pool.Pool):  # type: ignore[unresolved-attribute]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["context"] = NoDaemonContext()
        super().__init__(*args, **kwargs)


class ShmDataLoader(Iterable[list[T_co]]):
    """A :class:`DataLoader` that uses shared memory to efficiently manage and prefetch data batches.

    Enables double-buffered micro-batch processing and fetching that overlaps with model
    forward/backward passes, minimizing dataloading overhead.

    Args:
        seed (int): Random seed for reproducibility. Use ${seed} at top level in config.yaml.
        dataloader_factory (DataLoaderFactory[T_co]): Factory for creating DataLoaders.
        num_microbatch_prefetches (int, optional): Number of microbatches to prefetch.
            Defaults to -1.
        dp_rank (int, optional): Rank of the current process. Defaults to 0.
        profiler (Optional[Profiler], optional): Profiler for profiling.
            Defaults to None.
        device (Optional[torch.device]): device to move the microbatches to in the background
        multiprocessing (Optional[True]): whether to instantiate DataLoader in a separate process.
            Defaults to True to relieve pressure from the training process, use False to debug and profile
    """

    def __init__(
        self,
        seed: int,
        dataloader_factory: DataLoaderFactory[T_co],
        num_microbatch_prefetches: int = -1,
        dp_rank: int = 0,
        profiler: Profiler | None = None,
        device: torch.device | None = None,
        multiprocessing: bool = True,
    ) -> None:
        self.microbatches: Iterator[list[T_co]] | None = None
        self.path: Path | None = None
        self.device: torch.device | None = device
        self.cleanup: set[Path] = set()
        self.shutdown = mp.Event()
        self.dataloader_factory = dataloader_factory
        self.dataloader_iter_generator = DataLoaderIterGenerator(
            dataloader_factory,
            num_microbatch_prefetches,
        )
        self.data_jobs: deque[tuple[Path, mp.pool.AsyncResult[Any]]] = deque()  # type: ignore[unresolved-attribute]

        # Initialize a new ProcessPoolExecutor instance for prefetching if necessary
        signal.signal(signal.SIGTERM, self.teardown)  # terminate signal
        signal.signal(signal.SIGINT, self.teardown)  # keyboard interrupt
        atexit.register(self.teardown)
        self.writing_pool: NoDaemonPool | ThreadPool = (
            NoDaemonPool(1, initializer=initialize, initargs=(seed, dp_rank, self.shutdown, profiler))
            if multiprocessing
            else ThreadPool(
                max_workers=1,
                thread_name_prefix="ShmDataWriter",
                initializer=initialize,
                initargs=(seed, dp_rank, self.shutdown, profiler),
            )
        )
        self.reading_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ShmDataReader")
        self.next_batch: Future[Any] | None = None

    def __iter__(self) -> Iterator[list[T_co]]:
        # all reading needs to go through the same TPE to avoid contention
        if not self.reading_pool._shutdown:
            self.reading_pool.submit(self._cleanup).result()
        return self

    def __next__(self) -> list[T_co]:
        if not self.next_batch:
            self.load_batch()
        assert self.next_batch
        batch = self.next_batch.result()
        self.load_batch()  # double-buffering
        return batch

    def load_batch(self) -> None:
        if not self.reading_pool._shutdown:
            self.next_batch = self.reading_pool.submit(self.load_batch_sync)

    def load_batch_sync(self) -> list[T_co]:
        while True:
            if self.microbatches:
                # Fetch the next microbatch if available
                try:
                    microbatch = next(self.microbatches)
                    if self.device:
                        microbatch = move_data_to_device(microbatch, self.device)
                    return microbatch
                # If no microbatches are available, which means all microbatches from current until exhausted
                except StopIteration:
                    if self.path:
                        self.cleanup.remove(self.path)
                        self.path = None
                    self.microbatches = None
                    raise

            if len(self.data_jobs) == 0:
                logger.debug("load iter scheduling ...")
                self.prefetch()
            path, data_job = self.data_jobs.popleft()
            logger.debug(f"load iter to {path} ...")

            def wait_callback() -> None:
                if not data_job.ready():  # noqa: B023
                    # Job is still running
                    return None
                else:
                    # Job is finished, raise exception if job failed.
                    data_job.get()  # noqa: B023
                    # Return whether the call completed without raising an exception.
                    assert data_job.successful()  # noqa: B023

            self.microbatches = shm.load_iter(path, wait_callback=wait_callback)
            self.cleanup.add(path)
            self.path = path

    def set_device(self, device: torch.device | None) -> None:
        self.device = device

    def prefetch(self) -> None:
        path = shm.generate_path()
        data_job = self.writing_pool.apply_async(self.dataloader_iter_generator, (path,))
        self.data_jobs.append((path, data_job))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"queued {path}, {len(self.data_jobs)}")

    def _cleanup(self, stop_pool: bool = False) -> None:
        self.microbatches = None
        if self.next_batch:
            try:
                # if called on teardown/on_exception/__del__ will wait for the pending work to finish
                # if called on __init__ it's already done
                self.next_batch.result()
            except Exception:
                pass
            self.next_batch = None
        self.shutdown.set()  # signal running tasks to stop
        if stop_pool:
            self.writing_pool.close()  # no new tasks can run
            self.reading_pool.shutdown()
        for path, result in self.data_jobs:
            logger.debug(f"waiting for {path} to stop ...")
            self.cleanup.add(path)
            try:
                result.wait(timeout=DEFAULT_SHUTDOWN_TIMEOUT)
            except Exception:
                pass
            logger.debug(f"{path} stopped ...")
        self.data_jobs.clear()
        self.shutdown.clear()
        for path in self.cleanup:
            logger.debug(f"removing {path} ...")
            shutil.rmtree(path, ignore_errors=True)

    # called when fit/validate/predict/test is complete
    def teardown(self, *args: Any) -> None:
        logger.debug("teardown ...")
        # all reading needs to go through the same TPE to avoid contention
        self._cleanup(stop_pool=True)
        logger.debug("teardown complete")

    # will be used once https://github.com/Lightning-AI/pytorch-lightning/pull/19601 is in effect
    # once the below callback is operational we no longer need __del__ override
    def on_exception(self, exception: BaseException) -> None:
        self.teardown()

    # called when the iterable link pointing to this object goes out of scope
    # e.g. when exception happens
    def __del__(self) -> None:
        self.teardown()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import atexit
import fnmatch
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event
from concurrent.futures import ThreadPoolExecutor, Future

import os
import random
import shutil
import signal
from pathlib import Path
from typing import Any, Generic, TypeVar
from collections import deque
from collections.abc import Callable, Iterable, Iterator

import numpy as np
from pyarrow.fs import FileSelector, FileSystem
from lightning.pytorch.core.hooks import CheckpointHooks
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.utilities import move_data_to_device
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from typing_extensions import override

from fkat.data import PersistStates, RestoreStates
from fkat.utils import shm
from fkat.utils.pool import ThreadPool, NoDaemonPool
from fkat.utils.profiler import profile_until_exit

logger = logging.getLogger(__name__)

_shutdown: Event | None = None

DEFAULT_SHUTDOWN_TIMEOUT = 60  # time for shard workers to gracefully shutdown


def initialize(seed: int, dp_rank: int, shutdown: Event, profiler: Profiler | None = None) -> None:
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
    logger.debug(f"shard worker init {pid} ...")
    if profiler:
        action = f"ShardedDataLoader[worker_pid={pid}]"
        profile_until_exit(profiler, action=action, filename_suffix=f"_{pid}")

    # Set RNG seed ensure TP rank within same DP group load and iterate same data
    # in same order with consistent RNG states
    rng_seed = seed + dp_rank
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    logger.info(f"RNG seed is set with {rng_seed}")
    logger.debug(f"shard worker init {pid} complete")


T_co = TypeVar("T_co", covariant=True)
Shard = str | list[str]


class ShardSampler(Iterable[Shard], PersistStates, RestoreStates):
    def __init__(self) -> None:
        self.shards: list[Shard] = []
        self.index: int = -1
        self.all_rank_indices: list[int] = []

    def __iter__(self) -> Iterator[Shard]:
        # Iterate over shards and yield them one by one
        yield from self.shards

    def state_dict(self) -> dict[str, Any]:
        """Converts the current state to a dictionary saving the sampler states.

        Returns:
            Dict[str, Any]: dict object representing the state.
        """
        # Raise error if torch.distributed is not initialized
        if not dist.is_initialized():  # type: ignore[possibly-unbound-attribute]
            raise RuntimeError("torch.distributed is not initialized.")

        # Get local rank and world size
        world_size = dist.get_world_size()  # type: ignore[possibly-unbound-attribute]

        # Get device
        device = "cpu" if dist.get_backend() == "gloo" else "cuda"  # type: ignore[possibly-unbound-attribute]

        # Create a torch tensor with index defined
        local_index = torch.tensor(self.index, dtype=torch.int, device=device)

        # Prepare a list of tensors to hold the indices from all ranks
        # i.e. rank 0 to rank 3 would have access to
        # [torch.tensor(1), torch.tensor(1), torch.tensor(5), torch.tensor(6)]
        all_rank_indices = [torch.zeros_like(local_index) for _ in range(world_size)]

        # Gather the indices from all ranks so all ranks have access to the same list of indices
        dist.all_gather(all_rank_indices, local_index)  # type: ignore[possibly-unbound-attribute]

        # Return all rank indices
        sampler_states = {"all_rank_indices": all_rank_indices}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DataModule sampler states are {sampler_states}")

        return sampler_states

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the sampler state dict with serialized state_dict

        Args:
            state_dict (Dict[str, Any]): serialized sampler states
        """
        # Raise error if torch.distributed is not initialized
        if not dist.is_initialized():  # type: ignore[possibly-unbound-attribute]
            raise RuntimeError("torch.distributed is not initialized.")

        # Get the all_rank_indices
        all_rank_indices = state_dict["all_rank_indices"]

        # Convert list of tensor indices to list of integer indices
        # if all_rank_indices = [torch.tensor(1), torch.tensor(1), torch.tensor(5), torch.tensor(6)]
        # then self.all_rank_indices = [1, 1, 5, 6]
        self.all_rank_indices = [tensor_index.item() for tensor_index in all_rank_indices]
        logger.info(f"All rank indices are {self.all_rank_indices}")

        # Check if the number of ranks in the state_dict matches the current distributed setup.
        # This ensures consistency when resuming training, preventing issues from mismatched
        # configurations (e.g., different number of nodes or devices).
        world_size = dist.get_world_size()  # type: ignore[possibly-unbound-attribute]
        num_saved_ranks = len(self.all_rank_indices)
        if num_saved_ranks != world_size:
            raise ValueError(
                f"Inconsistent distributed training configuration: the loaded state_dict contains "
                f"checkpoint data for {num_saved_ranks} ranks, but the current world size is {world_size}. "
                "Ensure that you are resuming from a checkpoint with the same distributed setup "
                "(number of nodes and devices)."
            )

        # Get the local rank and set the corresponding index from saved state_dict
        # Each rank loads its corresponding saved index from self.all_rank_indices
        local_rank = dist.get_rank()  # type: ignore[possibly-unbound-attribute]
        self.index = self.all_rank_indices[local_rank]
        logger.info(f"Rank {local_rank} has {self.index}")

        # Set self.reset as False so it doesn't reset the index
        self.reset = False


class DataLoaderFactory(Generic[T_co]):
    """Factory class for creating DataLoaders.

    Args:
        dataset_generator (Callable): A function that generates a dataset given a shard.
        batch_size (Optional[int]): The batch size.
        sampler_generator (Optional[Callable]): An optional function that generates a sampler for the dataset.
        batch_sampler_generator (Optional[Callable]): An optional function that generates a batch sampler.
        dataloader_generator (Optional[Callable]): An optional function that generates a DataLoader instance.
    """

    def __init__(
        self,
        dataset_generator: Callable[[Shard], Dataset[T_co]],
        sampler_generator: Callable[[Dataset[T_co]], Sampler[T_co]] | None = None,
        batch_sampler_generator: Callable[[Sampler[Any] | Dataset[T_co]], Iterable[list[Any]]] | None = None,
        dataloader_generator: Callable[[Any], Iterable[list[T_co]]] | None = None,
        batch_size: int = 1,
    ) -> None:
        assert batch_size or sampler_generator or batch_sampler_generator, (
            "either batch_size, sampler_generator or batch_sampler_generation must be provided"
        )
        self.dataset_generator = dataset_generator
        self.sampler_generator = sampler_generator
        self.batch_sampler_generator = batch_sampler_generator
        self.dataloader_generator = dataloader_generator or DataLoader
        self.batch_size = batch_size

    def __call__(self, shard: Shard) -> Iterable[list[T_co]]:
        """Generates a DataLoader for the given shard.

        Args:
            shard (Shard): Represents a subset of the dataset.

        Returns:
            Iterable[List[T_co]]: An iterable of batches of data.
        """
        # Generate dataset using dataset_generator
        dataset = self.dataset_generator(shard)

        # Generate sampler if sampler_generator is provided
        sampler = self.sampler_generator(dataset) if self.sampler_generator else None

        # Generate batch sampler if batch_sampler_generator is provided
        if self.batch_sampler_generator:
            batch_sampler = self.batch_sampler_generator(sampler if sampler else dataset)
            sampler = None  # mutually exclusive
        else:
            batch_sampler = None

        # Generate DataLoader instance using dataloader_generator
        dataloader = self.dataloader_generator(  # type: ignore[call-arg]
            dataset, batch_size=self.batch_size, shuffle=None, sampler=sampler, batch_sampler=batch_sampler
        )
        return dataloader


class DataLoaderIterGenerator(Generic[T_co]):
    """Generates and saves an iterator over DataLoaders.

    Args:
        dataloader_factory (DataLoaderFactory): An instance of DataLoaderFactory responsible for generating DataLoaders.
        num_microbatches_prefetch (int, optional): The number of microbatches to prefetch.
            Defaults to -1.
    """

    def __init__(
        self,
        dataloader_factory: DataLoaderFactory[T_co],
        num_microbatch_prefetches: int = -1,
    ) -> None:
        """Initializes the DataLoaderIterGenerator.

        Args:
            dataloader_factory (DataLoaderFactory): An instance of DataLoaderFactory responsible
                for generating DataLoaders.
            num_microbatches_prefetch (int, optional): The number of microbatches to prefetch.
                Defaults to -1.
        """
        self.dataloader_factory = dataloader_factory
        self.num_microbatch_prefetches = num_microbatch_prefetches

    def __call__(self, shard: Shard, path: Path) -> None:
        """Generates and saves an iterator over DataLoaders.

        Args:
            shard (Shard): A subset of the dataset.
            path (Path): The path where the iterator will be saved.
        """
        # Log debug message indicating the start of the process
        logger.debug("shard generate ...")

        # Generate a DataLoader using the provided shard
        dataloader = self.dataloader_factory(shard)

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
        logger.debug("shard generate complete")


class DistributedDataParallelShardSampler(ShardSampler, CheckpointHooks):
    """Distributed Data Parallel Shard Sampler.

    This sampler distributes shards evenly among processes in a distributed training setup.

    Args:
        sampler (ShardSampler): An instance of ShardSampler containing shards.
        dp_size (int): Total number of processes in the distributed training setup.
        dp_rank (int): Rank of the current process among the distributed processes.
        state_dict (dict, optional): A dictionary object serialized from a Sampler object. If provided,
            the sampler will be reconstructed using the dictionary state object and recovered to the previous state.
        drop_last (bool, optional): Whether to drop last shards if number of shards can't be divided by dp_size.
            Recommend to set this as False for evaluation and prediction tasks.
            Default to ``True``.
        num_uri_merge (int, optional): merge how many uri into a shard,
            default to ``0``; if setting ``-1``, then all uri will be merged into a shard.
    """

    def __init__(
        self,
        sampler: ShardSampler,
        dp_size: int,
        dp_rank: int,
        drop_last: bool = True,
        num_uri_merge: int = 0,
    ) -> None:
        super().__init__()
        # Convert sampler to list and determine the total number of shards
        shards: list[str | list[str]] = list(sampler)
        num_shards = len(shards)

        # Ensure that the number of shards is compatible with dp_size
        if num_shards < dp_size:
            raise ValueError(
                f"Only datasets with num_shards >= dp_size are supported, "
                f"got num_shards={num_shards}, dp_size={dp_size}"
            )

        # Distribute shards evenly among each DP group
        dp_shards, rem = divmod(num_shards, dp_size)
        if rem > 0 and drop_last:
            logger.warning(f"Truncating not even distribution of {num_shards} shards across dp_size={dp_size}")
            shards = shards[:-rem]
        else:
            dp_shards, _ = divmod(num_shards + dp_size - 1, dp_size)
        shards = shards[dp_rank * dp_shards : (dp_rank + 1) * dp_shards]  # offset

        if num_uri_merge != 0:
            merged_shards: list[list[str]] = []
            sub_shards: list[str] = []
            for i in range(len(shards)):
                if isinstance(shards[i], str):
                    sub_shards.append(str(shards[i]))
                else:
                    sub_shards.extend(shards[i])
                if len(sub_shards) == num_uri_merge:
                    merged_shards.append(sub_shards.copy())
                    sub_shards.clear()
            if sub_shards:
                merged_shards.append(sub_shards)
            self.shards = merged_shards  # type: ignore[assignment]
        else:
            self.shards = shards

        # Ensure that assigned shards are not empty
        assert self.shards

        # Initialize vars
        self.reset = True
        self.index = -1

    @override
    def __iter__(self) -> Iterator[Shard]:
        """Returns an iterator over the shards.

        Returns:
            Iterator[Shard]: Iterator over the shards.
        """
        # Reset iterator if reset flag is set
        if self.reset:
            self.index = -1
        self.reset = True
        return self

    def __next__(self) -> Shard:
        """Returns the next shard.

        Returns:
            Shard: Next shard.
        """
        # Increment index and return the corresponding shard
        self.index += 1
        if self.index >= len(self.shards):
            raise StopIteration
        return self.shards[self.index]


class ShuffledShardSampler(ShardSampler, CheckpointHooks):
    """Sampler for shuffling shards.

    Args:
        sampler (ShardSampler): An instance of ShardSampler containing shards.
        state_dict (Dict[str, Any], optional): A dictionary object containing the state of the sampler.
            Defaults to None.
    """

    def __init__(self, sampler: ShardSampler) -> None:
        # Convert sampler to list and assert non-emptiness
        self.shards = list(sampler)
        assert self.shards

        # Initialize variables
        self.reset = True
        self.index = -1

        # Create indices for shuffling
        self.indices = list(range(len(self.shards)))

    @override
    def __iter__(self) -> Iterator[Shard]:
        """Returns an iterator over the shards.

        Returns:
            Iterator[Shard]: Iterator over the shards.
        """
        # If reset flag is set, shuffle indices
        if self.reset:
            self.index = -1
            random.shuffle(self.indices)
        self.reset = True
        return self

    def __next__(self) -> Shard:
        """Returns the next shard.

        Returns:
            Shard: Next shard.
        """
        # Increment index and return the corresponding shard
        self.index += 1
        if self.index >= len(self.indices):
            raise StopIteration
        return self.shards[self.indices[self.index]]


class FsShardSampler(ShardSampler, CheckpointHooks):
    """Sampler for shuffling shards based on file system paths.

    Args:
        uri (str): The URI specifying the file system path.
        glob (Optional[str], optional): A glob pattern to filter files.
            Defaults to None.
        recursive (Optional[bool], optional): Whether to recursively search for files.
            Defaults to True.
        state_dict (Optional[Dict[str, Any]], optional): A dictionary object containing the state of the sampler.
            Defaults to None.
        num_uri_merge (int, optional): merge how many uri into a shard,
            default to ``0``; if setting ``-1``, then all uri will be merged into a shard.
    """

    def __init__(self, uri: str, glob: str | None = None, recursive: bool = True, num_uri_merge: int = 0) -> None:
        # from_uri is a static method, but pyarrow-stubs says it's an instance one
        # Extract file system and path from URI
        fs: FileSystem
        path: str
        fs, path = FileSystem.from_uri(uri)

        # Define a selector for files in the specified path
        selector = FileSelector(path, recursive=recursive)

        # Initialize shards list
        self.shards = []
        # Populate shards list with files matching the glob pattern (if provided)
        for file in fs.get_file_info(selector):
            path = f"{fs.type_name}://{file.path}"
            if not glob or fnmatch.fnmatch(path, glob):
                self.shards.append(path)
        if num_uri_merge != 0:
            merged_shards: list[list[str]] = []
            shards: list[str] = []
            for i in range(len(self.shards)):
                shards.append(str(self.shards[i]))
                if len(shards) == num_uri_merge:
                    merged_shards.append(shards.copy())
                    shards.clear()
            if shards:
                merged_shards.append(shards)
            self.shards = merged_shards  # type: ignore[assignment]
        # Ensure that shards list is not empty
        assert self.shards

        # Initialize reset and index vars
        self.reset = True
        self.index = -1

    @override
    def __iter__(self) -> Iterator[Shard]:
        """Returns an iterator over the shards.

        Returns:
            Iterator[Shard]: Iterator over the shards.
        """
        # Reset iterator if reset flag is set
        if self.reset:
            self.index = -1
        self.reset = True
        return self

    def __next__(self) -> Shard:
        """Returns the next shard.

        Returns:
            Shard: Next shard.
        """
        # Increment index and return the corresponding shard
        self.index += 1
        if self.index >= len(self.shards):
            raise StopIteration
        return self.shards[self.index]


class ShardedDataLoader(Iterable[list[T_co]]):
    """A :class:`DataLoader` that processes data in shards, designed for distributed training scenarios.

    Enables double-buffered micro-batch processing and fetching that overlaps with model
    forward/backward passes, minimizing dataloading overhead.

    Args:
        seed (int): Random seed for reproducibility. Use ${seed} at top level in config.yaml.
        shard_sampler (ShardSampler): Sampler for generating shards.
        dataloader_factory (DataLoaderFactory[T_co]): Factory for creating DataLoaders.
        num_shard_prefetches (int, optional): Number of shards to prefetch.
            Defaults to 0.
        num_microbatch_prefetches (int, optional): Number of microbatches to prefetch.
            Defaults to -1.
        dp_rank (int, optional): Rank of the current process.
            Defaults to 0.
        profiler (Profiler, optional): Profiler for profiling.
            Defaults to None.
        device (Optional[torch.device]): device to move the microbatches to in the background
        multiprocessing (Optional[True]): whether to instantiate DataLoader in a separate process.
            Defaults to True to relieve pressure from the training process, use False to debug and profile
    """

    def __init__(
        self,
        seed: int,
        shard_sampler: ShardSampler,
        dataloader_factory: DataLoaderFactory[T_co],
        num_shard_prefetches: int = 0,
        num_microbatch_prefetches: int = -1,
        dp_rank: int = 0,
        profiler: Profiler | None = None,
        device: torch.device | None = None,
        multiprocessing: bool = True,
    ) -> None:
        # Initialize
        self.microbatches: Iterator[list[T_co]] | None = None
        self.path: Path | None = None
        self.device: torch.device | None = device
        self.cleanup: set[Path] = set()
        self.shutdown = mp.Event()
        self.shard_sampler = shard_sampler
        self.shard_sampler_iter: Iterator[Shard]
        self.dataloader_factory = dataloader_factory
        self.dataloader_iter_generator = DataLoaderIterGenerator(
            dataloader_factory,
            num_microbatch_prefetches,
        )
        self.data_jobs: deque[tuple[Shard, Path, mp.pool.AsyncResult[Any]]] = deque()  # type: ignore[unresolved-attribute]

        # Initialize a new ProcessPoolExecutor instance for prefetching shards if necessary
        signal.signal(signal.SIGTERM, self.teardown)  # terminate signal
        signal.signal(signal.SIGINT, self.teardown)  # keyboard interrupt
        atexit.register(self.teardown)
        self.writing_pool: NoDaemonPool | ThreadPool = (
            NoDaemonPool(
                max(1, num_shard_prefetches),
                initializer=initialize,
                initargs=(seed, dp_rank, self.shutdown, profiler),
            )
            if multiprocessing
            else ThreadPool(
                max_workers=1,
                thread_name_prefix="ShardedDataWriter",
                initializer=initialize,
                initargs=(seed, dp_rank, self.shutdown, profiler),
            )
        )
        self.num_shard_prefetches = num_shard_prefetches
        self.reading_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ShardedDataReader")
        self.next_batch: Future[Any] | None = None

    def __iter__(self) -> Iterator[list[T_co]]:
        # all reading needs to go through the same TPE to avoid contention
        if not self.reading_pool._shutdown:
            self.reading_pool.submit(self._cleanup).result()
        self.shard_sampler_iter = iter(self.shard_sampler)
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
                    # Move to target device in advance
                    if self.device:
                        microbatch = move_data_to_device(microbatch, self.device)
                    return microbatch
                # If no microbatches are available, which means all microbatches from current shard are exhausted
                except StopIteration:
                    if self.path:
                        self.cleanup.remove(self.path)
                    self.microbatches = None
                    self.path = None

            if len(self.data_jobs) == 0:
                logger.debug("load iter scheduling ...")
                self.prefetch_shards(max(1, self.num_shard_prefetches))
                if len(self.data_jobs) == 0:
                    raise StopIteration
            shard, path, data_job = self.data_jobs.popleft()
            logger.debug(f"load iter {shard} to {path} ...")
            self.prefetch_shards(min(1, self.num_shard_prefetches))  # prefetch next shard in parallel

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

    def state_dict(self) -> dict[str, Any]:
        """
        Returns the shard sampler state dict with adjusted shard indices,
        accounting for shard prefetches and prefetch backfill in parallel.

        Example:
        If num_shard_prefetches is 3 and the original state dict is
        {"all_rank_indices": [torch.tensor(4), torch.tensor(5)]},
        it will be updated to {"all_rank_indices": [torch.tensor(0), torch.tensor(1)]}.
        This ensures that each rank resumes training from the correct shard index,
        preventing reprocessing of shards that have already been trained on.
        """
        # Invoke the shard_sampler's state_dict method when saving the data shard indices for each rank
        shard_sampler_state_dict = self.shard_sampler.state_dict()

        # Define the minimum prefetched shard backfill
        min_prefetch_shard_backfill = 1

        # Adjust each rank's shard index in place
        for _, rank_indices in shard_sampler_state_dict.items():
            for i, idx_tensor in enumerate(rank_indices):
                rank_indices[i] = idx_tensor - self.num_shard_prefetches - min_prefetch_shard_backfill

        return shard_sampler_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Loads the state dict into the shard sampler, restoring the data shard indices for each rank.

        The state dict should look like {"all_rank_indices":
        [torch.tensor(1), torch.tensor(1), torch.tensor(5), torch.tensor(6)]},
        where each tensor corresponds to the indices of data shards for specific ranks.
        """
        # Restore the shard sampler's state from the given state_dict
        self.shard_sampler.load_state_dict(state_dict)

    def prefetch_shards(self, count: int) -> None:
        try:
            for _ in range(count):
                shard = next(self.shard_sampler_iter)
                path = shm.generate_path()
                # append data_job to job pool
                data_job = self.writing_pool.apply_async(self.dataloader_iter_generator, (shard, path))
                self.data_jobs.append((shard, path, data_job))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"queued {path}, {len(self.data_jobs)}")
        except StopIteration:
            pass

    def _cleanup(self, stop_pool: bool = False) -> None:
        self.microbatches = None
        self.shutdown.set()  # signal running tasks to stop
        if self.next_batch:
            try:
                # if called on teardown/on_exception/__del__ will wait for the pending work to finish
                # if called on __init__ it's already done
                self.next_batch.result()
            except Exception:
                pass
            self.next_batch = None
        if stop_pool:
            self.writing_pool.close()  # no new tasks can run
            self.reading_pool.shutdown()
        for _, path, result in self.data_jobs:
            logger.debug(f"waiting for {path} to stop ...")
            self.cleanup.add(path)
            try:
                result.wait(timeout=DEFAULT_SHUTDOWN_TIMEOUT)
            except Exception:
                pass
            logger.debug(f"{path} stopped ...")
        self.data_jobs.clear()
        self.shutdown.clear()
        if stop_pool:
            self.writing_pool.join()  # make sure atexit is triggered in each subprocess
        for path in self.cleanup:
            logger.debug(f"removing {path} ...")
            shutil.rmtree(path, ignore_errors=True)

    def set_device(self, device: torch.device | None) -> None:
        self.device = device

    # called when fit/validate/predict/test is complete
    def teardown(self, *args: Any) -> None:
        logger.debug("teardown ...")
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

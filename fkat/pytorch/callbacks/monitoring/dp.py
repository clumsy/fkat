# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import time
from typing import Any, Protocol
from typing_extensions import override

import lightning as L
import torch
import torch.distributed as dist

from fkat.pytorch.schedule import Schedule, Never
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger
from fkat.utils.logging import rank0_logger

logger = rank0_logger(__name__)


class DpGroupStrategy(Protocol):
    """Protocol for getting DP group info for the current rank."""

    def dp_group_info(self) -> tuple[int, int]:
        """Return (group_id, rank_in_group) for the current rank."""
        ...


class DistDpGroup(DpGroupStrategy):
    """Calculates DP group info based on dp_size using distributed rank."""

    def __init__(self, dp_size: int) -> None:
        self.dp_size = dp_size

    def dp_group_info(self) -> tuple[int, int]:
        return divmod(dist.get_rank(), self.dp_size)  # type: ignore[possibly-unbound-attribute]


class EnvDpGroup(DpGroupStrategy):
    """Calculates DP group info based on dp_size using environment variables."""

    def __init__(self, dp_size: int) -> None:
        self.dp_size = dp_size

    def dp_group_info(self) -> tuple[int, int]:
        rank = int(os.environ.get("RANK", 0))
        return divmod(rank, self.dp_size)


class MegatronDpGroup(DpGroupStrategy):
    """Gets DP group info from Megatron parallel_state."""

    def dp_group_info(self) -> tuple[int, int]:
        from megatron.core import parallel_state  # type: ignore[import-not-found]

        group = parallel_state.get_data_parallel_group()
        rank_in_group = dist.get_rank(group)  # type: ignore[possibly-unbound-attribute]
        # For Megatron, we need to calculate group_id differently
        # This assumes we can derive it from global rank and group size
        global_rank = dist.get_rank()  # Get global rank separately  # type: ignore[possibly-unbound-attribute]
        group_size = dist.get_world_size(group)  # type: ignore[possibly-unbound-attribute]
        group_id = global_rank // group_size
        return group_id, rank_in_group


class DpSyncMonitor(L.Callback):
    """
    Monitors time for each DP group to reach synchronization point.
    Measures from batch start to before optimizer step to identify slow/fast groups.
    """

    def __init__(self, dp_group: DpGroupStrategy, schedule: Schedule | None = None) -> None:
        """
        Initialize the DP synchronization monitor.

        Args:
            dp_group: Strategy for determining DP group info (required).
            schedule: Controls when logging occurs. Defaults to ``Never``.
        """
        self.dp_group = dp_group
        self.schedule = schedule or Never()
        self.batch_start_time: float | None = None
        self._cb_logger: LightningLogger | None = None

    @override
    def setup(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        stage: str,
    ) -> None:
        if not self._cb_logger:
            self._cb_logger = CallbackLogger(trainer)

    @override
    def on_train_batch_start(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Start timing when batch processing begins."""
        self.batch_start_time = time.perf_counter()

    @override
    def on_before_optimizer_step(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """End timing when ready for sync (before optimizer step) and log if needed."""
        if self.batch_start_time is not None:
            sync_time_s = time.perf_counter() - self.batch_start_time
            # Log immediately since we're at the sync point, before any DP comms
            self._log_statistics(trainer, "train", 0, sync_time_s)
        self.batch_start_time = None

    def _log_statistics(self, trainer: "L.Trainer", stage: str, batch_idx: int, sync_time_s: float) -> None:
        """Log current group timing if schedule permits and this is DP group rank 0."""
        group_id, rank_in_group = self.dp_group.dp_group_info()
        if rank_in_group != 0:
            return

        if not self.schedule.check(stage=stage, batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            return

        if self._cb_logger:
            metrics = {f"dp_sync/group{group_id}/sync_s": sync_time_s}
            self._cb_logger.log_batch(metrics=metrics, timestamp=int(time.time()), step=trainer.global_step)

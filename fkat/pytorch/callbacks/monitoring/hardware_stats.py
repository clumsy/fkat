# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from time import time_ns
from typing import Any
from typing_extensions import override

import lightning as L
import psutil
import torch

from fkat.pytorch.callbacks.loggers import CallbackLogger
from fkat.pytorch.schedule import Schedule, Every

logger = logging.getLogger(__name__)


class HardwareStats(L.Callback):
    """Monitor and log hardware usage (CPU, RAM, and GPU) during training.

    Args:
        accelerator: Hardware accelerator to monitor ("gpu" or "cpu").
            If None or unsupported, only CPU/RAM are monitored.
        schedule: Controls when hardware stats are logged. Defaults to every batch.
    """

    def __init__(self, accelerator: str | None = None, schedule: Schedule | None = None) -> None:
        if accelerator not in ("gpu", "cpu", None):
            logger.warning(f"Unsupported accelerator: {accelerator}. Monitoring CPU/RAM only.")
            accelerator = "cpu"
        self.accelerator = accelerator
        self.schedule = schedule or Every(n_batches=1)
        self._cb_logger: CallbackLogger | None = None
        self._total_gpu_memory_gb: float | None = None

    @override
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        self._cb_logger = CallbackLogger(trainer)
        if self.accelerator == "gpu":
            if torch.cuda.is_available():
                self._total_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                torch.cuda.reset_peak_memory_stats()
            else:
                logger.warning("GPU accelerator requested but CUDA not available. Monitoring CPU/RAM only.")

    @L.pytorch.utilities.rank_zero_only
    @override
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.accelerator == "gpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _get_stats(self) -> dict[str, float]:
        stats = {
            "cpu_usage_percent": psutil.cpu_percent(),
            "ram_used_percent": psutil.virtual_memory().percent,
            "ram_used_gb": psutil.virtual_memory().used / 1e9,
        }
        if self.accelerator == "gpu" and torch.cuda.is_available():
            stats.update(
                {
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu_peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "gpu_memory_total_per_rank_gb": self._total_gpu_memory_gb or 0.0,
                }
            )
        return stats

    def _log_stats(self, trainer: L.Trainer) -> None:
        if self._cb_logger:
            self._cb_logger.log_batch(metrics=self._get_stats(), step=trainer.global_step, timestamp=int(time_ns()))
        if self.accelerator == "gpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @L.pytorch.utilities.rank_zero_only
    @override
    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        if self.schedule.check(stage="train", batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            self._log_stats(trainer)

    @L.pytorch.utilities.rank_zero_only
    @override
    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if self.schedule.check(stage="train", batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            self._log_stats(trainer)

    @L.pytorch.utilities.rank_zero_only
    @override
    def on_validation_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.schedule.check(stage="validation", batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            self._log_stats(trainer)

    @L.pytorch.utilities.rank_zero_only
    @override
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.schedule.check(stage="validation", batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            self._log_stats(trainer)

    @L.pytorch.utilities.rank_zero_only
    @override
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_stats(trainer)

    @L.pytorch.utilities.rank_zero_only
    @override
    def on_before_zero_grad(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Any) -> None:
        if self.schedule.check(
            stage="train", batch_idx=trainer.fit_loop.batch_idx, step=trainer.global_step, trainer=trainer
        ):
            self._log_stats(trainer)

    @L.pytorch.utilities.rank_zero_only
    @override
    def teardown(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        self._log_stats(trainer)

    @L.pytorch.utilities.rank_zero_only
    @override
    def on_exception(self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException) -> None:
        self._log_stats(trainer)

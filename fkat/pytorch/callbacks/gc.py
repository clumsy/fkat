# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import gc
from time import perf_counter
from typing import Any
from typing_extensions import override

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT

from fkat.pytorch.schedule import (
    Schedule,
    Never,
)
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger


class ManualGc(L.Callback):
    def __init__(
        self,
        schedule: Schedule | None = None,
        stats: dict[str, list[str]] | None = None,
    ) -> None:
        """
        PyTorch Lightning callback for manual garbage collection (GC) control.

        This callback allows fine-grained control over Python's garbage collection during training,
        validation, testing, and prediction. It can disable automatic garbage collection and instead
        perform manual collection at specified batch intervals.

        Args:
            schedule (Optional[Schedule]): When to invoke manual GC, defaults to class:`Never`
            stats (Optional[dict[str, list[str]]]): The list of stats to log per generation.
                Defaults to all stats for all generations

        Example:
            >>> trainer = Trainer(callbacks=[ManualGc()])
        """
        self.schedule = schedule or Never()
        self.stats = stats or dict(
            zip("012", (["collections", "collected", "uncollected"] for _ in range(3)), strict=True)
        )
        self._cb_logger: LightningLogger | None = None

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._cb_logger = self._cb_logger or CallbackLogger(trainer)
        if not isinstance(self.schedule, Never):
            gc.disable()

    @override
    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Perform GC after training epoch."""
        self.maybe_collect()

    @override
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform GC after training batch if needed."""
        self.maybe_gc(trainer, "train", batch_idx)

    @override
    def on_validation_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Perform GC after validation epoch."""
        self.maybe_collect()

    @override
    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform GC after validation batch if needed."""
        self.maybe_gc(trainer, "validation", batch_idx)

    @override
    def on_test_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Perform GC after test epoch."""
        self.maybe_collect()

    @override
    def on_predict_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform GC after prediction batch if needed."""
        self.maybe_gc(trainer, "predict", batch_idx)

    @override
    def on_predict_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Perform GC after predict epoch."""
        self.maybe_collect()

    @override
    def on_test_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform GC after test batch if needed."""
        self.maybe_gc(trainer, "test", batch_idx)

    def maybe_collect(self) -> None:
        if not isinstance(self.schedule, Never):
            gc.collect()

    def maybe_gc(self, trainer: "L.Trainer", stage: str, batch_idx: int) -> None:
        """
        Perform garbage collection if conditions are met.

        Args:
            batch_idx (int): Current batch index
        """
        if self.schedule.check(stage=stage, batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            now = perf_counter()
            gc.collect()
            if self._cb_logger is not None:
                metrics = {f"gc/rank{trainer.global_rank}/time": perf_counter() - now}
                for i, stats in enumerate(gc.get_stats()):
                    gen = str(i)
                    for key, value in stats.items():
                        if key in self.stats[gen]:
                            metrics[f"gc/rank{trainer.global_rank}/gen{gen}/{key}"] = float(value)
                self._cb_logger.log_batch(
                    metrics=metrics, timestamp=int(now or perf_counter()), step=trainer.global_step
                )

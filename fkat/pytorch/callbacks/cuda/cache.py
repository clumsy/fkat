# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import torch
import lightning as L

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

from fkat.pytorch.schedule import (
    Schedule,
    Never,
)


class EmptyCache(L.Callback):
    def __init__(self, schedule: Schedule | None = None) -> None:
        """
        PyTorch Lightning callback to trigger ``torch.cuda.empty_cache()``.

        This callback allows fine-grained control over CUDA's Caching Allocator during training,
        validation, testing, and prediction.

        Args:
            schedule (Optional[Schedule]): When to invoke ``torch, defaults to class:`Never`

        Example:
            >>> trainer = Trainer(callbacks=[EmptyCache()])
        """
        self.schedule = schedule or Never()

    @override
    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Perform empty cache after training epoch."""
        self.maybe_empty_cache(trainer, "train")

    @override
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform empty cache after training batch if needed."""
        self.maybe_empty_cache(trainer, "train", batch_idx)

    @override
    def on_validation_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Perform empty cache after validation epoch."""
        self.maybe_empty_cache(trainer, "validation")

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
        """Perform empty cache after validation batch if needed."""
        self.maybe_empty_cache(trainer, "validation", batch_idx)

    @override
    def on_test_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Perform empty cache after test epoch."""
        self.maybe_empty_cache(trainer, "test")

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
        """Perform empty cache after prediction batch if needed."""
        self.maybe_empty_cache(trainer, "predict", batch_idx)

    @override
    def on_predict_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Perform empty cache after predict epoch."""
        self.maybe_empty_cache(trainer, "predict")

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
        """Perform empty cache after test batch if needed."""
        self.maybe_empty_cache(trainer, "test", batch_idx)

    def maybe_empty_cache(self, trainer: "L.Trainer", stage: str, batch_idx: int | None = None) -> None:
        """
        Perform empty cache if conditions are met.

        Args:
            trainer (L.Trainer): Lightning Trainer
            stage (str): training stage
            batch_idx (int | None): Current batch index if available
        """
        if self.schedule.check(stage=stage, batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            torch.cuda.empty_cache()

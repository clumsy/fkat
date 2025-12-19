# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from typing_extensions import override
import datetime as dt

import torch
import fsspec
import lightning as L

from fkat.pytorch.schedule import Schedule, Never


class OptimizerSnapshot(L.Callback):
    """
    Callback that saves optimizer state at specified intervals during training.

    This callback allows you to capture the state of optimizers at specific points
    during training, which can be useful for debugging, analysis, or resuming training
    from specific optimization states.

    Args:
        output_path_prefix (str): Output path prefix for generated optimizer snapshots.
        schedule (Optional[Schedule]): Schedule at which to take a snapshot of optimizers.
            Defaults to ``Never``
    """

    def __init__(
        self,
        output_path_prefix: str,
        schedule: Schedule | None = None,
    ) -> None:
        self.output_path_prefix = output_path_prefix
        self.schedule = schedule or Never()

    @override
    def on_train_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if self.schedule.check(trainer=trainer, stage="train", batch_idx=batch_idx, step=trainer.global_step):
            timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
            for i, opt in enumerate(trainer.optimizers):
                path = f"{self.output_path_prefix}rank{trainer.global_rank}_opt{i}_{timestamp}.pt"
                with fsspec.open(path, "wb", makedirs=True) as f:
                    torch.save(opt, f)

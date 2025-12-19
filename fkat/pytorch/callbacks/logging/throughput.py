# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from time import time
from typing import Any, TYPE_CHECKING

import lightning as L
from lightning.pytorch.callbacks import LearningRateFinder, BatchSizeFinder
from lightning.pytorch.utilities.data import extract_batch_size
from lightning.pytorch.utilities import rank_zero_only

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

import torch
from fkat.pytorch.schedule import (
    Schedule,
    Never,
)
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger
from fkat.utils.logging import rank0_logger

logger = rank0_logger(__name__)


class Throughput(L.Callback):
    def __init__(self, dp_ranks: int | None = None, schedule: Schedule | None = None) -> None:
        """
        Throughput logging callback that measures the time spent processing the microbatches.
        Args:
            schedule (Optional[Schedule]): Controls when logging occurs. Defaults to ``Never``.
        """
        self.schedule = schedule or Never()
        self.dp_ranks: int | None = dp_ranks
        self.was_last_step_val = False
        self.publish = False
        self.step_start_time: dict[str, float] = {}
        self.step_time: dict[str, float] = {}
        self.total_time: dict[str, float] = {}
        self.step_samples: dict[str, float] = {}
        self.total_samples: dict[str, float] = {}
        self.epoch_start_time: dict[str, float] = {}
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
            # ignoring special callbacks used for tuning
            callbacks = [
                c
                for c in trainer.callbacks  # type: ignore[attr-defined]
                if not isinstance(c, LearningRateFinder | BatchSizeFinder)
            ]
            tput_callbacks = [i for i, c in enumerate(callbacks) if isinstance(c, Throughput)]
            assert len(tput_callbacks) == 1, "There can only be one Throughput logging callback in operation"
        self.dp_ranks: int = self.dp_ranks or trainer.world_size
        self.step_start_time = {}
        self.step_time = {}
        self.total_time = {}
        self.step_samples = {}
        self.total_samples = {}
        self.epoch_start_time = {}

    def _start_epoch(self, stage: str) -> None:
        self.epoch_start_time[stage] = self.step_start_time[stage] = time()

    @override
    @rank_zero_only
    def on_train_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._start_epoch("train")

    @override
    @rank_zero_only
    def on_validation_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._start_epoch("validation")

    @override
    @rank_zero_only
    def on_test_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._start_epoch("test")

    @override
    @rank_zero_only
    def on_predict_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._start_epoch("predict")

    def _report_epoch(self, trainer: "L.Trainer", stage: str) -> None:
        self.step_start_time[stage] = (now := time())
        metrics = {
            f"{stage}/epochs/epoch_time": now - self.epoch_start_time[stage],
        }
        if self._cb_logger:
            self._cb_logger.log_batch(metrics=metrics, timestamp=int(now), step=trainer.global_step)
        else:
            logger.info(metrics)

    @override
    @rank_zero_only
    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._report_epoch(trainer, "train")

    @override
    @rank_zero_only
    def on_validation_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._report_epoch(trainer, "validation")

    @override
    @rank_zero_only
    def on_test_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._report_epoch(trainer, "test")

    @override
    @rank_zero_only
    def on_predict_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._report_epoch(trainer, "predict")

    def _update(self, stage: str, batch: Any, batch_idx: int, step: int | None, trainer: "L.Trainer") -> None:
        # because of other callbacks we want to only measure within batch start/end
        # and make sure this callback is the first in the list
        now = time()
        self.step_time[stage] = self.step_time.get(stage, 0) + (now - self.step_start_time[stage])
        self.step_start_time[stage] = now
        num_samples = extract_batch_size(batch) if batch else 0
        self.step_samples[stage] = self.step_samples.get(stage, 0) + num_samples
        # train data points have to be at step boundaries or we will have multiple datapoints for the same step
        self.publish = stage != "train" and self.schedule.check(
            stage=stage, batch_idx=batch_idx, step=step, trainer=trainer
        )

    @override
    @rank_zero_only
    def on_before_zero_grad(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", optimizer: "torch.optim.Optimizer"
    ) -> None:
        """
        Report metrics for individual steps during training.
        """
        self.publish = True  # always log on step

    @override
    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # throughput is the first callback so it's safe to capture time here
        self._update("train", batch, batch_idx, trainer.global_step, trainer)
        self._report(trainer, "train")

    @override
    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._update("validation", batch, batch_idx, trainer.global_step, trainer)
        self._report(trainer, "validation")

    @override
    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._update("test", batch, batch_idx, trainer.global_step, trainer)
        self._report(trainer, "test")

    @override
    @rank_zero_only
    def on_predict_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._update("predict", batch, batch_idx, trainer.global_step, trainer)
        self._report(trainer, "predict")

    def _report(self, trainer: "L.Trainer", stage: str) -> None:
        if not self.publish:
            return
        if not self.step_time.get(stage):
            # can end up here outside of training loop e.g. when initializing precision plugin
            return
        self.total_time[stage] = self.total_time.get(stage, 0) + self.step_time[stage]
        self.total_samples[stage] = self.total_samples.get(stage, 0) + self.step_samples[stage]
        rank0_avg_tput = self.total_samples[stage] / self.total_time[stage]
        assert self.dp_ranks
        metrics = {
            f"{stage}/throughput/running_avg_rank0": rank0_avg_tput,
            f"{stage}/throughput/running_avg": self.dp_ranks * rank0_avg_tput,
        }
        if stage == "train":
            # we only have steps during fit
            metrics[f"{stage}/steps/step_time"] = self.step_time[stage]
        rank0_tput = self.step_samples[stage] / self.step_time[stage]
        metrics[f"{stage}/throughput/current_rank0"] = rank0_tput
        metrics[f"{stage}/throughput/current"] = self.dp_ranks * rank0_tput
        if self._cb_logger:
            self._cb_logger.log_batch(metrics=metrics, timestamp=int(time()), step=trainer.global_step)
        else:
            logger.info(metrics)
        self.step_time[stage] = 0.0
        self.step_samples[stage] = 0

    @override
    @rank_zero_only
    def on_train_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.publish = True
        self._report(trainer, "train")

    @override
    @rank_zero_only
    def on_validation_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.publish = True
        self._report(trainer, "validation")

    @override
    @rank_zero_only
    def on_test_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.publish = True
        self._report(trainer, "test")

    @override
    @rank_zero_only
    def on_predict_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.publish = True
        self._report(trainer, "predict")

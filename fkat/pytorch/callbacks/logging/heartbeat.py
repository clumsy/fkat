# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime as dt
from typing import Any
from typing_extensions import override

import lightning as L
from lightning.pytorch.utilities import rank_zero_only

from fkat.pytorch.schedule import (
    Schedule,
    Elapsed,
)
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger
from fkat.utils.logging import rank0_logger

log = rank0_logger(__name__)


class Heartbeat(L.Callback):
    """Publishes tags indicating the time and step of the last heartbeat with the provided schedule."""

    def __init__(
        self,
        schedule: Schedule | None = None,
        last_check_in_time_tag: str = "last_check_in_time",
        last_check_in_step_tag: str = "last_check_in_step",
    ) -> None:
        self.last_check_in_time_tag = last_check_in_time_tag
        self.last_check_in_step_tag = last_check_in_step_tag
        self.schedule = schedule or Elapsed(interval=dt.timedelta(minutes=15))
        self._cb_logger: LightningLogger | None = None

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._cb_logger = CallbackLogger(trainer)

    def _publish_tags(self, stage: str, batch_idx: int, trainer: "L.Trainer") -> None:
        if self.schedule.check(stage=stage, batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            assert self._cb_logger
            time = dt.datetime.now(dt.timezone.utc)
            self._cb_logger.log_batch(
                tags={
                    self.last_check_in_time_tag: str(time),
                    self.last_check_in_step_tag: str(trainer.global_step),
                }
            )

    @override
    @rank_zero_only
    def on_train_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        self._publish_tags("train", batch_idx, trainer)

    @override
    @rank_zero_only
    def on_test_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._publish_tags("test", batch_idx, trainer)

    @override
    @rank_zero_only
    def on_validation_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._publish_tags("validation", batch_idx, trainer)

    @override
    @rank_zero_only
    def on_predict_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._publish_tags("predict", batch_idx, trainer)

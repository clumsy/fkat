# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import time
import random
import signal
import logging
import ast
import multiprocessing
import datetime as dt
from typing import Any
from typing_extensions import override

import lightning as L
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder
from lightning.pytorch.utilities import rank_zero_only

from fkat.pytorch.schedule import (
    Schedule,
    Elapsed,
)
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger

log = logging.getLogger(__name__)


def start_shutdown_detection_process(
    logger: LightningLogger | None,
    shutdown_tag: str,
    trainer: "L.Trainer",
) -> multiprocessing.Process | None:
    """
    Create a process for monitoring trainer errors by periodically detecting
    the ``shutdown`` tag. Terminate the application if the tag is detected.
    The process is spawn on local rank 0 to minimize the overhead.
    """
    process: multiprocessing.Process | None = None
    if logger is not None and trainer.local_rank == 0:
        log.info("Starting a shutdown detection process...")
        process = multiprocessing.Process(
            target=detect_shutdown_from_logger, args=(logger, shutdown_tag, os.getpid(), 60)
        )
        process.daemon = True
        process.start()
    return process


def detect_shutdown_from_logger(
    logger: LightningLogger,
    shutdown_tag: str,
    pid: int,
    detection_interval_secs: int,
) -> None:
    """
    Detect ``shutdown_tag`` tag periodically. If the tag is found, sends a SIGABRT to the
    process with the provided pid to shutdown the training process.
    """
    sleep_duration = int(os.getenv("SHUTDOWN_DETECTION_INTERVAL", default=str(detection_interval_secs)))
    log.debug(f"Shutdown detection frequency is {sleep_duration} secs")
    try:
        while True:
            random_delay = random.uniform(0, sleep_duration * 0.5)
            time.sleep(random_delay)
            tags = logger.tags()
            if shutdown_tag in tags:
                log.info(f"Found {shutdown_tag}={tags[shutdown_tag]} tag. Shutting down process {pid}.")
                os.kill(pid, signal.SIGABRT)
            time.sleep(sleep_duration - random_delay)
    except Exception as e:
        log.error(f"Got error when querying mlflow SHUTDOWN tag: {e}")


class GracefulShutdown(L.Callback):
    def __init__(
        self,
        schedule: Schedule | None = None,
        shutdown_tag: str = "shutdown",
        shutdown_info_tag: str = "shutdown_info",
    ) -> None:
        self.shutdown_tag = shutdown_tag
        self.shutdown_info_tag = shutdown_info_tag
        self.schedule = schedule or Elapsed(dt.timedelta(minutes=5))
        self._cb_logger: LightningLogger | None = None
        self._process: multiprocessing.Process | None = None

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._cb_logger = CallbackLogger(trainer)
        self._process = start_shutdown_detection_process(self._cb_logger, self.shutdown_tag, trainer)

    def _maybe_stop(self, stage: str, trainer: "L.Trainer", batch_idx: int) -> None:
        if (
            trainer.should_stop
            or not self._cb_logger
            or not self.schedule.check(stage=stage, batch_idx=batch_idx, step=trainer.global_step, trainer=trainer)
        ):
            return
        tags = self._cb_logger.tags()
        shutdown_tag = tags.get(self.shutdown_tag)
        trainer.should_stop = shutdown_tag is not None
        if trainer.should_stop:
            info_tag = tags.get(self.shutdown_info_tag)
            if info_tag:
                info = ast.literal_eval(info_tag)[-1]
                strategy = info["Strategy"].upper()
                log.info(f"Shutdown signal received. Using shutdown strategy {strategy}")
                self._cb_logger.log_tag(self.shutdown_tag, "SHUTTING_DOWN")

    def _update_shutdown_status(self, trainer: "L.Trainer", status: str) -> None:
        if self._cb_logger:
            log.info(f"update shutdown status {status} indicate job finished.")
            self._cb_logger.log_tag(self.shutdown_tag, status)

    @override
    @rank_zero_only
    def on_train_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        self._maybe_stop("train", trainer, batch_idx)

    @rank_zero_only
    def on_test_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._maybe_stop("test", trainer, batch_idx)

    @override
    @rank_zero_only
    def on_validation_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._maybe_stop("validation", trainer, batch_idx)

    @rank_zero_only
    def on_predict_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._maybe_stop("predict", trainer, batch_idx)

    @override
    def teardown(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        if trainer.global_rank == 0:
            if not self._tuning(trainer):
                log.info("update job status before job finished")
                self._update_shutdown_status(trainer, "JOB_FINISHED")
        self._terminate_monitor()

    @override
    def on_exception(self, trainer: "L.Trainer", pl_module: "L.LightningModule", exception: BaseException) -> None:
        self._terminate_monitor()

    def _terminate_monitor(self) -> None:
        """
        Terminates the separate process used for monitoring trainer errors if it is alive.
        """
        if self._process and self._process.is_alive():
            log.info("\nTerminating error monitor...")
            self._process.kill()

    def _tuning(self, trainer: "L.Trainer") -> bool:
        num_tuning_cbs = sum(
            isinstance(
                cb,
                LearningRateFinder | BatchSizeFinder,
            )
            for cb in trainer.callbacks  # type: ignore[attr-defined]
        )
        return num_tuning_cbs > 0

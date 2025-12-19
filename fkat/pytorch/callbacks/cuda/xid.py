# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import logging
import multiprocessing
from typing import Any
from typing_extensions import override

import lightning as L
from lightning.pytorch.utilities import rank_zero_only

from fkat.utils.cuda.xid import detect_xid_errors
from fkat.pytorch.actions import LightningAction
from fkat.pytorch.schedule import Schedule
from fkat.pytorch.utilities import local_rank_zero_only

log = logging.getLogger(__name__)


class Xid(L.Callback):
    """
    A callback to monitor and log Xid errors in a separate process during training.

    It utilizes a separate process to monitor these errors, ensuring that the main training process remains unaffected.
    The monitoring process is started at the beginning of training and terminated either
    upon an exception in training or at the end of the training/validation stage.
    """

    monitor: multiprocessing.Process | None = None

    def __init__(self, actions: dict[str, LightningAction], schedule: Schedule) -> None:
        """
        Arguments:
            actions: Dictionary mapping Xid ranges to actions
                     Format: {
                         "0-100": fkat.actions.log,
                         "13,43,63-64,48,79,95": fkat.actions.ec2.reboot,
                         "81": fkat.actions.ec2.terminate,
                     }
        """
        super().__init__()
        self.actions = self._parse_xid_ranges(actions)
        self.schedule = schedule
        self.xid_errors: multiprocessing.Queue[set[int]] = multiprocessing.Queue()
        self.xid_check: multiprocessing.Event = multiprocessing.Event()  # type: ignore[attr-defined]

    def _parse_xid_ranges(self, xid_actions: dict[str, LightningAction]) -> dict[int, LightningAction]:
        actions = {}
        for xid_range, action in xid_actions.items():
            parts = xid_range.split(",")
            for part in parts:
                part = part.strip()
                is_range = "-" in part
                if is_range:
                    start, end = map(int, part.split("-"))
                    for xid in range(start, end + 1):
                        actions[xid] = action
                else:
                    try:
                        actions[int(part)] = action
                    except ValueError:
                        print(f"Warning: Invalid XID format: {part}")
        return actions

    @override
    @local_rank_zero_only
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """
        Initializes the Xid error monitoring process at the start of the training stage.

        This method is automatically invoked by the PyTorch Lightning framework. It starts
        a separate background process dedicated to monitoring Xid errors.

        Args:
            trainer (L.Trainer): The PyTorch Lightning Trainer instance.
            module (L.LightningModule): The PyTorch Lightning module being trained.
            stage (str): The stage of the training process (e.g., 'fit', 'test').

        Returns:
            None.
        """
        log.info("Checking for Xid errors ...")
        self.monitor = multiprocessing.Process(target=detect_xid_errors, args=(self.xid_check, self.xid_errors))
        self.monitor.start()

    def _terminate_monitor(self) -> None:
        """
        Terminates the separate process used for monitoring Xid errors if it is alive.

        This is an internal method that checks if the monitoring process is active and,
        if so, terminates it to clean up resources.

        Returns:
            None.
        """
        if self.monitor and self.monitor.is_alive():
            log.info("Terminating Xid errors monitor")
            self.monitor.kill()

    @override
    @local_rank_zero_only
    def on_exception(self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException) -> None:
        """
        Callback method to handle exceptions during training.

        If an exception occurs during the training process, this method ensures that the
        Xid error monitoring process is terminated to prevent resource leakage.

        Args:
            trainer (L.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (L.LightningModule): The PyTorch Lightning module being trained.
            exception (BaseException): The exception that occurred during training.

        Returns:
            None.
        """
        self._terminate_monitor()

    @override
    @local_rank_zero_only
    def teardown(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """
        Ensures the Xid error monitoring process is terminated at the end of training.

        This method is automatically called by the PyTorch Lightning framework at the
        end of the training or validation stage to clean up the monitoring process.

        Args:
            trainer (L.Trainer): The PyTorch Lightning Trainer instance.
            module (L.LightningModule): The PyTorch Lightning module being trained.
            stage (str): The stage of the training process (e.g., 'fit', 'test').

        Returns:
            None.
        """
        self._terminate_monitor()

    @override
    @rank_zero_only
    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self.check(trainer, "train", batch_idx)

    @override
    @rank_zero_only
    def on_test_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.check(trainer, "test", batch_idx)

    @override
    @rank_zero_only
    def on_validation_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.check(trainer, "validation", batch_idx)

    @override
    @rank_zero_only
    def on_predict_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.check(trainer, "predict", batch_idx)

    def check(self, trainer: L.Trainer, stage: str, batch_idx: int) -> None:
        if self.schedule.check(stage=stage, batch_idx=batch_idx, step=trainer.global_step):
            self.xid_check.set()
        while not self.xid_errors.empty():
            xids = self.xid_errors.get()
            for xid in xids:
                if action := self.actions.get(xid):
                    action.perform(trainer=trainer, xid=xid)

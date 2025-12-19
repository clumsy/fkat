# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
import atexit
import signal
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import lightning as L

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

from fkat.pytorch.schedule import (
    Schedule,
    Never,
)
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger

memray = None


class Memray(L.Callback):
    def __init__(
        self,
        ranks: Sequence[int] | None = None,
        flamegraph: bool = False,
        output_path_prefix: str | None = None,
        schedule: Schedule | None = None,
        compress: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        [Memray](https://bloomberg.github.io/memray/api.html) PyTorch Lightning callback.
        This callbacks traces host RAM (DRAM) allocations and publishes a report to help identify
        potential memory leaks and investigate OOM errors.

        Args:
            ranks (Optional[Sequence[int]]): only trace the provided ranks, defaults to all ranks
            flamegraph (bool): whether to generate [Flamegraph](https://www.brendangregg.com/flamegraphs.html)
                for the traced allocations, generates HTML report that van be viewed without installing `memray`
            output_path_prefix (Optional[str]): output path prefix for generated reports,
                use to persist these files locally, defaults to temporary location that is cleaned as soon as possible
            schedule (Optional[Schedule]): Controls when logging occurs during training.
                Defaults to Never - no logging
            compress (bool): publish reports as compressed files defaults to publishing raw files
        """
        self.ranks = ranks
        self.flamegraph = flamegraph
        self.compress = compress
        self.rank: int | None = None
        self.stage: str | None = None
        self.kwargs = kwargs

        self.output_path_prefix = output_path_prefix
        self.schedule = schedule or Never()
        self._cb_logger: LightningLogger | None = None

        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Memray")

        global memray
        import memray  # type: ignore[unresolved-import]

        self.tracker: memray.Tracker | None = None  # type: ignore
        self.dir = self.tmp_dir = "/tmp"

        signal.signal(signal.SIGTERM, self._terminate)  # terminate signal
        signal.signal(signal.SIGINT, self._terminate)  # keyboard interrupt
        atexit.register(self._terminate)

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._cb_logger = CallbackLogger(trainer)
        self.rank = trainer.global_rank
        self.stage = stage

    @override
    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._start(trainer)

    @override
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._on_batch_end(trainer, "train", batch_idx + 1)

    @override
    def on_validation_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._start(trainer)

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
        self._on_batch_end(trainer, "validation", batch_idx + 1)

    @override
    def on_predict_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._start(trainer)

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
        self._on_batch_end(trainer, "predict", batch_idx + 1)

    @override
    def on_test_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self._start(trainer)

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
        self._on_batch_end(trainer, "test", batch_idx + 1)

    def _on_batch_end(self, trainer: "L.Trainer", stage: str, batch_idx: int) -> None:
        if self.schedule.check(stage=stage, batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            self._stop(str(batch_idx))
            self._start(trainer)

    @override
    def teardown(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._terminate()

    @override
    def on_exception(self, trainer: "L.Trainer", pl_module: "L.LightningModule", exception: BaseException) -> None:
        self._terminate()

    def _terminate(self, *args: Any, **kwargs: Any) -> None:
        # calling synchronously since this can be called during interpreter shutdown
        self._stop("last", sync=True)
        self.executor.shutdown()

    def _start(self, trainer: "L.Trainer") -> None:
        if self.ranks is not None and trainer.global_rank not in self.ranks:
            return
        if not self.tracker:
            self.tmp_dir = tempfile.mkdtemp()
            self.dir = self.output_path_prefix or self.tmp_dir
            path = os.path.join(self.dir, f"rank{trainer.global_rank}.bin")
            assert memray
            self.tracker = memray.Tracker(path, **self.kwargs)
            self.tracker.__enter__()
        assert self.tracker

    def _stop(self, suffix: str, sync: bool = False) -> None:
        if not self.tracker:
            return
        # create reports synchronously
        self.tracker.__exit__(None, None, None)
        self.tracker = None
        if self.flamegraph:
            for f in os.listdir(self.dir):
                results = os.path.join(self.dir, f)
                from memray.commands.flamegraph import FlamegraphCommand  # type: ignore[unresolved-import]

                # creating this report synchronously because it uses a global memray lock
                FlamegraphCommand().write_report(Path(results), Path(results + ".html"), True, -1, False)
        # process reports asynchronously
        artifacts_path = f"memray/{self.stage}/{suffix}"
        if sync:
            self._process(artifacts_path, self.dir, self.tmp_dir)
        else:
            self.executor.submit(self._process, artifacts_path, self.dir, self.tmp_dir)

    def _process(
        self,
        artifacts_path: str,
        report_dir: str,
        tmp_dir: str,
    ) -> None:
        assert self._cb_logger
        for f in os.listdir(report_dir):
            output_file = os.path.join(report_dir, f)
            if self.compress:
                with open(output_file, "rb") as f_in:
                    output_file = output_file + ".gz"
                    with gzip.open(output_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            self._cb_logger.log_artifact(output_file, artifacts_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)

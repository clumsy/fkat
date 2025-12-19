# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import gzip
import shutil
import tempfile
import atexit
import signal
from typing import Any, TYPE_CHECKING
from typing_extensions import override
from collections.abc import Sequence

import torch
import lightning as L

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

from fkat.pytorch.schedule import (
    Schedule,
    Never,
)
from fkat.pytorch.utilities import get_rank
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger


def exec_with_nsys(kwargs: dict[str, str]) -> None:
    """Replace current process with nsys profiling of the specified script."""
    # only capture between explicit API calls to start/stop profiling
    kwargs["capture-range"] = "cudaProfilerApi"
    kwargs["capture-range-end"] = "stop"

    script_path, args = sys.argv[0], sys.argv[1:]
    nsys_cmd = ["nsys", "profile", *[f"--{k}={v}" for k, v in kwargs.items()], "python", script_path] + args

    # add current working dir for module resolution
    os.environ["PYTHONPATH"] = os.path.join(
        os.getcwd(), *([os.environ["PYTHONPATH"]] if "PYTHONPATH" in os.environ else [])
    )

    # replace current process with nsys
    os.execvp("nsys", nsys_cmd)


class Nsys(L.Callback):
    def __init__(
        self,
        ranks: Sequence[int] | None = None,
        output_path_prefix: str | None = None,
        schedule: Schedule | None = None,
        compress: bool = True,
        record_shapes: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        [Nsys](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) PyTorch Lightning callback.
        This :class:`L.Callback` continiously traces the training process and publishes a report
        that helps examining the duration of individual calls through time.

        Args:
            ranks (Optional[Sequence[int]]): Only trace the provided ranks, defaults to all ranks
            output_path_prefix (Optional[str]): output path prefix for generated reports,
                use to persist these files locally, defaults to temporary location that is cleaned as soon as possible
            schedule (Optional[Schedule]): Controls when tracing occurs during training.
                Defaults to :class:`Never` - no tracing
            compress (bool): Whether to compress the report.
                Defaults to ``True``
            record_shapes (bool): Whether to include tensor shapes in the report.
                Defaults to ``False``
            **kwargs (Any): Arbitrary keyword arguments passed as is to Nsys.
        """
        self.rank = get_rank()
        self.schedule = schedule or Never()
        self.output_path_prefix = output_path_prefix
        self.compress = compress
        self.record_shapes = record_shapes
        self._enabled = False

        if ranks is None or self.rank in ranks:
            # break infinite recusion
            self.output_file = os.environ.pop("NSYS_OUTPUT", None)
            if self.output_file is None:
                output_file = os.path.join(self.output_path_prefix or tempfile.mkdtemp(), f"rank{self.rank}.nsys-rep")
                os.environ["NSYS_OUTPUT"] = kwargs["output"] = output_file
                exec_with_nsys(kwargs)
            self._maybe_trace()
        self._cb_logger: LightningLogger | None = None
        self.stage: str | None = None

        signal.signal(signal.SIGTERM, self._terminate)  # terminate signal
        signal.signal(signal.SIGINT, self._terminate)  # keyboard interrupt
        atexit.register(self._terminate)

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._cb_logger = CallbackLogger(trainer)
        self.stage = stage
        self._maybe_trace(stage=stage)

    def _maybe_trace(
        self, trainer: "L.Trainer | None" = None, stage: str | None = None, batch_idx: int | None = None
    ) -> None:
        should_run = self.schedule.check(
            stage=stage, batch_idx=batch_idx, step=trainer.global_step if trainer else None, trainer=trainer
        )
        if should_run:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        if self._enabled:
            return
        self._enabled = True
        torch.cuda.cudart().cudaProfilerStart()
        torch.autograd.profiler.emit_nvtx(record_shapes=self.record_shapes).__enter__()

    def _stop(self) -> None:
        if not self._enabled:
            return
        torch.cuda.cudart().cudaProfilerStop()
        torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)
        self._enabled = False

    @override
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._maybe_trace(trainer, "train", batch_idx + 1)

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
        self._maybe_trace(trainer, "validation", batch_idx + 1)

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
        self._maybe_trace(trainer, "predict", batch_idx + 1)

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
        self._maybe_trace(trainer, "test", batch_idx + 1)

    def _publish(self) -> None:
        self._stop()
        assert self.output_file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        if self.compress:
            with open(self.output_file, "rb") as f_in:
                output_file = self.output_file + ".gz"
                with gzip.open(output_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                shutil.rmtree(self.output_file, ignore_errors=True)
        assert self._cb_logger
        self._cb_logger.log_artifact(output_file, "nsys")
        if not self.output_path_prefix:
            shutil.rmtree(output_file, ignore_errors=True)

    @override
    def teardown(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._terminate()

    @override
    def on_exception(self, trainer: "L.Trainer", pl_module: "L.LightningModule", exception: BaseException) -> None:
        self._terminate()

    def _terminate(self, *_: Any) -> None:
        if self.stage:
            self._publish()

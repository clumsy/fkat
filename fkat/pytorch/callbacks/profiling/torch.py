# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import gzip
import shutil
import tempfile
import atexit
import signal
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence
from typing_extensions import override

import lightning as L
import torch

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

from fkat.pytorch.schedule import (
    Schedule,
    Never,
)
from fkat.pytorch.utilities import get_rank
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger


class PyTorch(L.Callback):
    def __init__(
        self,
        ranks: Sequence[int] | None = None,
        output_path_prefix: str | None = None,
        schedule: Schedule | None = None,
        compress: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) Lightning callback.
        This :class:`L.Callback` continiously traces the training process and publishes a report
        that helps examining the duration of individual calls through time.

        Args:
            ranks (Optional[Sequence[int]]): only trace the provided ranks, defaults to all ranks
            output_path_prefix (Optional[str]): output path prefix for generated reports,
                use to persist these files locally, defaults to temporary location that is cleaned as soon as possible
            schedule (Optional[Schedule]): Controls when logging occurs during training.
                Defaults to :class:`Never` - no intermediate logging
            compress (bool): compress the report
                Defaults to ``True``
            **kwargs (Any): Arbitrary keyword arguments passed as is to PyTorch Profiler
                except for ``execution_trace_observer`` and ``on_trace_ready``.
        """
        self.rank = get_rank()
        self.compress = compress
        self.schedule = schedule or Never()
        self.output_path_prefix = output_path_prefix

        self.trace_observer: torch.profiler.ExecutionTraceObserver | None = None
        self.trace_file: str | None
        self.profiler: torch.profiler.profile | None = None
        if ranks is None or self.rank in ranks:
            self.trace_file = os.path.join(self.output_path_prefix or tempfile.mkdtemp(), f"rank{self.rank}.json")
            self.trace_observer = torch.profiler.ExecutionTraceObserver()
            kwargs.pop("execution_trace_observer", None)
            kwargs.pop("on_trace_ready", None)
            self.profiler = torch.profiler.profile(
                schedule=lambda step: torch.profiler.ProfilerAction.RECORD_AND_SAVE,
                on_trace_ready=self._publish,
                execution_trace_observer=self.trace_observer,
                **kwargs,
            )
            self._start_profiler()
        self._cb_logger: LightningLogger | None = None
        self.stage: str | None = None
        self.batch_idx = "?"
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PyTorchProfiler")

        signal.signal(signal.SIGTERM, self._terminate)  # terminate signal
        signal.signal(signal.SIGINT, self._terminate)  # keyboard interrupt
        atexit.register(self._terminate)

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._cb_logger = CallbackLogger(trainer)
        self.stage = stage

    def _on_batch_end(self, trainer: "L.Trainer", stage: str, batch_idx: int) -> None:
        if self.profiler and self.schedule.check(
            stage=stage, batch_idx=batch_idx + 1, step=trainer.global_step if trainer else None, trainer=trainer
        ):
            self.batch_idx = str(batch_idx + 1)
            self.profiler.step()

    @override
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._on_batch_end(trainer, "train", batch_idx)

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
        self._on_batch_end(trainer, "validation", batch_idx)

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
        self._on_batch_end(trainer, "predict", batch_idx)

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
        self._on_batch_end(trainer, "test", batch_idx)

    def _publish(self, prof: torch.profiler.profile) -> None:
        # create report synchronously
        assert self.trace_file
        prof.export_chrome_trace(self.trace_file)
        base_path = self.output_path_prefix or os.path.dirname(self.trace_file)
        output_file = os.path.join(base_path, self.batch_idx, os.path.basename(self.trace_file))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shutil.move(self.trace_file, output_file)
        self._start_profiler()
        # process report asynchronously
        artifact_path = f"pt_profiler/{self.stage}/{self.batch_idx}"
        sync = self.profiler is None
        if sync:
            # calling synchronously since this can be called during interpreter shutdown
            self._process(output_file, artifact_path)
        else:
            self.executor.submit(self._process, output_file, artifact_path)

    def _process(self, output_file: str, artifacts_path: str) -> None:
        assert self._cb_logger
        if self.compress:
            with open(output_file, "rb") as f_in:
                output_file = output_file + ".gz"
                with gzip.open(output_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        self._cb_logger.log_artifact(output_file, artifacts_path)
        shutil.rmtree(output_file, ignore_errors=True)

    @override
    def teardown(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._terminate()

    @override
    def on_exception(self, trainer: "L.Trainer", pl_module: "L.LightningModule", exception: BaseException) -> None:
        self._terminate()

    def _terminate(self, *_: Any) -> None:
        if self.profiler and self.stage:
            self.batch_idx = "last"
            self._stop_profiler()
            self.profiler = None
            self.executor.shutdown()
            if self.trace_file:
                shutil.rmtree(self.trace_file, ignore_errors=True)

    def _start_profiler(self) -> None:
        assert self.trace_file and self.trace_observer and self.profiler
        shutil.rmtree(self.trace_file, ignore_errors=True)
        self.trace_observer.register_callback(self.trace_file)
        self.profiler.start()

    def _stop_profiler(self) -> None:
        if self.profiler:
            self.profiler.stop()
        if self.trace_observer:
            self.trace_observer.unregister_callback()

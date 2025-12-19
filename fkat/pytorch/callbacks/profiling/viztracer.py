# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import json
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

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

from fkat.pytorch.schedule import (
    Schedule,
    Never,
)
from fkat.pytorch.utilities import get_rank
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger

if TYPE_CHECKING:
    import viztracer
else:
    viztracer = None


class VizTracer(L.Callback):
    def __init__(
        self,
        ranks: Sequence[int] | None = None,
        output_path_prefix: str | None = None,
        schedule: Schedule | None = None,
        compress: bool = False,
        patch: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        [VizTracer](https://viztracer.readthedocs.io/en/latest/) PyTorch Lightning callback.
        This :class:`L.Callback` continiously traces the training process and publishes a report
        that helps examining the duration of individual calls through time.

        Args:
            ranks (Optional[Sequence[int]]): only trace the provided ranks, defaults to all ranks
            output_path_prefix (Optional[str]): output path prefix for generated reports,
                use to persist these files locally, defaults to temporary location that is cleaned as soon as possible
            schedule (Optional[Schedule]): Controls when logging occurs during training.
                Defaults to :class:`Never` - no logging
            compress (bool): publish reports as compressed binaries
                (need to be decompressed via `viztracer --decompress <REPORT>`),
                if ``True``` saves reports using viztracer's own compression that requires `viztracer` installation,
                defaults to ``False`` and publishes gzipped HTML reports which require no `viztracer` installation
            patch (bool): whether to let VizTracer patch internal Python hooks: subprocess, multiprocessing, etc.
                Defaults to ``False``
            **kwargs (Any): Arbitrary keyword arguments passed as is to VizTracer.
        """
        self.rank = get_rank()
        self.schedule = schedule or Never()
        self.output_path_prefix = output_path_prefix

        global viztracer
        import viztracer
        from viztracer.vcompressor import VCompressor

        self.compressor = VCompressor() if compress else None

        self.tracer: viztracer.VizTracer | None = None  # type: ignore[no-any-unimported]
        if ranks is None or self.rank in ranks:
            kwargs["output_file"] = f"rank{self.rank}.json"
            kwargs["verbose"] = 0
            self.tracer = viztracer.VizTracer(**kwargs)
            assert self.tracer
            if patch:
                args = [v for k, v in kwargs.items() for v in ("--" * min(2, len(k)) + k, v)]
                from viztracer.patch import install_all_hooks

                install_all_hooks(self.tracer, args)
            self.tracer.start()
        self._cb_logger: LightningLogger | None = None
        self.stage: str | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="VizTracer")

        signal.signal(signal.SIGTERM, self._terminate)  # terminate signal
        signal.signal(signal.SIGINT, self._terminate)  # keyboard interrupt
        atexit.register(self._terminate)

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._cb_logger = CallbackLogger(trainer)
        self.stage = stage

    def _on_batch_end(self, trainer: "L.Trainer", stage: str, batch_idx: int) -> None:
        if self.tracer and self.schedule.check(
            stage=stage, batch_idx=batch_idx, step=trainer.global_step, trainer=trainer
        ):
            self._publish(str(batch_idx))
            self.tracer.start()

    @override
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._on_batch_end(trainer, "train", batch_idx + 1)

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

    def _publish(self, suffix: str, sync: bool = False) -> None:
        assert self.tracer
        self.tracer.stop()
        # create report synchronously
        tmp_dir = tempfile.mkdtemp()
        output_stem = os.path.join(self.output_path_prefix or tmp_dir, suffix, f"rank{self.rank}")
        output_file = output_stem + (".json" if self.compressor else ".html")
        self.tracer.save(output_file=output_file, verbose=0)
        self.tracer.clear()
        # process report asynchronously
        artifact_path = f"viztracer/{self.stage}/{suffix}"
        if sync:
            self._process(tmp_dir, output_file, artifact_path)
        else:
            self.executor.submit(self._process, tmp_dir, output_file, artifact_path)

    def _process(self, tmp_dir: str, output_file: str, artifacts_path: str) -> None:
        if not self.compressor:
            with open(output_file, "rb") as f_in:
                output_file = output_file + ".gz"
                with gzip.open(output_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            with open(output_file) as f:
                data = json.load(f)
            output_file = os.path.splitext(output_file)[0] + ".cvf"
            self.compressor.compress(data, output_file)
        assert self._cb_logger
        self._cb_logger.log_artifact(output_file, artifacts_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    @override
    def teardown(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._terminate()

    @override
    def on_exception(self, trainer: "L.Trainer", pl_module: "L.LightningModule", exception: BaseException) -> None:
        self._terminate()

    def _terminate(self, *_: Any) -> None:
        if self.tracer and self.stage:
            # calling synchronously since this can be called during interpreter shutdown
            self._publish("last", sync=True)
            self.tracer.terminate()
            self.tracer = None
            self.executor.shutdown()

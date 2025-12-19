# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import pickle
import tempfile
from datetime import datetime, timezone
from typing import Any
from typing_extensions import override

import lightning as L
import torch
from torch.cuda import memory

from fkat.pytorch.schedule import (
    Schedule,
    Never,
)
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger

logger: logging.Logger = logging.getLogger(__name__)


def _artifact_path(root_dir: str, rank: int, file_type: str, ext: str) -> tuple[str, str]:
    base_dir = os.path.join(root_dir, "torch.cuda.memory")
    now = datetime.now(timezone.utc).isoformat()
    file_path = os.path.join(base_dir, f"rank{rank}/{file_type}/rank{rank}_{now}.{ext}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return base_dir, file_path


def _reset_recording(kwargs: dict[str, Any]) -> None:
    if torch.cuda.is_available():
        memory._record_memory_history(enabled=None)
        # set the limitation of ring buffer ~100 G. Otherwise, the buffer might be too large and trigger CPU OOM.
        kwargs.setdefault("max_entries", 1000000)
        memory._record_memory_history(**kwargs)


def _detect_tensor_cycles(cb_logger: CallbackLogger, rank: int) -> None:
    from torch.utils.viz import _cycles

    def is_cuda_tensor(obj: Any) -> bool:
        try:
            return (
                isinstance(obj, torch.Tensor)
                and obj.device.type == "cuda"
                and not isinstance(obj, torch._subclasses.FakeTensor)
            )
        except:  # noqa: E722
            return False

    _cycles.is_cuda_tensor = is_cuda_tensor  # type: ignore[invalid-assignment]

    def observer(garbage: Any) -> None:
        if garbage:
            if not any(_cycles.is_cuda_tensor(obj) for obj in garbage):
                logger.debug("No CUDA Tensors found in garbage")
                return
            logger.warning("Reference cycle includes a CUDA Tensor")
            with tempfile.TemporaryDirectory() as temp_dir:
                base_dir, html_path = _artifact_path(temp_dir, rank, "cycles", "html")
                logger.debug(f"Saving tensor cycles to {html_path}")
                with open(html_path, "wb") as f:
                    f.write(_cycles.to_html(_cycles.create_graph(garbage)))
                cb_logger.log_artifact(base_dir)

    _cycles.observe_garbage(observer)


class MemoryObserver(L.Callback):
    """This callback registers an observer to dump and log the CUDA memory snapshot.

    Args:
        oom: (bool): whether to dump memory snapshot on Out-of-Memory (OOM) event. Defaults to ``True``
        flamegraph (bool): whether to save memory snapshot in flamegraph format. Defaults to ``True``
        reset_memory_history (bool): whether to reset memory history after snapshot. Defaults to ``False``
        snapshot_pickle (bool): whether to dump memory snapshot in pickle format. Defaults to ``False``
        tensor_cycles (bool): whether to detect and dump graphs with cycles containing tensors in the garbage.
            Defaults to ``False``.
        schedule (Optional[Schedule]): Controls when logging occurs besides OOM event. Defaults to :class:`Never`
        **kwargs (Any): Arbitrary keyword arguments passed as is to ``memory._record_memory_history``.
    """

    def __init__(
        self,
        flamegraph: bool = True,
        reset_memory_history: bool = False,
        snapshot_pickle: bool = False,
        tensor_cycles: bool = False,
        schedule: Schedule | None = None,
        oom: bool = True,
        **kwargs: Any,
    ) -> None:
        self.flamegraph = flamegraph
        self.reset_memory_history = reset_memory_history
        self.snapshot_pickle = snapshot_pickle
        self.tensor_cycles = tensor_cycles
        self.schedule = schedule or Never()
        self.oom = oom
        self.kwargs = kwargs
        self._cb_logger: LightningLogger | None = None
        _reset_recording(kwargs)

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        if not torch.cuda.is_available():
            logger.warning("No CUDA device is available")
            return
        self._cb_logger = CallbackLogger(trainer)
        if self.tensor_cycles:
            _detect_tensor_cycles(self._cb_logger, trainer.global_rank)
        if self.oom:
            if hasattr(torch._C, "_cuda_attach_out_of_memory_observer"):

                def oom_observer_func(device: Any, alloc: Any, device_alloc: Any, device_free: Any) -> None:
                    logger.warning("OOM observer triggered")
                    return self.dump_memory_snapshot(trainer.global_rank)

                torch._C._cuda_attach_out_of_memory_observer(oom_observer_func)
                logger.info("OOM observer registered successfully")
            else:
                logger.warning(
                    f"Failed to register OOM observer because torch._C._cuda_attach_out_of_memory_observer "
                    f"is missing in torch=={torch.__version__}"
                )

    def maybe_dump_memory_snapshot(
        self, trainer: "L.Trainer", stage: str | None = None, batch_idx: int | None = None
    ) -> None:
        if not torch.cuda.is_available():
            return
        if self.schedule.check(stage="train", batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            self.dump_memory_snapshot(trainer.global_rank)

    def dump_memory_snapshot(self, rank: int) -> None:
        if not hasattr(memory, "_snapshot"):
            logger.warning(
                f"Failed to capture memory snapshot because memory._snapshot is missing in torch=={torch.__version__}"
            )
            return
        now = datetime.now(timezone.utc).isoformat()
        logger.debug(f"Capturing memory snapshot on rank {rank} at {now}")
        snapshot = memory._snapshot()
        if self.reset_memory_history:
            _reset_recording(self.kwargs)
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir: str | None = None
            if self.snapshot_pickle:
                base_dir, snapshot_path = _artifact_path(temp_dir, rank, "snapshot", "pickle")
                logger.debug(f"Saving memory snapshot to {snapshot_path}")
                with open(snapshot_path, "wb") as f:
                    pickle.dump(snapshot, f)
            if self.flamegraph:
                if hasattr(torch.cuda, "_memory_viz"):
                    flamegraph = torch.cuda._memory_viz.memory(snapshot)
                    base_dir, flamegraph_path = _artifact_path(temp_dir, rank, "flamegraph", "svg")
                    logger.debug(f"Saving memory flamegraph to {flamegraph_path}")
                    with open(flamegraph_path, "w") as f:
                        print(flamegraph, file=f)
                else:
                    logger.warning(
                        f"Failed to create flamegraph because torch.cuda._memory_viz "
                        f"is missing in torch=={torch.__version__}"
                    )
            if base_dir is not None:
                logger.debug(f"Logging memory snapshot files with {self._cb_logger}")
                assert self._cb_logger
                self._cb_logger.log_artifact(base_dir)
        logger.debug("Finished capturing memory snapshot")

    @override
    def on_train_batch_start(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.maybe_dump_memory_snapshot(trainer, stage="train", batch_idx=batch_idx)

    @override
    def on_test_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.maybe_dump_memory_snapshot(trainer, stage="test", batch_idx=batch_idx)

    @override
    def on_validation_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.maybe_dump_memory_snapshot(trainer, stage="validation", batch_idx=batch_idx)

    @override
    def on_predict_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.maybe_dump_memory_snapshot(trainer, stage="predict", batch_idx=batch_idx)

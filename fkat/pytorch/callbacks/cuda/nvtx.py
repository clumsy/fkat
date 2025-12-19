# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Any, TYPE_CHECKING
from typing_extensions import override
import inspect

import lightning as L
import torch

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

try:
    import nvtx
except ImportError:
    from torch.cuda import nvtx

    _mark = nvtx.mark

    def _conditional_mark(message: str, *args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(_mark)
        filtered_kwargs = {}

        if "domain" in kwargs and "color" not in kwargs:
            kwargs["color"] = DOMAIN_COLORS[kwargs["domain"]]

        for param in ["color", "domain"]:
            if param in sig.parameters and param in kwargs:
                filtered_kwargs[param] = kwargs[param]
        return _mark(message, **filtered_kwargs)

    nvtx.mark = _conditional_mark  # type: ignore[invalid-assignment]


class Domain(str, Enum):
    INIT = "init"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"
    TUNE = "tune"
    ERROR = "error"
    CHECKPOINT = "checkpoint"

    @staticmethod
    def from_stage(s: str) -> "Domain":
        if s == "fit" or s == "train":
            return Domain.TRAIN
        if s == "validation":
            return Domain.VALIDATION
        if s == "test":
            return Domain.TEST
        if s == "predict":
            return Domain.PREDICT
        if s == "tune":
            return Domain.TUNE
        raise NotImplementedError(f"Unsupported stage: {s}")


DOMAIN_COLORS = {
    Domain.INIT: "white",
    Domain.TUNE: "pink",
    Domain.TRAIN: "green",
    Domain.VALIDATION: "blue",
    Domain.TEST: "purple",
    Domain.PREDICT: "yellow",
    Domain.ERROR: "red",
    Domain.CHECKPOINT: "orange",
}


class Nvtx(L.Callback):
    def __init__(self) -> None:
        nvtx.mark("__init__()", domain=Domain.INIT)  # type: ignore[unknown-argument]

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        domain = Domain.from_stage(stage)
        nvtx.mark(f"setup(stage={stage})", domain=domain)  # type: ignore[unknown-argument]

    @override
    def teardown(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        domain = Domain.from_stage(stage)
        nvtx.mark(f"teardown(stage={stage})", domain=domain)  # type: ignore[unknown-argument]

    @override
    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_train_start()", domain=Domain.TRAIN)  # type: ignore[unknown-argument]

    @override
    def on_train_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_train_epoch_start()", domain=Domain.TRAIN)  # type: ignore[unknown-argument]

    @override
    def on_train_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        nvtx.mark(
            f"on_train_batch_start(batch_idx={batch_idx})",
            domain=Domain.TRAIN,  # type: ignore[unknown-argument]
        )

    @override
    def on_before_zero_grad(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", optimizer: "torch.optim.Optimizer"
    ) -> None:
        nvtx.mark("on_before_zero_grad()", domain=Domain.TRAIN)  # type: ignore[unknown-argument]

    @override
    def on_before_backward(self, trainer: "L.Trainer", pl_module: "L.LightningModule", loss: "torch.Tensor") -> None:
        nvtx.mark("on_before_backward()", domain=Domain.TRAIN)  # type: ignore[unknown-argument]

    @override
    def on_after_backward(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_after_backward()", domain=Domain.TRAIN)  # type: ignore[unknown-argument]

    @override
    def on_before_optimizer_step(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", optimizer: "torch.optim.Optimizer"
    ) -> None:
        nvtx.mark("on_before_optimizer_step()", domain=Domain.TRAIN)  # type: ignore[unknown-argument]

    @override
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        nvtx.mark(f"on_train_batch_end(batch_idx={batch_idx})", domain=Domain.TRAIN)  # type: ignore[unknown-argument]

    @override
    def on_train_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_train_epoch_end()", domain=Domain.TRAIN)  # type: ignore[unknown-argument]

    @override
    def on_train_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_train_end()", domain=Domain.TRAIN)  # type: ignore[unknown-argument]

    @override
    def on_sanity_check_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_validation_start()", domain=Domain.VALIDATION)  # type: ignore[unknown-argument]

    def on_sanity_check_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_sanity_check_start()", domain=Domain.VALIDATION)  # type: ignore[unknown-argument]

    @override
    def on_validation_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_sanity_check_end()", domain=Domain.VALIDATION)  # type: ignore[unknown-argument]

    @override
    def on_validation_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_validation_epoch_start()", domain=Domain.VALIDATION)  # type: ignore[unknown-argument]

    @override
    def on_validation_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        nvtx.mark(
            f"on_validation_batch_start(batch_idx={batch_idx})",
            domain=Domain.VALIDATION,  # type: ignore[unknown-argument]
        )

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
        nvtx.mark(
            f"on_validation_batch_end(batch_idx={batch_idx})",
            domain=Domain.VALIDATION,  # type: ignore[unknown-argument]
        )

    @override
    def on_validation_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_validation_epoch_end()", domain=Domain.VALIDATION)  # type: ignore[unknown-argument]

    @override
    def on_validation_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_validation_end()", domain=Domain.VALIDATION)  # type: ignore[unknown-argument]

    @override
    def on_test_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_test_start()", domain=Domain.TEST)  # type: ignore[unknown-argument]

    @override
    def on_test_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_test_epoch_start()", domain=Domain.TEST)  # type: ignore[unknown-argument]

    @override
    def on_test_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        nvtx.mark(f"on_test_batch_start(batch_idx={batch_idx})", domain=Domain.TEST)  # type: ignore[unknown-argument]

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
        nvtx.mark(f"on_test_batch_end(batch_idx={batch_idx})", domain=Domain.TEST)  # type: ignore[unknown-argument]

    @override
    def on_test_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_test_epoch_end()", domain=Domain.TEST)  # type: ignore[unknown-argument]

    @override
    def on_test_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_test_end()", domain=Domain.TEST)  # type: ignore[unknown-argument]

    @override
    def on_predict_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_predict_start()", domain=Domain.PREDICT)  # type: ignore[unknown-argument]

    @override
    def on_predict_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_predict_epoch_start()", domain=Domain.PREDICT)  # type: ignore[unknown-argument]

    @override
    def on_predict_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        nvtx.mark(
            f"on_predict_batch_start(batch_idx={batch_idx})",
            domain=Domain.PREDICT,  # type: ignore[unknown-argument]
        )

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
        nvtx.mark(
            f"on_predict_batch_end(batch_idx={batch_idx})",
            domain=Domain.PREDICT,  # type: ignore[unknown-argument]
        )

    @override
    def on_predict_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_predict_epoch_end()", domain=Domain.PREDICT)  # type: ignore[unknown-argument]

    @override
    def on_predict_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        nvtx.mark("on_predict_end()", domain=Domain.PREDICT)  # type: ignore[unknown-argument]

    @override
    def state_dict(self) -> dict[str, Any]:
        nvtx.mark("state_dict()", domain=Domain.CHECKPOINT)  # type: ignore[unknown-argument]
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        nvtx.mark("load_state_dict()", domain=Domain.CHECKPOINT)  # type: ignore[unknown-argument]

    @override
    def on_save_checkpoint(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        nvtx.mark("on_save_checkpoint()", domain=Domain.CHECKPOINT)  # type: ignore[unknown-argument]

    @override
    def on_load_checkpoint(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        nvtx.mark("on_load_checkpoint()", domain=Domain.CHECKPOINT)  # type: ignore[unknown-argument]

    @override
    def on_exception(self, trainer: "L.Trainer", pl_module: "L.LightningModule", exception: BaseException) -> None:
        nvtx.mark(f"on_exception({type(exception)})", domain=Domain.ERROR)  # type: ignore[unknown-argument]

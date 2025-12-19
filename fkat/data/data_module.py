# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from functools import partial
from typing import Any
from collections.abc import Callable, Iterable

import lightning as L
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.core.hooks import CheckpointHooks
from torch.utils.data import DataLoader
from typing_extensions import Protocol, override, runtime_checkable

from fkat.utils.rng import get_rng_states, set_rng_states
from fkat.utils.profiler import profile_until_exit

logger = logging.getLogger(__name__)


def _call(dataloader: Iterable[Any] | None, attr: str, *args: Any, **kwargs: Any) -> None:
    if dataloader is None:
        return
    for obj in (
        dataloader,
        getattr(dataloader, "dataset", None),
        getattr(dataloader, "sampler", None),
        getattr(dataloader, "batch_sampler", None),
    ):
        if not obj:
            continue
        if impl := getattr(obj, attr, None):
            impl(*args, **kwargs)


def worker_init_fn(
    profiler: Profiler,
    stage: str,
    init_fn: Callable[[int], Any] | None,
    worker_id: int,
) -> None:
    action = f"DataWorker[{stage}][{worker_id}]"
    profile_until_exit(profiler, action=action, filename_suffix=f"_{stage}_{worker_id}")

    if init_fn is not None:
        init_fn(worker_id)

    # TODO: Add Dataloader worker have consistent seed based on worker_id
    # Reference: https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359


def instrument(
    cfg: dict[str, Any],
    profiler: Profiler | None,
    stage: str,
) -> dict[str, Any]:
    if not profiler:
        return cfg
    cfg["worker_init_fn"] = partial(worker_init_fn, profiler, stage, cfg.get("worker_init_fn"))
    return cfg


@runtime_checkable
class PersistStates(Protocol):
    def state_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class RestoreStates(Protocol):
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class DataModule(L.LightningDataModule, CheckpointHooks):
    """A :class:`LightningDataModule` that manages multiple :class:`DataLoader`\\s for different stages.

    Args:
        dataloaders (dict[str, dict[str, Any] | Callable[[], Iterable[Any]]]): Dataloaders for different stages.
        profiler (Profiler | None): Profiler instance for worker initialization.
    """

    SUPPORTED_STAGES = ("train", "test", "val", "predict")

    def __init__(
        self,
        dataloaders: dict[str, dict[str, Any] | Callable[[], Iterable[Any]]],
        profiler: Profiler | None = None,
    ) -> None:
        super().__init__()
        self.profiler = profiler
        self.dataloader_factory: dict[str, Callable[[], Iterable[Any]]] = {}
        for stage, cfg in dataloaders.items():
            if stage not in DataModule.SUPPORTED_STAGES:
                raise ValueError(f"Unsupported stage {stage}, use one of {DataModule.SUPPORTED_STAGES}")
            dataloader_factory: Callable[[], Iterable[Any]]
            if isinstance(cfg, dict):
                cfg = instrument(cfg, profiler, stage)  # type: ignore[arg-type]
                dataloader_factory = partial(DataLoader, **cfg)
            else:
                dataloader_factory = cfg
            self.dataloader_factory[stage] = dataloader_factory
        self.dataloaders: dict[str, Iterable[Any] | None] = {}

    def _new_dataloader(self, stage: str) -> Iterable[Any] | None:
        dataloader_factory = self.dataloader_factory.get(stage, lambda: None)
        self.dataloaders[stage] = (dataloader := dataloader_factory())
        return dataloader

    @override
    def prepare_data(self) -> None:
        for dataloader in self.dataloaders.values():
            _call(dataloader, "prepare_data")

    def _dataloader(self, stage: str) -> Iterable[Any] | None:
        stage = "train" if stage == "fit" else "val" if stage == "validation" else stage
        return self.dataloaders.get(stage)

    @override
    def setup(self, stage: str) -> None:
        device = self.trainer and self.trainer.strategy and self.trainer.strategy.root_device
        _call(self._dataloader(stage), "set_device", device)
        _call(self._dataloader(stage), "setup", stage)

    # will be used once https://github.com/Lightning-AI/pytorch-lightning/pull/19601 is in effect
    def on_exception(self, exception: BaseException) -> None:
        for dataloader in self.dataloaders.values():
            _call(dataloader, "on_exception", exception)

    @override
    def teardown(self, stage: str | None) -> None:
        # this is it, terminating everything regardless of which stage we received this
        for dataloader in self.dataloaders.values():
            _call(dataloader, "teardown", stage)

    @override
    def train_dataloader(self) -> Iterable[Any] | None:
        return self._new_dataloader("train")

    @override
    def val_dataloader(self) -> Iterable[Any] | None:
        return self._new_dataloader("val")

    @override
    def predict_dataloader(self) -> Iterable[Any] | None:
        return self._new_dataloader("predict")

    @override
    def test_dataloader(self) -> Iterable[Any] | None:
        return self._new_dataloader("test")

    @override
    def state_dict(self) -> dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state for ShardedDataLoader.

        This method iterates over each stage's dataloader to retrieve its state using the `state_dict()` method,
        and saves it along with the RNG states. If a dataloader does not implement the `PersistStates` protocol,
        it sets its `state_dict` attribute to `vanilla_dataloader_state_dict` to allow saving its state.

        Returns:
            dict[str, Any]: A dictionary containing the dataloader states and RNG states.
        """
        dataloader_states = {}
        for stage, dataloader in self.dataloaders.items():
            dataloader_states[stage] = get_rng_states()
            if isinstance(dataloader, PersistStates):
                try:
                    result_dict = dataloader.state_dict()
                    dataloader_states[stage].update(result_dict)
                except Exception as e:
                    logger.warning(f"{dataloader} states can't be persisted yet: {e}.")
        return dataloader_states

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint to reload the DataModule's state for a ShardedDataLoader.

        This method iterates over each stage's dataloader, loads its state from the provided `state_dict`,
        and sets the RNG states. If a dataloader does not implement the `RestoreStates` protocol,
        it sets its `load_state_dict` attribute to `vanilla_dataloader_load_state_dict` to allow loading its state.

        Args:
            state_dict (Dict[str, Any]): A dictionary containing the dataloader states and RNG states.
        """
        for stage, dataloader in self.dataloaders.items():
            set_rng_states(state_dict[stage])
            if isinstance(dataloader, RestoreStates):
                try:
                    dataloader.load_state_dict(state_dict[stage])
                except Exception as e:
                    logger.warning(f"{dataloader} states can't be restored yet: {e}.")

    @override
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        for dataloader in self.dataloaders.values():
            _call(dataloader, "on_save_checkpoint", checkpoint)

    @override
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        for dataloader in self.dataloaders.values():
            _call(dataloader, "on_load_checkpoint", checkpoint)

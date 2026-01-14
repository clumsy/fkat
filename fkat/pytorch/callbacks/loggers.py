# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from mlflow.entities import Metric, RunTag, Param
from mlflow.tracking import MlflowClient  # type: ignore[possibly-unbound-import]

if TYPE_CHECKING:
    pass

from fkat.pytorch.loggers import LightningLogger, _is_logger_type
from fkat.utils import assert_not_none
from fkat.utils.logging import rank0_logger
from fkat.utils.mlflow import broadcast_mlflow_run_id, mlflow_logger

log = rank0_logger(__name__)


class MLFlowCallbackLogger(LightningLogger):
    """
    Mlflow logger class that supports distributed logging of tags, metrics and artifacts.

    Args:
        trainer (L.Trainer): PTL trainer object
    """

    def __init__(
        self,
        trainer: "L.Trainer | None" = None,
        client: MlflowClient | None = None,
        synchronous: bool | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__()
        if trainer:
            # Initialize logger and broadcast run_ids to all ranks
            logger = assert_not_none(mlflow_logger(trainer))
            broadcast_mlflow_run_id(logger, trainer)  # type: ignore[arg-type]
            # Set client and run_id
            self._client: MlflowClient = assert_not_none(getattr(logger, "_mlflow_client", None))
            self._synchronous = getattr(logger, "_log_batch_kwargs", {}).get("synchronous")
            self._run_id: str = assert_not_none(getattr(logger, "_run_id", None))
        else:
            assert client
            self._client = client
            self._synchronous = synchronous
            assert run_id
            self._run_id = run_id

    def log_tag(self, key: str, value: str) -> None:
        self._client.set_tag(run_id=self._run_id, key=key, value=value, synchronous=self._synchronous)

    def tags(self) -> dict[str, Any]:
        run = self._client.get_run(self._run_id)
        return run.data.tags

    def log_batch(
        self,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        ms = [Metric(k, v, timestamp, step) for k, v in metrics.items()] if metrics else []
        ps = [Param(k, v) for k, v in params.items()] if params else []
        ts = [RunTag(k, v) for k, v in tags.items()] if tags else []
        self._client.log_batch(
            run_id=self._run_id,
            metrics=ms,
            params=ps,
            tags=ts,
            synchronous=self._synchronous,
        )

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        # TODO: log directly to s3 uri
        # TODO: support async logging
        self._client.log_artifact(
            run_id=self._run_id,
            local_path=local_path,
            artifact_path=artifact_path,
        )


class TensorBoardCallbackLogger(LightningLogger):
    """TensorBoard logger for distributed logging."""

    def __init__(self, logger: TensorBoardLogger) -> None:
        self._logger = logger

    def log_tag(self, key: str, value: str) -> None:
        self._logger.experiment.add_text(key, value)

    def tags(self) -> dict[str, Any]:
        return {}

    def log_batch(
        self,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        if metrics:
            for k, v in metrics.items():
                self._logger.experiment.add_scalar(k, v, step)
        if tags:
            for k, v in tags.items():
                self._logger.experiment.add_text(k, v, step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        pass


class WandbCallbackLogger(LightningLogger):
    """WandB logger for distributed logging."""

    def __init__(self, logger: WandbLogger) -> None:
        self._logger = logger

    def log_tag(self, key: str, value: str) -> None:
        self._logger.experiment.config.update({key: value})

    def tags(self) -> dict[str, Any]:
        return dict(self._logger.experiment.config)

    def log_batch(
        self,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        log_dict = {}
        if metrics:
            log_dict.update(metrics)
        if tags:
            log_dict.update(tags)
        if log_dict:
            self._logger.experiment.log(log_dict, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self._logger.experiment.save(local_path)


class CallbackLogger(LightningLogger):
    """
    A wrapper on top of the collection of Logger instances,
    providing methods to log metrics, artifacts, and tags across all registered loggers
    simultaneously.

    Attributes:
        loggers (list[LightningLogger]): List of loggers

    Args:
        trainer (L.Trainer): PyTorch Lightning trainer instance used to initialize loggers
    """

    loggers: list[LightningLogger]

    def __init__(self, trainer: "L.Trainer | None", loggers: list[LightningLogger] | None = None) -> None:
        if trainer:
            self.loggers = []
            for logger in trainer.loggers:
                if _is_logger_type(logger, "MLFlowLogger"):
                    self.loggers.append(MLFlowCallbackLogger(trainer=trainer))
                elif _is_logger_type(logger, "TensorBoardLogger"):
                    self.loggers.append(TensorBoardCallbackLogger(logger=logger))  # type: ignore[arg-type]
                elif _is_logger_type(logger, "WandbLogger"):
                    self.loggers.append(WandbCallbackLogger(logger=logger))  # type: ignore[arg-type]
        else:
            assert loggers
            self.loggers = loggers

    def __str__(self) -> str:
        return str([type(obj).__name__ for obj in self.loggers])

    @override
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        for logger in self.loggers:
            logger.log_artifact(local_path=local_path, artifact_path=artifact_path)

    @override
    def log_batch(
        self,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        for logger in self.loggers:
            logger.log_batch(metrics=metrics, tags=tags, timestamp=timestamp, step=step)

    @override
    def tags(self) -> dict[str, Any]:
        tags = {}
        for logger in self.loggers:
            tags.update(logger.tags())
        return tags

    @override
    def log_tag(self, key: str, value: str) -> None:
        for logger in self.loggers:
            logger.log_tag(key=key, value=value)

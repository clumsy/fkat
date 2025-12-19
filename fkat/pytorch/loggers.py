# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Protocol, TYPE_CHECKING
from typing_extensions import override

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only
from mlflow.entities import Metric, RunTag, Param
from mlflow.tracking import MlflowClient  # type: ignore[possibly-unbound-import]

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger

from fkat.utils import assert_not_none
from fkat.utils.mlflow import mlflow_logger


class LightningLogger(Protocol):
    """
    Protocol defining the interface for logging that handle metrics, tags, and artifacts.
    """

    def tags(self) -> dict[str, Any]:
        """Get current tags"""
        ...

    def log_tag(self, key: str, value: str) -> None:
        """
        Log a single key-value tag.

        Args:
            key (str): The identifier/name of the tag
            value (str): The value associated with the tag
        """
        ...

    def log_batch(
        self,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        """
        Log multiple metrics and/or tags in a single batch operation.

        Args:
            metrics (dict[str, float], optional): Dictionary mapping metric names to their float values
            params (dict[str, Any], optional): Dictionary mapping params names to their values
            tags (dict[str, str], optional): Dictionary mapping tag names to their string values
            timestamp (int, optional): Unix timestamp for when the batch was logged
            step (int, optional): Training step or iteration number
        """
        ...

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """
        Log a local file as an artifact.

        Args:
            local_path (str): Path to the file on the local filesystem to be logged
            artifact_path (str, optional): Remote path where the artifact should be stored
                If None, a default location should be used
        """
        ...


class MLFlowLogger:
    """
    Mlflow logger class that supports rank_zero logging of tags, metrics and distributed
    logging of artifacts.

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

    @rank_zero_only
    def log_tag(self, key: str, value: str) -> None:
        self._client.set_tag(run_id=self._run_id, key=key, value=value, synchronous=self._synchronous)

    def tags(self) -> dict[str, Any]:
        run = self._client.get_run(self._run_id)
        return run.data.tags

    @rank_zero_only
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


class CompositeLogger(LightningLogger):
    """
    A wrapper on top of the collection of :class:`Logger` instances,
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
            for logger in trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]:
                if isinstance(logger, MLFlowLogger):
                    self.loggers.append(MLFlowLogger(trainer=trainer))  # type: ignore[arg-type]
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

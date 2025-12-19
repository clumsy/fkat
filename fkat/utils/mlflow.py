# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import TYPE_CHECKING

from lightning.pytorch.loggers import MLFlowLogger

if TYPE_CHECKING:
    from lightning import Trainer

log = logging.getLogger(__name__)


def mlflow_logger(trainer: "Trainer") -> "MLFlowLogger | None":
    """
    Returns MLFlowLogger from trainer as constructed by PyTorch Lightning.
    """
    loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
    for logger in loggers:
        if isinstance(logger, MLFlowLogger):
            return logger
    return None


def broadcast_mlflow_run_id(mlflow: "MLFlowLogger", trainer: "Trainer") -> None:
    """
    Broadcast mlflow run_id from rank0 to all ranks and setup the mlflow logger.
    We assume PTL mlflow logger is only initialized with a run_id on rank0 via
    logger.experiment.
    """
    mlflow._run_id = trainer.strategy.broadcast(mlflow.run_id, src=0)
    log.debug(f"Received mlflow run_id: {mlflow.run_id}")

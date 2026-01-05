# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
from typing_extensions import override

import fsspec
import lightning as L
from lightning.pytorch.utilities import rank_zero_only

from fkat.utils.logging import rank0_logger
from fkat.pytorch.callbacks.loggers import CallbackLogger

logger = rank0_logger(__name__)


class ValidationMetrics(L.Callback):
    """
    Saves validation metrics after each validation epoch.

    This callback persists validation metrics in JSON format, creating both
    a versioned file (with epoch and step) and a "latest" file for easy access.

    Args:
        output_path (str | None): Directory path where metrics will be saved.
            Supports any fsspec-compatible filesystem (local, s3://, gcs://, etc.).
            If None, logs to MLflow artifacts. Defaults to None.

    Example:
        >>> # MLflow artifacts (default)
        >>> callback = ValidationMetrics()
        >>> # Local storage
        >>> callback = ValidationMetrics(output_path="/tmp/metrics")
        >>> # S3 storage
        >>> callback = ValidationMetrics(output_path="s3://my-bucket/metrics")
        >>> trainer = L.Trainer(callbacks=[callback])
    """

    def __init__(self, output_path: str | None = None) -> None:
        self.output_path = output_path
        self._cb_logger: CallbackLogger | None = None

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        if self.output_path is None:
            self._cb_logger = CallbackLogger(trainer)

    @override
    @rank_zero_only
    def on_validation_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if trainer.sanity_checking:
            logger.info("Skipping validation metrics save during sanity checking")
            return

        # Extract metrics
        metrics_dict = {k: v.item() for k, v in trainer.logged_metrics.items()}
        metrics_json = json.dumps(metrics_dict)

        # Filenames
        versioned_name = f"validation-metrics-epoch={pl_module.current_epoch}-step={pl_module.global_step}.json"
        latest_name = "validation-metrics-latest.json"

        if self.output_path is None:
            # Log to MLflow artifacts
            logger.info("Saving validation metrics to MLflow artifacts")
            with tempfile.TemporaryDirectory() as tmpdir:
                versioned_file = f"{tmpdir}/{versioned_name}"
                latest_file = f"{tmpdir}/{latest_name}"

                with open(versioned_file, "w") as f:
                    f.write(metrics_json)
                with open(latest_file, "w") as f:
                    f.write(metrics_json)

                if self._cb_logger:
                    self._cb_logger.log_artifact(versioned_file, "validation_metrics")
                    self._cb_logger.log_artifact(latest_file, "validation_metrics")

            logger.info("Validation metrics saved to MLflow artifacts")
        else:
            # Use fsspec for filesystem
            logger.info(f"Saving validation metrics to {self.output_path}")
            fs, _, paths = fsspec.get_fs_token_paths(self.output_path)
            base_path = paths[0] if paths else self.output_path

            versioned_file = f"{base_path}/{versioned_name}"
            latest_file = f"{base_path}/{latest_name}"

            with fs.open(versioned_file, "w") as f:
                f.write(metrics_json)
            with fs.open(latest_file, "w") as f:
                f.write(metrics_json)

            logger.info(f"Validation metrics saved to {versioned_file} and {latest_file}")

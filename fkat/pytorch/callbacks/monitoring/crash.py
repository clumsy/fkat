# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import logging
import traceback
import multiprocessing
import datetime as dt
import tempfile
from pathlib import Path
from typing_extensions import override

import lightning as L
from lightning.pytorch.utilities import rank_zero_only

from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger

log = logging.getLogger(__name__)


def _monitor_process(queue: multiprocessing.Queue, parent_pid: int, rank: int) -> None:
    """Monitor parent process and report crash info."""
    try:
        _, status = os.waitpid(parent_pid, 0)
        exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
        signal_num = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None

        crash_info = {
            "pid": parent_pid,
            "rank": rank,
            "exit_code": exit_code,
            "signal": signal_num,
            "timestamp": str(dt.datetime.now(dt.timezone.utc)),
        }
        queue.put(crash_info)
    except Exception as e:
        log.error(f"Error monitoring process {parent_pid}: {e}")


class CrashDetector(L.Callback):
    """
    Detects process crashes and logs detailed error information.

    Monitors the training process and any spawned subprocesses for crashes.
    Captures PID, rank, error details, and stack traces, logging them to
    the configured Lightning logger.

    Args:
        error_tag: Tag for error messages (default: "error")
        crash_info_tag: Tag for crash details (default: "crash_info")

    Example:
        >>> from fkat.pytorch.callbacks.monitoring import CrashDetector
        >>> callback = CrashDetector()
        >>> trainer = L.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        error_tag: str = "error",
        crash_info_tag: str = "crash_info",
    ) -> None:
        self.error_tag = error_tag
        self.crash_info_tag = crash_info_tag
        self._cb_logger: LightningLogger | None = None
        self._processes: list[multiprocessing.Process] = []
        self._queue: multiprocessing.Queue | None = None

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        """Initialize crash detection."""
        if trainer.local_rank == 0:
            self._cb_logger = CallbackLogger(trainer)
            self._queue = multiprocessing.Queue()

            # Monitor main process
            process = multiprocessing.Process(
                target=_monitor_process, args=(self._queue, os.getpid(), trainer.global_rank)
            )
            process.daemon = True
            process.start()
            self._processes.append(process)

    @override
    @rank_zero_only
    def on_exception(self, trainer: "L.Trainer", pl_module: "L.LightningModule", exception: BaseException) -> None:
        """Log exception details."""
        if not self._cb_logger:
            return

        exc_type = type(exception)
        stacktrace = "".join(traceback.format_exception(exc_type, exception, exception.__traceback__))

        error_msg = f"[{exc_type.__name__}]: {exception}"
        crash_info = {
            "pid": os.getpid(),
            "rank": trainer.global_rank,
            "error": error_msg,
            "stacktrace": stacktrace,
            "timestamp": str(dt.datetime.now(dt.timezone.utc)),
        }

        log.error(f"Exception: {error_msg}\n{stacktrace}")
        self._cb_logger.log_tag(self.error_tag, error_msg)
        self._cb_logger.log_tag(self.crash_info_tag, str(crash_info))

        # Log to MLflow artifacts if available
        self._log_to_mlflow_artifact(trainer, crash_info)

    @override
    def teardown(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        """Check for crashes and cleanup."""
        if trainer.global_rank == 0 and self._queue:
            # Check for any crash reports
            while not self._queue.empty():
                crash_info = self._queue.get_nowait()
                log.warning(f"Detected crash: {crash_info}")
                if self._cb_logger:
                    self._cb_logger.log_tag(self.crash_info_tag, str(crash_info))
                    self._log_to_mlflow_artifact(trainer, crash_info)

        self._terminate_monitors()

    def _log_to_mlflow_artifact(self, trainer: "L.Trainer", crash_info: dict) -> None:
        """Log crash info to MLflow artifacts."""
        try:
            from lightning.pytorch.loggers import MLFlowLogger

            mlflow_logger = next((logger for logger in trainer.loggers if isinstance(logger, MLFlowLogger)), None)
            if not mlflow_logger:
                return

            # Create filename: rank0-timestamp.txt
            rank = crash_info.get("rank", 0)
            timestamp = crash_info.get("timestamp", "unknown")
            # Convert timestamp to filename-safe format
            timestamp_safe = timestamp.replace(" ", "_").replace(":", "-")
            filename = f"rank{rank}-{timestamp_safe}.txt"

            # Create temp file with crash info
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / filename

            with open(temp_path, "w") as f:
                f.write("Crash Information\n")
                f.write("=" * 80 + "\n\n")
                for key, value in crash_info.items():
                    f.write(f"{key}: {value}\n")

            # Log as artifact
            mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(temp_path), "crashes")
            temp_path.unlink()
            log.info(f"Logged crash info to MLflow artifacts: {filename}")
        except Exception as e:
            log.warning(f"Failed to log crash info to MLflow: {e}")

    def _terminate_monitors(self) -> None:
        """Terminate all monitoring processes."""
        for process in self._processes:
            if process.is_alive():
                process.kill()
        self._processes.clear()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import lightning as L
from typing_extensions import override

from fkat.pytorch.schedule import Schedule
from fkat.pytorch.callbacks.loggers import CallbackLogger

logger: logging.Logger = logging.getLogger(__name__)


class HasTag(Schedule):
    """
    A schedule that activates only when a specific MLflow tag is present AND the trigger schedule is satisfied.

    This schedule combines another schedule (the trigger schedule) with MLflow tag validation.
    It allows callbacks to be dynamically enabled or disabled through experiment configuration rather than
    code changes, which is particularly useful for performance-intensive callbacks like FLOP measurement
    or detailed logging that should only run conditionally.

    The schedule checks two conditions:
    1. If the trigger schedule is satisfied
    2. If the specified MLflow tag exists in the current MLflow run

    Both conditions must be true for the schedule to activate. If the trigger schedule doesn't activate or
    the trainer is not provided, the schedule will never activate.

    Note:
        - The trainer can be optionally provided to the ``check`` method for MLflow tag validation. If trainer is None,
          the schedule will never activate.
        - Tag checking occurs only when the trigger schedule condition is already satisfied,
          minimizing MLflow API calls.
        - If an exception occurs during tag checking, it will be logged and the schedule will not activate.

    Example:
        Python code example::

            # Create a schedule that checks every 5 batches if the 'enable_flops' tag exists
            from fkat.pytorch.schedule import Every

            trigger = Every(n_batches=5)
            flops_schedule = HasTag(tag="enable_flops", schedule=trigger)
            flops_callback = Flops(schedule=flops_schedule)
            trainer = L.Trainer(callbacks=[flops_callback])

        Hydra configuration example:

        .. code-block:: yaml

            # In your config.yaml file
            callbacks:
              - _target_: fkat.pytorch.callbacks.profiling.Flops
                schedule:
                  _target_: fkat.pytorch.schedule.mlflow.HasTag
                  tag: ENABLE_FLOPS
                  schedule:
                    _target_: fkat.pytorch.schedule.Every
                    n_steps: 20

            # Another example using Fixed schedule
            callbacks:
              - _target_: fkat.pytorch.callbacks.heartbeat.Heartbeat
                schedule:
                  _target_: fkat.pytorch.schedule.mlflow.HasTag
                  tag: ENABLE_HEARTBEAT
                  schedule:
                    _target_: fkat.pytorch.schedule.Fixed
                    warmup_steps: 100
                    active_steps: 1000

            # Example with Elapsed time-based schedule
            callbacks:
              - _target_: fkat.pytorch.callbacks.custom_logging.DetailedMetrics
                schedule:
                  _target_: fkat.pytorch.schedule.mlflow.HasTag
                  tag: DETAILED_LOGGING
                  schedule:
                    _target_: fkat.pytorch.schedule.Elapsed
                    interval: ${timedelta:minutes=15}
    """

    def __init__(self, tag: str, schedule: Schedule) -> None:
        """
        Initialize a new MLflow HasTag schedule.

        Args:
            tag (str): The name of the tag that must be present in the MLflow run.
            schedule (Schedule): The schedule that determines when to check for the tag.
                This can be any implementation of the Schedule protocol (e.g., Every, Fixed, Elapsed).
        """
        self._tag: str = tag
        self._schedule: Schedule = schedule
        self._cb_logger: CallbackLogger | None = None

    @override
    def check(
        self,
        *,
        stage: str | None = None,
        batch_idx: int | None = None,
        step: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> bool:
        """
        Check if the schedule should activate based on the trigger schedule and MLflow tag presence.

        This method first checks if the trigger schedule is satisfied.
        If this condition is met, it then checks if the specified MLflow tag exists in the current run.
        Both conditions must be true for the method to return True.

        Args:
            stage (str, optional): Current training stage (e.g., "train", "validate", "test").
                Passed to the trigger schedule.
            batch_idx (int, optional): Current batch index within the epoch.
                Passed to the trigger schedule.
            step (int, optional): Current global step (cumulative across epochs).
                Passed to the trigger schedule.
            trainer (L.Trainer, optional): The Lightning Trainer instance.
                Required for MLflow tag validation.

        Returns:
            bool: True if both the trigger schedule is satisfied AND the specified tag is present,
                  False otherwise.

        Note:
            - The trainer must be provided for MLflow tag validation.
            - Tag checking occurs only when the trigger schedule is already satisfied.
            - If an exception occurs during tag checking, it will be logged as a warning
              and the method will return False.
        """
        triggered = self._schedule.check(stage=stage, batch_idx=batch_idx, step=step, trainer=trainer)
        if not triggered or trainer is None:
            return False
        try:
            if self._cb_logger is None:
                self._cb_logger = CallbackLogger(trainer)

            tags = self._cb_logger.tags()
            return self._tag in tags
        except Exception as e:
            logger.warning(f"Error when checking if tag {self._tag} exists: {e}")
            return False

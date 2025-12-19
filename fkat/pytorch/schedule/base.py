# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime as dt
import operator
from collections.abc import Callable, Sequence
from functools import reduce
from typing import Protocol
from typing_extensions import override

import lightning as L


class Schedule(Protocol):
    """
    Protocol defining a generic PyTorch schedule
    """

    def check(
        self,
        *,
        stage: str | None = None,
        batch_idx: int | None = None,
        step: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> bool:
        """Checks the schedule for the given moment.

        Args:
            stage (str): current trainer stage. eg. train/test/validate/predict
            batch_idx (Optional[int]): current batch_idx
            step (Optional[int]): current step
            trainer (Optional[Trainer]): lightning trainer of callback
        Returns:
            bool: True if schedule passed the check, False otherwise
        """
        ...

    def __and__(self, first: "Schedule", second: "Schedule") -> "CombinedSchedule":
        return self._combine(operator.and_, first, second)

    def __or__(self, first: "Schedule", second: "Schedule") -> "CombinedSchedule":
        return self._combine(operator.or_, first, second)

    def __invert__(self, other: "Schedule") -> "InvertedSchedule":
        return InvertedSchedule(other)

    def _combine(self, fn: Callable[[bool, bool], bool], first: "Schedule", second: "Schedule") -> "CombinedSchedule":
        return CombinedSchedule(fn, (first, second))


class InvertedSchedule(Schedule):
    def __init__(self, other: Schedule) -> None:
        self.other = other

    @override
    def check(
        self,
        *,
        stage: str | None = None,
        batch_idx: int | None = None,
        step: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> bool:
        return not self.other.check(stage=stage, batch_idx=batch_idx, step=step, trainer=trainer)


class CombinedSchedule(Schedule):
    def __init__(self, fn: Callable[[bool, bool], bool], schedules: Sequence[Schedule]) -> None:
        self.fn = fn
        self.schedules = schedules

    @override
    def check(
        self,
        *,
        stage: str | None = None,
        batch_idx: int | None = None,
        step: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> bool:
        return reduce(
            self.fn, (s.check(stage=stage, batch_idx=batch_idx, step=step, trainer=trainer) for s in self.schedules)
        )


class GlobalRank(Schedule):
    """
    A schedule that only executes on specific global ranks in a distributed training setup.

    This is useful for operations that should only be performed on certain ranks,
    such as logging, checkpointing, or other operations that would be redundant
    or conflicting if performed on all ranks.

    Attributes:
        ranks (tuple[int, ...]): The global ranks on which this schedule should execute.
    """

    def __init__(self, *ranks: int) -> None:
        """
        Initialize a GlobalRank schedule.

        Args:
            *ranks: Variable number of integer ranks. The schedule will only execute
                   on these global ranks.
        """
        self.ranks = ranks

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
        Check if the current global rank is in the specified ranks.

        Args:
            stage: Current trainer stage (ignored by this schedule).
            batch_idx: Current batch index (ignored by this schedule).
            step: Current step (ignored by this schedule).
            trainer: Lightning trainer instance, used to get the global rank.

        Returns:
            bool: True if the trainer's global rank is in the specified ranks, False otherwise.
                 Always returns False if trainer is None.
        """
        if trainer is None:
            return False
        return trainer.global_rank in self.ranks


class LocalRank(Schedule):
    """
    A schedule that only executes on specific local ranks in a distributed training setup.

    This is useful for node-specific operations that should only be performed on certain
    ranks within each node, such as local logging or monitoring.

    Attributes:
        ranks (tuple[int, ...]): The local ranks on which this schedule should execute.
    """

    def __init__(self, *ranks: int) -> None:
        """
        Initialize a LocalRank schedule.

        Args:
            *ranks: Variable number of integer ranks. The schedule will only execute
                   on these local ranks.
        """
        self.ranks = ranks

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
        Check if the current local rank is in the specified ranks.

        Args:
            stage: Current trainer stage (ignored by this schedule).
            batch_idx: Current batch index (ignored by this schedule).
            step: Current step (ignored by this schedule).
            trainer: Lightning trainer instance, used to get the local rank.

        Returns:
            bool: True if the trainer's local rank is in the specified ranks, False otherwise.
                 Always returns False if trainer is None.
        """
        if trainer is None:
            return False
        return trainer.local_rank in self.ranks


class Never(Schedule):
    """
    A schedule for an event that never happens.
    """

    @override
    def check(
        self,
        *,
        stage: str | None = None,
        batch_idx: int | None = None,
        step: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> bool:
        return False


class Always(Schedule):
    """
    A schedule for an event that always happens.
    """

    @override
    def check(
        self,
        *,
        stage: str | None = None,
        batch_idx: int | None = None,
        step: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> bool:
        return True


class Fixed(Schedule):
    """
    A schedule for an event that happens after a warmup period
    and lasts for a fixed number of steps.

    Attributes:
        warmup_steps (int): Number of initial steps to skip before logging starts.
        active_steps (int): Number of steps to log after warmup period.
    """

    def __init__(self, warmup_steps: int, active_steps: int) -> None:
        self._warmup_steps: int = warmup_steps
        self._active_steps: int = active_steps

    @override
    def check(
        self,
        *,
        stage: str | None = None,
        batch_idx: int | None = None,
        step: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> bool:
        assert step is not None, "step must be provided"
        if step < self._warmup_steps:
            return False

        if step - self._warmup_steps >= self._active_steps:
            return False
        return True


class Every(Schedule):
    """
    A schedule for an event that happens every specified number of batches and/or steps.

    Attributes:
        n_batches (Optional[int]): A positive number of batches between logging events.
            Defaults to 0 - use only n_steps
        n_steps (Optional[int]): A positive number of (train) steps between logging events.
            Defaults to 0 - use only n_batches
        stage (Optional[str]): The stage this schedule applies to ('train', 'validation', 'test', 'predict').
            If None, applies to all stages.
    """

    def __init__(self, *, n_batches: int = 0, n_steps: int = 0, stage: str | None = None) -> None:
        assert n_batches or n_steps, "either n_batches or n_steps has to be a positive number"
        self._n_batches = n_batches
        self._n_steps = n_steps
        self._stage = stage

    @override
    def check(
        self,
        *,
        stage: str | None = None,
        batch_idx: int | None = None,
        step: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> bool:
        # If stage is specified and doesn't match, return False
        if self._stage is not None and stage != self._stage:
            return False

        return (batch_idx is not None and self._n_batches > 0 and batch_idx % self._n_batches == 0) or (
            step is not None and self._n_steps > 0 and step % self._n_steps == 0
        )


class Elapsed(Schedule):
    """
    A schedule for an event that happens after the provided time interval has elapsed.
    """

    def __init__(self, interval: dt.timedelta) -> None:
        self.interval = interval
        self.last_triggered: dt.datetime | None = None

    @override
    def check(
        self,
        *,
        stage: str | None = None,
        batch_idx: int | None = None,
        step: int | None = None,
        trainer: L.Trainer | None = None,
    ) -> bool:
        now = dt.datetime.now(dt.timezone.utc)
        if self.last_triggered is None or now - self.last_triggered >= self.interval:
            self.last_triggered = now
            return True
        return False

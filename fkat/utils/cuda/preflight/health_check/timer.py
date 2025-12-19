# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from contextlib import contextmanager
from collections.abc import Iterator, Mapping


class Timer(Mapping[str, float]):
    def __init__(self) -> None:
        self._times: dict[str, list[float]] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._times.setdefault(name, []).append(1000 * (end - start))

    def __getitem__(self, name: str) -> float:
        if len(self._times[name]) == 1:
            return self._times[name][0]
        else:
            return max(self._times[name])

    def __iter__(self) -> Iterator[str]:
        return iter(self._times)

    def __len__(self) -> int:
        return len(self._times)

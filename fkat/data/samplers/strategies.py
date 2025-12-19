# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from collections.abc import Iterator

import numpy as np
from typing_extensions import Protocol, override


class SamplerStrategy(Protocol):
    """
    This strategy decides which Sampler to use next and returns its label
    """

    def __iter__(self) -> Iterator[str]: ...


class Weighted(SamplerStrategy):
    """
    Sample the label to generate next microbatch from using the provided weight distribution.
    For uniform distribution use the same weight for all labels.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self.names = []
        self.weights = []
        for name, weight in weights.items():
            self.names.append(name)
            self.weights.append(weight)

    @override
    def __iter__(self) -> Iterator[str]:
        while True:
            yield np.random.choice(self.names, p=self.weights)


class RoundRobin(SamplerStrategy):
    """
    Specifies the order of labels to generate microbatches from.
    """

    def __init__(self, names: list[str]) -> None:
        self.names = names

    @override
    def __iter__(self) -> Iterator[str]:
        i = -1
        while True:
            i = (i + 1) % len(self.names)
            yield self.names[i]


class Frequency(SamplerStrategy):
    """
    Specifies the order and number of microbatches to generate for specific labels.
    E.g. [["first", 2], ["second", 1], ["first", 3], ["third", 1]]
    """

    def __init__(self, freq: list[list[Any]]) -> None:
        assert all(isinstance(e[0], str) and isinstance(e[1], int) for e in freq)
        self.freq = freq

    @override
    def __iter__(self) -> Iterator[str]:
        while True:
            for name, count in self.freq:
                for _ in range(count):
                    yield name

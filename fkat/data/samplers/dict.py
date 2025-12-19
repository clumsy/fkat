# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterator

from typing_extensions import override

from fkat.data.samplers.sized import SizedSampler
from fkat.data.samplers.strategies import SamplerStrategy


class DictBatchSampler(SizedSampler[tuple[str, list[int]]]):
    def __init__(self, strategy: SamplerStrategy, samplers: dict[str, SizedSampler[list[int]]]) -> None:
        self.strategy = strategy
        self.samplers = samplers
        self.len = sum(len(sampler) for sampler in samplers.values())

    @override
    def __len__(self) -> int:
        return self.len

    @override
    def __iter__(self) -> Iterator[tuple[str, list[int]]]:
        rem_samplers = {name: iter(sampler) for name, sampler in self.samplers.items()}
        for key in self.strategy:
            if not rem_samplers:
                # no more iterators left, stopping iteration
                return
            if key not in rem_samplers:
                # this sampler is exhausted, skipping to sample the remaining ones
                # this for example will adjust the effective weights for the remaining ones when sampling
                # or sample next ones for sequential or frequency-based samplers
                continue
            try:
                batch = next(rem_samplers[key])
                yield (key, batch)
            except StopIteration:
                # this sampler is exhausted, removing from consideration
                del rem_samplers[key]

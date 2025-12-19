# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable
from unittest import TestCase

from fkat.data.samplers.strategies import Frequency, RoundRobin, SamplerStrategy, Weighted


class SamplerStrategyTest(TestCase):
    def test_is_iterable(self):
        assert issubclass(SamplerStrategy, Iterable)


class WeightedTest(TestCase):
    def test_uses_weights(self):
        weighted = Weighted(
            weights={
                "first": 0.75,
                "second": 0.25,
                "missing": 0,
            }
        )
        it = iter(weighted)
        assert next(it) in {"first", "second"}


class RoundRobinTest(TestCase):
    def test_uses_order(self):
        round_robin = RoundRobin(names=["a", "b", "c", "b"])
        it = iter(round_robin)
        assert ["a", "b", "c", "b", "a", "b", "c"] == [next(it) for _ in range(7)]


class FrequencyTest(TestCase):
    def test_uses_freq(self):
        freq = Frequency(freq=[["a", 1], ["b", 3], ["a", 1], ["c", 2]])
        it = iter(freq)
        assert ["a", "b", "b", "b", "a", "c", "c"] == [next(it) for _ in range(7)]

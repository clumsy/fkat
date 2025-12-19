# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from fkat.data.samplers import DictBatchSampler


class DictBatchSamplerTest(TestCase):
    def test_len(self):
        # Arrange
        key, sampler = "some_key", MagicMock(__len__=lambda _: 42)
        strategy = MagicMock()
        dict_sampler = DictBatchSampler(strategy=strategy, samplers={key: sampler})
        # Act
        length = len(dict_sampler)
        # Assert
        assert len(sampler) == length

    def test_iter(self):
        # Arrange
        key1, sampler1 = "some_key1", MagicMock(__len__=lambda _: 1, __iter__=lambda _: iter([1]))
        key2, sampler2 = "some_key2", MagicMock(__len__=lambda _: 2, __iter__=lambda _: iter([2, 2]))
        key3, sampler3 = "some_key3", MagicMock(__len__=lambda _: 3, __iter__=lambda _: iter([3, 3, 3]))
        strategy_iter = MagicMock()
        strategy_iter.__next__ = MagicMock(
            side_effect=lambda: [key1, key2, key3][(strategy_iter.__next__.call_count - 1) % 3]
        )
        strategy = MagicMock(__iter__=lambda _: strategy_iter)
        dict_sampler = DictBatchSampler(strategy=strategy, samplers={key1: sampler1, key2: sampler2, key3: sampler3})
        # Act
        it = iter(dict_sampler)
        keys = [next(it)[0] for _ in range(6)]
        # Assert
        assert [key1, key2, key3, key2, key3, key3] == keys
        with pytest.raises(StopIteration):
            next(it)
        assert it != iter(dict_sampler)

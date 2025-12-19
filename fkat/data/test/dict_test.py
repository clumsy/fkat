# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import TestCase
from unittest.mock import MagicMock

import pytest

from fkat.data import DictDataLoader


class DictDataLoaderTest(TestCase):
    def test_dataloader(self):
        # Arrange
        key1, dataloader1 = "some_key1", MagicMock(__len__=lambda _: 1, __iter__=lambda _: iter([{"id": 1}]))
        key2, dataloader2 = "some_key2", MagicMock(__len__=lambda _: 2, __iter__=lambda _: iter([{"id": 2}, {"id": 2}]))
        key3, dataloader3 = (
            "some_key3",
            MagicMock(__len__=lambda _: 3, __iter__=lambda _: iter([{"id": 3}, {"id": 3}, {"id": 3}])),
        )
        strategy_iter = MagicMock()
        strategy_iter.__next__ = MagicMock(
            side_effect=lambda: [key1, key2, key3][(strategy_iter.__next__.call_count - 1) % 3]
        )
        strategy = MagicMock(__iter__=lambda _: strategy_iter)
        key = "dataloader"
        dict_dataloader = DictDataLoader(
            strategy=strategy, dataloaders={key1: dataloader1, key2: dataloader2, key3: dataloader3}, key=key
        )
        # Act
        it = iter(dict_dataloader)
        keys = [next(it)[key] for _ in range(6)]
        # Assert
        assert [key1, key2, key3, key2, key3, key3] == keys
        with pytest.raises(StopIteration):
            next(it)
        assert it != iter(dict_dataloader)

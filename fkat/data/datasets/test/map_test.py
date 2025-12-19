# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import TestCase
from unittest.mock import MagicMock

from fkat.data.datasets import MapDataset, IterableMapDataset, SizedDataset

MODULE = "fkat.data.datasets.map"


class IterableMapDatasetTest(TestCase):
    def test_len(self):
        # Arrange
        dataset = MagicMock(__iter__=lambda _: iter([42]))
        # Act
        ds = IterableMapDataset(dataset, lambda x: -x)
        # Assert
        assert list(iter(ds)) == [-i for i in iter(dataset)]


class MapDatasetTest(TestCase):
    def test_len(self):
        # Arrange
        dataset = MagicMock(__len__=lambda _: 42)
        # Act
        ds = MapDataset(dataset, lambda _: None)
        # Assert
        assert len(ds) == len(dataset)

    def test_getitems(self):
        # Arrange
        dataset = MagicMock(spec=SizedDataset)
        dataset.__getitems__ = MagicMock(return_value=list(range(7)))
        # Act
        ds = MapDataset(dataset, lambda x: x + 42)
        # Assert
        assert ds.__getitems__(list(range(7))) == [i + 42 for i in range(7)]

    def test_getitem(self):
        # Arrange
        dataset = MagicMock(spec=SizedDataset)
        dataset.__getitem__.return_value = 7
        # Act
        ds = MapDataset(dataset, lambda x: x + 42)
        # Assert
        assert ds.__getitem__(7) == 7 + 42

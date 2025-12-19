# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import TestCase
from unittest.mock import MagicMock

from fkat.data.datasets.dict import DictDataset


class DictDatasetTest(TestCase):
    def test_len(self):
        # Arrange
        dataset1 = MagicMock()
        dataset1.__len__.return_value = 42
        dataset2 = MagicMock()
        dataset2.__len__.return_value = 1
        dataset = DictDataset(datasets={"1": dataset1, "2": dataset2})
        # Act
        length = len(dataset)
        # Assert
        assert length == len(dataset1) + len(dataset2)

    def test_getitem(self):
        # Arrange
        dataset1 = MagicMock()
        dataset1.__len__.return_value = 42
        dataset1.__getitem__.return_value = {"inputs_ids": [42]}
        dataset2 = MagicMock()
        dataset2.__len__.return_value = 1
        dataset = DictDataset(datasets={"1": dataset1, "2": dataset2})
        # Act
        sample = dataset[((key := "1"), (idx := 17))]
        # Assert
        assert sample == {**dataset1[idx], dataset.key: key}

    def test_getitems(self):
        # Arrange
        dataset1 = MagicMock()
        dataset1.__len__.return_value = 42
        dataset1.__getitems__ = MagicMock(return_value=[{"inputs_ids": [42]}])
        dataset2 = MagicMock()
        dataset2.__len__.return_value = 1
        dataset = DictDataset(datasets={"1": dataset1, "2": dataset2})
        # Act
        batch = dataset.__getitems__(((key := "1"), (idxs := [17])))
        # Assert
        assert batch == [{**b, dataset.key: key} for b in dataset1.__getitems__(idxs)]

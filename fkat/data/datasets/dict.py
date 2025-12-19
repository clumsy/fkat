# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from typing_extensions import override

from fkat.data.datasets import SizedDataset
from fkat.utils.config import to_primitive_container


class DictDataset(SizedDataset[tuple[str, Any], dict[str, Any]]):
    """:class:`Dataset` that can get samples from one of the :class:`Dataset` using a mapping."""

    def __init__(
        self,
        datasets: dict[str, SizedDataset[Any, dict[str, Any]]],
        key: str = "dataset",
    ) -> None:
        """Create a :class:`Dataset` that can get samples from one of the :class:`Dataset` using a mapping.

        Args:
            datasets (Dict[str, SizedDataset[Any, Dict[str, Any]]]): A mapping from labels to :class:`Dataset`\\s.
            key (str): The name of the field to reflect the :class:`Dataset` the samples were provided from.
                Defaults to "dataset".

        Returns:
            None
        """
        self.datasets = to_primitive_container(datasets)
        self.len = sum(len(dataset) for dataset in datasets.values())
        self.key = key

    @override
    def __len__(self) -> int:
        """Get :class:`Dataset` size.

        Returns:
            int: :class:`Dataset` size.
        """
        return self.len

    def _wrap(self, name: str, item: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(item, dict) or self.key in item:
            raise RuntimeError(f"Datasets must return a dict without {self.key} key")
        item[self.key] = name
        return item

    def __getitems__(self, name_and_idxs: tuple[str, list[Any]]) -> list[dict[str, Any]]:
        """Get a batch of samples from the target :class:`Dataset` at the specified indices.

        Args:
            name_and_idxs (Tuple[str, List[Any]]): Samples' :class:`Dataset` and indices.

        Returns:
            List[Dict[str, Any]]: A batch of samples.
        """
        name, idxs = name_and_idxs
        if getitems := getattr(self.datasets[name], "__getitems__", None):
            batch = getitems(idxs)
        else:
            batch = [self.datasets[name][idx] for idx in idxs]
        for b in batch:
            self._wrap(name, b)
        return batch

    @override
    def __getitem__(self, idx: tuple[str, Any]) -> dict[str, Any]:
        """Get a sample from the target :class:`Dataset` at the specified index.

        Args:
            idx (Tuple[str, Any]): Sample :class:`Dataset` and index.

        Returns:
            Dict[str, Any]: A sample.
        """
        name, idx_ = idx
        sample = self.datasets[name][idx_]
        return self._wrap(name, sample)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TypeVar
from collections.abc import Callable, Iterator

from fkat.data.datasets import SizedDataset
from torch.utils.data import IterableDataset

T_in = TypeVar("T_in", contravariant=True)
T_from = TypeVar("T_from", covariant=True)
T_to = TypeVar("T_to", covariant=True)


class MapDataset(SizedDataset[T_in, T_to]):
    """A :class:`Dataset` that transforms the samples from another :class:`Dataset` using a function."""

    def __init__(
        self,
        dataset: SizedDataset[T_in, T_from],
        fn: Callable[[T_from], T_to],
    ) -> None:
        """Create a :class:`Dataset` that maps samples of another :class:`Dataset` using a function.

        Args:
            dataset (SizedDataset): Source :class:`Dataset`.
            fn (Callable[[T_from], T_to]): Sample transformation function.

        Returns:
            None
        """
        self.dataset = dataset
        self.fn = fn

    def __len__(self) -> int:
        """Get :class:`Dataset` size.

        Returns:
            int: :class:`Dataset` size.
        """
        return len(self.dataset)

    def __getitems__(self, idxs: list[T_in]) -> list[T_to]:
        """Get a batch of samples at the specified indices.

        Args:
            idxs (List[T_in]): Samples' indices.

        Returns:
            List[T_to]: A batch of samples.
        """
        if getitems := getattr(self.dataset, "__getitems__", None):
            batch = getitems(idxs)
        else:
            batch = [self.dataset[idx] for idx in idxs]
        samples = [self.fn(sample) for sample in batch]
        return samples

    def __getitem__(self, idx: T_in) -> T_to:
        """Get a sample at the specified index.

        Args:
            idx (T_in): Sample index.

        Returns:
            T_to: A sample.
        """
        sample = self.fn(self.dataset[idx])
        return sample


class IterableMapDataset(IterableDataset[T_to]):
    """An :class:`IterableDataset` that transforms the samples from another
    :class:`IterableDataset` using a function."""

    def __init__(
        self,
        dataset: IterableDataset[T_from],
        fn: Callable[[T_from], T_to],
    ) -> None:
        self.dataset = dataset
        self.fn = fn

    def __iter__(self) -> Iterator[T_to]:
        """Get :class:`IterableDataset` iterator.

        Yields:
            T_to: A sample.
        """
        for sample in iter(self.dataset):
            yield self.fn(sample)

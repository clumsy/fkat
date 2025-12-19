# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from collections.abc import Iterable, Iterator

from typing_extensions import override

from fkat.data.samplers.strategies import SamplerStrategy
from fkat.utils.config import to_primitive_container


def wrap(key: str, name: str, batch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(batch, dict) or key in batch:
        raise RuntimeError(f"DataLoaders must return a batch dict without {key} key")
    batch[key] = name
    return batch


class DictDataLoader(Iterable[dict[str, Any]]):
    """A :class:`LightningDataModule` that manages multiple :class:`DataLoader`\\s for different stages."""

    def __init__(
        self,
        dataloaders: dict[str, Iterable[dict[str, Any]]],
        strategy: SamplerStrategy,
        key: str = "dataset",
    ) -> None:
        self.dataloaders = to_primitive_container(dataloaders)
        self.strategy = strategy
        self.key = key

    @override
    def __iter__(self) -> Iterator[dict[str, Any]]:
        iters = {k: iter(self.dataloaders[k]) for k in self.dataloaders}
        for name in self.strategy:
            if not iters:
                return
            if it := iters.get(name):
                try:
                    yield wrap(self.key, name, next(it))
                except StopIteration:
                    del iters[name]

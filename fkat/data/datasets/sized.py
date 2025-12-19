# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TypeVar

from typing_extensions import Protocol

T_in = TypeVar("T_in", contravariant=True)
T_out = TypeVar("T_out", covariant=True)


class SizedDataset(Protocol[T_in, T_out]):
    """A :class:`Dataset`  with a known size."""

    def __len__(self) -> int:
        """Get :class:`Dataset` size.

        Returns:
            int: :class:`Dataset` size.
        """
        ...

    def __getitem__(self, idx: T_in) -> T_out:
        """Get a sample at the specified index.

        Args:
            idx (T_in): Sample index.

        Returns:
            T_out: A sample.
        """
        ...

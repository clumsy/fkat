# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TypeVar
from collections.abc import Iterator

from typing_extensions import Protocol

T_co = TypeVar("T_co", covariant=True)


class SizedSampler(Protocol[T_co]):
    """A Sampler with a known size."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[T_co]: ...

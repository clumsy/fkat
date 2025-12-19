# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from functools import wraps
from typing import TypeVar, ParamSpec
from collections.abc import Callable
from typing_extensions import overload


def get_rank() -> int | None:
    return int(os.getenv("RANK", "0"))


def get_local_rank() -> int | None:
    return int(os.getenv("LOCAL_RANK", "0"))


T = TypeVar("T")
P = ParamSpec("P")


@overload
def local_rank_zero_only(fn: Callable[P, T]) -> Callable[P, T | None]: ...


@overload
def local_rank_zero_only(fn: Callable[P, T], default: T) -> Callable[P, T]: ...


def local_rank_zero_only(fn: Callable[P, T], default: T | None = None) -> Callable[P, T | None]:
    """Wrap a function to call internal function only in rank zero.

    Function that can be used as a decorator to enable a function/method being called only on global rank 0.

    """

    @wraps(fn)
    def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> T | None:
        local_rank = getattr(local_rank_zero_only, "local_rank", None)
        if local_rank is None:
            raise RuntimeError("The `local_rank_zero_only.local_rank` needs to be set before use")
        if local_rank == 0:
            return fn(*args, **kwargs)
        return default

    return wrapped_fn


local_rank_zero_only.local_rank = getattr(local_rank_zero_only, "local_rank", get_local_rank() or 0)  # type: ignore[attr-defined]

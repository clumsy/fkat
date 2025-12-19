# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TypeVar

T = TypeVar("T")


def assert_not_none(obj: T | None, name: str = "obj") -> T:
    assert obj is not None, f"{name} cannot be None"
    return obj

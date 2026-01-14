# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timezone
from typing import TypeVar

T = TypeVar("T")


def assert_not_none(obj: T | None, name: str = "obj") -> T:
    assert obj is not None, f"{name} cannot be None"
    return obj


def safe_timestamp(dt: datetime | None = None) -> str:
    """
    Generate a filesystem-safe timestamp string.

    Format: YYYY-MM-DD_HH-MM-SS-mmm (e.g., 2026-01-14_16-09-15-123)

    Args:
        dt: datetime object to format. If None, uses current UTC time.

    Returns:
        Formatted timestamp string safe for use in filenames
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d_%H-%M-%S-") + f"{dt.microsecond // 1000:03d}"

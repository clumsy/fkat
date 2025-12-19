# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest.mock import patch

import pytest

from fkat.pytorch.utilities import local_rank_zero_only, get_local_rank


class TestLocalRankZeroOnly(unittest.TestCase):
    def setUp(self):
        # Save original local_rank value to restore after tests
        self.original_local_rank = getattr(local_rank_zero_only, "local_rank", 0)

    def tearDown(self):
        # Restore original local_rank value after tests
        local_rank_zero_only.local_rank = self.original_local_rank  # type: ignore[attr-defined]

    def test_get_local_rank_default(self):
        """Test that _get_local_rank returns 0 when LOCAL_RANK is not set."""
        with patch.dict(os.environ, clear=True):
            assert get_local_rank() == 0

    def test_get_local_rank_custom(self):
        """Test that _get_local_rank returns the correct value when LOCAL_RANK is set."""
        with patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            assert get_local_rank() == 2

    def test_function_called_on_rank_zero(self):
        """Test that decorated function is called when local_rank is 0."""
        local_rank_zero_only.local_rank = 0  # type: ignore[attr-defined]

        @local_rank_zero_only
        def test_fn() -> str:
            return "function called"

        assert test_fn() == "function called"

    def test_function_with_arguments_non_zero_rank(self):
        """Test that decorated function with arguments returns default on non-zero rank."""
        local_rank_zero_only.local_rank = 1  # type: ignore[attr-defined]

        # Correct way to use the decorator with a default value
        def test_fn(a: int, b: int, c: int = 3) -> int:
            return a + b + c

        decorated_fn = local_rank_zero_only(test_fn, default=0)  # type: ignore[arg-type]
        assert decorated_fn(1, 2) == 0

        # Alternative way using decorator syntax
        @local_rank_zero_only
        def test_fn2(a: int, b: int, c: int = 3) -> int:
            return a + b + c

        # Without default, it should return None
        assert test_fn2(1, 2) is None

    def test_function_with_arguments(self):
        """Test that decorated function works with arguments."""
        local_rank_zero_only.local_rank = 0  # type: ignore[attr-defined]

        @local_rank_zero_only
        def test_fn(a: int, b: int, c: int = 3) -> int:
            return a + b + c

        assert test_fn(1, 2) == 6
        assert test_fn(1, 2, c=4) == 7

    def test_missing_local_rank(self):
        """Test that RuntimeError is raised when local_rank is not set."""
        # Temporarily remove the local_rank attribute
        delattr(local_rank_zero_only, "local_rank")

        @local_rank_zero_only
        def test_fn() -> str:
            return "function called"

        with pytest.raises(RuntimeError):
            test_fn()

    def test_decorator_preserves_function_metadata(self):
        """Test that the decorator preserves the function's metadata."""

        @local_rank_zero_only
        def test_fn():
            """Test function docstring."""
            pass

        assert test_fn.__name__ == "test_fn"  # type: ignore[attr-defined]
        assert test_fn.__doc__ == "Test function docstring."

    def test_environment_variable_initialization(self):
        """Test that local_rank is initialized from environment variable."""
        # Reset the module to test initialization
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            # Re-initialize the local_rank attribute
            local_rank_zero_only.local_rank = get_local_rank() or 0  # type: ignore[attr-defined]
            assert local_rank_zero_only.local_rank == 3  # type: ignore[attr-defined]

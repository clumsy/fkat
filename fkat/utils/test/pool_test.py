# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor, Future

from fkat.utils import pool
from fkat.utils.pool import ThreadPool, FutureResult


class TestThreadPool(unittest.TestCase):
    def setUp(self):
        class MockExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                self._submit.submit(fn, *args, **kwargs)
                res = fn(*args, **kwargs)
                fut = Future()
                fut.set_result(res)
                return fut

            shutdown = MagicMock()
            _submit = MagicMock()

        with patch(f"{pool.__name__}.ThreadPoolExecutor", MockExecutor):
            self.it = ThreadPool()

    def test_future_result_ready(self):
        # Arrange
        mock_future = MagicMock(spec=Future)
        mock_future.done.return_value = True
        future_result = FutureResult(mock_future)
        # Act
        res = future_result.ready()
        # Assert
        assert res
        mock_future.done.assert_called_once()

    def test_future_result_get(self):
        # Arrange
        mock_future = MagicMock(spec=Future)
        mock_future.result.return_value = 42
        future_result = FutureResult(mock_future)
        # Act
        res = future_result.get()
        # Assert
        assert res == mock_future.result.return_value

    def test_future_result_get_with_timeout(self):
        # Arrange
        mock_future = MagicMock(spec=Future)
        mock_future.result.return_value = "result"
        future_result = FutureResult(mock_future)
        # Act
        res = future_result.get(timeout=1.0)
        # Assert
        assert res == mock_future.result.return_value

    def test_future_result_wait(self):
        # Arrange
        mock_future = MagicMock(spec=Future)
        future_result = FutureResult(mock_future)
        # Act
        future_result.wait(timeout=1.0)
        # Assert
        mock_future.exception.assert_called_once_with(1.0)

    def test_future_result_successful(self):
        # Arrange
        mock_future = MagicMock(spec=Future)
        mock_future.exception.return_value = None
        future_result = FutureResult(mock_future)
        # Act
        res = future_result.successful()
        # Assert
        assert res
        mock_future.exception.assert_called_once()

    def test_future_result_not_successful(self):
        # Arrange
        mock_future = MagicMock(spec=Future)
        mock_future.exception.return_value = ValueError()
        future_result = FutureResult(mock_future)
        # Act
        res = future_result.successful()
        # Assert
        assert not res
        mock_future.exception.assert_called_once()

    def test_apply_async_with_args(self):
        # Arrange
        def dummy_func(a: int, b: int) -> int:
            return a + b

        # Act
        res = self.it.apply_async(dummy_func, args=(1, 2))
        # Assert
        assert res.get() == dummy_func(1, 2)

    def test_apply_async_with_kwargs(self):
        # Arrange
        def dummy_func(a: int, b: int) -> int:
            return a + b

        kwds = {"a": 1, "b": 2}
        # Act
        res = self.it.apply_async(dummy_func, kwds=kwds)
        # Assert
        assert res.get() == dummy_func(**kwds)  # type: ignore[missing-argument]

    def test_apply_async_with_args_and_kwargs(self):
        # Arrange
        def dummy_func(a: int, b: int, c: int) -> int:
            return a + b + c

        args, kwds = (1,), {"b": 2, "c": 3}
        # Act
        res = self.it.apply_async(dummy_func, args=args, kwds=kwds)
        # Assert
        assert res.get() == dummy_func(*args, **kwds)  # type: ignore[missing-argument]

    def test_pool_close(self):
        # Act
        self.it.close()
        # Assert
        self.it.pool.shutdown.assert_called_once()  # type: ignore[attr-defined]

    def test_pool_join(self):
        # Act
        self.it.join()
        # Assert
        self.it.pool._submit.submit.assert_called_once()  # type: ignore[attr-defined]

    def test_pool_join_shutdown(self):
        # Act
        self.it.pool._shutdown = True
        self.it.join()
        # Assert
        self.it.pool.shutdown.assert_called_once()  # type: ignore[attr-defined]

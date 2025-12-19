# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, TypeVar
import multiprocessing as mp
from multiprocessing.pool import AsyncResult
from collections.abc import Callable, Iterable, Mapping


T = TypeVar("T", covariant=False)
T_co = TypeVar("T_co", covariant=True)


class FutureResult(AsyncResult[T]):
    """An AsyncResult implementation for concurrent.future Future object"""

    def __init__(self, fut: Future[T]) -> None:
        self.fut = fut

    def ready(self) -> bool:
        return self.fut.done()

    def get(self, timeout: float | None = None) -> T:
        return self.fut.result(timeout)

    def wait(self, timeout: float | None = None) -> None:
        self.fut.exception(timeout)

    def successful(self) -> bool:
        return self.fut.exception() is None


class ThreadPool:
    """A multiprocessing Pool-like implementation that uses ThreadPoolExecutor"""

    def __init__(self, **kwargs: Any) -> None:
        self.pool = ThreadPoolExecutor(**kwargs)

    def apply_async(
        self,
        func: Callable[..., T_co],
        args: Iterable[Any] | None = None,
        kwds: Mapping[str, Any] | None = None,
    ) -> FutureResult[T_co]:
        fut = self.pool.submit(func, *(args or ()), **(kwds or {}))
        return FutureResult(fut)

    def close(self) -> None:
        self.pool.shutdown()

    def join(self) -> None:
        if self.pool._shutdown:
            self.close()
        else:
            self.pool.submit(lambda: None).result()


class NoDaemonProcess(mp.Process):
    """A Process implementation that never runs in daemon mode"""

    @property
    def daemon(self) -> bool:
        return False

    @daemon.setter
    def daemon(self, value: bool) -> None:
        pass


class NoDaemonContext(type(mp.get_context())):  # type: ignore[misc]
    """A multiprocessing Context that uses NoDaemonProcess"""

    Process = NoDaemonProcess


class NoDaemonPool(mp.pool.Pool):  # type: ignore[unresolved-attribute]
    """A multiprocessing Pool that uses NoDaemonContext"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["context"] = NoDaemonContext()
        super().__init__(*args, **kwargs)

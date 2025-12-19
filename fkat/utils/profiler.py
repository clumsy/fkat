# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import atexit

from lightning.pytorch.profilers import Profiler


def profile_until_exit(profiler: Profiler, action: str, filename_suffix: str | None = None) -> None:
    def stop_profiler() -> None:
        profiler.stop(action)
        profiler.summary()
        profiler.describe()

    atexit.register(stop_profiler)

    if profiler.filename and filename_suffix:
        profiler.filename += filename_suffix
    profiler.start(action)

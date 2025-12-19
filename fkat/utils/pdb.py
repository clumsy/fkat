# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import pdb
import sys
from typing import Any
from types import FrameType, TracebackType
from typing_extensions import override


class ForkedPdb(pdb.Pdb):
    def __init__(self) -> None:
        super().__init__()
        self.rank = os.environ.get(
            "RANK",  # PyTorch DDP
            os.environ.get(
                "PMI_RANK",  # MPI
                os.environ.get(
                    "OMPI_COMM_WORLD_RANK",  # OpenMPI
                    "unknown",
                ),
            ),
        )

    @override
    def interaction(self, frame: FrameType | None, traceback: TracebackType | None, *args: Any, **kwargs: Any) -> None:
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            self.print_rank_info()
            pdb.Pdb.interaction(self, frame, traceback, *args, **kwargs)
        finally:
            sys.stdin = _stdin

    def print_rank_info(self) -> None:
        print(f"\n[RANK={self.rank}, PID={os.getpid()}]:")

    @override
    def default(self, line: str) -> None:
        self.print_rank_info()
        super().default(line)

    @override
    def do_continue(self, arg: str) -> bool | None:
        self.print_rank_info()
        return super().do_continue(arg)

    @override
    def do_next(self, arg: str) -> bool | None:
        self.print_rank_info()
        return super().do_next(arg)

    @override
    def do_step(self, arg: str) -> bool | None:
        self.print_rank_info()
        return super().do_step(arg)

    @override
    def do_return(self, arg: str) -> bool | None:
        self.print_rank_info()
        return super().do_return(arg)

    @override
    def do_quit(self, arg: str) -> bool | None:
        self.print_rank_info()
        return super().do_quit(arg)

    @override
    def do_jump(self, arg: str) -> bool | None:
        self.print_rank_info()
        return super().do_jump(arg)

    @override
    def precmd(self, line: str) -> str:
        self.print_rank_info()
        return line

    def post_mortem(self, tb: TracebackType | None) -> None:
        self.reset()
        self.interaction(None, tb)


def post_mortem() -> None:
    sys.excepthook = lambda t, v, tb: ForkedPdb().post_mortem(tb)

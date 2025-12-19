# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import logging
import re
import subprocess
import sys
import multiprocessing

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

XID_PAT = re.compile(r"\[(.*)\] NVRM: Xid \(.*\): (\d+),")


def detect_xid_errors(
    xid_check: multiprocessing.synchronize.Event,  # type: ignore[unresolved-attribute]
    xid_errors: multiprocessing.Queue[set[int]],
) -> None:
    """
    Detect XID errors by monitoring system logs.

    Args:
        xid_check: Event to trigger checking for XID errors
        xid_errors: Queue to put detected XID errors
    """
    try:
        log.info("\nChecking for Xid errors in a background process ...")
        while True:
            xid_check.wait()
            xid_check.clear()
            xids: set[int] = set()
            f = subprocess.check_output("dmesg -Tc", shell=True)
            lines = f.decode("utf8", errors="ignore").split("\n")
            for line in lines:
                res = XID_PAT.match(line)
                if res:
                    xid = int(res.group(2))
                    xids.add(xid)
            xid_errors.put(xids)
    except Exception as e:
        if hasattr(e, "returncode") and e.returncode == 1:
            log.info(
                "Xid monitoring requires running in privileged mode example "
                "ensure privileged access is available to access dmesg"
            )
        log.info(f"error executing command: {e}")

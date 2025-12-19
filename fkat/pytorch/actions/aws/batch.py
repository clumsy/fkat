# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from fkat.utils import boto3
from fkat.pytorch.actions import LightningAction

if TYPE_CHECKING:
    from types_boto3_batch import BatchClient


class TerminateJob(LightningAction):
    """This action calls Batch.TerminateJob."""

    def __init__(self, job_id: str | None = None) -> None:
        self.job_id = job_id

    @override
    def perform(self, **kwargs: Any) -> Any:
        """Calls Batch.TerminateJob."""
        job_id = self.job_id or os.getenv("AWS_BATCH_JOB_ID")
        if job_id:
            reason = ",".join(f"{k}={v}" for k, v in kwargs.items() if isinstance(v, str))
            batch: BatchClient = boto3.session().client("batch")  # type: ignore[assignment]
            batch.terminate_job(jobId=job_id, reason=reason)

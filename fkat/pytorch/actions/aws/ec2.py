# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

from fkat.utils import boto3, assert_not_none
from fkat.utils.aws import imds
from fkat.pytorch.loggers import LightningLogger, CompositeLogger
from fkat.pytorch.actions import LightningAction

if TYPE_CHECKING:
    from types_boto3_ec2 import EC2Client


class TerminateInstances(LightningAction):
    """This action calls EC2.TerminateInstances."""

    def __init__(self, instance_ids: list[str] | None = None) -> None:
        self.instance_ids = instance_ids or []

    @override
    def perform(self, **kwargs: Any) -> Any:
        """Calls EC2.TerminateInstances with the provided ``instance_id`` or the current node's instance_id"""
        instance_ids = self.instance_ids or kwargs.get("instance_ids") or [imds.instance_metadata().instance_id]
        ec2: EC2Client = boto3.session().client("ec2")  # type: ignore[assignment]
        ec2.terminate_instances(InstanceIds=instance_ids)


class RebootInstances(LightningAction):
    """This action calls EC2.RebootInstances."""

    def __init__(self, instance_ids: list[str] | None = None) -> None:
        self.instance_ids = instance_ids or []

    @override
    def perform(self, **kwargs: Any) -> Any:
        """Calls EC2.RebootInstances with the provided ``instance_id`` or the current node's instance_id"""
        instance_ids = self.instance_ids or kwargs.get("instance_ids") or [imds.instance_metadata().instance_id]
        ec2: EC2Client = boto3.session().client("ec2")  # type: ignore[assignment]
        ec2.reboot_instances(InstanceIds=instance_ids)


class LogInstanceTags(LightningAction):
    """This action logs tags suffixed with EC2 instance-id."""

    def __init__(self, instance_id: str | None = None, tags: list[str] | None = None) -> None:
        self.instance_id = instance_id
        self.tags = tags or []
        self.logger: LightningLogger | None = None

    @override
    def perform(self, **kwargs: Any) -> Any:
        """Calls LightningLogger.log_tag with all string-valued keys, requires ``trainer`` to be provided"""
        instance_id = self.instance_id or kwargs.get("instance_id")
        for t in self.tags:
            if v := kwargs.get(t):
                instance_id = instance_id or imds.instance_metadata().instance_id
                self.logger = self.logger or CompositeLogger(assert_not_none(kwargs.get("trainer"), "trainer"))
                self.logger.log_tag(f"{instance_id}/{t}/{v}", "True")

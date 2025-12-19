# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch

from fkat.pytorch.actions.aws import ec2
from fkat.pytorch.actions.aws.ec2 import TerminateInstances, RebootInstances, LogInstanceTags
from fkat.utils.aws.imds import InstanceMetadata


class TestEc2Actions(unittest.TestCase):
    @patch(f"{ec2.__name__}.boto3.session")
    def test_terminate_instances_with_provided_ids(self, mock_session):
        # Arrange
        mock_ec2 = MagicMock()
        mock_session.return_value.client.return_value = mock_ec2
        instance_ids = ["i-1234567890abcdef0", "i-0987654321fedcba0"]
        action = TerminateInstances(instance_ids=instance_ids)

        # Act
        action.perform()

        # Assert
        mock_session.return_value.client.assert_called_once_with("ec2")
        mock_ec2.terminate_instances.assert_called_once_with(InstanceIds=instance_ids)

    @patch(f"{ec2.__name__}.boto3.session")
    @patch(f"{ec2.__name__}.imds.instance_metadata")
    def test_terminate_instances_with_current_instance(self, mock_metadata, mock_session):
        # Arrange
        mock_ec2 = MagicMock()
        mock_session.return_value.client.return_value = mock_ec2
        mock_metadata.return_value = InstanceMetadata(
            instance_id="i-current",
            instance_type="",
            hostname="",
            public_hostname="",
            local_hostname="",
            availability_zone="",
            local_ipv4="",
            region="",
            ami_id="",
        )
        action = TerminateInstances()

        # Act
        action.perform()

        # Assert
        mock_ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-current"])

    @patch(f"{ec2.__name__}.boto3.session")
    def test_reboot_instances_with_provided_ids(self, mock_session):
        # Arrange
        mock_ec2 = MagicMock()
        mock_session.return_value.client.return_value = mock_ec2
        instance_ids = ["i-1234567890abcdef0"]
        action = RebootInstances(instance_ids=instance_ids)

        # Act
        action.perform()

        # Assert
        mock_session.return_value.client.assert_called_once_with("ec2")
        mock_ec2.reboot_instances.assert_called_once_with(InstanceIds=instance_ids)

    @patch(f"{ec2.__name__}.boto3.session")
    @patch(f"{ec2.__name__}.imds.instance_metadata")
    def test_reboot_instances_with_kwargs(self, mock_metadata, mock_session):
        # Arrange
        mock_ec2 = MagicMock()
        mock_session.return_value.client.return_value = mock_ec2
        instance_ids = ["i-from-kwargs"]
        action = RebootInstances()

        # Act
        action.perform(instance_ids=instance_ids)

        # Assert
        mock_ec2.reboot_instances.assert_called_once_with(InstanceIds=instance_ids)
        mock_metadata.assert_not_called()

    @patch(f"{ec2.__name__}.CompositeLogger")
    @patch(f"{ec2.__name__}.imds.instance_metadata")
    def test_log_instance_tags(self, mock_metadata, mock_logger_cls):
        # Arrange
        mock_logger = MagicMock()
        mock_logger_cls.return_value = mock_logger
        mock_metadata.return_value = InstanceMetadata(
            instance_id="i-test",
            instance_type="",
            hostname="",
            public_hostname="",
            local_hostname="",
            availability_zone="",
            local_ipv4="",
            region="",
            ami_id="",
        )
        mock_trainer = MagicMock()
        action = LogInstanceTags(tags=["accuracy", "loss"])

        # Act
        action.perform(trainer=mock_trainer, accuracy="0.95", loss="0.1")

        # Assert
        mock_logger_cls.assert_called_once_with(mock_trainer)
        mock_logger.log_tag.assert_any_call("i-test/accuracy/0.95", "True")
        mock_logger.log_tag.assert_any_call("i-test/loss/0.1", "True")
        assert mock_logger.log_tag.call_count == 2

    @patch(f"{ec2.__name__}.CompositeLogger")
    def test_log_instance_tags_with_provided_instance_id(self, mock_logger_cls):
        # Arrange
        mock_logger = MagicMock()
        mock_logger_cls.return_value = mock_logger
        mock_trainer = MagicMock()
        action = LogInstanceTags(instance_id="i-custom", tags=["tag"])

        # Act
        action.perform(trainer=mock_trainer, tag="42")

        # Assert
        mock_logger.log_tag.assert_called_once_with("i-custom/tag/42", "True")

    def test_log_instance_tags_no_matching_tags(self):
        # Arrange
        action = LogInstanceTags(instance_id="i-test", tags=["tag"])
        mock_trainer = MagicMock()

        # Act
        action.perform(trainer=mock_trainer)  # No matching tag values

        # Assert
        # No logger should be created since no tags matched
        assert action.logger is None

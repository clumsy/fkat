# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch

from fkat.pytorch.actions.aws import batch
from fkat.pytorch.actions.aws.batch import TerminateJob


class TestBatchActions(unittest.TestCase):
    @patch(f"{batch.__name__}.boto3.session")
    def test_terminate_job_with_provided_job_id(self, mock_session):
        # Arrange
        mock_batch = MagicMock()
        mock_session.return_value.client.return_value = mock_batch
        job_id = "job-12345"
        action = TerminateJob(job_id=job_id)

        # Act
        action.perform(error="Out of memory")

        # Assert
        mock_session.return_value.client.assert_called_once_with("batch")
        mock_batch.terminate_job.assert_called_once_with(jobId=job_id, reason="error=Out of memory")

    @patch(f"{batch.__name__}.boto3.session")
    @patch(f"{batch.__name__}.os.getenv")
    def test_terminate_job_with_env_job_id(self, mock_getenv, mock_session):
        # Arrange
        mock_batch = MagicMock()
        mock_session.return_value.client.return_value = mock_batch
        mock_getenv.return_value = "env-job-67890"
        action = TerminateJob()

        # Act
        action.perform()

        # Assert
        mock_getenv.assert_called_once_with("AWS_BATCH_JOB_ID")
        mock_batch.terminate_job.assert_called_once_with(jobId="env-job-67890", reason="")

    @patch(f"{batch.__name__}.boto3.session")
    @patch(f"{batch.__name__}.os.getenv")
    def test_terminate_job_with_multiple_kwargs(self, mock_getenv, mock_session):
        # Arrange
        mock_batch = MagicMock()
        mock_session.return_value.client.return_value = mock_batch
        mock_getenv.return_value = "env-job-67890"
        action = TerminateJob()

        # Act
        action.perform(error="Failed", step="100", metric="0.75")

        # Assert
        mock_batch.terminate_job.assert_called_once_with(
            jobId="env-job-67890", reason="error=Failed,step=100,metric=0.75"
        )

    @patch(f"{batch.__name__}.boto3.session")
    @patch(f"{batch.__name__}.os.getenv")
    def test_terminate_job_filters_non_string_kwargs(self, mock_getenv, mock_session):
        # Arrange
        mock_batch = MagicMock()
        mock_session.return_value.client.return_value = mock_batch
        mock_getenv.return_value = "env-job-67890"
        action = TerminateJob()

        # Act
        action.perform(
            error="Failed",
            step=100,  # int, should be filtered
            data={"key": "value"},  # dict, should be filtered
            message="Terminating",
        )

        # Assert
        mock_batch.terminate_job.assert_called_once_with(
            jobId="env-job-67890", reason="error=Failed,message=Terminating"
        )

    @patch(f"{batch.__name__}.boto3.session")
    @patch(f"{batch.__name__}.os.getenv")
    def test_terminate_job_no_job_id(self, mock_getenv, mock_session):
        # Arrange
        mock_batch = MagicMock()
        mock_session.return_value.client.return_value = mock_batch
        mock_getenv.return_value = None  # No job ID in environment
        action = TerminateJob()  # No job ID provided

        # Act
        action.perform(error="Test error")

        # Assert
        mock_getenv.assert_called_once_with("AWS_BATCH_JOB_ID")
        mock_batch.terminate_job.assert_not_called()  # Should not call terminate_job

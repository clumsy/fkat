# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch

from requests import HTTPError

from fkat.utils.aws import imds
from fkat.utils.aws.imds import fetch, token, instance_metadata, IMDS_METADATA_URL, IMDS_V2_TOKEN_URL, NULL


class TestImds(unittest.TestCase):
    def setUp(self):
        """Clear the LRU cache before each test."""
        # Clear the cache for both cached functions
        fetch.cache_clear()
        instance_metadata.cache_clear()

    @patch(f"{imds.__name__}.requests.get")
    def test_fetch_success(self, mock_get):
        # Arrange
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = "i-1234567890abcdef0"
        mock_get.return_value = mock_response
        test_token = "test-token-value"

        # Act
        result = fetch("instance-id", test_token)

        # Assert
        mock_get.assert_called_once_with(
            f"{IMDS_METADATA_URL}/instance-id", headers={"X-aws-ec2-metadata-token": test_token}
        )
        assert result == "i-1234567890abcdef0"

    @patch(f"{imds.__name__}.requests.get")
    def test_fetch_with_empty_token(self, mock_get):
        # Arrange
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = "i-1234567890abcdef0"
        mock_get.return_value = mock_response

        # Act
        result = fetch("instance-id", None)

        # Assert
        mock_get.assert_called_once_with(f"{IMDS_METADATA_URL}/instance-id", headers={"X-aws-ec2-metadata-token": ""})
        assert result == "i-1234567890abcdef0"

    @patch(f"{imds.__name__}.requests.get")
    @patch(f"{imds.__name__}.log")
    def test_fetch_exception(self, mock_log, mock_get):
        # Arrange
        mock_get.side_effect = Exception("Connection error")

        # Act
        result = fetch("instance-id")

        # Assert
        mock_get.assert_called_once()
        mock_log.warning.assert_called_once()
        assert result is None

    @patch(f"{imds.__name__}.requests.get")
    def test_fetch_response_not_ok(self, mock_get):
        # Arrange
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found", response=mock_response)
        mock_get.return_value = mock_response

        # Act
        result = fetch("invalid-metadata")

        # Assert
        mock_get.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
        assert result is None

    @patch(f"{imds.__name__}.requests.put")
    def test_token_success(self, mock_put):
        # Arrange
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = "test-token-value"
        mock_put.return_value = mock_response

        # Act
        result = token()

        # Assert
        mock_put.assert_called_once_with(
            IMDS_V2_TOKEN_URL, headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, timeout=60
        )
        assert result == "test-token-value"

    @patch(f"{imds.__name__}.requests.put")
    def test_token_with_custom_timeout(self, mock_put):
        # Arrange
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = "test-token-value"
        mock_put.return_value = mock_response
        custom_timeout = 30

        # Act
        result = token(timeout=custom_timeout)

        # Assert
        mock_put.assert_called_once_with(
            IMDS_V2_TOKEN_URL, headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, timeout=custom_timeout
        )
        assert result == "test-token-value"

    @patch(f"{imds.__name__}.requests.put")
    @patch(f"{imds.__name__}.log")
    def test_token_exception(self, mock_log, mock_put):
        # Arrange
        mock_put.side_effect = Exception("Connection timeout")

        # Act
        result = token()

        # Assert
        mock_put.assert_called_once()
        mock_log.warning.assert_called_once()
        assert result == ""

    @patch(f"{imds.__name__}.requests.put")
    @patch(f"{imds.__name__}.log")
    def test_token_response_not_ok(self, mock_log, mock_put):
        # Arrange
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 403
        mock_put.return_value = mock_response

        # Act
        result = token()

        # Assert
        mock_put.assert_called_once()
        mock_log.warning.assert_called_once()
        assert result == ""

    @patch(f"{imds.__name__}.token")
    @patch(f"{imds.__name__}.fetch")
    def test_instance_metadata_success(self, mock_fetch, mock_token):
        # Arrange
        mock_token.return_value = "test-token"

        # Configure fetch to return different values for different metadata paths
        def mock_fetch_side_effect(metadata_path: str, token_value: str) -> str:
            metadata_values = {
                "instance-id": "i-1234567890abcdef0",
                "instance-type": "t3.large",
                "placement/availability-zone": "us-west-2a",
                "placement/region": "us-west-2",
                "hostname": "ip-10-0-0-1.us-west-2.compute.internal",
                "public-hostname": "ec2-203-0-113-1.us-west-2.compute.amazonaws.com",
                "local-hostname": "ip-10-0-0-1.us-west-2.compute.internal",
                "ami-id": "ami-1234567890abcdef0",
                "local-ipv4": "10.1.11.111",
            }
            return metadata_values.get(metadata_path, NULL)

        mock_fetch.side_effect = mock_fetch_side_effect

        # Act
        result = instance_metadata()

        # Assert
        mock_token.assert_called_once()
        assert mock_fetch.call_count == 9  # One call for each metadata field

        # Verify the InstanceMetadata object has the correct values
        assert result.instance_id == "i-1234567890abcdef0"
        assert result.instance_type == "t3.large"
        assert result.availability_zone == "us-west-2a"
        assert result.region == "us-west-2"
        assert result.hostname == "ip-10-0-0-1.us-west-2.compute.internal"
        assert result.public_hostname == "ec2-203-0-113-1.us-west-2.compute.amazonaws.com"
        assert result.local_hostname == "ip-10-0-0-1.us-west-2.compute.internal"
        assert result.ami_id == "ami-1234567890abcdef0"
        assert result.local_ipv4 == "10.1.11.111"

    @patch(f"{imds.__name__}.token")
    @patch(f"{imds.__name__}.fetch")
    @patch(f"{imds.__name__}.socket.gethostname")
    def test_instance_metadata_fallbacks(self, mock_gethostname, mock_fetch, mock_token):
        # Arrange
        mock_token.return_value = "test-token"
        mock_gethostname.return_value = "local-hostname"

        # Configure fetch to return None for all metadata paths
        mock_fetch.return_value = None

        # Act
        result = instance_metadata()

        # Assert
        mock_token.assert_called_once()
        assert mock_fetch.call_count == 9  # One call for each metadata field

        # Verify the InstanceMetadata object has the fallback values
        assert result.instance_id == "localhost"
        assert result.instance_type == NULL
        assert result.availability_zone == NULL
        assert result.region == NULL
        assert result.hostname == "local-hostname"  # From socket.gethostname()
        assert result.public_hostname == NULL
        assert result.local_hostname == NULL
        assert result.ami_id == NULL
        assert result.local_ipv4 == NULL

    @patch(f"{imds.__name__}.token")
    @patch(f"{imds.__name__}.fetch")
    def test_instance_metadata_caching(self, mock_fetch, mock_token):
        # Arrange
        mock_token.return_value = "test-token"
        mock_fetch.return_value = "test-value"

        # Act
        result1 = instance_metadata()

        # Reset the mocks to verify they're not called again
        mock_token.reset_mock()
        mock_fetch.reset_mock()

        # Get the result again (should use cached value)
        result2 = instance_metadata()

        # Assert
        mock_token.assert_not_called()  # Token should not be fetched again
        mock_fetch.assert_not_called()  # Fetch should not be called again

        # Both results should be the same object (due to caching)
        assert result1 is result2

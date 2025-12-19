# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import TestCase
from unittest.mock import patch

from fkat.utils import profiler


class ProfilerTest(TestCase):
    @patch(f"{profiler.__name__}.atexit")
    @patch(f"{profiler.__name__}.Profiler")
    def test_profiles_until_exit(self, mock_profiler, mock_atexit):
        # Arrange
        action = "some_action"
        filename_suffix = "some_suffix"
        mock_profiler.filename = (filename := "some_filename")

        # Act
        profiler.profile_until_exit(mock_profiler, action, filename_suffix)

        # Assert
        mock_profiler.start.assert_called_with(action)
        assert mock_profiler.filename == filename + filename_suffix
        mock_atexit.register.assert_called()
        stop_profiler = mock_atexit.register.call_args[0][0]
        stop_profiler()
        mock_profiler.stop.assert_called_with(action)
        mock_profiler.summary.assert_called()
        mock_profiler.describe.assert_called()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from unittest import TestCase

from lightning.pytorch.utilities import rank_zero_only

from fkat.utils.logging import rank0_logger


class Rank0LoggerTest(TestCase):
    @rank_zero_only
    def test_rank0_logger(self):
        # Test that the logger returns a logging.Logger instance
        logger = rank0_logger()
        assert isinstance(logger, logging.Logger)

    @rank_zero_only
    def test_rank0_logger_name(self):
        # Test that the logger has the correct name
        logger = rank0_logger("test")
        assert logger.name == "test"

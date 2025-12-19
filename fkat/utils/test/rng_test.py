# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from random import getstate as python_get_rng_state
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from fkat.utils.rng import get_rng_states, set_rng_states

logger = logging.getLogger(__name__)


class TestRandomStateFunctions(TestCase):
    def setUp(self):
        # Setting up initial random states for testing
        self.initial_states = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": python_get_rng_state(),
        }

    @patch("torch.cuda.is_available", MagicMock(return_value=True))
    def test_get_rng_states(self):
        states = get_rng_states()
        assert isinstance(states, dict)
        initial_torch_state = self.initial_states["torch"]
        assert isinstance(initial_torch_state, torch.Tensor)
        assert torch.equal(torch.get_rng_state(), initial_torch_state)
        assert np.random.get_state()[0] == self.initial_states["numpy"][0]  # type: ignore[index]
        assert np.array_equal(np.random.get_state()[1], self.initial_states["numpy"][1])  # type: ignore[index]
        assert python_get_rng_state() == self.initial_states["python"]

    def test_set_rng_states(self):
        rng_state_dict = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": python_get_rng_state(),
        }
        set_rng_states(rng_state_dict)
        torch_state = rng_state_dict["torch"]
        assert isinstance(torch_state, torch.Tensor)
        assert torch.equal(torch.get_rng_state(), torch_state)
        assert np.random.get_state()[0] == rng_state_dict["numpy"][0]  # type: ignore[index]
        assert np.array_equal(np.random.get_state()[1], rng_state_dict["numpy"][1])  # type: ignore[index]
        assert python_get_rng_state() == rng_state_dict["python"]

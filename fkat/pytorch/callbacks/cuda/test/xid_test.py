# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch, call

import lightning as L

from fkat.pytorch.callbacks.cuda import xid
from fkat.pytorch.callbacks.cuda.xid import Xid
from fkat.pytorch.actions import LightningAction
from fkat.pytorch.schedule import Schedule


class TestXidCallback(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        # Create mock actions
        self.mock_log_action = MagicMock(spec=LightningAction)
        self.mock_reboot_action = MagicMock(spec=LightningAction)
        self.mock_terminate_action = MagicMock(spec=LightningAction)

        # Create mock schedule
        self.mock_schedule = MagicMock(spec=Schedule)

        # Create actions dictionary
        self.actions = {
            "0-10": self.mock_log_action,
            "13,43,48": self.mock_reboot_action,
            "81": self.mock_terminate_action,
        }

        # Create the callback
        self.callback = Xid(actions=self.actions, schedule=self.mock_schedule)

        # Replace the multiprocessing objects with mocks
        self.callback.xid_check = MagicMock()
        self.callback.xid_errors = MagicMock()

    def test_parse_xid_ranges(self):
        # Arrange
        expected_actions = {
            0: self.mock_log_action,
            1: self.mock_log_action,
            2: self.mock_log_action,
            3: self.mock_log_action,
            4: self.mock_log_action,
            5: self.mock_log_action,
            6: self.mock_log_action,
            7: self.mock_log_action,
            8: self.mock_log_action,
            9: self.mock_log_action,
            10: self.mock_log_action,
            13: self.mock_reboot_action,
            43: self.mock_reboot_action,
            48: self.mock_reboot_action,
            81: self.mock_terminate_action,
        }

        # Act
        parsed_actions = self.callback.actions

        # Assert
        assert parsed_actions == expected_actions
        assert len(parsed_actions) == 15  # 11 from range 0-10, plus 13, 43, 48, 81

    @patch(f"{xid.__name__}.multiprocessing.Process")
    @patch(f"{xid.__name__}.detect_xid_errors")
    @patch(f"{xid.__name__}.local_rank_zero_only")
    def test_setup(self, mock_rank_zero, mock_detect_xid, mock_process):
        # Arrange
        mock_rank_zero.return_value = lambda func: func  # Make decorator pass-through
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_module = MagicMock(spec=L.LightningModule)
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        # Act
        self.callback.setup(mock_trainer, mock_module, "fit")

        # Assert
        mock_process.assert_called_once_with(
            target=mock_detect_xid, args=(self.callback.xid_check, self.callback.xid_errors)
        )
        mock_process_instance.start.assert_called_once()
        assert self.callback.monitor == mock_process_instance

    @patch(f"{xid.__name__}.log")
    def test_terminate_monitor(self, mock_log):
        # Arrange
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        self.callback.monitor = mock_process

        # Act
        self.callback._terminate_monitor()

        # Assert
        mock_process.kill.assert_called_once()
        mock_log.info.assert_called_once_with("Terminating Xid errors monitor")

    @patch(f"{xid.__name__}.log")
    def test_terminate_monitor_not_alive(self, mock_log):
        # Arrange
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        self.callback.monitor = mock_process

        # Act
        self.callback._terminate_monitor()

        # Assert
        mock_process.kill.assert_not_called()

    @patch(f"{xid.__name__}.local_rank_zero_only")
    @patch.object(Xid, "_terminate_monitor")
    def test_on_exception(self, mock_terminate, mock_rank_zero):
        # Arrange
        mock_rank_zero.return_value = lambda func: func  # Make decorator pass-through
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_module = MagicMock(spec=L.LightningModule)

        # Act
        self.callback.on_exception(mock_trainer, mock_module, Exception("Test exception"))

        # Assert
        mock_terminate.assert_called_once()

    @patch(f"{xid.__name__}.local_rank_zero_only")
    @patch.object(Xid, "_terminate_monitor")
    def test_teardown(self, mock_terminate, mock_rank_zero):
        # Arrange
        mock_rank_zero.return_value = lambda func: func  # Make decorator pass-through
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_module = MagicMock(spec=L.LightningModule)

        # Act
        self.callback.teardown(mock_trainer, mock_module, "fit")

        # Assert
        mock_terminate.assert_called_once()

    @patch(f"{xid.__name__}.rank_zero_only")
    def test_check_with_xid_errors(self, mock_rank_zero):
        # Arrange
        mock_rank_zero.return_value = lambda func: func  # Make decorator pass-through
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_trainer.global_step = 100

        # Set up schedule to return True
        self.mock_schedule.check.return_value = True

        # Set up xid_errors queue with mock methods
        mock_empty = MagicMock(side_effect=[False, False, True])
        mock_get = MagicMock(side_effect=[{13, 81}, {48}])
        self.callback.xid_errors.empty = mock_empty  # type: ignore[method-assign]
        self.callback.xid_errors.get = mock_get  # type: ignore[method-assign]

        # Act
        self.callback.check(mock_trainer, "train", 5)

        # Assert
        self.mock_schedule.check.assert_called_once_with(stage="train", batch_idx=5, step=100)
        self.callback.xid_check.set.assert_called_once()

        # Check that actions were performed for each XID
        self.mock_reboot_action.perform.assert_has_calls(
            [call(trainer=mock_trainer, xid=13), call(trainer=mock_trainer, xid=48)]
        )
        self.mock_terminate_action.perform.assert_called_once_with(trainer=mock_trainer, xid=81)

    @patch(f"{xid.__name__}.rank_zero_only")
    def test_check_no_schedule_match(self, mock_rank_zero):
        # Arrange
        mock_rank_zero.return_value = lambda func: func  # Make decorator pass-through
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_trainer.global_step = 100

        # Set up schedule to return False
        self.mock_schedule.check.return_value = False

        # Act
        self.callback.check(mock_trainer, "train", 5)

        # Assert
        self.mock_schedule.check.assert_called_once_with(stage="train", batch_idx=5, step=100)
        # Use assert_not_called() on the MagicMock object, not the function
        self.callback.xid_check.set.assert_not_called()

    @patch(f"{xid.__name__}.rank_zero_only")
    @patch.object(Xid, "check")
    def test_on_train_batch_start(self, mock_check, mock_rank_zero):
        # Arrange
        mock_rank_zero.return_value = lambda func: func  # Make decorator pass-through
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_module = MagicMock(spec=L.LightningModule)
        mock_batch = {"data": [1, 2, 3]}
        batch_idx = 10

        # Act
        self.callback.on_train_batch_start(mock_trainer, mock_module, mock_batch, batch_idx)

        # Assert
        mock_check.assert_called_once_with(mock_trainer, "train", batch_idx)

    @patch(f"{xid.__name__}.rank_zero_only")
    @patch.object(Xid, "check")
    def test_on_validation_batch_start(self, mock_check, mock_rank_zero):
        # Arrange
        mock_rank_zero.return_value = lambda func: func  # Make decorator pass-through
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_module = MagicMock(spec=L.LightningModule)
        mock_batch = {"data": [1, 2, 3]}
        batch_idx = 5

        # Act
        self.callback.on_validation_batch_start(mock_trainer, mock_module, mock_batch, batch_idx)

        # Assert
        mock_check.assert_called_once_with(mock_trainer, "validation", batch_idx)

    @patch(f"{xid.__name__}.rank_zero_only")
    def test_unknown_xid(self, mock_rank_zero):
        # Arrange
        mock_rank_zero.return_value = lambda func: func  # Make decorator pass-through
        mock_trainer = MagicMock(spec=L.Trainer)

        # Set up schedule to return True
        self.mock_schedule.check.return_value = True

        # Set up xid_errors queue with mock methods
        mock_empty = MagicMock(side_effect=[False, True])
        mock_get = MagicMock(return_value={999})
        self.callback.xid_errors.empty = mock_empty  # type: ignore[method-assign]
        self.callback.xid_errors.get = mock_get  # type: ignore[method-assign]

        # Act
        self.callback.check(mock_trainer, "train", 5)

        # Assert
        # No actions should be performed for unknown XIDs
        self.mock_log_action.perform.assert_not_called()
        self.mock_reboot_action.perform.assert_not_called()
        self.mock_terminate_action.perform.assert_not_called()

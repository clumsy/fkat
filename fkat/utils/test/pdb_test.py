# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from unittest import TestCase
from unittest.mock import patch, MagicMock

from fkat.utils.pdb import ForkedPdb, post_mortem  # adjust import path as needed


class TestForkedPdb(TestCase):
    def setUp(self):
        self.pdb = ForkedPdb()
        self.original_stdin = sys.stdin
        # Save original environ
        self.original_environ = dict(os.environ)

    def tearDown(self):
        sys.stdin = self.original_stdin
        # Restore original environ
        os.environ.clear()
        os.environ.update(self.original_environ)

    def test_init_rank_detection(self):
        # Test PyTorch DDP rank
        os.environ["RANK"] = "0"
        pdb = ForkedPdb()
        assert pdb.rank == "0"

        # Test MPI rank
        os.environ.clear()
        os.environ["PMI_RANK"] = "1"
        pdb = ForkedPdb()
        assert pdb.rank == "1"

        # Test OpenMPI rank
        os.environ.clear()
        os.environ["OMPI_COMM_WORLD_RANK"] = "2"
        pdb = ForkedPdb()
        assert pdb.rank == "2"

        # Test fallback
        os.environ.clear()
        pdb = ForkedPdb()
        assert pdb.rank == "unknown"

    @patch("builtins.print")
    def test_print_rank_info(self, mock_print):
        self.pdb.rank = "0"
        with patch("os.getpid", return_value=12345):
            self.pdb.print_rank_info()
            mock_print.assert_called_once_with("\n[RANK=0, PID=12345]:")

    @patch("builtins.open")
    @patch("pdb.Pdb.interaction")
    @patch.object(ForkedPdb, "print_rank_info")
    def test_interaction(self, mock_print_rank, mock_pdb_interaction, mock_open):
        # Arrange
        mock_file = MagicMock()
        mock_open.return_value = mock_file

        # Act
        self.pdb.interaction(None, None, kwarg1="value1")

        # Assert
        mock_open.assert_called_once_with("/dev/stdin")
        mock_print_rank.assert_called_once()
        mock_pdb_interaction.assert_called_once_with(self.pdb, None, None, kwarg1="value1")
        assert sys.stdin == self.original_stdin

    @patch.object(ForkedPdb, "print_rank_info")
    def test_debug_commands(self, mock_print_rank):
        commands = [
            ("do_continue", "c"),
            ("do_next", "n"),
            ("do_step", "s"),
            ("do_return", "r"),
            ("do_quit", "q"),
            ("do_jump", "j"),
        ]

        for method_name, arg in commands:
            with patch("pdb.Pdb." + method_name) as mock_method:
                # Act
                method = getattr(self.pdb, method_name)
                method(arg)

                # Assert
                mock_print_rank.assert_called_once()
                mock_method.assert_called_once_with(arg)
                mock_print_rank.reset_mock()

    @patch.object(ForkedPdb, "print_rank_info")
    def test_precmd(self, mock_print_rank):
        test_line = "next"
        result = self.pdb.precmd(test_line)
        mock_print_rank.assert_called_once()
        assert result == test_line

    @patch.object(ForkedPdb, "print_rank_info")
    def test_default(self, mock_print_rank):
        test_line = "invalid_command"
        with patch("pdb.Pdb.default") as mock_default:
            self.pdb.default(test_line)
            mock_print_rank.assert_called_once()
            mock_default.assert_called_once_with(test_line)

    @patch("pdb.Pdb.reset")
    @patch.object(ForkedPdb, "interaction")
    def test_post_mortem(self, mock_interaction, mock_reset):
        test_tb = MagicMock()
        self.pdb.post_mortem(test_tb)  # type: ignore[arg-type]
        mock_reset.assert_called_once()
        mock_interaction.assert_called_once_with(None, test_tb)

    def test_post_mortem_function(self):
        original_excepthook = sys.excepthook
        post_mortem()
        assert sys.excepthook != original_excepthook

        test_type = Exception
        test_value = Exception("test")
        test_tb = MagicMock()

        with patch.object(ForkedPdb, "post_mortem") as mock_post_mortem:
            sys.excepthook(test_type, test_value, test_tb)  # type: ignore[arg-type]
            mock_post_mortem.assert_called_once_with(test_tb)

    def test_real_exception_handling(self):
        post_mortem()

        with patch.object(ForkedPdb, "post_mortem") as mock_post_mortem:
            try:
                raise ValueError("Test exception")
            except ValueError:
                type_, value_, traceback_ = sys.exc_info()
                assert type_ is not None and value_ is not None
                sys.excepthook(type_, value_, traceback_)
                mock_post_mortem.assert_called_once()
                assert mock_post_mortem.call_args[0][0] == traceback_

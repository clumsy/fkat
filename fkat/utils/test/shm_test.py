# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import ANY, MagicMock, call, mock_open, patch

from fkat.utils import shm


class ShmTest(TestCase):
    @patch(f"{shm.__name__}.pickle")
    @patch("builtins.open", new_callable=mock_open)
    def test_save(self, mock_open, mock_pickle):
        # Arrange
        path = MagicMock(spec=Path)
        obj = object()
        buf = MagicMock()
        # Act
        shm.save(obj, path)
        # Assert
        path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_pickle.dump.assert_called_once_with(obj, mock_open.return_value, protocol=5, buffer_callback=ANY)
        mock_open.return_value.write.assert_not_called()
        args, kwargs = mock_pickle.dump.call_args_list[0]
        buffer_callback = kwargs["buffer_callback"]
        buffer_callback(buf)
        mock_open.return_value.write.assert_called_once_with(buf)
        path.__truediv__.return_value.touch.assert_called_once()

    @patch(f"{shm.__name__}.pickle")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_iter(self, mock_open, mock_pickle):
        # Arrange
        it = iter([1, 2, 3])
        path = MagicMock(spec=Path)
        max_items = 3

        def should_stop():
            try:
                next(it)
                return False
            except StopIteration:
                return True

        # Act
        res = shm.save_iter(it, path, max_items, should_stop)
        # Assert
        path.__truediv__.return_value.__truediv__.return_value.touch.assert_any_call()
        assert path == res

    @patch(f"{shm.__name__}.pickle")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_iter_truncate(self, mock_open, mock_pickle):
        # Arrange
        it = iter(list(range(100)))
        path = MagicMock(spec=Path)
        max_items = 3

        def should_stop():
            try:
                next(it)
                return False
            except StopIteration:
                return True

        # Act
        res = shm.save_iter(it, path, max_items, should_stop, truncation_threshold=5)
        # Assert
        assert path.__truediv__.call_count == 5 + 1  # 5 iterations plus POISON_PILL
        assert path == res

    @patch(f"{shm.__name__}.pickle")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_iter_empty(self, mock_open, mock_pickle):
        # Arrange
        it = iter([])
        path = MagicMock(spec=Path)
        # Act
        res = shm.save_iter(it, path, 3, lambda: False)
        # Assert
        path.__truediv__.return_value.__truediv__.return_value.touch.assert_any_call()
        assert path == res

    @patch(f"{shm.__name__}.mmap")
    @patch(f"{shm.__name__}.pickle.load")
    @patch(f"{shm.__name__}.os")
    @patch(f"{shm.__name__}.shutil")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_mmap(self, mock_open, mock_shutil, mock_os, mock_pickle_load, mock_mmap):
        # Arrange
        path = MagicMock(spec=Path)
        path.iterdir.return_value = [0, 1, 2]
        mock_os.stat.return_value = (stats := MagicMock(spec=os.stat_result))
        stats.st_size = 42
        mb = MagicMock()
        mb.close = MagicMock()
        mock_mmap.mmap.return_value = mb

        # Act
        res = shm.load(path)

        # Assert
        mock_shutil.rmtree.assert_called_once_with(path)
        assert res == mock_pickle_load.return_value
        mb.close.assert_called_once()
        mb.close.assert_called()

    @patch(f"{shm.__name__}.mmap")
    @patch(f"{shm.__name__}.pickle.load")
    @patch(f"{shm.__name__}.os")
    @patch(f"{shm.__name__}.shutil")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_pickle(self, mock_open, mock_shutil, mock_os, mock_pickle_load, mock_mmap):
        # Arrange
        path = MagicMock(spec=Path)
        path.iterdir.return_value = [0, 1, 2]
        mock_os.stat.return_value = (stats := MagicMock(spec=os.stat_result))
        stats.st_size = 0

        # Create a mock PickleBuffer class that isinstance will recognize
        class MockPickleBuffer:
            def __init__(self, data):
                self.data = data
                self.release = MagicMock()

        with patch(f"{shm.__name__}.pickle.PickleBuffer", MockPickleBuffer):
            # Act
            res = shm.load(path)

            # Assert
            mock_shutil.rmtree.assert_called_once_with(path)
            assert res == mock_pickle_load.return_value
            # The buffer should have been created and release called
            # Since st_size is 0, a PickleBuffer is created

    @patch(f"{shm.__name__}.mmap")
    @patch(f"{shm.__name__}.pickle")
    @patch(f"{shm.__name__}.os")
    @patch(f"{shm.__name__}.shutil")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_iter(self, mock_open, mock_shutil, mock_os, mock_pickle, mock_mmap):
        # Arrange
        path = MagicMock(spec=Path)
        num_chunks = 3
        path.iterdir.return_value = list(range(num_chunks))
        chunk_path = path / "test"
        (chunk_path / shm.COMPLETE).exists.side_effect = [False] + [True] * 100
        chunk_path.iterdir.return_value = [0]
        chunks = [f"chunk{i}" for i in range(num_chunks)]
        mock_pickle.load.side_effect = [*chunks, shm.POISON_PILL]
        mock_os.stat.return_value = (stats := MagicMock(spec=os.stat_result))
        stats.st_size = 42
        wait_callback = MagicMock()
        # Act
        it = shm.load_iter(path, wait_callback=wait_callback)
        # Assert
        path.assert_not_called()
        assert chunks == list(it)
        for i in range(num_chunks + 1):
            path.__truediv__.assert_any_call(str(i))
        mock_pickle.load.assert_has_calls([call(mock_open.return_value, buffers=[]) for _ in range(num_chunks + 1)])
        wait_callback.assert_called_once()

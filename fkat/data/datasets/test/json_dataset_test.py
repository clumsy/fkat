# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import TestCase
from unittest.mock import MagicMock, patch
from pyarrow.fs import S3FileSystem  # type: ignore[possibly-unbound-import]

from fkat.data.datasets import json
from fkat.data.datasets.json import JsonDataset, IterableJsonDataset


class IterableJsonDatasetTest(TestCase):
    @patch(f"{json.__name__}.pa_iter_rows")
    @patch(f"{json.__name__}.pj")
    @patch(f"{json.__name__}.FileSystem")
    def test_iter_non_s3(self, mock_fs, mock_pj, mock_iter_rows):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        fs.open_input_file.return_value = MagicMock(spec=["read", "__enter__", "__exit__"])
        tbl = mock_pj.read_json.return_value
        chunk_size = 42
        # Act
        ds = IterableJsonDataset("some://uri", chunk_size=chunk_size)
        # Assert
        assert iter(ds) == mock_iter_rows.return_value
        mock_iter_rows.assert_called_once_with(tbl, chunk_size=chunk_size)

    @patch(f"{json.__name__}.pd_iter_rows")
    @patch(f"{json.__name__}.session")
    @patch(f"{json.__name__}.s3wr")
    @patch(f"{json.__name__}.FileSystem")
    def test_iter(self, mock_fs, mock_s3wr, mock_configure_fs, mock_iter_rows):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        chunk_size = 42
        # Act
        ds = IterableJsonDataset("some://uri", chunk_size=chunk_size)
        # Assert
        assert iter(ds) == mock_iter_rows.return_value


class JsonDatasetTest(TestCase):
    @patch(f"{json.__name__}.pd")
    @patch(f"{json.__name__}.pj")
    @patch(f"{json.__name__}.pa")
    @patch(f"{json.__name__}.FileSystem")
    def test_len_non_s3(self, mock_fs, mock_pa, mock_pj, mock_pd):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        fs.open_input_file.return_value = MagicMock(spec=["read", "__enter__", "__exit__"])
        df = mock_pd.concat.return_value
        # Act
        ds = JsonDataset("some://uri")
        # Assert
        assert len(ds) == len(df)

    @patch(f"{json.__name__}.session")
    @patch(f"{json.__name__}.s3wr")
    @patch(f"{json.__name__}.FileSystem")
    def test_len(self, mock_fs, mock_s3wr, mock_configure_fs):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        df = mock_s3wr.s3.read_json.return_value
        # Act
        ds = JsonDataset("some://uri")
        # Assert
        assert len(ds) == len(df)
        mock_s3wr.s3.read_json.assert_called_once_with(
            "some://uri", lines=True, boto3_session=mock_configure_fs.return_value, path_suffix="json"
        )

    @patch(f"{json.__name__}.pd")
    @patch(f"{json.__name__}.pj")
    @patch(f"{json.__name__}.pa")
    @patch(f"{json.__name__}.FileSystem")
    def test_getitem_non_s3(self, mock_fs, mock_pa, mock_pj, mock_pd):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        fs.open_input_file.return_value = MagicMock(spec=["read", "__enter__", "__exit__"])
        df = mock_pd.concat.return_value
        # Act
        ds = JsonDataset("some://uri")
        # Assert
        assert ds[7] == df.iloc.__getitem__.return_value.to_dict.return_value

    @patch(f"{json.__name__}.session")
    @patch(f"{json.__name__}.s3wr")
    @patch(f"{json.__name__}.FileSystem")
    def test_getitem(self, mock_fs, mock_s3wr, mock_configure_fs):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        df = mock_s3wr.s3.read_json.return_value.replace.return_value
        # Act
        ds = JsonDataset("some://uri")
        # Assert
        assert ds[7] == df.iloc.__getitem__.return_value.to_dict.return_value
        mock_s3wr.s3.read_json.assert_called_once_with(
            "some://uri", lines=True, boto3_session=mock_configure_fs.return_value, path_suffix="json"
        )

    @patch(f"{json.__name__}.pd")
    @patch(f"{json.__name__}.pj")
    @patch(f"{json.__name__}.FileSystem")
    def test_getitems_non_s3(self, mock_fs, mock_pj, mock_pd):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        fs.open_input_file.return_value = MagicMock(spec=["read", "__enter__", "__exit__"])
        df = mock_pd.concat.return_value
        # Act
        ds = JsonDataset("some://uri")
        # Assert
        assert ds.__getitems__(list(range(7))) == [
            df.iloc.__getitem__.return_value.iloc.__getitem__.return_value.to_dict.return_value for _ in range(7)
        ]

    @patch(f"{json.__name__}.pd")
    @patch(f"{json.__name__}.pj")
    @patch(f"{json.__name__}.pa")
    @patch(f"{json.__name__}.FileSystem")
    def test_getitems_non_s3_by_list(self, mock_fs, mock_pa, mock_pj, mock_pd):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        fs.open_input_file.return_value = MagicMock(spec=["read", "__enter__", "__exit__"])
        df = mock_pd.concat.return_value
        # Act
        ds = JsonDataset(["some://uri"])
        # Assert
        assert ds.__getitems__(list(range(7))) == [
            df.iloc.__getitem__.return_value.iloc.__getitem__.return_value.to_dict.return_value for _ in range(7)
        ]

    @patch(f"{json.__name__}.session")
    @patch(f"{json.__name__}.s3wr")
    @patch(f"{json.__name__}.FileSystem")
    @patch(f"{json.__name__}.pa")
    def test_getitems(self, mock_pa, mock_fs, mock_s3wr, mock_configure_fs):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        df = mock_s3wr.s3.read_json.return_value.replace.return_value
        # Act
        ds = JsonDataset("some://uri")
        # Assert
        assert ds.__getitems__(list(range(7))) == [
            df.iloc.__getitem__.return_value.iloc.__getitem__.return_value.to_dict.return_value for _ in range(7)
        ]
        mock_s3wr.s3.read_json.assert_called_once_with(
            "some://uri", lines=True, boto3_session=mock_configure_fs.return_value, path_suffix="json"
        )

    @patch(f"{json.__name__}.session")
    @patch(f"{json.__name__}.s3wr")
    @patch(f"{json.__name__}.FileSystem")
    @patch(f"{json.__name__}.pa")
    def test_getitems_from_uri_list(self, mock_pa, mock_fs, mock_s3wr, mock_configure_fs):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        df = mock_s3wr.s3.read_json.return_value.replace.return_value
        # Act
        ds = JsonDataset(["some://uri"])
        # Assert
        assert ds.__getitems__(list(range(7))) == [
            df.iloc.__getitem__.return_value.iloc.__getitem__.return_value.to_dict.return_value for _ in range(7)
        ]
        mock_s3wr.s3.read_json.assert_called_once_with(
            ["some://uri"], lines=True, boto3_session=mock_configure_fs.return_value, path_suffix="json"
        )

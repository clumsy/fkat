# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import TestCase
from unittest.mock import MagicMock, patch
from pyarrow.fs import S3FileSystem  # type: ignore[possibly-unbound-import]
from fkat.data.datasets import parquet
from fkat.data.datasets.parquet import ParquetDataset, IterableParquetDataset


class IterableJsonDatasetTest(TestCase):
    @patch(f"{parquet.__name__}.pa_iter_rows")
    @patch(f"{parquet.__name__}.FileSystem")
    @patch(f"{parquet.__name__}.pa")
    def test_iter_non_s3(self, mock_pa, mock_fs, mock_iter_rows):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        fs.open_input_file.return_value = MagicMock(spec=["read", "__enter__", "__exit__"])
        pds = mock_pa.parquet.ParquetDataset.return_value
        tbl = pds.read.return_value
        chunk_size = 42
        # Act
        ds = IterableParquetDataset("some://uri", chunk_size=chunk_size)
        # Assert
        assert iter(ds) == mock_iter_rows.return_value
        mock_iter_rows.assert_called_once_with(tbl, chunk_size=chunk_size)

    @patch(f"{parquet.__name__}.pa_iter_rows")
    @patch(f"{parquet.__name__}.pa")
    @patch(f"{parquet.__name__}.FileSystem")
    def test_iter_non_s3_by_list(self, mock_fs, mock_pa, mock_iter_rows):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        fs.open_input_file.return_value = MagicMock(spec=["read", "__enter__", "__exit__"])
        pds = mock_pa.parquet.ParquetDataset.return_value
        tbl = pds.read.return_value
        chunk_size = 42

        # Act
        ds = IterableParquetDataset(["some://uri"], chunk_size=chunk_size)

        # Assert
        assert iter(ds) == mock_iter_rows.return_value
        mock_iter_rows.assert_called_once_with(tbl, chunk_size=chunk_size)

    @patch(f"{parquet.__name__}.pd_iter_rows")
    @patch(f"{parquet.__name__}.session")
    @patch(f"{parquet.__name__}.s3wr")
    @patch(f"{parquet.__name__}.FileSystem")
    @patch(f"{parquet.__name__}.pa")
    def test_iter(self, mock_pa, mock_fs, mock_s3wr, mock_configure_fs, mock_iter_rows):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        chunk_size = 42
        # Act
        ds = IterableParquetDataset("some://uri", chunk_size=chunk_size)
        # Assert
        assert iter(ds) == mock_iter_rows.return_value
        mock_s3wr.s3.read_parquet.assert_called_once_with(
            "some://uri",
            use_threads=True,
            columns=None,
            chunked=42,
            boto3_session=mock_configure_fs.return_value,
            path_suffix="parquet",
        )


class ParquetDatasetTest(TestCase):
    @patch(f"{parquet.__name__}.FileSystem")
    @patch(f"{parquet.__name__}.pa")
    def test_len_non_s3(self, mock_pa, mock_fs):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        pds = mock_pa.parquet.ParquetDataset.return_value
        tbl = pds.read.return_value
        df = tbl.to_pandas.return_value
        # Act
        ds = ParquetDataset("some://uri")
        # Assert
        assert len(ds) == len(df)

    @patch(f"{parquet.__name__}.session")
    @patch(f"{parquet.__name__}.s3wr")
    @patch(f"{parquet.__name__}.FileSystem")
    @patch(f"{parquet.__name__}.pa")
    def test_len(self, mock_pa, mock_fs, mock_s3wr, mock_configure_fs):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        df = mock_s3wr.s3.read_parquet.return_value.replace.return_value
        mock_configure_fs.return_value = "mock_configure_fs"
        # Act
        ds = ParquetDataset("some://uri")
        # Assert
        mock_configure_fs.assert_called_once_with(clients=["s3"])
        assert len(ds) == len(df)
        mock_s3wr.s3.read_parquet.assert_called_once_with(
            "some://uri", use_threads=True, columns=None, boto3_session="mock_configure_fs", path_suffix="parquet"
        )

    @patch(f"{parquet.__name__}.pa")
    @patch(f"{parquet.__name__}.FileSystem")
    def test_getitem_non_s3(self, mock_fs, mock_pa):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        pds = mock_pa.parquet.ParquetDataset.return_value
        tbl = pds.read.return_value
        df = tbl.to_pandas.return_value
        # Act
        ds = ParquetDataset("some://uri")
        # Assert
        assert ds[7] == df.iloc.__getitem__.return_value.to_dict.return_value

    @patch(f"{parquet.__name__}.session")
    @patch(f"{parquet.__name__}.s3wr")
    @patch(f"{parquet.__name__}.FileSystem")
    def test_getitem(self, mock_fs, mock_s3wr, mock_configure_fs):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        df = mock_s3wr.s3.read_parquet.return_value.replace.return_value
        # Act
        ds = ParquetDataset("some://uri")
        # Assert
        mock_configure_fs.assert_called_once_with(clients=["s3"])
        assert ds[7] == df.iloc.__getitem__.return_value.to_dict.return_value
        mock_s3wr.s3.read_parquet.assert_called_once_with(
            "some://uri",
            use_threads=True,
            columns=None,
            boto3_session=mock_configure_fs.return_value,
            path_suffix="parquet",
        )

    @patch(f"{parquet.__name__}.FileSystem")
    @patch(f"{parquet.__name__}.pa")
    def test_getitems_non_s3(self, mock_pa, mock_fs):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        pds = mock_pa.parquet.ParquetDataset.return_value
        tbl = pds.read.return_value
        df = tbl.to_pandas.return_value
        # Act
        ds = ParquetDataset("some://uri")
        # Assert
        assert ds.__getitems__(list(range(7))) == [
            df.iloc.__getitem__.return_value.iloc.__getitem__.return_value.to_dict.return_value for _ in range(7)
        ]

    @patch(f"{parquet.__name__}.FileSystem")
    @patch(f"{parquet.__name__}.pa")
    def test_getitems_non_s3_by_list(self, mock_pa, mock_fs):
        # Arrange
        fs, path = MagicMock(), "some/path"
        mock_fs.from_uri.return_value = fs, path
        pds = mock_pa.parquet.ParquetDataset.return_value
        tbl = pds.read.return_value
        df = tbl.to_pandas.return_value
        # Act
        ds = ParquetDataset(["some://uri"])
        # Assert
        assert ds.__getitems__(list(range(7))) == [
            df.iloc.__getitem__.return_value.iloc.__getitem__.return_value.to_dict.return_value for _ in range(7)
        ]

    @patch(f"{parquet.__name__}.session")
    @patch(f"{parquet.__name__}.s3wr")
    @patch(f"{parquet.__name__}.FileSystem")
    def test_getitems(self, mock_fs, mock_s3wr, mock_configure_fs):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        df = mock_s3wr.s3.read_parquet.return_value.replace.return_value
        # Act
        ds = ParquetDataset("some://uri")
        # Assert
        mock_configure_fs.assert_called_once_with(clients=["s3"])
        assert ds.__getitems__(list(range(7))) == [
            df.iloc.__getitem__.return_value.iloc.__getitem__.return_value.to_dict.return_value for _ in range(7)
        ]
        mock_s3wr.s3.read_parquet.assert_called_once_with(
            "some://uri",
            use_threads=True,
            columns=None,
            boto3_session=mock_configure_fs.return_value,
            path_suffix="parquet",
        )

    @patch(f"{parquet.__name__}.session")
    @patch(f"{parquet.__name__}.s3wr")
    @patch(f"{parquet.__name__}.FileSystem")
    @patch(f"{parquet.__name__}.pa")
    def test_getitems_from_uri_list(self, mock_pa, mock_fs, mock_s3wr, mock_configure_fs):
        # Arrange
        fs, path = MagicMock(spec=S3FileSystem), "some/path"
        mock_fs.from_uri.return_value = fs, path
        df = mock_s3wr.s3.read_parquet.return_value.replace.return_value
        # Act
        ds = ParquetDataset(["some://uri"])
        # Assert
        mock_configure_fs.assert_called_once_with(clients=["s3"])
        assert ds.__getitems__(list(range(7))) == [
            df.iloc.__getitem__.return_value.iloc.__getitem__.return_value.to_dict.return_value for _ in range(7)
        ]
        mock_s3wr.s3.read_parquet.assert_called_once_with(
            ["some://uri"],
            use_threads=True,
            columns=None,
            boto3_session=mock_configure_fs.return_value,
            path_suffix="parquet",
        )

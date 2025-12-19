# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from collections.abc import Iterator
import awswrangler as s3wr
from pyarrow.fs import FileSystem, S3FileSystem  # type: ignore[possibly-unbound-import]
import numpy as np
import pyarrow as pa
from torch.utils.data import IterableDataset

from fkat.data.datasets import SizedDataset
from fkat.utils.pyarrow import iter_rows as pa_iter_rows
from fkat.utils.pandas import iter_rows as pd_iter_rows
from fkat.utils.boto3 import session


class IterableParquetDataset(IterableDataset[dict[str, Any]]):
    """
    An :class:`IterableDataset` backed by Parquet data.

    .. note:: If you want to keep the original type from reading parquet,
        you should set ``dtype_backend='pyarrow'``.

        example config:

        .. code-block:: yaml

            _target_: fkat.data.datasets.parquet.IterableParquetDataset
            uri: s3://path/to/fkat.parquet
            dtype_backend: pyarrow

    Args:
        uri (str or list[str]): URI of Parquet data.
        columns (List[str], optional): Columns to load.
            Default to ``None``
        use_threads (bool): Use multi-threaded processing.
            Default to ``True``.
        chunk_size (int): An iterable of DataFrames is returned with maximum rows equal to the received INTEGER.
            Default to ``10000``
        replace_nan (bool): Whether to replace np.nan as None.
            Default to ``True``
        s3wr_args (dict): config for s3wr.s3.read_parquet,
            refer to https://aws-sdk-pandas.readthedocs.io/en/3.5.1/stubs/awswrangler.s3.read_parquet.html
    """

    def __init__(
        self,
        uri: str | list[str],
        columns: list[str] | None = None,
        use_threads: bool = True,
        chunk_size: int = 10000,
        replace_nan: bool = True,
        **s3wr_args: Any,
    ) -> None:
        fs: FileSystem
        path: str
        if isinstance(uri, str):
            fs, path = FileSystem.from_uri(uri)
        else:
            fs, path = FileSystem.from_uri(uri[0])
        self.s3_file = isinstance(fs, S3FileSystem)
        if self.s3_file:
            self.uri = uri
            self.use_threads = use_threads
            self.columns = columns
            self.chunk_size = chunk_size
            self.replace_nan = replace_nan
            self.s3wr_args = s3wr_args
        else:
            # otherwise, use pyarrow
            path_list = []
            if isinstance(uri, str):
                _, path = FileSystem.from_uri(uri)
                path_list.append(path)
            else:
                path_list = []
                for each in uri:
                    _, path = FileSystem.from_uri(each)
                    path_list.append(path)
            pds = pa.parquet.ParquetDataset(path_list, filesystem=fs)  # type: ignore
            self.tbl = pds.read(columns, use_threads=use_threads, use_pandas_metadata=False)
            self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Creates dataset iterator.
        Returns:
            Iterator[dict[str, Any]]: dataset iterator
        """
        if self.s3_file:
            return pd_iter_rows(
                s3wr.s3.read_parquet(
                    self.uri,
                    use_threads=self.use_threads,
                    columns=self.columns,
                    chunked=self.chunk_size,
                    boto3_session=session(clients=["s3"]),
                    path_suffix="parquet",
                    **self.s3wr_args,
                ),
                self.replace_nan,
            )
        else:
            return pa_iter_rows(self.tbl, chunk_size=self.chunk_size)


class ParquetDataset(SizedDataset[int, dict[str, Any]]):
    """
    A :class:`Dataset` backed by Parquet data.

    Create a :class:`Dataset` from Parquet data at the specified URI.

    .. note:: If you want to keep the original type from reading parquet,
        you should set ``dtype_backend='pyarrow'``.

        example config:

        .. code-block:: yaml

            _target_: fkat.data.datasets.parquet.ParquetDataset
            uri: s3://path/to/fkat.parquet
            dtype_backend: pyarrow

        Difference for ``dtype_backend`` between ``pyarrow`` and ``numpy_nullable``

        .. code-block:: python

            from fkat.data.datasets.parquet import ParquetDataset
            from fkat.utils.s3_utils import fs_save_prediction_output_parquet

            uri = "s3://path/to/fkat.parquet"

            saved_data = {
                "purchased_items": [
                    [
                        {"product_id": "PROD001", "item_index": 12345, "quantity": "1"},
                        {"product_id": "PROD002", "item_index": None, "quantity": "1"},
                    ],
                    [{"product_id": "PROD001", "item_index": 12345, "quantity": "1"}],
                ],
                "ground_truth": [[1, 2, 3], [1, 2]],
                "embeddings": [np.random.randn(128), np.random.randn(128)],
            }
            fs_save_prediction_output_parquet()(saved_data, uri)

            dataset = ParquetDataset(uri)  # dtype_backend: numpy_nullable
            print(type(dataset[0]["embeddings"]))  # type: numpy.ndarray
            print(type(dataset[0]["purchased_items"]))  # type: numpy.ndarray of object
            print(type(dataset[0]["ground_truth"]))  # type: numpy.ndarray of object

            pyarrow_dataset = ParquetDataset(uri, dtype_backend="pyarrow")  # dtype_backend: pyarrow
            print(type(pyarrow_dataset[0]["embeddings"]))  # type: list
            print(type(pyarrow_dataset[0]["purchased_items"]))  # type: list of dictionary
            print(type(pyarrow_dataset[0]["ground_truth"]))  # type: list of int

    Args:
        uri (str | list[str]): URI of Parquet data.
        columns (list[str] | None): Columns to load.
        use_threads (bool): Use multi-threaded processing.
            Defaults to ``True``.
        replace_nan (bool): Whether to replace np.nan as None.
            Default to ``True``
        s3wr_args (dict): config for s3wr.s3.read_parquet,
            refer to https://aws-sdk-pandas.readthedocs.io/en/3.5.1/stubs/awswrangler.s3.read_parquet.html
    """

    def __init__(
        self,
        uri: str | list[str],
        columns: list[str] | None = None,
        use_threads: bool = True,
        replace_nan: bool = True,
        **s3wr_args: Any,
    ) -> None:
        # check FileSystem
        fs: FileSystem
        path: str
        if isinstance(uri, str):
            fs, path = FileSystem.from_uri(uri)
        else:
            fs, path = FileSystem.from_uri(uri[0])
        if isinstance(fs, S3FileSystem):
            # if file is in S3, then use awswrangler
            self.df = s3wr.s3.read_parquet(
                uri,
                use_threads=use_threads,
                columns=columns,
                boto3_session=session(clients=["s3"]),
                path_suffix="parquet",
                **s3wr_args,
            )
            if replace_nan:
                self.df = self.df.replace({np.nan: None})
        else:
            # otherwise, use pyarrow
            path_list = []
            if isinstance(uri, str):
                fs, path = FileSystem.from_uri(uri)
                path_list.append(path)
            elif isinstance(uri, list):
                path_list = []
                for each in uri:
                    _, path = FileSystem.from_uri(each)
                    path_list.append(path)
            else:
                raise Exception(f"ParquetDataset can't support uri as {type(uri)}")
            pds = pa.parquet.ParquetDataset(path_list, filesystem=fs)  # type: ignore
            tbl = pds.read(columns, use_threads=use_threads, use_pandas_metadata=False)
            self.df = tbl.to_pandas()

    def __len__(self) -> int:
        """Get :class:`Dataset` size.

        Returns:
            int: :class:`Dataset` size.
        """
        return len(self.df)

    def __getitems__(self, idxs: list[int]) -> list[dict[str, Any]]:
        """Get a batch of samples at the specified indices.

        Args:
            idxs (list[int]): Samples' indices.

        Returns:
            list[dict[str, Any]]: A batch of samples.
        """
        series = self.df.iloc[idxs]
        samples = [series.iloc[i].to_dict() for i in range(len(idxs))]
        return samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample at the specified index.

        Args:
            idx (int): Sample index.

        Returns:
            dict[str, Any]: A sample.
        """
        series = self.df.iloc[idx]
        sample = series.to_dict()
        return sample

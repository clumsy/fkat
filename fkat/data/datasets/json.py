# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import awswrangler as s3wr
from typing import Any
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.json as pj
from pyarrow.fs import FileSystem, S3FileSystem  # type: ignore[possibly-unbound-import]
import pandas as pd
import numpy as np
from fkat.data.datasets import SizedDataset
from fkat.utils.pyarrow import iter_rows as pa_iter_rows
from fkat.utils.pandas import iter_rows as pd_iter_rows
from fkat.utils.boto3 import session
from torch.utils.data import IterableDataset


class IterableJsonDataset(IterableDataset[dict[str, Any]]):
    """
    An :class:`IterableDataset` backed by Json data.

    Args:
        uri (str | list[str]): URI of Parquet data.
        read_options: pyarrow.json.ReadOptions, optional
            Options for the JSON reader (see ReadOptions constructor for defaults).
        parse_options: pyarrow.json.ParseOptions, optional
            Options for the JSON parser
            (see ParseOptions constructor for defaults).
        memory_pool: MemoryPool, optional
            Pool to allocate Table memory from.
        chunk_size (int): An iterable of DataFrames is returned with maximum rows equal to the received INTEGER.
        replace_nan (bool): Whether to replace np.nan as None.
            Default to ``True``
        s3wr_args (dict): config for s3wr.s3.read_json,
            refer to https://aws-sdk-pandas.readthedocs.io/en/3.5.1/stubs/awswrangler.s3.read_parquet.html

    """

    def __init__(
        self,
        uri: str | list[str],
        read_options: pa.json.ReadOptions | None = None,
        parse_options: pa.json.ParseOptions | None = None,
        memory_pool: pa.MemoryPool | None = None,
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
            self.chunk_size = chunk_size
            self.replace_nan = replace_nan
            self.s3wr_args = s3wr_args
        else:
            assert isinstance(uri, str), "IterableJsonDataset can only accept uri as str"
            with fs.open_input_file(path) as f:
                self.tbl = pj.read_json(
                    f, read_options=read_options, parse_options=parse_options, memory_pool=memory_pool
                )
            self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Creates dataset iterator.
        Returns:
            Iterator[dict[str, Any]]: dataset iterator
        """
        if self.s3_file:
            return pd_iter_rows(
                s3wr.s3.read_json(
                    self.uri,
                    lines=True,
                    chunksize=self.chunk_size,
                    boto3_session=session(clients=["s3"]),
                    path_suffix="json",
                    **self.s3wr_args,
                ),
                self.replace_nan,
            )
        else:
            return pa_iter_rows(self.tbl, chunk_size=self.chunk_size)


class JsonDataset(SizedDataset[int, dict[str, Any]]):
    """
    Create a :class:`Dataset` from JSON data at the specified URI.

    Args:
        uri (str | list[str]): URI of JSON data.
        read_options (pa.json.ReadOptions | None): JSON read options.
        parse_options (pa.json.ParseOptions | None): JSON parse options.
        memory_pool (pa.MemoryPool | None): JSON processing memory pool configuration.
        replace_nan (bool): Whether to replace np.nan as None.
            Default to ``True``
        s3wr_args (Any): config for s3wr.s3.read_json,
            refer to https://aws-sdk-pandas.readthedocs.io/en/3.5.1/stubs/awswrangler.s3.read_json.html
    """

    def __init__(
        self,
        uri: str | list[str],
        read_options: pa.json.ReadOptions | None = None,
        parse_options: pa.json.ParseOptions | None = None,
        memory_pool: pa.MemoryPool | None = None,
        replace_nan: bool = True,
        **s3wr_args: Any,
    ) -> None:
        fs: FileSystem
        path: str
        if isinstance(uri, str):
            fs, path = FileSystem.from_uri(uri)
        else:
            fs, path = FileSystem.from_uri(uri[0])
        if isinstance(fs, S3FileSystem):
            self.df = s3wr.s3.read_json(
                uri, lines=True, boto3_session=session(clients=["s3"]), path_suffix="json", **s3wr_args
            )
            if replace_nan:
                self.df = self.df.replace({np.nan: None})
        else:
            path_list: list[str] = []
            if isinstance(uri, str):
                path_list.append(path)
            else:
                for each in uri:
                    _, path = FileSystem.from_uri(each)
                    path_list.append(path)
            df = []
            for each in path_list:
                with fs.open_input_file(each) as f:
                    tbl = pj.read_json(
                        f, read_options=read_options, parse_options=parse_options, memory_pool=memory_pool
                    )
                    df.append(tbl.to_pandas())
            self.df = pd.concat(df)

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

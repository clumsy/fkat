# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pyarrow as pa
from typing import Any
from collections.abc import Iterator


def iter_rows(table: pa.Table, chunk_size: int) -> Iterator[dict[str, Any]]:
    """
    Generator function to iterate over rows of a PyArrow table in chunks.

    Args:
        table (pa.Table): PyArrow table.
        chunk_size (int): The number of rows per chunk for processing.
    Yields:
        Dict[str, Any]: Dictionary representing each row.
    """
    for chunk in table.to_batches(chunk_size):
        columns = chunk.to_pydict()
        for i in range(chunk.num_rows):
            yield {col: columns[col][i] for col in columns}

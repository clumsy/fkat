# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
from typing import Any
from collections.abc import Iterator


def iter_rows(table: Iterator[pd.DataFrame], replace_nan: bool = True) -> Iterator[dict[str, Any]]:
    """
    Generator function to iterate over rows of a Pandas :class:`DataFrame`\\s in chunks.

    Args:
        table (Iterator[pd.DataFrame]): Pandas :class:`DataFrame`\\s.
        replace_nan (bool): Whether to replace NaN with None.
            Defaults to `True`.
    Yields:
        dict[str, Any]: Dictionary representing each row.
    """
    for chunk in table:
        if replace_nan:
            chunk = chunk.replace({np.nan: None})
        columns = chunk.to_dict()
        for idx in sorted(columns[list(columns.keys())[0]].keys()):
            yield {str(col): columns[col][idx] for col in columns}

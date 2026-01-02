# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from torch.utils.data import DataLoader


def tokenize_and_load(dataset: Any, tokenizer: Any, batch_size: int = 2, shuffle: bool = False) -> DataLoader:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples: Any) -> dict:
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    return DataLoader(tokenized, batch_size=batch_size, shuffle=shuffle)

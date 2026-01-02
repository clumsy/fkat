# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from torch.utils.data import DataLoader


def get_dataloader(
    dataset: Any, tokenizer: Any, batch_size: int = 2, shuffle: bool = False, num_workers: int = 0
) -> DataLoader:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples: Any) -> dict:
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokenized["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in ids] for ids in tokenized["input_ids"]
        ]
        return tokenized

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    return DataLoader(tokenized, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

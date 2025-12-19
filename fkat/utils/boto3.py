# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import boto3
from botocore.config import Config


def session(
    max_attempts: int = 6,
    mode: Literal["legacy", "standard", "adaptive"] = "standard",
    clients: list[str] | None = None,
) -> boto3.Session:
    config = Config(
        retries={
            "max_attempts": max_attempts,
            "mode": mode,
        }
    )
    session = boto3.Session()
    if clients:
        for client in clients:
            session.client(client, config=config)  # type: ignore
    return session

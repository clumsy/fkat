# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Protocol, Any


class LightningAction(Protocol):
    """A generic action to be executed given the context provided via key-value arguments."""

    def perform(self, **kwargs: Any) -> Any:
        """Performs the action with the context provided via key-value arguments."""
        ...

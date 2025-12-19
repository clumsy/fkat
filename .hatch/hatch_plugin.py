"""
Noop hatch plugin for environment collection
"""

from typing import Any
from hatch.env.collectors.plugin.interface import EnvironmentCollectorInterface


class CustomEnvironmentCollector(EnvironmentCollectorInterface):
    def finalize_config(self, config: dict[str, dict[str, Any]]) -> None:
        pass

    def finalize_environments(self, environments: dict[str, dict[str, Any]]) -> None:
        pass

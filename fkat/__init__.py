# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import asyncio
from collections.abc import Callable

import hydra
import omegaconf as oc
import lightning as L
import torch
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import record

from fkat.utils import config, pdb
from fkat.utils.logging import rank0_logger
from fkat.utils.config import SingletonResolver

log = rank0_logger(__name__)


def run_main(main: Callable[[], None]) -> None:
    patch_args()

    @record
    async def async_main() -> None:
        try:
            main()
        except Exception as e:
            import traceback

            traceback.print_tb(e.__traceback__)
            raise e

    asyncio.run(async_main())  # type: ignore[arg-type]


def patch_args() -> None:
    """
    In case we need to pass wildcard arguments (e.g. overrides) as expected by Hydra,
    but the runtime only allows named arguments we pass them using a bogus "--overrides" flag.
    This function will take care of removing this flag by the time we call Hydra.
    """
    overrides_pos = -1
    for i, a in enumerate(sys.argv):
        if a == "--overrides":
            overrides_pos = i
            break
    if overrides_pos >= 0:
        overrides = sys.argv[overrides_pos + 1] if overrides_pos + 1 < len(sys.argv) else ""
        # skipping overrides when constructing new args, there could be more args ahead
        sys.argv = sys.argv[:overrides_pos] + (
            sys.argv[overrides_pos + 2 :] if overrides_pos + 2 < len(sys.argv) else []
        )
        if overrides:
            # adding overrides at the end
            sys.argv.extend(overrides.split(" "))


def setup(
    cfg: oc.DictConfig | None = None,
    print_config: bool = False,
    multiprocessing: str = "spawn",
    seed: int | None = None,
    post_mortem: bool = False,
    determinism: bool = False,
    resolvers: dict[str, "oc.Resolver"] | None = None,
) -> SingletonResolver:
    """Setup the training environment.

    Args:
        cfg (oc.OmegaConf | None): Full configuration
        print_config (bool): Whether to print configuration to output. Defaults to ``False``
        multiprocessing (str): Multiprocessing mode. Defaults to ``spawn``
        seed (int | None): Random number generator seed to start off when set
        post_mortem (bool): Whether to start pdb debugger when an uncaught exception encoutered. Defaults to ``False``
        determinism (bool): Whether to enforce deterministric algorithms. Defaults to ``False```
        resolvers (dict[str, oc.Resover] | None): Custom resolvers to register for configuration processing

    Returns:
        :class:`SingletonResolver` object that holds initialized data, trainer, model, etc.
    """
    if print_config:
        log.info(config.to_str(cfg))

    mp.set_start_method(multiprocessing, force=True)

    if seed:
        L.seed_everything(seed)

    if post_mortem:
        pdb.post_mortem()

    if determinism:  # Enable deterministic algorithms globally
        assert seed is not None, "seed has to be set for deterministic runs"
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            deterministic_env_vars = {
                "CUBLAS_WORKSPACE_CONFIG": [":16:8", ":4096:8"],
                "NCCL_ALGO": ["^NVLS"],
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": ["0"],
            }
            for var, vals in deterministic_env_vars.items():
                if (val := os.environ.get(var, vals[-1])) not in vals:
                    raise ValueError(f"{var} has to be set to one of {vals} for deterministic runs, got: {val}")
                os.environ[var] = val
    for rn, fn in (resolvers or {}).items():
        oc.OmegaConf.register_new_resolver(rn, fn, replace=True)

    s = config.register_singleton_resolver()
    return s


def initialize(cfg: oc.DictConfig) -> SingletonResolver:
    """Initialize data, model and trainer with supplied configurations.

    Args:
        cfg (oc.DictConfig): Configurations supplied by user through yaml file.

    Returns:
        :class:`SingletonResolver` object that holds initialized data, trainer, model, etc.
    """
    # 0. setup the training environment
    s = setup(cfg, **(hydra.utils.instantiate(cfg["setup"]) if "setup" in cfg else {}))

    # 1. instantiate `trainer`
    s.trainer = hydra.utils.instantiate(cfg.trainer)

    # 2. instantiate optional `data`
    s.data = hydra.utils.instantiate(cfg.get("data"))

    # 3. instantiate `model` after `trainer`
    s.model = hydra.utils.instantiate(cfg.model)

    # 4. obtain optional `ckpt_path`, `return_predictions` after `model`
    s.ckpt_path = hydra.utils.call(cfg.get("ckpt_path"))
    s.return_predictions = hydra.utils.call(cfg.get("return_predictions"))

    # 5. save and upload the config
    config.save(cfg, s.trainer)

    # 6. run tuners
    s.tuners = hydra.utils.instantiate(cfg.get("tuners"))

    return s

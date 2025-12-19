# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python

"""
The ``fkat.train`` entrypoint processes the provided config,
instatiates the ``trainer``, ``model`` and ``data`` sections and calls ``trainer.fit()``.
"""

import hydra
import lightning as L
from omegaconf import DictConfig

from fkat import initialize, run_main


@hydra.main(version_base="1.3")
def main(cfg: DictConfig) -> None:
    s = initialize(cfg)
    kwargs = {
        "ckpt_path": s.ckpt_path,
    }
    if isinstance(s.data, L.LightningDataModule):
        kwargs["datamodule"] = s.data
    else:
        kwargs["train_dataloaders"] = s.data.train_dataloader() if s.data else None
        kwargs["val_dataloaders"] = s.data.val_dataloader() if s.data else None
    s.trainer.fit(s.model, **kwargs)


if __name__ == "__main__":
    run_main(main)

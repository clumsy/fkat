# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python

"""
The ``fkat.predict`` entrypoint processes the provided config,
instantiates the ``trainer``, ``model`` and ``data`` sections and calls ``trainer.predict()``.
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
        "return_predictions": s.return_predictions,
    }
    if isinstance(s.data, L.LightningDataModule):
        kwargs["datamodule"] = s.data
    else:
        kwargs["predict_dataloader"] = s.data.predict_dataloader() if s.data else None
    s.trainer.predict(s.model, **kwargs)


if __name__ == "__main__":
    run_main(main)

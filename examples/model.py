# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import lightning as L
import torch


class CausalLM(L.LightningModule):
    def __init__(self, model, lr=5e-5):
        super().__init__()
        self.model = model
        self.lr = lr
        
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("train_loss", outputs.loss)
        return outputs.loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log("val_loss", outputs.loss)
        return outputs.loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

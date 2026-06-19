from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelF1Score,
)

import torchvision.models as tvm

from utils.data_utils import parse_batch


class MultiLabelResNet50(pl.LightningModule):
    def __init__(
        self,
        num_labels: int,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 50,
        dataset_name: str = None,
        pos_weight: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])

        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_labels)
        self.model = m
        self.dataset_name = dataset_name
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.register_buffer("pos_weight", torch.tensor([]))

        self.map_macro = MultilabelAveragePrecision(
            num_labels=num_labels, average="macro"
        )
        self.f1_micro = MultilabelF1Score(
            num_labels=num_labels, average="micro", threshold=threshold
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        # logits

    def _get_targets(self, batch: Dict[str, Any]) -> torch.Tensor:
        if self.dataset_name == "CUB":
            y = batch["attrs"]
        elif self.dataset_name == "COCO":
            y = batch["labels"]
        else:
            # fall back: try common keys
            if "attrs" in batch:
                y = batch["attrs"]
            elif "labels" in batch:
                y = batch["labels"]
            else:
                raise KeyError("Batch must contain 'attrs' or 'labels' for targets.")
        # Ensure float targets for BCEWithLogitsLoss
        return y.float()

    def _bce_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.pos_weight.numel() > 0:
            return F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=self.pos_weight
            )
        return F.binary_cross_entropy_with_logits(logits, y)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = parse_batch(batch, self.dataset_name)

        x = batch["features"]
        y = self._get_targets(batch)
        logits = self(x)
        loss = self._bce_loss(logits, y)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        batch = parse_batch(batch, self.dataset_name)

        x = batch["features"]
        y = self._get_targets(batch)
        logits = self(x)
        loss = self._bce_loss(logits, y)

        probs = torch.sigmoid(logits)
        self.map_macro.update(probs, y.int())
        self.f1_micro.update(probs, y.int())

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        batch = parse_batch(batch, self.dataset_name)

        x = batch["features"]
        y = self._get_targets(batch)
        logits = self(x)
        loss = self._bce_loss(logits, y)

        probs = torch.sigmoid(logits)
        self.map_macro.update(probs, y.int())
        self.f1_micro.update(probs, y.int())

        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

    def on_validation_epoch_end(self) -> None:
        self.log("val/mAP", self.map_macro.compute(), prog_bar=True)
        self.log("val/F1_micro", self.f1_micro.compute(), prog_bar=True)
        self.map_macro.reset()
        self.f1_micro.reset()

    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = CosineAnnealingLR(opt, T_max=int(self.hparams.max_epochs))
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
        }

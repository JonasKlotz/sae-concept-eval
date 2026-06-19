from __future__ import annotations

import os
import sys
from pprint import pprint

from tqdm import tqdm

sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")


from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl
from torchvision import transforms
import rootutils
from pytorch_lightning.loggers import CSVLogger

root = Path(rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False))

from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.scripts.classifier_for_syn_data.classifier import MultiLabelResNet50
from src.utils.data_utils import parse_batch, load_image_datamodule


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=(0.7, 1.0), ratio=(0.75, 1.3333333)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def unify_targets(batch: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    if dataset_name == "CUB":
        targets = batch["attrs"]
    elif dataset_name == "COCO":
        targets = batch["labels"]
    else:
        raise RuntimeError(f"Unknown dataset: {dataset_name}")
    batch["targets"] = targets
    return batch


@torch.no_grad()
def estimate_pos_weight(
    train_loader,
    dataset_name: str,
    max_batches: int = 400,
    device: str = "cpu",
) -> torch.Tensor:
    pos_sum: Optional[torch.Tensor] = None
    n_total = 0

    for i, raw in tqdm(
        enumerate(train_loader), total=max_batches, desc="Estimating pos_weight"
    ):
        if i >= max_batches:
            break
        batch = parse_batch(raw, dataset_name)
        batch = unify_targets(batch, dataset_name)
        y = batch["targets"].to(device).float()

        pos_sum = y.sum(dim=0) if pos_sum is None else (pos_sum + y.sum(dim=0))
        n_total += int(y.size(0))

    if pos_sum is None:
        raise RuntimeError(
            "Could not estimate pos_weight, train loader returned no batches."
        )

    neg_sum = float(n_total) - pos_sum
    pos_sum = torch.clamp(pos_sum, min=1.0)
    pos_weight = neg_sum / pos_sum
    return pos_weight.float()


@hydra.main(
    config_path=str(root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_classifier(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.get("seed", 0)), workers=True)
    print("Running main with config:")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    dataset_name = str(cfg.dataset.name)
    max_epochs = 10

    out_root = Path(cfg.outputs) / "classifier" / dataset_name.lower()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}")
    train_transform = build_transforms(train=True)
    val_transform = build_transforms(train=False)
    cfg.dataset["batch_size"] = (
        64  # override batch size for faster pos_weight estimation and training
    )
    data_module = load_image_datamodule(
        cfg.dataset,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=0,
    )
    data_module.setup()
    print("Data module loaded")

    raw_batch = next(iter(data_module.val_dataloader()))
    batch = unify_targets(parse_batch(raw_batch, dataset_name), dataset_name)
    num_labels = int(batch["targets"].shape[1])
    pos_path = out_root / "pos_weight.pt"
    if pos_path.exists():
        pos_weight = torch.load(pos_path, map_location="cpu")
    else:
        pos_weight = estimate_pos_weight(
            data_module.train_dataloader(),
            dataset_name=dataset_name,
            max_batches=400,
            device="cpu",
        )
        torch.save(pos_weight, pos_path)

    # print(f"Estimated pos_weight: {pos_weight.tolist()}")

    model = MultiLabelResNet50(
        num_labels=num_labels,
        lr=3e-4,
        weight_decay=1e-2,
        max_epochs=max_epochs,
        pos_weight=pos_weight,  # registered buffer -> Lightning moves it with the model
        threshold=0.5,
        dataset_name=dataset_name,
    )

    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # inside run_classifier(...)
    log_dir = out_root / "lightning_csv"
    logger = CSVLogger(
        save_dir=str(log_dir), name=""
    )  # writes metrics.csv into log_dir/version_0/

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor="val/mAP",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=50,
        callbacks=callbacks,
        logger=logger,
    )
    print("Starting training...")

    trainer.fit(model, datamodule=data_module)

    print("Training complete. Starting testing with best checkpoint...")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    run_classifier()

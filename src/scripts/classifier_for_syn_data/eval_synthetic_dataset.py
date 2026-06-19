from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, Any, List

import hydra
import pandas as pd
import pytorch_lightning as pl
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelAUROC,
)
from torchvision import transforms
from tqdm import tqdm

# project paths (match your setup)
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")

root = Path(rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False))

from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.scripts.classifier_for_syn_data.classifier import MultiLabelResNet50
from datamodule.CUB_syn_dataset import CUBSyntheticDataset
from datamodule.coco_dataset import COCOSynDataset


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
        batch["targets"] = batch["attrs"]
    elif dataset_name == "COCO":
        batch["targets"] = batch["labels"]
    else:
        raise RuntimeError(f"Unknown dataset: {dataset_name}")
    return batch


def _find_best_checkpoint(ckpt_dir: Path) -> Path:
    """
    Returns the newest/most relevant checkpoint path.
    Preference order:
      1) any file containing 'best' in name
      2) any file containing 'val' / metric-like name (ModelCheckpoint default)
      3) otherwise newest .ckpt
    """
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = sorted(
        ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in: {ckpt_dir}")

    for p in ckpts:
        if "best" in p.name.lower():
            return p
    for p in ckpts:
        n = p.name.lower()
        if ("val" in n) or ("map" in n) or ("epoch" in n):
            return p
    return ckpts[0]


def _ensure_bool_targets(y: torch.Tensor) -> torch.Tensor:
    # Accept 0/1 floats or ints
    if y.dtype != torch.bool:
        y = y > 0.5
    return y


def _binarize_logits(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return logits.sigmoid() >= threshold


@torch.no_grad()
def evaluate_on_syn_dataset(
    model: torch.nn.Module,
    syn_dataset,
    syn_loader: DataLoader,
    num_labels: int,
    dataset_name: str,
    threshold: float,
    out_dir: Path,
    device: torch.device | str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.to(device)
    model.eval()

    # Separate metrics: orig vs syn
    orig_metrics = {
        "f1_micro": MultilabelF1Score(num_labels=num_labels, average="micro").to(
            device
        ),
        "f1_macro": MultilabelF1Score(num_labels=num_labels, average="macro").to(
            device
        ),
        "precision_micro": MultilabelPrecision(
            num_labels=num_labels, average="micro"
        ).to(device),
        "recall_micro": MultilabelRecall(num_labels=num_labels, average="micro").to(
            device
        ),
        "auroc_macro": MultilabelAUROC(num_labels=num_labels, average="macro").to(
            device
        ),
    }
    syn_metrics = {
        "f1_micro": MultilabelF1Score(num_labels=num_labels, average="micro").to(
            device
        ),
        "f1_macro": MultilabelF1Score(num_labels=num_labels, average="macro").to(
            device
        ),
        "precision_micro": MultilabelPrecision(
            num_labels=num_labels, average="micro"
        ).to(device),
        "recall_micro": MultilabelRecall(num_labels=num_labels, average="micro").to(
            device
        ),
        "auroc_macro": MultilabelAUROC(num_labels=num_labels, average="macro").to(
            device
        ),
    }

    rows: List[Dict[str, Any]] = []
    num_pairs = 0

    for batch in tqdm(syn_loader, desc="Evaluating synthetic dataset", unit="batch"):
        if dataset_name == "syn_cub":
            orig_images, _, orig_attrs, syn_images, _, syn_attrs, _, _, idx = batch
        else:  # syn_coco
            orig_images, orig_attrs, syn_images, syn_attrs, _, idx = batch

        pair_paths = syn_dataset.image_paths_for_index(idx)
        num_pairs += len(pair_paths)

        orig_images = orig_images.to(device, non_blocking=True)
        syn_images = syn_images.to(device, non_blocking=True)

        orig_targets = _ensure_bool_targets(orig_attrs.to(device, non_blocking=True))
        syn_targets = _ensure_bool_targets(syn_attrs.to(device, non_blocking=True))

        orig_logits = model(orig_images)
        syn_logits = model(syn_images)

        orig_pred = _binarize_logits(orig_logits, threshold=threshold)
        syn_pred = _binarize_logits(syn_logits, threshold=threshold)

        # Update orig metrics with orig only
        orig_metrics["f1_micro"].update(orig_pred, orig_targets)
        orig_metrics["f1_macro"].update(orig_pred, orig_targets)
        orig_metrics["precision_micro"].update(orig_pred, orig_targets)
        orig_metrics["recall_micro"].update(orig_pred, orig_targets)
        orig_metrics["auroc_macro"].update(orig_logits.sigmoid(), orig_targets.int())

        # Update syn metrics with syn only
        syn_metrics["f1_micro"].update(syn_pred, syn_targets)
        syn_metrics["f1_macro"].update(syn_pred, syn_targets)
        syn_metrics["precision_micro"].update(syn_pred, syn_targets)
        syn_metrics["recall_micro"].update(syn_pred, syn_targets)
        syn_metrics["auroc_macro"].update(syn_logits.sigmoid(), syn_targets.int())

        # Per-image error counts and correctness flags
        orig_wrong = (orig_pred != orig_targets).sum(dim=1).detach().cpu().tolist()
        syn_wrong = (syn_pred != syn_targets).sum(dim=1).detach().cpu().tolist()

        orig_correct = (orig_pred == orig_targets).all(dim=1).detach().cpu().tolist()
        syn_correct = (syn_pred == syn_targets).all(dim=1).detach().cpu().tolist()

        orig_gt_pos = orig_targets.sum(dim=1).detach().cpu().tolist()
        syn_gt_pos = syn_targets.sum(dim=1).detach().cpu().tolist()
        orig_pred_pos = orig_pred.sum(dim=1).detach().cpu().tolist()
        syn_pred_pos = syn_pred.sum(dim=1).detach().cpu().tolist()

        # Identify which attributes differ between orig and syn (expected 1 or 2)
        # diff_mask[b, j] == True  <=>  attribute j differs between the pair
        diff_mask = orig_targets ^ syn_targets  # bool, (B, L)
        diff_count = diff_mask.sum(dim=1)  # int,  (B,)

        # Correctness restricted to the differing attributes only
        # For each sample b:
        #   orig_diff_correct[b] = all(orig_pred[b, diff] == orig_targets[b, diff])
        #   syn_diff_correct[b]  = all(syn_pred[b,  diff] == syn_targets[b,  diff])
        # If (unexpectedly) diff_count == 0, we define correctness as True.
        orig_diff_correct = torch.zeros(
            orig_targets.size(0), dtype=torch.bool, device=device
        )
        syn_diff_correct = torch.zeros(
            orig_targets.size(0), dtype=torch.bool, device=device
        )

        for b in range(orig_targets.size(0)):
            m = diff_mask[b]
            if bool(m.any()):
                orig_diff_correct[b] = (orig_pred[b, m] == orig_targets[b, m]).all()
                syn_diff_correct[b] = (syn_pred[b, m] == syn_targets[b, m]).all()
            else:
                orig_diff_correct[b] = True
                syn_diff_correct[b] = True

        both_diff_correct = orig_diff_correct & syn_diff_correct

        # Optional: store which attributes differed (as a compact string)
        # This can be useful for debugging and analysis later.
        diff_indices = [
            ",".join(
                map(
                    str,
                    torch.nonzero(diff_mask[b], as_tuple=False)
                    .squeeze(1)
                    .detach()
                    .cpu()
                    .tolist(),
                )
            )
            for b in range(orig_targets.size(0))
        ]

        for b in range(len(pair_paths)):
            orig_path, syn_path = pair_paths[b]
            rows.append(
                {
                    "orig_path": orig_path,
                    "syn_path": syn_path,
                    "orig_wrong_labels": int(orig_wrong[b]),
                    "syn_wrong_labels": int(syn_wrong[b]),
                    "orig_correct": bool(orig_correct[b]),
                    "syn_correct": bool(syn_correct[b]),
                    "orig_gt_pos": int(orig_gt_pos[b]),
                    "syn_gt_pos": int(syn_gt_pos[b]),
                    "orig_pred_pos": int(orig_pred_pos[b]),
                    "syn_pred_pos": int(syn_pred_pos[b]),
                    "pair_correct": bool(orig_correct[b] and syn_correct[b]),
                    "diff_count": int(diff_count[b].detach().cpu()),
                    "diff_indices": diff_indices[b],  # comma-separated label indices
                    "orig_diff_detected": bool(orig_diff_correct[b].detach().cpu()),
                    "syn_diff_detected": bool(syn_diff_correct[b].detach().cpu()),
                    "both_diff_detected": bool(both_diff_correct[b].detach().cpu()),
                }
            )

    # Compute summaries
    metrics = {
        "dataset_name": dataset_name,
        "threshold": float(threshold),
        "num_pairs": int(num_pairs),
        "orig": {k: float(m.compute().detach().cpu()) for k, m in orig_metrics.items()},
        "syn": {k: float(m.compute().detach().cpu()) for k, m in syn_metrics.items()},
    }

    print("Metrics:")
    print("  orig:")
    for k, v in metrics["orig"].items():
        print(f"    {k}: {v}")
    print("  syn:")
    for k, v in metrics["syn"].items():
        print(f"    {k}: {v}")

    # ---- save metrics ----
    torch.save(metrics, out_dir / "metrics.pt")

    # save metrics also as CSV (one row)
    metrics_csv_row = {
        "dataset_name": metrics["dataset_name"],
        "threshold": metrics["threshold"],
        "num_pairs": metrics["num_pairs"],
        **{f"orig_{k}": v for k, v in metrics["orig"].items()},
        **{f"syn_{k}": v for k, v in metrics["syn"].items()},
    }
    pd.DataFrame([metrics_csv_row]).to_csv(out_dir / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_pair_predictions.csv", index=False)
    torch.save(
        {"metrics": metrics, "per_pair_df": df}, out_dir / "evaluation_bundle.pt"
    )

    print(f"Saved evaluation results to: {out_dir}")


@hydra.main(
    config_path=str(root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def load_best_classifier(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.get("seed", 0)), workers=True)

    print("Loading classifier with config:")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    dataset_name = str(cfg.dataset.name)
    out_root = Path(cfg.outputs) / "classifier" / dataset_name.lower()
    ckpt_dir = out_root / "checkpoints"
    out_root = (
        Path("/home/htc/jklotz/git/rs_concepts/plots/classifier") / dataset_name.lower()
    )

    val_transform = build_transforms(train=False)
    if cfg.dataset.name == "CUB":
        num_labels = 312
        syn_dataset = CUBSyntheticDataset(transform=val_transform)
        dataset_name = "syn_cub"
    elif cfg.dataset.name == "COCO":
        num_labels = 80
        syn_dataset = COCOSynDataset(
            transform=val_transform,
        )
        dataset_name = "syn_coco"
    else:
        raise RuntimeError(f"Unknown dataset: {cfg.dataset.name}")

    best_ckpt = _find_best_checkpoint(ckpt_dir)
    print(f"Best checkpoint: {best_ckpt}")

    # load weights (Lightning-style)
    model = MultiLabelResNet50.load_from_checkpoint(
        checkpoint_path=str(best_ckpt),
        num_labels=num_labels,
        threshold=0.5,
        dataset_name=dataset_name,
        pos_weight=torch.ones(num_labels, dtype=torch.float32),
    )

    model.eval()
    print("Classifier loaded and set to eval().")
    out_eval = out_root / "eval_syn"  # or any dir you want
    syn_loader = DataLoader(syn_dataset, batch_size=64, shuffle=False, num_workers=0)

    evaluate_on_syn_dataset(
        model=model,
        syn_dataset=syn_dataset,
        syn_loader=syn_loader,
        num_labels=num_labels,
        dataset_name=dataset_name,
        threshold=0.5,
        out_dir=out_eval,
    )


if __name__ == "__main__":
    load_best_classifier()

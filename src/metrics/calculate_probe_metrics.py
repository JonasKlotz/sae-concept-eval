"""
Linear probe upper bound for MATCHScore and TAPAScore.

Trains one logistic regression per attribute (multi-label, single model) on raw
CLIP embeddings, then runs the identical FBMP/F1-coalition matching and TAPAScore
pipeline as the SAE experiments.  Results land in:

  metrics/{dataset}/{model}/{seed}/linear_probe/ground_truth/   ← same .pt keys as SAE
  metrics/{dataset}/{model}/{seed}/linear_probe/pert/           ← same CSV columns as SAE
"""

import sys
from pathlib import Path

sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")

import hydra
import numpy as np
import rootutils
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)
from src.utils import resolvers  # noqa: F401

from src.datamodule.CUB_syn_dataset import CUBSyntheticDataset
from src.datamodule.coco_dataset import COCOSynDataset
from src.metrics.calculate_metrics_targeted_perturbation import (
    calculate_perturbation_metric_cub,
    calculate_perturbation_metric_coco,
)
from src.metrics.gt_metrics.metrics_gt_concept import (
    calculate_topk_f1_coalitions,
    compute_bmp_per_class,
    compute_max_f1_per_class,
)
from src.metrics.metric_utils import get_topk_matching, load_metric_results
from src.metrics.probe_based.linear_probe import (
    LinearAttrPredictor,
    get_or_train_linear_predictor,
)
from src.utils.data_utils import (
    get_eval_emb_dataloader,
    load_embedding_datamodule,
    parse_batch,
    save_metric_results,
)
from src.utils.model_load_utils import get_image_encoder


class ProbeAsLatent:
    """
    Wraps LinearAttrPredictor to satisfy the SAE interface used by the
    TAPAScore pipeline (_get_normed_encodings, calculate_perturbation_metric_*).

    W_enc is a dummy (1, C) ones matrix so that atom_norms = ones(C) and the
    division z / (atom_norms + 1e-8) in _get_normed_encodings is a no-op.
    encode() returns (binary_preds, probs) matching the (codes, pre_codes)
    convention of TopKSAE / JumpReLUSAE.
    """

    def __init__(self, probe: LinearAttrPredictor, threshold: float = 0.5):
        self.probe = probe
        self.threshold = threshold
        self.nb_concepts = probe.nb_concepts
        self._W_enc = torch.ones(1, probe.nb_concepts)

    @property
    def W_enc(self):
        device = next(self.probe.parameters()).device
        return self._W_enc.to(device)

    @torch.no_grad()
    def encode(self, x):
        probs = torch.sigmoid(self.probe.linear(x))
        binary = (probs >= self.threshold).float()
        return binary, probs

    def to(self, device):
        self.probe = self.probe.to(device)
        return self

    def eval(self):
        self.probe.eval()
        return self


@torch.no_grad()
def extract_probe_concept_matrix(
    cfg: DictConfig,
    data_loader: DataLoader,
    probe: LinearAttrPredictor,
    threshold: float = 0.5,
):
    """
    Returns binary probe predictions and GT labels for the full dataset split.

    Returns:
        binary_matrix:  (N, C) float32, threshold-binarized sigmoid outputs
        gt_matrix:      (N, C) float32, ground-truth attribute annotations
    """
    N = len(data_loader.dataset)
    C = probe.nb_concepts
    gt_matrix = None
    binary_matrix = torch.zeros((N, C), device=cfg.device)

    probe.to(cfg.device).eval()

    for idx, batch in enumerate(tqdm(data_loader, desc="Probe concept extraction")):
        batch_dict = parse_batch(batch, dataset_name=cfg.dataset.name, embedding=True)
        embeddings = batch_dict["features"].to(cfg.device).float()
        labels = batch_dict["labels"].to(cfg.device)
        bs = embeddings.shape[0]

        probs = torch.sigmoid(probe.linear(embeddings))
        binary = (probs >= threshold).float()
        binary_matrix[idx * bs : (idx + 1) * bs] = binary

        if gt_matrix is None:
            gt_matrix = torch.zeros((N, labels.shape[-1]), device=cfg.device)
        gt_matrix[idx * bs : (idx + 1) * bs] = labels

    return binary_matrix.cpu(), gt_matrix.cpu()


def run_probe_metrics(cfg: DictConfig):
    """
    Full probe upper-bound pipeline: GT matching (MATCHScore) + TAPAScore.
    Results are saved under metrics_dir.parent / "linear_probe/".
    """
    probe_metrics_dir = Path(cfg.paths.metrics_dir).parent / "linear_probe"
    probe_save_path = (
        Path(cfg.paths.metrics_dir).parent
        / "matching"
        / "linear_probe"
        / "linear_attr_probe.pt"
    )
    gt_dir = probe_metrics_dir / "ground_truth"
    pert_dir = probe_metrics_dir / "pert"

    gt_done = gt_dir.exists() and any(gt_dir.glob("*.pt"))
    pert_done = (
        cfg.dataset.name not in ["CUB", "COCO"]
        or (pert_dir.exists() and any(pert_dir.glob("*.csv")))
    )
    if gt_done and pert_done:
        print(f"Probe metrics already exist at {probe_metrics_dir}, skipping.")
        return

    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = get_eval_emb_dataloader(
        data_module, cfg.get("embedding_dataloader_for_eval", "test")
    )

    # ── 1. Train / load probe ──────────────────────────────────────────────────
    probe = get_or_train_linear_predictor(
        train_loader,
        val_loader=val_loader,
        save_path=str(probe_save_path),
        epochs=100,
        lr=1e-3,
        device=cfg.device,
    )
    probe.eval().to(cfg.device)

    # ── 2. GT matching metrics ─────────────────────────────────────────────────
    binary_matrix, gt_matrix = extract_probe_concept_matrix(cfg, test_loader, probe)

    _, f1_inv_order_matrix, f1_rec_F1_matrix = calculate_topk_f1_coalitions(
        binary_matrix, gt_matrix, top_k=10
    )
    f1_max_scores, _, _ = compute_max_f1_per_class(gt_matrix, binary_matrix)
    print(f"Probe mean best-F1: {np.mean(f1_max_scores):.4f}")
    print(f"Probe mean coalition-F1 @1: {np.mean(f1_rec_F1_matrix[:, 0]):.4f}")

    results = {
        "f1_inv_order_matrix": f1_inv_order_matrix,
        "f1_rec_F1_matrix": f1_rec_F1_matrix,
    }
    for f_beta in [0.25, 0.5, 1.0]:
        _, bmp_inv, bmp_rec = compute_bmp_per_class(
            gt_matrix, binary_matrix, f_beta=f_beta, criterion="fbeta"
        )
        results[f"bmp_inv_order_matrix_f{f_beta}"] = bmp_inv
        results[f"bmp_rec_F1_matrix_f{f_beta}"] = bmp_rec

    gt_dir.mkdir(parents=True, exist_ok=True)
    save_metric_results(gt_dir, results)
    print(f"Probe GT matching metrics saved to {gt_dir}")

    # ── 3. TAPAScore ───────────────────────────────────────────────────────────
    if cfg.dataset.name not in ["CUB", "COCO"]:
        print("TAPAScore not implemented for this dataset, skipping.")
        return

    image_encoder, _, _ = get_image_encoder(cfg.model, device=cfg.device)
    transform_img = T.Compose([*image_encoder.preprocess.transforms])

    data_root = Path("/data/jonas/datasets/")
    if not data_root.exists():
        data_root = Path("/scratch/htc/jklotz/data")

    if cfg.dataset.name == "CUB":
        syn_dataset = CUBSyntheticDataset(
            root=data_root / "syn_cub_dataset", transform=transform_img
        )
        calculate_perturbation_metric_func = calculate_perturbation_metric_cub
    else:
        syn_dataset = COCOSynDataset(
            root=data_root / "syn_coco_dataset", transform=transform_img
        )
        calculate_perturbation_metric_func = calculate_perturbation_metric_coco

    probe_adapter = ProbeAsLatent(probe)

    cfg = cfg.copy()
    cfg.dataset.name = "syn_cub" if cfg.dataset.name == "CUB" else "syn_coco"

    loaded_results = load_metric_results(gt_dir)

    for matching_style in ["bmp_f0.25", "bmp_f0.5", "bmp_f1.0", "f1"]:
        for k in [10, 5, 3, 1]:
            topk_concepts = get_topk_matching(matching_style, loaded_results, top_k=k)
            calculate_perturbation_metric_func(
                syn_dataset,
                cfg.device,
                topk_concepts,
                image_encoder,
                probe_adapter,
                cfg,
                pert_dir,
                f"{matching_style}_top{k}",
            )

    print(f"Probe TAPAScore results saved to {pert_dir}")


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig):
    run_probe_metrics(cfg)


if __name__ == "__main__":
    main()

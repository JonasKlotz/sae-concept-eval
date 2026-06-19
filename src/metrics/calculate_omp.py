import os
import sys
from pathlib import Path

sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
import hydra
import numpy as np
import rootutils
import torch
from scipy.optimize import nnls
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Dict
from tqdm import tqdm


project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)
from src.utils import resolvers  # noqa: F401 ensures resolver is registered

from src.metrics.metric_utils import extract_concept_matrix
from src.utils.model_load_utils import load_sae
from src.utils.data_utils import (
    load_embedding_datamodule,
    save_metric_results,
    get_eval_emb_dataloader,
)


def compute_omp_per_class(
    ground_truth_concept_matrix,
    concept_matrix,
    max_coalition_size=20,
    nonneg=True,
    threshold=0.5,
    ztol=1e-12,
):
    """
    OMP-based matching of GT binary attributes to continuous SAE latents.

    Designed to be directly comparable to compute_bmp_per_class:
      - atoms  = continuous SAE activations (column-normalized for selection)
      - target = binary GT attribute vector
      - sparsity capped at max_coalition_size (same default as FBMP)
      - F1 computed at each step by thresholding ypred at `threshold`
      - rec_F1_matrix has the same (C, max_coalition_size) shape/semantics as FBMP's

    Args:
        ground_truth_concept_matrix: (N, C) binary GT
        concept_matrix:              (N, D) continuous SAE activations
        max_coalition_size:          max atoms per attribute
        nonneg:                      enforce non-negative LS coefficients (NNLS)
        threshold:                   binarization threshold on ypred
        ztol:                        stop if max residual covariance < ztol * ||y||

    Returns:
        order_matrix:     (C, D) selection order (1-indexed; 0 = not selected)
        inv_order_matrix: (C, D) 1/order (0 = not selected); topk-friendly like FBMP
        rec_F1_matrix:    (C, max_coalition_size) F1 at each step, padded with last value
    """
    if isinstance(ground_truth_concept_matrix, torch.Tensor):
        ground_truth_concept_matrix = ground_truth_concept_matrix.detach().cpu().numpy()
    if isinstance(concept_matrix, torch.Tensor):
        concept_matrix = concept_matrix.detach().cpu().numpy()

    gt = ground_truth_concept_matrix.astype(np.float64)
    X = concept_matrix.astype(np.float64)

    C = gt.shape[1]
    D = X.shape[1]
    max_k = min(max_coalition_size, D)

    # Column-normalize atoms so argmax(rcov) selects on correlation, not energy.
    col_norms = np.linalg.norm(X, axis=0)
    col_norms_safe = np.where(col_norms < 1e-12, 1.0, col_norms)
    Xn = X / col_norms_safe

    order_matrix = np.zeros((C, D))
    inv_order_matrix = np.zeros_like(order_matrix)
    rec_F1_matrix = np.zeros((C, max_k))

    for i_class in tqdm(range(C), desc="OMP matching"):
        y = gt[:, i_class]
        ynorm = np.linalg.norm(y)
        if ynorm < 1e-10:
            continue

        active = []
        residual = y.copy()
        last_f1 = 0.0

        for step in range(max_k):
            rcov = Xn.T @ residual
            i = int(np.argmax(rcov) if nonneg else np.argmax(np.abs(rcov)))
            rc = rcov[i] if nonneg else abs(rcov[i])

            if rc < ztol * ynorm:
                rec_F1_matrix[i_class, step:max_k] = last_f1
                break

            if i in active:
                # NNLS reselection: support didn't grow → no progress possible.
                rec_F1_matrix[i_class, step:max_k] = last_f1
                break

            active.append(i)

            if nonneg:
                coefi, _ = nnls(Xn[:, active], y)
            else:
                coefi, _, _, _ = np.linalg.lstsq(Xn[:, active], y, rcond=None)

            residual = y - Xn[:, active] @ coefi
            ypred = y - residual

            f1 = f1_score(
                y.astype(int),
                (ypred > threshold).astype(int),
            )

            order_matrix[i_class, i] = step + 1
            inv_order_matrix[i_class, i] = 1.0 / (step + 1)
            rec_F1_matrix[i_class, step] = f1
            last_f1 = f1

    return order_matrix, inv_order_matrix, rec_F1_matrix


def calculate_omp(
    cfg: DictConfig,
    sae: torch.nn.Module,
    embedding_val_loader: DataLoader,
) -> Dict:
    concept_matrix, ground_truth_concept_matrix, _ = (
        extract_concept_matrix(cfg, embedding_val_loader, sae)
    )
    if isinstance(concept_matrix, np.ndarray):
        concept_matrix = torch.from_numpy(concept_matrix)
    if isinstance(ground_truth_concept_matrix, np.ndarray):
        ground_truth_concept_matrix = torch.from_numpy(ground_truth_concept_matrix)

    results = {}
    _, inv_order, rec_F1 = compute_omp_per_class(
        ground_truth_concept_matrix,
        concept_matrix,
        nonneg=True,
    )
    results[f"nnomp_inv_order_matrix"] = inv_order
    results[f"nnomp_rec_F1_matrix"] = rec_F1

    return results


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_calculate_omp(cfg: DictConfig):
    print("Calculating OMP...")
    metrics_dir = Path(cfg.paths.metrics_dir) / "ground_truth"

    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    data_loader = get_eval_emb_dataloader(
        data_module, cfg.get("embedding_dataloader_for_eval", "test")
    )

    sae = load_sae(cfg)
    sae.eval().to(cfg.device)

    results = calculate_omp(cfg, sae, data_loader)

    os.makedirs(metrics_dir, exist_ok=True)
    save_metric_results(metrics_dir, results)

    return results


if __name__ == "__main__":
    run_calculate_omp()

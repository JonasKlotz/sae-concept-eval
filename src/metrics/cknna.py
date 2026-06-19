import torch
from tqdm import tqdm
from torch.nn import functional as F

from metrics.metric_utils import extract_concept_matrix
from src.models import sae

import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import rootutils
import pandas as pd

from models.new_sae_lightning import LitSparseAutoencoder

# Set up project root
project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)

from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.utils.model_load_utils import (
    load_sae,
)
from src.utils.data_utils import (
    parse_batch,
    load_embedding_datamodule,
    save_metric_results,
    get_eval_emb_dataloader,
)

"""
    Taken from Matryoska repo https://github.com/WolodjaZ/MSAE/blob/main/metrics.py
    _
"""


def hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Taken from Matryoska repo https://github.com/WolodjaZ/MSAE/blob/main/metrics.py

    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC).

    This implementation follows Equation 5 in Song et al. (2012), which provides
    an unbiased estimator of HSIC. This measure quantifies the dependency between
    two sets of variables represented by their kernel matrices.

    Reference:
        Song, L., Smola, A., Gretton, A., & Borgwardt, K. (2012).
        "A dependence maximization view of clustering."
        https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf

    Args:
        K (torch.Tensor): First kernel matrix of shape [n, n]
        L (torch.Tensor): Second kernel matrix of shape [n, n]

    Returns:
        torch.Tensor: Unbiased HSIC value (scalar)
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute the biased Hilbert-Schmidt Independence Criterion (HSIC).

    This is the original form used in Centered Kernel Alignment (CKA).
    It's computationally simpler than the unbiased version but may have
    statistical bias, especially for small sample sizes.

    Args:
        K (torch.Tensor): First kernel matrix of shape [n, n]
        L (torch.Tensor): Second kernel matrix of shape [n, n]

    Returns:
        torch.Tensor: Biased HSIC value (scalar)
    """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def cknna(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    topk: int = 10,
    distance_agnostic: bool = False,
    unbiased: bool = True,
) -> float:
    """

    Compute the Centered Kernel Nearest Neighbor Alignment (CKNNA). From:
    https://github.com/minyoungg/platonic-rep/blob/4dd084e1b96804ddd07ae849658fbb69797e319b/metrics.py#L180

    CKNNA is a variant of CKA that only considers k-nearest neighbors when computing
    similarity. This makes it more robust to outliers and more sensitive to local
    structure in the data.

    Args:
        feats_A (torch.Tensor): First feature matrix of shape [n_samples, n_features_A]
        feats_B (torch.Tensor): Second feature matrix of shape [n_samples, n_features_B]
        topk (int, optional): Number of nearest neighbors to consider. Defaults to 10.
        distance_agnostic (bool, optional): If True, only considers binary neighborhood
                                           membership without weighting by similarity.
                                           Defaults to False.
        unbiased (bool, optional): If True, uses unbiased HSIC estimator.
                                  Defaults to True.

    Returns:
        float: CKNNA similarity score between 0 and 1, where higher values
               indicate greater similarity between the feature spaces

    Raises:
        ValueError: If topk is less than 2
    """
    n = feats_A.shape[0]

    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")

    if topk is None:
        topk = feats_A.shape[0] - 1

    # Compute kernel matrices (linear kernels)
    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    def similarity(K, L, topk):
        """
        Compute similarity based on nearest neighbor intersection.

        This inner function computes similarity between two kernel matrices
        based on their shared nearest neighbor structure.
        """
        if unbiased:
            # Fill diagonal with -inf to exclude self-similarity when finding topk
            K_hat = K.clone().fill_diagonal_(float("-inf"))
            L_hat = L.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K, L

        # Get topk indices for each row
        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)

        # Create masks for nearest neighbors
        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)

        # Intersection of nearest neighbors
        mask = mask_K * mask_L

        if distance_agnostic:
            # Simply count shared neighbors without considering similarity values
            sim = mask * 1.0
        else:
            # Compute HSIC on the masked kernel matrices
            if unbiased:
                sim = hsic_unbiased(mask * K, mask * L)
            else:
                sim = hsic_biased(mask * K, mask * L)
        return sim

    # Compute similarities
    sim_kl = similarity(K, L, topk)  # Cross-similarity
    sim_kk = similarity(K, K, topk)  # Self-similarity of K
    sim_ll = similarity(L, L, topk)  # Self-similarity of L

    # Normalized similarity (similar to correlation)
    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def compute_cknna_wrapper(cfg, data_loader, sae):
    concept_matrix, _, _, embedding_matrix = extract_concept_matrix(
        cfg, data_loader, sae, return_embedding=True
    )
    # convert to tensor
    concept_matrix = torch.tensor(concept_matrix, device=cfg.device)
    embedding_matrix = torch.tensor(embedding_matrix, device=cfg.device)
    cknna_score = cknna(concept_matrix, embedding_matrix)
    print(f"CKNNA Score: {cknna_score:.4f}")
    return {"CKNNA": cknna_score}


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_calculate_cknna(cfg: DictConfig):
    """
    Main function to calculate and optionally display cknna Scores.

    Args:
        cfg (DictConfig): Configuration object.
    """
    print("Calculating CKNNA Scores...")
    metrics_dir = Path(cfg.paths.metrics_dir) / "CKNNA"
    if metrics_dir.exists():
        print(f"Metrics directory {metrics_dir} already exists.")
        return {}

    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    data_loader = get_eval_emb_dataloader(
        data_module, cfg.get("embedding_dataloader_for_eval", "test")
    )
    sae = load_sae(cfg)
    sae.eval().to(cfg.device)

    results = compute_cknna_wrapper(cfg, data_loader, sae)

    # save it to torch dict
    save_metric_results(metrics_dir, results)

    return results


if __name__ == "__main__":
    run_calculate_cknna()

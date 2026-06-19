import os
import sys
from pathlib import Path

sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
import hydra
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import rootutils
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Dict



# Set up project root
project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)
from src.utils import resolvers  # noqa: F401 ensures resolver is registered

from src.metrics.metric_utils import extract_concept_matrix
from src.utils.model_load_utils import load_sae
from src.metrics.gt_metrics.metrics_gt_concept import (
    compute_max_jaccard_per_class,
    compute_max_f1_per_class,
    calculate_topk_f1_coalitions,
    compute_bmp_per_class
)
from src.metrics.calculate_omp import compute_omp_per_class
from src.utils.data_utils import (
    load_embedding_datamodule,
    save_metric_results,
    get_eval_emb_dataloader,
)


def calculate_mi(codes, factors):
    codes = codes.numpy()
    factors = factors.numpy()

    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    mi_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            mi_row = normalized_mutual_info_score(factors[:, f], codes[:, c])
            # mi_row = get_mutual_information(factors[:, f], codes[:, c], normalize=True)

            mi_matrix[f, c] = mi_row
            if mi_matrix[f, c] is None or np.isnan(mi_matrix[f, c]):
                mi_matrix[f, c] = 0

    mi_best_concepts = np.argmax(mi_matrix, axis=1)
    mi_max_scores = mi_matrix[np.arange(mi_matrix.shape[0]), mi_best_concepts]
    return mi_max_scores, mi_best_concepts, mi_matrix


def calculate_gt_metric(
    cfg: DictConfig,
    sae: torch.nn.Module,
    embedding_val_loader: DataLoader,
    compute_bmp=True,
    compute_omp=True,
    metrics_dir=None,
) -> Dict:
    """
    Calculates ground truth concept metrics for a given model and validation data.

    Args:
        cfg (DictConfig): Configuration object containing dataset and device information.
        sae (torch.nn.Module): The trained SAE (Sparse Autoencoder) model.
        embedding_val_loader (DataLoader): EMBEDDING DataLoader for the validation dataset.
        embedding: embedding for parsing batch


    Returns:
        Tuple containing:
            - C (int): Number of ground truth concepts/classes.
            - best_concepts (Tensor): Indices of best matching concepts per class.
            - f1_matrix (Tensor): F1 score matrix (classes x concepts).
            - f1_max_scores (Tensor): Maximum Jaccard scores per class.
            - jac_max_scores (Tensor): Maximum F1 scores per class.
            - jaccard (Tensor): Jaccard score matrix (classes x concepts).
    """
    concept_matrix, ground_truth_concept_matrix, unsparse_concept_matrix = (
        extract_concept_matrix(cfg, embedding_val_loader, sae)
    )
    # convert to numpy for metric calculations
    if isinstance(concept_matrix, np.ndarray):
        concept_matrix = torch.from_numpy(concept_matrix)
    if isinstance(ground_truth_concept_matrix, np.ndarray):
        ground_truth_concept_matrix = torch.from_numpy(ground_truth_concept_matrix)

    # concept_matrix is now filled with the SAE output for each sample
    # binarize the concept matrix
    binarized_concept_matrix = (concept_matrix > 1e-8).float()
    # compute metrics
    #
    _, f1_inv_order_matrix, f1_rec_F1_matrix = calculate_topk_f1_coalitions(binarized_concept_matrix,
                                                                            ground_truth_concept_matrix,
                                                                            top_k=10)
    meanf1_coalition_top1 = np.mean(f1_rec_F1_matrix[:, 0])
    print(f"Mean F1 coalition across classes: {meanf1_coalition_top1}")

    f1_max_scores, f1_best_concepts, f1_matrix = compute_max_f1_per_class(
        ground_truth_concept_matrix, binarized_concept_matrix
    )
    mean_f1 = np.mean(f1_max_scores)
    results ={}
    results.update({
            "f1_inv_order_matrix": f1_inv_order_matrix,
            "f1_rec_F1_matrix": f1_rec_F1_matrix,
        })
    if compute_bmp:
        criterions = ["fbeta"] #, "jaccard", "mi"]
        for criterion in criterions:
            if criterion == "fbeta":
                betas = [0.25, 0.5, 1.0]
                for f_beta in betas:
                    _, bmp_inv_order_matrix, bmp_rec_F1_matrix = compute_bmp_per_class(
                        ground_truth_concept_matrix,
                        binarized_concept_matrix,
                        f_beta=f_beta,
                        criterion=criterion,
                    )
                    results.update(
                        {
                            f"bmp_inv_order_matrix_f{f_beta}": bmp_inv_order_matrix,
                            f"bmp_rec_F1_matrix_f{f_beta}": bmp_rec_F1_matrix,
                        }
                    )
            else:
                _, bmp_inv_order_matrix, bmp_rec_F1_matrix = compute_bmp_per_class(
                    ground_truth_concept_matrix,
                    binarized_concept_matrix,
                    criterion=criterion,
                )
                results.update(
                    {
                        f"bmp_inv_order_matrix_{criterion}": bmp_inv_order_matrix,
                        f"bmp_rec_F1_matrix_{criterion}": bmp_rec_F1_matrix,
                    }
                )
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
                save_metric_results(metrics_dir, results)

    if compute_omp:
        _, nnomp_inv_order_matrix, nnomp_rec_F1_matrix = compute_omp_per_class(
            ground_truth_concept_matrix.numpy(force=True),
            concept_matrix.numpy(force=True),
            nonneg=True,
        )
        results.update(
            {
                "nnomp_inv_order_matrix": nnomp_inv_order_matrix,
                "nnomp_rec_F1_matrix": nnomp_rec_F1_matrix,
            }
        )
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
            save_metric_results(metrics_dir, results)

    return results


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_calculate_matching_metrics(cfg: DictConfig):
    print("Calculating Ground Truth Concept Metrics...")
    metrics_dir = Path(cfg.paths.metrics_dir) / "ground_truth"
    # if metrics_dir.exists() and len(list(metrics_dir.glob("*.pt"))) > 0:
    #     print(f"Metrics directory {metrics_dir} already exists.")
    #     return {}

    # Load data and model
    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    data_loader = get_eval_emb_dataloader(
        data_module, cfg.get("embedding_dataloader_for_eval", "test")
    )

    sae = load_sae(cfg)
    sae.eval().to(cfg.device)

    results = calculate_gt_metric(cfg,
                                  sae,
                                  data_loader,
                                  metrics_dir=metrics_dir,
                                  compute_bmp=True,
                                  compute_omp=True,
                                  )

    # Define where to save metrics

    os.makedirs(metrics_dir, exist_ok=True)
    save_metric_results(metrics_dir, results)

    return results


if __name__ == "__main__":
    run_calculate_matching_metrics()

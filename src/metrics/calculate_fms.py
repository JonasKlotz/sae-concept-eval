import json
import os
import pickle
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import rootutils
import torch
from PIL.ImagePalette import negative
from numpy import dtype, ndarray
from omegaconf import DictConfig
from rasterio.crs import defaultdict
from sklearn import tree
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple

from metrics.fms.fml_helpers import (
    load_tree_stats_data,
    load_local_tree_stats_data,
)

# Set up project root
project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)

from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.utils.model_load_utils import (
    load_sae,
)
from src.utils.data_utils import (
    load_embedding_datamodule,
    save_metric_results,
    get_eval_emb_dataloader,
    parse_batch,
)
from metrics.fms.tree_loader import get_root_node, get_tree_stats
from metrics.metric_utils import extract_concept_matrix


def compute_fms(cfg, data_loader, sae):
    # concept matrix are SAE embeddings and ground truth concepts are the labels/attrs
    concept_matrix, ground_truth_concept_matrix, _ = extract_concept_matrix(
        cfg, data_loader, sae
    )
    N = len(data_loader.dataset)  # = number of samples
    D = ground_truth_concept_matrix.shape[1]  # = number of ground truth concepts
    root_nodes = {}

    final_df = pd.DataFrame()
    for concept_idx in tqdm(range(D), desc="FMS computation over concepts"):
        ground_truth_vector = ground_truth_concept_matrix[:, concept_idx]
        if np.sum(ground_truth_vector) == 0 or np.sum(ground_truth_vector) == N:
            print(f"Skipping concept {concept_idx} due to only one class present.")
            continue
        # count of positive samples for this concept
        sampled_concept_matrix, sampled_ground_truth_vector = balance_dataset(
            N, concept_matrix, ground_truth_vector
        )

        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,  # unrestricted depth
            class_weight=None,  # no class weighting
        )
        clf = clf.fit(
            sampled_concept_matrix,
            sampled_ground_truth_vector,
        )
        root_nodes[concept_idx] = get_root_node(tree_model=clf)

        # Compute and save tree stats (e.g., accuracy, F1, etc.), implementation-specific.
        tree_stats = get_tree_stats(clf=clf)
        # tree_stats.to_csv(metrics_dir / f"tree_stats_{concept_idx}.csv")
        # s = pickle.dumps(clf)
        # with open(
        #         metrics_dir / f"tree_{concept_idx}.pkl",
        #         "wb",
        # ) as f:
        #     f.write(s)

        df_global = load_tree_stats_data(tree_stats, concept_idx)

        cut_stats = cut_tree(sampled_concept_matrix, sampled_ground_truth_vector)
        df_local = load_local_tree_stats_data(cut_stats, concept_idx)
        df = pd.merge(df_local, df_global)
        df["FMS"] = df.apply(
            lambda x: x["Accuracy"] * ((x["MS_local"] + x["MS_global"]) / 2), axis=1
        )
        final_df = pd.concat([final_df, df])

    metrics_dir = Path(cfg.paths.metrics_dir) / "fms"
    os.makedirs(metrics_dir, exist_ok=True)
    with open(
        metrics_dir / "root_nodes.json",
        "w",
    ) as f:
        json.dump(root_nodes, f)

    return final_df


def cut_tree(
    sampled_concept_matrix: ndarray[Any, dtype[Any]] | list[Any],
    sampled_ground_truth_vector: ndarray[Any, dtype[Any]],
):
    res = pd.DataFrame()
    root_node = None
    cut_root_nodes = []
    for _ in range(10):
        if root_node is not None:
            sampled_concept_matrix[:, root_node] = 0

        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=3,
        )
        clf = clf.fit(
            sampled_concept_matrix,
            sampled_ground_truth_vector,
        )

        tree_stats = get_tree_stats(clf=clf)
        root_node = get_root_node(tree_model=clf)["feature_index"]

        tree_stats["num_cuts"] = len(cut_root_nodes)
        res = pd.concat([res, tree_stats])

        cut_root_nodes.append(root_node)

    # res.to_csv(
    #     metrics_dir / f"tree_{concept_idx}_cut.csv"
    # )
    return res


def balance_dataset(
    N: int,
    concept_matrix: NDArray[Any],
    ground_truth_vector: NDArray[Any],
    rng: np.random.Generator | None = None,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Balance binary labels by subsampling the majority class (NumPy-only).

    Parameters
    ----------
    N : int
        Total number of samples. Must match len(ground_truth_vector).
    concept_matrix : ndarray
        Shape (N, ...) array of features.
    ground_truth_vector : ndarray
        Shape (N,) binary labels (0/1 or False/True).
    rng : np.random.Generator, optional
        RNG for reproducibility.

    Returns
    -------
    sampled_concept_matrix, sampled_ground_truth_vector : ndarrays
    """
    if rng is None:
        rng = np.random.default_rng()

    if ground_truth_vector.shape[0] != N:
        raise ValueError(
            f"N={N} does not match len(ground_truth_vector)={ground_truth_vector.shape[0]}."
        )
    if concept_matrix.shape[0] != N:
        raise ValueError(
            f"N={N} does not match concept_matrix.shape[0]={concept_matrix.shape[0]}."
        )

    # Ensure 1D labels
    y = np.asarray(ground_truth_vector).reshape(-1)

    # Treat True as 1, False as 0; if already 0/1
    positive_count = int(np.sum(y == 1))
    negative_count = int(N - positive_count)

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Concept has only one class, cannot balance dataset.")

    min_count = min(positive_count, negative_count)

    positive_indices = np.nonzero(y == 1)[0]
    negative_indices = np.nonzero(y == 0)[0]

    sampled_positive_indices = rng.permutation(positive_indices)[:min_count]
    sampled_negative_indices = rng.permutation(negative_indices)[:min_count]

    sampled_indices = np.concatenate(
        [sampled_positive_indices, sampled_negative_indices], axis=0
    )

    return concept_matrix[sampled_indices], y[sampled_indices]


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_calculate_fms(cfg: DictConfig):
    print("Calculating FMS...")
    metrics_dir = Path(cfg.paths.metrics_dir) / "fms"
    if (metrics_dir / "final_fms.csv").exists():
        print(f"FMS metrics already exist at {metrics_dir}, skipping computation.")
        return pd.read_csv(metrics_dir / "final_fms.csv")

    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    data_loader = get_eval_emb_dataloader(
        data_module, cfg.get("embedding_dataloader_for_eval", "test")
    )
    sae = load_sae(cfg)
    sae.eval().to(cfg.device)

    results_df = compute_fms(cfg, data_loader, sae)
    os.makedirs(metrics_dir, exist_ok=True)

    # get mean df["FMS"]
    mean_fms = results_df["FMS"].mean()
    # save with torch
    torch.save(mean_fms, metrics_dir / "mean_fms.pt")
    # save it to torch dict
    results_df.to_csv(metrics_dir / f"final_fms.csv")

    return results_df


if __name__ == "__main__":
    run_calculate_fms()

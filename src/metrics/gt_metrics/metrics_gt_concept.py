import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    jaccard_score,
    normalized_mutual_info_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm




""" - Compute Jaccard index and F1 score between class and concept matrices.
    - Cluster labels based on mean concept activations (sparse codes).
"""


def compute_max_jaccard_per_class(class_matrix, concept_matrix):
    """
    Vectorized version: Compute Jaccard index between each class and each concept.

    Args:
        class_matrix (torch.Tensor): (N, C) one-hot class matrix
        concept_matrix (torch.Tensor): (N, D)  concept matrix

    Returns:
        max_scores (np.ndarray): shape (C,)
        best_concepts (np.ndarray): shape (C,)
        jaccard_matrix (np.ndarray): shape (C, D)
    """
    A = class_matrix.int().cpu().numpy()  # (N, C)
    B = concept_matrix.int().cpu().numpy()  # (N, D)

    # Intersection: (C, D) = (N, C)^T @ (N, D)
    intersection = A.T @ B

    # Union: |A| + |B| - |A ∩ B|
    A_sum = A.sum(axis=0, keepdims=True).T  # (C, 1)
    B_sum = B.sum(axis=0, keepdims=True)  # (1, D)
    # Add a small epsilon (1e-6) to avoid division by zero in the Jaccard calculation
    union = A_sum + B_sum - intersection + 1e-6

    jaccard = intersection / union  # (C, D)

    best_concepts = jaccard.argmax(axis=1)
    max_scores = jaccard[np.arange(jaccard.shape[0]), best_concepts]

    return max_scores, best_concepts, jaccard


def f1_score_matrix(class_matrix, concept_matrix):
    """
    Vectorized computation of F1 scores between each class and each concept.

    Args:
        class_matrix (torch.Tensor): (N, C) binary
        concept_matrix (torch.Tensor): (N, D) binary

    Returns:
        f1_matrix (np.ndarray): (C, D) F1 scores
    """
    # Convert to numpy and int32
    Y = class_matrix.int().cpu().numpy()  # (N, C)
    Z = concept_matrix.int().cpu().numpy()  # (N, D)

    TP = np.dot(Y.T, Z)  # (C, D): class & concept both 1
    FP = np.dot(1 - Y.T, Z)  # (C, D): class 0, concept 1
    FN = np.dot(Y.T, 1 - Z)  # (C, D): class 1, concept 0

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_matrix = 2 * precision * recall / (precision + recall + 1e-8)

    return f1_matrix

def compute_max_f1_per_class(class_matrix, concept_matrix):
    f1_matrix = f1_score_matrix(class_matrix, concept_matrix)  # (C, D)

    best_concepts = np.argmax(f1_matrix, axis=1)
    max_scores = f1_matrix[np.arange(f1_matrix.shape[0]), best_concepts]

    return (
        max_scores,
        best_concepts,
        f1_matrix,
    )


def _score(
    residual,
    concept_col,
    criterion,
    f_beta=0.5,
) -> float:
    if criterion == "fbeta":
        return fbeta_score(
            residual.astype(np.int32),
            concept_col.astype(np.int32),
            beta=f_beta,
            zero_division=0,
        )
    if criterion == "jaccard":
        return jaccard_score(
            residual.astype(np.int32),
            concept_col.astype(np.int32),
            average="binary",
            zero_division=0,
        )
    return normalized_mutual_info_score(
        residual,
        concept_col,
    )


def compute_bmp_per_class(
    class_matrix,
    concept_matrix,
    f_beta=0.5,
    max_coalition_size=20,
    verbose=False,
    criterion="f1",
):
    """
    Computates a type of Binary Matching Pursuit reconstruction
    of each input (each "class" in class_matrix) as the best combination
    of concepts. It returns the order where each concept is selected.

    Args:
        class_matrix (torch.Tensor): (N, C) binary
        concept_matrix (torch.Tensor): (N, D) binary
        f_beta (scalar): F_beta score parameter for atom selection. Favors precision if beta<1
        max_coalition_size (int): maximum number of concepts to combine for reconstruction (stopping criterion)
        verbose (bool): whether to print detailed logs during the process
        criterion (str): "fbeta", "jaccard" or "mi" for the scoring function used to select the next concept in the
            pursuit.

    Returns:
        order_matrix (np.ndarray): (C, D) concept order of selection (0 for not selected)
        inv_order_matrix (np.ndarray): (C, D) 1 over order of selection (0 for not selected)
                                       ==> index of first k picks available via topk on this matrix
        rec_F1_matrix (np.ndarray): (C, nnz) F1 score between GT and reconstruction binary vectors
    """
    assert criterion in ["fbeta", "jaccard", "mi"], (
        "Invalid criterion. Must be 'fbeta', 'jaccard' or 'mi'."
    )

    if isinstance(concept_matrix, torch.Tensor):
        concept_matrix = concept_matrix.detach().cpu().numpy()
    if isinstance(class_matrix, torch.Tensor):
        class_matrix = class_matrix.detach().cpu().numpy()

    concept_matrix = concept_matrix.astype(bool)
    class_matrix = class_matrix.astype(bool)

    nnz = min(
        max_coalition_size, *class_matrix.shape
    )  # Max nonzero coefficients on reconstruction
    order_matrix = np.zeros([class_matrix.shape[1], concept_matrix.shape[1]])
    inv_order_matrix = np.zeros_like(order_matrix)
    rec_F1_matrix = np.zeros([class_matrix.shape[1], nnz])

    for i_class, y in enumerate(tqdm(class_matrix.T, desc="BMP computation")):
        residual = y.copy()  # Lists all remaining positives to cover
        reconstruction = np.zeros_like(residual).astype(
            bool
        )  # Sum (logical OR) of selected atoms activations
        if verbose:
            print(f"GT nnz = {residual.sum()}")

        for step in range(nnz):
            # Compute similarity criterion (correlations in standard MP)
            if verbose:
                print(f"\n------ STEP {step + 1} --------")
            correlations = np.array(
                [
                    _score(
                        residual,
                        concept_matrix[:, jcol],
                        f_beta=f_beta,
                        criterion=criterion,
                    )
                    for jcol in range(concept_matrix.shape[1])
                ]
            )  # size (D,)

            # Select atom with maximum absolute correlation
            j = np.argmax(np.abs(correlations))

            # Update residual (binary adaptation)
            residual = residual & ~concept_matrix[:, j]

            # Check stopping criterion (reconstruction F1 does not improve)
            reconstruction = np.logical_or(reconstruction, concept_matrix[:, j])
            F1_rec = f1_score(y, reconstruction)
            if verbose:
                print(f"selected concept {j} w/ nnz = {concept_matrix[:, j].sum()} ")
                print(f"residual nnz = {residual.sum()}")
                print(f"F1 rec {F1_rec}")
            if step > 0 and F1_rec <= rec_F1_matrix[i_class, step - 1]:
                for remaining_step in range(
                    step, nnz
                ):  # Replicate previous best F1 for remaining steps
                    rec_F1_matrix[i_class, remaining_step] = rec_F1_matrix[
                        i_class, step - 1
                    ]
                break

            # Update matrices
            order_matrix[i_class, j] = step + 1
            inv_order_matrix[i_class, j] = 1.0 / (step + 1)
            rec_F1_matrix[i_class, step] = F1_rec

    return order_matrix, inv_order_matrix, rec_F1_matrix


def label_concept_clustering(
    sparse_codes, labels, class_names=None, n_clusters=10, plot=True
):
    """
    Cluster labels based on mean concept activations (sparse codes)

    Returns:
        cluster_df: DataFrame with class index, name, cluster
        sim_matrix: cosine similarity matrix
    """
    unique_labels = torch.unique(labels)
    mean_vectors = []

    for label in unique_labels:
        idxs = labels == label
        class_sparse = sparse_codes[idxs]
        mean_activation = class_sparse.mean(dim=0)  # [D]
        mean_vectors.append(mean_activation)

    mean_vectors = torch.stack(mean_vectors).cpu().numpy()  # [num_classes, D]

    # Compute cosine similarity
    sim_matrix = cosine_similarity(mean_vectors)

    # Clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="cosine", linkage="average"
    )
    cluster_labels = clustering.fit_predict(mean_vectors)

    class_names = class_names or [f"Class {i}" for i in unique_labels]
    cluster_df = pd.DataFrame(
        {
            "class_idx": unique_labels.numpy(),
            "class_name": class_names,
            "cluster_id": cluster_labels,
        }
    )

    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            sim_matrix, xticklabels=class_names, yticklabels=class_names, cmap="viridis"
        )
        plt.title("Cosine Similarity Between Classes (Based on Concept Activations)")
        plt.tight_layout()
        plt.show()

    return cluster_df, sim_matrix


def compute_samplewise_class_activations(
    sparse_codes, labels, class_names=None, num_classes=50, plot=True
):
    unique_labels = torch.unique(labels)
    mean_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(unique_labels)):
        idxs_i = labels == unique_labels[i]
        class_sparse_i = sparse_codes[idxs_i]
        class_sparse_i /= class_sparse_i.norm(dim=1, keepdim=True)  # normalize

        for j in range(len(unique_labels)):
            idxs_j = labels == unique_labels[j]
            class_sparse_j = sparse_codes[idxs_j]
            class_sparse_j /= class_sparse_j.norm(dim=1, keepdim=True)  # normalize
            sim_ij = class_sparse_i @ class_sparse_j.T
            if i == j:
                sim_ij -= torch.eye(sim_ij.shape[0], device=sim_ij.device)
            mean_matrix[i, j] = sim_ij.mean(dim=1).mean().item()
    cos_sim = cosine_similarity(mean_matrix)
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            mean_matrix,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="viridis",
        )
        plt.title(
            "Cosine Similarity Between Classes (Based on Sample Wise Concept Activations)"
        )
        plt.tight_layout()
        plt.show()
    return mean_matrix


def compute_gt_attwise_class_sim(
    attr, labels, class_names=None, num_classes=50, plot=True
):
    unique_labels = torch.unique(labels)
    mean_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(unique_labels)):
        class_attr_i = attr[i].unsqueeze(0)  # Get the attributes for class i
        class_attr_i = class_attr_i / (
            class_attr_i.norm(dim=1, keepdim=True) + 1e-8
        )  # Normalize

        for j in range(len(unique_labels)):
            class_attr_j = attr[j].unsqueeze(0)
            class_attr_j = class_attr_j / (
                class_attr_j.norm(dim=1, keepdim=True) + 1e-8
            )

            sim_ij = class_attr_i @ class_attr_j.T
            # if i == j:
            #     sim_ij -= torch.eye(sim_ij.shape[0], device=sim_ij.device)
            mean_matrix[i, j] = sim_ij.mean(dim=1).mean().item()

    normed_matrix = cosine_similarity(mean_matrix)

    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            mean_matrix,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="viridis",
        )
        plt.title("Class-to-Class Similarity Based on Class-Level Attributes")
        plt.tight_layout()
        plt.show()

    return mean_matrix


def calculate_topk_f1_coalitions(latent_concepts, ground_truth_concepts, top_k=3, verbose=False):
    if isinstance(latent_concepts, torch.Tensor):
        latent_concepts = latent_concepts.detach().cpu().numpy().astype(bool)
    if isinstance(ground_truth_concepts, torch.Tensor):
        ground_truth_concepts = ground_truth_concepts.detach().cpu().numpy().astype(bool)

    num_classes = ground_truth_concepts.shape[1]
    num_concepts = latent_concepts.shape[1]

    order_matrix     = np.zeros((num_classes, num_concepts))   # which rank was each concept selected (0 = not selected)
    inv_order_matrix = np.zeros((num_classes, num_concepts))   # 1/rank (0 = not selected)
    rec_f1_matrix    = np.zeros((num_classes, top_k))          # coalition F1 at each coalition size

    for i in tqdm(range(num_classes), desc="Calculating top-k F1 coalitions"):
        class_labels = ground_truth_concepts[:, i].astype(bool)

        # Pairwise F1 for all concepts
        f1_scores = [
            f1_score(class_labels, latent_concepts[:, k], zero_division=0)
            for k in range(num_concepts)
        ]

        topk_indices = np.argsort(f1_scores)[-top_k:][::-1]  # best first

        coalition_labels = np.zeros_like(class_labels, dtype=bool)
        for rank, concept_idx in enumerate(topk_indices):
            coalition_labels = np.logical_or(coalition_labels, latent_concepts[:, concept_idx])

            coalition_f1 = f1_score(class_labels, coalition_labels, zero_division=0)
            rec_f1_matrix[i, rank] = coalition_f1
            order_matrix[i, concept_idx]     = rank + 1
            inv_order_matrix[i, concept_idx] = 1.0 / (rank + 1)

            if verbose:
                print(f"Class {i}, top {rank+1} concepts: {topk_indices[:rank+1]}, coalition F1: {coalition_f1:.4f}")

    print(f"\nOverall mean coalition F1: {rec_f1_matrix.mean():.4f}")
    return order_matrix, inv_order_matrix, rec_f1_matrix
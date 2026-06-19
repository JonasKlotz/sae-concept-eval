import os
from pathlib import Path
from typing import Dict, Any, List

import torch
from numpy import ndarray
from omegaconf import DictConfig
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_utils import parse_batch


def extract_concept_matrix(
    cfg: DictConfig, data_loader: DataLoader, sae: Module, return_embedding=False
):
    N = len(data_loader.dataset)  # = number of samples
    D = sae.nb_concepts  # = number of dimensions of the embedding space of the SAE
    ground_truth_concept_matrix = None
    concept_matrix = torch.zeros((N, D), device=cfg.device)
    unsparse_concept_matrix = torch.zeros((N, D), device=cfg.device)
    embedding_matrix = None

    for idx, batch in enumerate(tqdm(data_loader, desc="Concept Extraction")):
        batch_dict = parse_batch(batch, dataset_name=cfg.dataset.name, embedding=True)
        embeddings = batch_dict["features"]

        labels = batch_dict["labels"]
        embeddings = embeddings.to(cfg.device)
        labels = labels.to(cfg.device)
        batch_size = embeddings.shape[0]

        # get the SAE output
        with torch.no_grad():
            codes, codes_pre_sparsification = sae.encode(embeddings)

        # codes is the output of the SAE encoder, shape (N, D)
        concept_matrix[idx * batch_size : (idx + 1) * batch_size, :] = codes
        unsparse_concept_matrix[idx * batch_size : (idx + 1) * batch_size, :] = (
            codes_pre_sparsification
        )
        if return_embedding:
            if embedding_matrix is None:
                E = embeddings.shape[1]
                embedding_matrix = torch.zeros((N, E), device=cfg.device)
            embedding_matrix[idx * batch_size : (idx + 1) * batch_size, :] = embeddings

        # fill the ground truth concept matrix
        # size depends on whether we use attr or labels
        if ground_truth_concept_matrix is None:
            C = labels.shape[-1]
            ground_truth_concept_matrix = torch.zeros((N, C), device=cfg.device)

        ground_truth_concept_matrix[idx * batch_size : (idx + 1) * batch_size, :] = (
            labels
        )

    # to numpy
    concept_matrix = concept_matrix.numpy(force=True)
    ground_truth_concept_matrix = ground_truth_concept_matrix.numpy(force=True)
    unsparse_concept_matrix = unsparse_concept_matrix.numpy(force=True)
    if return_embedding:
        embedding_matrix = embedding_matrix.numpy(force=True)
        return (
            concept_matrix,
            ground_truth_concept_matrix,
            unsparse_concept_matrix,
            embedding_matrix,
        )
    return concept_matrix, ground_truth_concept_matrix, unsparse_concept_matrix


def get_topk_matching_bmp(
    results: Dict[str, Any],
    matching_style: str = "bmp_jaccard",
    top_k=1,
):
    bmp_inv_order_matching_dict = {
        "bmp_jaccard": "bmp_inv_order_matrix_jaccard",
        "bmp_mi": "bmp_inv_order_matrix_mi",
        "bmp_f0.25": "bmp_inv_order_matrix_f0.25",
        "bmp_f0.5": "bmp_inv_order_matrix_f0.5",
        "bmp_f1.0": "bmp_inv_order_matrix_f1.0",
        "f1": "f1_inv_order_matrix",
        "nnomp": "nnomp_inv_order_matrix",
    }
    bmp_rec_F1_matching_dict = {
        "bmp_jaccard": "bmp_rec_F1_matrix_jaccard",
        "bmp_mi": "bmp_rec_F1_matrix_mi",
        "bmp_f0.25": "bmp_rec_F1_matrix_f0.25",
        "bmp_f0.5": "bmp_rec_F1_matrix_f0.5",
        "bmp_f1.0": "bmp_rec_F1_matrix_f1.0",
        "f1": "f1_rec_F1_matrix",
        "nnomp": "nnomp_rec_F1_matrix",
    }

    bmp_rec_F1_matrix = torch.as_tensor(
        results[bmp_rec_F1_matching_dict[matching_style]]
    )
    bmp_inv_order_matrix = torch.as_tensor(
        results[bmp_inv_order_matching_dict[matching_style]]
    )

    C, K = bmp_rec_F1_matrix.shape

    # 1) effective k until convergence
    topk_count = torch.zeros(C, dtype=torch.long)
    for i in range(C):
        top_index_entry = torch.argmax(bmp_rec_F1_matrix[i]) + 1
        topk_count[i] = min(
            top_index_entry.item(), top_k
        )  #  ensure at most top_k concepts are selected, even if F1 converges later

    # 2) get selected concept indices per row from inv_order matrix
    # inv_order_matrix uses 1/order for selected concepts, 0 otherwise.
    # So "selected in first k picks" == take topk of inv_order values.
    topk_concepts: List[List[int]] = []
    for i in range(C):
        k = int(topk_count[i].item())

        scores = bmp_inv_order_matrix[i]  # (D,)
        idx = torch.topk(scores, k=k, largest=True).indices
        topk_concepts.append(idx.cpu().tolist())

    return topk_concepts

def get_topk_matching(matching_style: str, results: dict, top_k=1) -> List:
    matching_dict = {"f1": "f1_matrix", "jaccard": "jaccard_matrix", "mi": "mi_matrix"}

    if matching_style in ("jaccard", "mi"):
        matching_matrix = torch.as_tensor(results[matching_dict[matching_style]])
        # top_k = 1
        _, topk_concepts = torch.topk(matching_matrix, k=top_k, dim=1)
        return topk_concepts.tolist()

    elif matching_style.startswith("bmp_") or matching_style in ("f1", "nnomp"):
        return get_topk_matching_bmp(
            results=results, matching_style=matching_style, top_k=top_k
        )
    else:
        print(
            f"Unknown matching style '{matching_style}'. "
            f"Supported styles: {list(matching_dict.keys()) + ['bmp']}."
        )
        return []


def load_metric_results(matching_dir: Path):
    # get files
    files = os.listdir(matching_dir)
    results = {}
    # load files
    for file in files:
        if file.endswith(".pt"):
            metric_name = file.replace(".pt", "")
            file_path = matching_dir / file
            results[metric_name] = torch.load(
                file_path, map_location="cpu", weights_only=False
            )
    return results


def as_index_tensor(ix, device):
    if isinstance(ix, (list, tuple)):
        return torch.tensor(ix, device=device, dtype=torch.long)
    if isinstance(ix, torch.Tensor):
        return ix.to(device=device, dtype=torch.long).view(-1)
    return torch.tensor([ix], device=device, dtype=torch.long)

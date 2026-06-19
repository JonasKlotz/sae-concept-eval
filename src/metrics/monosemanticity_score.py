import torch
from tqdm import tqdm
from torch.nn import functional as F

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


def _minmax_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mn = x.min(dim=0, keepdim=True).values
    mx = x.max(dim=0, keepdim=True).values
    denom = (mx - mn).clamp_min(eps)
    return (x - mn) / denom


def compute_monosemanticity_wrapper(cfg, embedding_data_loader, sae):
    N = len(embedding_data_loader.dataset)  # = number of samples
    D = sae.nb_concepts  # = number of dimensions of the embedding space of the SAE
    # equation (5) from the paper
    dataset = embedding_data_loader.dataset

    embedding_matrix = torch.stack(
        [dataset[i][0] for i in range(len(dataset))], dim=0
    )  # (n, d)

    # extract concept strengths
    concept_matrix = torch.zeros((N, D), device=cfg.device)
    for idx, batch in enumerate(tqdm(embedding_data_loader, desc="Concept Extraction")):
        batch_dict = parse_batch(batch, dataset_name=cfg.dataset.name, embedding=True)
        embeddings = batch_dict["features"]
        embeddings = embeddings.to(cfg.device)
        batch_size = embeddings.shape[0]

        # get the SAE output
        with torch.no_grad():
            codes, _ = sae.encode(embeddings)

        # codes is the output of the SAE encoder, shape (N, D)
        concept_matrix[idx * batch_size : (idx + 1) * batch_size, :] = codes

    return compute_monosemanticity(embedding_matrix, concept_matrix, device=cfg.device)


def compute_monosemanticity(
    embeddings: torch.Tensor, activations: torch.Tensor, device=None
):
    # Scale to 0-1 per neuron
    min_values = activations.min(dim=0, keepdim=True)[0]
    max_values = activations.max(dim=0, keepdim=True)[0]
    activations = (activations - min_values) / (max_values - min_values)

    # embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    num_images, embed_dim = embeddings.shape
    num_neurons = activations.shape[1]

    # Initialize accumulators
    weighted_cosine_similarity_sum = torch.zeros(
        num_neurons, device=torch.device(device)
    )
    weight_sum = torch.zeros(num_neurons, device=torch.device(device))
    batch_size = 100  # Set batch size

    for i in tqdm(range(num_images), desc="Processing image pairs"):
        for j_start in range(i + 1, num_images, batch_size):  # Process in batches
            j_end = min(j_start + batch_size, num_images)

            embeddings_i = embeddings[i].to(device)  # (embedding_dim)
            embeddings_j = embeddings[j_start:j_end].to(
                device
            )  # (batch_size, embedding_dim)
            activations_i = activations[i].to(device)  # (num_neurons)
            activations_j = activations[j_start:j_end].to(
                device
            )  # (batch_size, num_neurons)

            # Compute cosine similarity
            cosine_similarities = F.cosine_similarity(
                embeddings_i.unsqueeze(0).expand(
                    j_end - j_start, -1
                ),  # Expanding to (batch_size, embedding_dim)
                embeddings_j,
                dim=1,
            )

            # Compute weights and weighted similarities
            # Expanding activations_i to (1, num_neurons)
            weights = (
                activations_i.unsqueeze(0) * activations_j
            )  # (batch_size, num_neurons)
            weighted_cosine_similarities = weights * cosine_similarities.unsqueeze(
                1
            )  # (batch_size, num_neurons)

            weighted_cosine_similarities = torch.sum(
                weighted_cosine_similarities, dim=0
            )  # (num_neurons)
            weighted_cosine_similarity_sum += weighted_cosine_similarities
            weights = torch.sum(weights, dim=0)
            weight_sum += weights

    monosemanticity = torch.where(
        weight_sum != 0, weighted_cosine_similarity_sum / weight_sum, torch.nan
    )

    is_nan = torch.isnan(monosemanticity)
    nan_count = is_nan.sum()
    monosemanticity_mean = torch.mean(monosemanticity[~is_nan])
    monosemanticity_std = torch.std(monosemanticity[~is_nan])
    print()
    print(
        f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()}"
    )
    print(f"Dead neurons:", nan_count.item())
    print(f"Total neurons:", num_neurons)

    # Filter out NaNs
    valid_indices = ~torch.isnan(monosemanticity)
    valid_monosemanticity = monosemanticity[valid_indices]
    valid_indices = torch.nonzero(valid_indices).squeeze()

    # Get top 10 highest and lowest monosemantic neurons
    top_10_values, top_10_indices = torch.topk(valid_monosemanticity, 10)
    bottom_10_values, bottom_10_indices = torch.topk(
        valid_monosemanticity, 10, largest=False
    )

    # Map indices back to original positions
    top_10_indices = valid_indices[top_10_indices]
    bottom_10_indices = valid_indices[bottom_10_indices]

    # Print results
    print("Top 10 most monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(top_10_indices, top_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    print("\nBottom 10 least monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(bottom_10_indices, bottom_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    return {
        "monosemanticity_mean": monosemanticity_mean,
        "monosemanticity_std": monosemanticity_std,
        "top_10_indices": top_10_indices,
        "bottom_10_indices": bottom_10_indices,
        "monosemanticity_scores": monosemanticity,
    }


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_calculate_monosemanticity(cfg: DictConfig):
    """
    Main function to calculate and optionally display Monosemanticity Scores.

    Args:
        cfg (DictConfig): Configuration object.
    """
    print("Calculating Monosemanticity Scores...")
    metrics_dir = Path(cfg.paths.metrics_dir) / "monosemanticity"
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

    results = compute_monosemanticity_wrapper(cfg, data_loader, sae)

    # save it to torch dict
    save_metric_results(metrics_dir, results)

    return results


if __name__ == "__main__":
    run_calculate_monosemanticity()

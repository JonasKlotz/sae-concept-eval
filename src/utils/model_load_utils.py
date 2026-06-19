import os.path
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
from omegaconf import DictConfig

from overcomplete.sae.matryoshka_sae import RandomSAE, FrozenEncoderSAE
from src.overcomplete import SAE
from src.overcomplete.sae.matryoshka_sae import (
    GlobalBatchTopKMatryoshkaSAE,
    BatchTopKSAE,
    TopKSAE,
    JumpReLUSAE,
)
from src.overcomplete.optimization.nmf_baseline import NMFBaseline
import open_clip
from src.overcomplete import DinoV2


def get_image_encoder(config, device=None):
    """
    Loads the model and attaches hooks for activation extraction.

    Returns:
        model: The image encoder model (with preprocess and tokenizer attributes).
        activations: Dictionary to store activations.
        handle: Hook handle (can be None if no hook is attached).
    """
    activations = None
    handle = None

    if config["name"] == "CLIP":
        backbone = config["backbone"]
        model, _, preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained="openai", force_quick_gelu=True
        )
        model.preprocess = preprocess
        tokenizer = open_clip.get_tokenizer(backbone)
        model.tokenizer = tokenizer

    elif config["name"] == "DINOV2":
        model = DinoV2(device=device)
        # TODO add preprocess and tokenizer if needed

    else:
        raise ValueError(f"Model {config['name']} is not supported.")

    if device is not None:
        model.to(device)

    return model, activations, handle


def load_concept_strengths_dict(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """Loads dictionary mapping image keys to concept strengths."""
    concept_path = Path(cfg.paths.learned_concepts_dir) / "all_concepts_dict.pth"
    return torch.load(concept_path)


def load_concept_names(cfg: DictConfig) -> np.ndarray:
    """Loads concept names from CSV."""
    path = Path(cfg.paths.learned_concepts_dir) / "concept_names.csv"
    with open(path, "r") as f:
        names = [line.strip() for line in f]
    return np.array(names)


def extract_concept_strengths(
    concept_dict: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, List]:
    """Stacks concept strengths into a tensor of shape (n_images, n_concepts)."""
    sorted_keys = sorted(concept_dict.keys())
    ordered_concept_strengths = [concept_dict[key] for key in sorted_keys]
    strengths = torch.stack(ordered_concept_strengths)  # (N_images, N_concepts)
    return strengths, sorted_keys


def load_sae(cfg, pretrained=True) -> SAE:
    """
    Loads the SAE model based on the configuration.

    Args:
        cfg (DictConfig): Configuration for the SAE model.
        pretrained (bool): Whether to load pretrained weights.
        device (torch.device): Device to load the model onto.

    Returns:
        SAE: The loaded SAE model.
    """

    sae_type = cfg.sae.model_type
    if cfg.model.name == "DINOV2":
        input_shape = 384
    else:
        input_shape = 768

    cfg.sae.input_shape = input_shape
    if sae_type == "topk":
        model = TopKSAE(cfg.sae)
    elif sae_type == "batchtopk":
        model = BatchTopKSAE(cfg.sae)

    elif sae_type == "matryoshka":
        model = GlobalBatchTopKMatryoshkaSAE(cfg.sae)

    elif sae_type == "random":
        model = RandomSAE(cfg.sae)
    elif sae_type == "frozen":
        model = FrozenEncoderSAE(cfg.sae)

    elif sae_type == "jumprelu":
        model = JumpReLUSAE(cfg.sae)

    elif sae_type == "nmf":
        model = NMFBaseline(
            input_dim=input_shape,
            nb_concepts=cfg.sae.dict_size,
            solver=cfg.sae.get("solver", "hals"),
            nnls_max_iter=cfg.sae.get("nnls_max_iter", 100),
        )

    if pretrained and not sae_type in ["random", "frozen"]:
        print(f"Load SAE from {cfg.sae.final_sae_path}")
        state_dict_path = cfg.sae.final_sae_path

        assert os.path.exists(state_dict_path), (
            f"State dict path does not exist: {state_dict_path}"
        )
        model.load_state_dict(torch.load(state_dict_path, map_location=cfg.device))
    return model

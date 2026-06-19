import os
import sys
from typing import Dict, Any, Sequence

from torch import device
from torch.nn import functional as F
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")

from datamodule.coco_dataset import COCOSynDataset
from metrics.metric_utils import load_metric_results, as_index_tensor


import hydra
import rootutils
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from pathlib import Path
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as T
import torch

# Set up project root
project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)

from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.utils.model_load_utils import load_sae, get_image_encoder
from src.utils.data_utils import load_embedding_datamodule
from src.metrics.calculate_metrics_gt_concept import calculate_gt_metric
from src.datamodule.CUB_syn_dataset import CUBSyntheticDataset
from src.metrics.metric_utils import get_topk_matching




def calc_rem_and_stay_metrics(
    z_orig: torch.Tensor,  # [1, K]
    z_syn: torch.Tensor,   # [1, K]
    concept_indices_to_remove: Sequence[int] | torch.Tensor,
    attrs: torch.Tensor = None,          # [1, A] original image attribute vector
    old_attr_id: int = None,             # attribute being removed
    new_attr_id: int = None,             # attribute being added (None for COCO)
    topk_concepts: list = None,          # List[List[int]], matched latents per attribute
) -> Dict[str, Any]:
    """
    Computes the "remove" and "stay" related metrics only.
    Returns:
      delta_rem, delta_stay, delta_rem_min, delta_stay_max,
      delta_rem_min_first, delta_stay_max_first,
      delta_rem_min_first_bin, delta_stay_max_first_bin,
      delta_stay_attr (mean |delta_stay| over untouched present attributes),
      n_stay_attrs (number of untouched present attributes contributing)
    """
    z_orig_s = z_orig.squeeze(0)  # [K]
    z_syn_s = z_syn.squeeze(0)    # [K]
    delta = (z_syn_s - z_orig_s)  # [K]

    I_rem = as_index_tensor(concept_indices_to_remove, delta.device)

    K = delta.numel()
    mask = torch.ones(K, dtype=torch.bool, device=delta.device)

    if I_rem.numel() > 0:
        mask[I_rem] = False

    # aggregated (continuous)
    delta_rem = delta[I_rem].mean() if I_rem.numel() > 0 else torch.tensor(0.0, device=delta.device)
    delta_stay = delta[mask].abs().mean() if mask.any() else torch.tensor(0.0, device=delta.device)

    delta_rem_min = delta[I_rem].min() if I_rem.numel() > 0 else torch.tensor(0.0, device=delta.device)
    delta_stay_max = delta[mask].abs().max() if mask.any() else torch.tensor(0.0, device=delta.device)

    # max or min first (continuous)
    delta_rem_min_first = (
        z_syn_s[I_rem].min() - z_orig_s[I_rem].min()
        if I_rem.numel() > 0
        else torch.tensor(0.0, device=delta.device)
    )
    delta_stay_max_first = (
        z_syn_s[mask].abs().max() - z_orig_s[mask].abs().max()
        if mask.any()
        else torch.tensor(0.0, device=delta.device)
    )

    # binarized max or min first
    z_orig_b = (z_orig_s > 0).float()
    z_syn_b = (z_syn_s > 0).float()
    delta_bin = (z_syn_b - z_orig_b)  # [K]
    delta_rem_bin = delta_bin[I_rem].mean() if I_rem.numel() > 0 else torch.tensor(0.0, device=delta.device)

    delta_rem_max_first_bin = (
        z_syn_b[I_rem].max() - z_orig_b[I_rem].max()
        if I_rem.numel() > 0
        else torch.tensor(0.0, device=delta.device)
    )

    delta_rem_min_first_bin = (
        z_syn_b[I_rem].min() - z_orig_b[I_rem].min()
        if I_rem.numel() > 0
        else torch.tensor(0.0, device=delta.device)
    )


    delta_stay_max_first_bin = (
        z_syn_b[mask].max() - z_orig_b[mask].max()
        if mask.any()
        else torch.tensor(0.0, device=delta.device)
    )

    # delta_stay_attr: per paper formulation — untouched present attributes
    delta_stay_attr = 0.0
    n_stay_attrs = 0
    if attrs is not None and topk_concepts is not None:
        attrs_1d = attrs.view(-1)
        vals = []
        for attr_id, attr_val in enumerate(attrs_1d):
            if attr_val.item() != 1:
                continue
            if attr_id == old_attr_id:
                continue
            if new_attr_id is not None and attr_id == new_attr_id:
                continue
            I_stay = as_index_tensor(topk_concepts[attr_id], z_orig.device)
            if I_stay.numel() == 0:
                continue
            d = (z_syn_b[I_stay].max() - z_orig_b[I_stay].max()).item()
            vals.append(abs(d))
        delta_stay_attr = sum(vals) / len(vals) if vals else 0.0
        n_stay_attrs = len(vals)

    return {
        "delta_rem": float(delta_rem.item()),
        "delta_rem_bin": float(delta_rem_bin.item()),
        "delta_stay": float(delta_stay.item()),
        "delta_rem_min": float(delta_rem_min.item()),
        "delta_stay_max": float(delta_stay_max.item()),
        "delta_rem_min_first": float(delta_rem_min_first.item()),
        "delta_stay_max_first": float(delta_stay_max_first.item()),
        "delta_rem_min_first_bin": float(delta_rem_min_first_bin.item()),
        "delta_stay_max_first_bin": float(delta_stay_max_first_bin.item()),
        "delta_rem_max_first_bin": float(delta_rem_max_first_bin.item()),
        "delta_stay_attr": delta_stay_attr,
        "n_stay_attrs": n_stay_attrs,
    }


def calc_add_metrics(
    z_orig: torch.Tensor,  # [1, K]
    z_syn: torch.Tensor,   # [1, K]
    concept_indices_to_add: Sequence[int] | torch.Tensor,
) -> Dict[str, Any]:
    """
    Computes the "add" related metrics only.
    Returns:
      delta_add, delta_add_max, delta_add_max_first, delta_add_max_first_bin
    """
    z_orig_s = z_orig.squeeze(0)  # [K]
    z_syn_s = z_syn.squeeze(0)    # [K]
    delta = (z_syn_s - z_orig_s)  # [K]

    I_add = as_index_tensor(concept_indices_to_add, delta.device)

    if I_add.numel() == 0:
        return {
            "delta_add": 0.0,
            "delta_add_max": 0.0,
            "delta_add_max_first": 0.0,
            "delta_add_max_first_bin": 0.0,
        }

    delta_add = delta[I_add].mean()
    delta_add_max = delta[I_add].max()

    delta_add_max_first = z_syn_s[I_add].max() - z_orig_s[I_add].max()

    z_orig_b = (z_orig_s > 0).float()
    z_syn_b = (z_syn_s > 0).float()
    delta_bin = (z_syn_b - z_orig_b)  # [K]
    delta_bin_add = delta_bin[I_add].mean()

    delta_add_max_first_bin = z_syn_b[I_add].max() - z_orig_b[I_add].max()

    return {
        "delta_add": float(delta_add.item()),
        "delta_bin_add" : float(delta_bin_add.item()),
        "delta_add_max": float(delta_add_max.item()),
        "delta_add_max_first": float(delta_add_max_first.item()),
        "delta_add_max_first_bin": float(delta_add_max_first_bin.item()),
    }

def _get_normed_encodings(device: device, image_encoder, orig_img, sae, syn_img):
    atom_norms = sae.W_enc.norm(dim=0)
    with torch.no_grad():
        enc_orig = image_encoder.encode_image(orig_img.to(device))
        z_orig, _ = sae.encode(enc_orig)
        z_orig = z_orig / (atom_norms + 1e-8)

        enc_syn = image_encoder.encode_image(syn_img.to(device))
        z_syn, _ = sae.encode(enc_syn)
        z_syn = z_syn / (atom_norms + 1e-8)

    # detach all
    z_orig = z_orig.detach()
    z_syn = z_syn.detach()

    embedding_difference = F.cosine_similarity(enc_orig, enc_syn).item()
    sae_difference = F.cosine_similarity(z_orig, z_syn).item()
    l1_sae_change = (z_syn - z_orig).abs().sum().item()
    return embedding_difference, l1_sae_change, sae_difference, z_orig, z_syn



def calculate_perturbation_metric_cub(
    syn_dataset,
    device: torch.device,
    topk_concepts: Dict[int, torch.Tensor],
    image_encoder,
    sae,
    cfg: DictConfig,
    metrics_dir: Path,
    file_name: str,
) -> pd.DataFrame:
    image_encoder.to(device)
    sae.to(device)

    save_path = Path(metrics_dir, f"{file_name}.csv")
    cub_syn_loader = DataLoader(syn_dataset, batch_size=1, shuffle=False, num_workers=0)
    #
    rows = []
    for batch in tqdm(cub_syn_loader, desc="Calculating perturbation metrics"):
        (
            orig_img,
            label,
            attrs,
            syn_img,
            label_c,
            attrs_c,
            old_attr_name,
            new_attr_name,
            idx,
        ) = batch
        old_attr_id = syn_dataset.reverse_attr_map[old_attr_name[0]] - 1
        new_attr_id = syn_dataset.reverse_attr_map[new_attr_name[0]] - 1
        attrs_diff = torch.where(attrs != attrs_c)[1].tolist()

        # assert that only one attribute changed
        assert len(attrs_diff) == 2, (
            f"Expected only one attribute to change, but got {len(attrs_diff)} changes."
        )
        assert old_attr_id in attrs_diff, (
            f"Old attribute id {old_attr_id} not in changed attributes {attrs_diff}."
        )
        assert new_attr_id in attrs_diff, (
            f"New attribute id {new_attr_id} not in changed attributes {attrs_diff}."
        )
        emb_diff, l1_sae_change, sae_difference, z_orig, z_syn = _get_normed_encodings(device,
                                                                                                   image_encoder,
                                                                                                   orig_img, sae,
                                                                                                   syn_img)

        # this is the concept index of the attribute that was removed
        concept_indices_to_remove = topk_concepts[old_attr_id]
        # this is the concept index of the attribute that was added
        concept_indices_to_add = topk_concepts[new_attr_id]
        # remove shared indices (if any) to isolate the effect of the changed attribute
        shared_indices = set(concept_indices_to_remove) & set(concept_indices_to_add)
        concept_indices_to_remove = [
            i for i in concept_indices_to_remove if i not in shared_indices
        ]
        concept_indices_to_add = [
            i for i in concept_indices_to_add if i not in shared_indices
        ]

        ############################################################################################################
        rem_stay = calc_rem_and_stay_metrics(
            z_orig=z_orig,
            z_syn=z_syn,
            concept_indices_to_remove=concept_indices_to_remove,
            attrs=attrs,
            old_attr_id=old_attr_id,
            new_attr_id=new_attr_id,
            topk_concepts=topk_concepts,
        )
        add_part = calc_add_metrics(
            z_orig=z_orig,
            z_syn=z_syn,
            concept_indices_to_add=concept_indices_to_add,
        )

        out = {
            **rem_stay,
            **add_part,
            "old_attr_name": old_attr_name[0],
            "new_attr_name": new_attr_name[0],
            "embedding_cosine": emb_diff,
            "sae_cosine": sae_difference,
            "l1_sae_change": l1_sae_change,
            "old_attr_id": old_attr_id,
            "new_attr_id": new_attr_id,
        }
        rows.append(out)

    # filter nan rows
    perturbation_info_df = pd.DataFrame(rows).dropna()

    os.makedirs(save_path.parent, exist_ok=True)
    perturbation_info_df.to_csv(save_path, index=False)
    print("Saved perturbation results to", save_path)

    return perturbation_info_df




def calculate_perturbation_metric_coco(
    syn_dataset,
    device: torch.device,
    topk_concepts: Dict[int, torch.Tensor],
    image_encoder,
    sae,
    cfg: DictConfig,
    metrics_dir: Path,
    file_name: str,
) -> pd.DataFrame:
    image_encoder.to(device)
    sae.to(device)

    save_path = Path(metrics_dir, f"{file_name}.csv")
    cub_syn_loader = DataLoader(syn_dataset, batch_size=1, shuffle=False, num_workers=0)
    #
    rows = []

    for batch in tqdm(cub_syn_loader, desc="Calculating perturbation metrics"):
        orig_img, attrs, syn_img, attrs_c, removed_attr_name, idx = batch
        attrs_diff_list = torch.where(attrs != attrs_c)[1].tolist()
        if len(attrs_diff_list) != 1:
            print(
                f"Skipping sample {idx.item()} with {len(attrs_diff_list)} changed attributes."
            )
            continue
        attrs_diff = attrs_diff_list[0]

        emb_diff, l1_sae_change, sae_difference, z_orig, z_syn = _get_normed_encodings(device,
                                                                                       image_encoder,
                                                                                       orig_img, sae,
                                                                                       syn_img)
        # this is the concept index of the attribute that was removed
        concept_indices_to_remove = topk_concepts[attrs_diff]

        # remove shared indices (if any) to isolate the effect of the changed attribute
        concept_indices_to_remove = [i for i in concept_indices_to_remove]

        ############################################################################################################
        rem_stay = calc_rem_and_stay_metrics(
            z_orig=z_orig,
            z_syn=z_syn,
            concept_indices_to_remove=concept_indices_to_remove,
            attrs=attrs,
            old_attr_id=attrs_diff,
            new_attr_id=None,
            topk_concepts=topk_concepts,
        )

        out = {
            **rem_stay,
            "removed_attr_name": removed_attr_name[0],
            "embedding_cosine": emb_diff,
            "sae_cosine": sae_difference,
            "l1_sae_change": l1_sae_change,
        }
        rows.append(out)

    # filter nan rows
    perturbation_info_df = pd.DataFrame(rows).dropna()

    os.makedirs(save_path.parent, exist_ok=True)
    perturbation_info_df.to_csv(save_path, index=False)
    print("Saved perturbation results to", save_path)

    return perturbation_info_df



@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_tapas(cfg: DictConfig):
    """
    Main function to calculate and optionally display Monosemanticity Scores.

    Args:
        cfg (DictConfig): Configuration object.
    """
    # Load data and model

    # resolve some metadata once

    if cfg.dataset.name not in ["CUB", "COCO"]:
        print("Targeted probe perturbation only implemented for CUB and COCO dataset.")
        return

    cfg = cfg.copy()  # to avoid modifying the original config

    device = cfg.device
    sae = load_sae(cfg)
    sae.eval().to(cfg.device)
    image_encoder, _, _ = get_image_encoder(
        cfg.model,
        device=cfg.device,
    )
    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    preprocess = image_encoder.preprocess.transforms
    transform_img = T.Compose([*preprocess])

    data_root = Path("/data/jonas/datasets/")
    if not data_root.exists():
        data_root = Path("/scratch/htc/jklotz/data")

    if cfg.dataset.name == "CUB":
        syn_dataset = CUBSyntheticDataset(
            root=data_root/ "syn_cub_dataset",
            transform=transform_img)
        syn_dataset_name = "syn_cub"
        calculate_perturbation_metric_func = calculate_perturbation_metric_cub
    else:
        syn_dataset = COCOSynDataset(
            root= data_root/ "syn_coco_dataset",
            transform=transform_img,
        )
        syn_dataset_name = "syn_coco"
        calculate_perturbation_metric_func = calculate_perturbation_metric_coco
    ########################################################################

    matching_dir = Path(cfg.paths.metrics_dir) / "ground_truth"
    if matching_dir.exists():
        results = load_metric_results(matching_dir)
    else:
        print("Calculating ground-truth matching metrics... as they do not exist yet.")
        results = calculate_gt_metric(cfg, sae, train_loader)
    ########################################################################
    # results = calculate_gt_metric(cfg, sae, train_loader, compute_bmp=False)

    # rename after loading of metrics
    cfg.dataset.name = syn_dataset_name

    # k_list = [1, 3,]
    k_list = [10, 5, 3, 1]
    matching_styles = ["f1",]
    bmp_styles = [
        "bmp_f0.25",
        "bmp_f0.5",
        "bmp_f1.0",
    ]
    omp = ["nnomp"]

    matching_styles = bmp_styles + matching_styles + omp
    # matching
    for matching_style in matching_styles:
        for k in k_list:
            print(
                f"\nCalculating perturbation metrics for matching style '{matching_style}' with top_k={k}..."
            )
            topk_concepts = get_topk_matching(matching_style, results, top_k=k)

            metrics_dir = cfg.paths.metrics_dir + f"/pert/"
            updated_df = calculate_perturbation_metric_func(
                syn_dataset,
                device,
                topk_concepts,
                image_encoder,
                sae,
                cfg,
                metrics_dir,
                f"{matching_style}_top{k}",
            )

            print(updated_df.head())


if __name__ == "__main__":
    run_tapas()

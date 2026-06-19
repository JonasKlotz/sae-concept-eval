import os
import sys

from torch import Tensor


sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
import hydra
import rootutils
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from pathlib import Path
from typing import Dict, Any
import torch.nn.functional as F
from pandas import DataFrame
from tqdm import tqdm
import torchvision.transforms as T
import torch
import lightning as L

# seed everything for reproducibility
L.seed_everything(42)
# Set up project root
project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)

from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.utils.model_load_utils import get_image_encoder
from src.utils.data_utils import load_embedding_datamodule, load_image_datamodule
from src.metrics.calculate_metrics_gt_concept import calculate_gt_metric
from src.datamodule.CUB_syn_dataset import CUBSyntheticDataset
from src.datamodule.cub_datamodule import CUBDataset
from utils.data_utils import save_metric_results

from src.metrics.probe_based.linear_probe import (
    LinearAttrPredictor,
    train_linear_predictor,
)
from src.utils.data_utils import load_results


sub_attr_names = [
    "has_back_color::grey",
    "has_belly_pattern::striped",
    "has_bill_color::grey",
    "has_bill_shape::needle",
    "has_breast_color::blue",
    "has_breast_color::red",
    "has_breast_color::white",
    "has_breast_pattern::spotted",
    "has_crown_color::black",
    "has_crown_color::grey",
    "has_crown_color::pink",
    "has_crown_color::white",
    "has_eye_color::blue",
    "has_eye_color::white",
    "has_eye_color::yellow",
    "has_forehead_color::grey",
    "has_leg_color::black",
    "has_leg_color::grey",
    "has_leg_color::pink",
    "has_primary_color::blue",
    "has_primary_color::brown",
    "has_primary_color::green",
    "has_primary_color::orange",
    "has_tail_pattern::multi-colored",
    "has_tail_pattern::solid",
    "has_throat_color::blue",
    "has_throat_color::yellow",
    "has_underparts_color::green",
    "has_underparts_color::red",
    "has_upperparts_color::yellow",
    "has_wing_color::black",
    "has_wing_color::grey",
    "has_wing_color::white",
]
ATTR_FALLBACK = {
    "has_breast_pattern::spotted": "has_breast_pattern::striped",
}


def family(attr: str) -> str:
    # "has_eye_color::black" -> "has_eye_color"
    return attr.replace("--", "::").split("::", 1)[0] if isinstance(attr, str) else ""


def remove_family(attrs, banned_family: str):
    return [a for a in attrs if family(a[0]) != banned_family]


def calculate_perturbation_metric(
    cub_syn_dataset,
    device: torch.device,
    topk_concepts: Dict[int, torch.Tensor],
    image_encoder,
    sae,
    cfg: DictConfig,
    metrics_dir: Path,
    file_name: str,
) -> DataFrame:
    image_encoder.to(device)
    sae.to(device)

    save_path = Path(metrics_dir, f"{file_name}.csv")
    cub_syn_loader = DataLoader(
        cub_syn_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    #
    perturbation_info_df = DataFrame(
        columns=[
            "delta_add_sum",
            "delta_rem_sum",
            "delta_stay",
            "combined_score",
            "old_attr_name",
            "new_attr_name",
            "embedding_cosine",
            "sae_cosine",
            "l1_sae_change",
            "old_attr_id",
            "new_attr_id",
        ]
    )
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
        old_attr_id = cub_syn_dataset.reverse_attr_map[old_attr_name[0]] - 1
        new_attr_id = cub_syn_dataset.reverse_attr_map[new_attr_name[0]] - 1
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

        with torch.no_grad():
            enc_orig = image_encoder.encode_image(orig_img.to(device))
            z_orig, _ = sae.encode(enc_orig)
            enc_syn = image_encoder.encode_image(syn_img.to(device))
            z_syn, _ = sae.encode(enc_syn)

        # detach all
        z_orig = z_orig.detach()
        z_syn = z_syn.detach()

        embedding_difference = F.cosine_similarity(enc_orig, enc_syn).item()
        sae_difference = F.cosine_similarity(z_orig, z_syn).item()
        l1_sae_change = (z_syn - z_orig).abs().sum().item()
        # this is the concept index of the attribute that was removed
        concept_indices_to_remove = topk_concepts[old_attr_id]

        # this is the concept index of the attribute that was added
        concept_indices_to_add = topk_concepts[new_attr_id]

        ############################################################################################################
        eps = 1e-8

        # z_orig, z_syn: [1, K]
        delta = (z_syn - z_orig).squeeze(0)  # [K]

        # allow single int or list/1D tensor of indices

        I_rem = as_index_tensor(concept_indices_to_remove, delta.device)
        I_add = as_index_tensor(concept_indices_to_add, delta.device)
        I_tgt = torch.unique(torch.cat([I_rem, I_add]))
        K = delta.numel()
        mask = torch.ones(K, dtype=torch.bool, device=delta.device)
        mask[I_tgt] = False  # everything NOT add/remove

        # 1) directionality (aggregated)   (did the “right” concepts move the “right” way?)
        delta_add = delta[I_add].mean()  # should be positive
        delta_rem = delta[I_rem].mean()  # should be negative
        delta_stay = delta[mask].abs().mean()
        S = delta_add - delta_rem

        out = {
            "delta_add_sum": delta_add.item(),
            "delta_rem_sum": delta_rem.item(),
            "delta_stay": delta_stay.item(),
            "combined_score": S.item(),
            "old_attr_name": old_attr_name[0],
            "new_attr_name": new_attr_name[0],
            "embedding_cosine": embedding_difference,
            "sae_cosine": sae_difference,
            "l1_sae_change": l1_sae_change,
            "old_attr_id": old_attr_id,
            "new_attr_id": new_attr_id,
        }

        # append row to df
        perturbation_info_df.loc[len(perturbation_info_df)] = out
        # print()
        # print(
        #     f"Summary for sample {idx.item()} | "
        #     f"old_attr='{old_attr_name[0]}' (id={old_attr_id}) -> "
        #     f"new_attr='{new_attr_name[0]}' (id={new_attr_id}) | "
        #     f"changed_attr_ids={attrs_diff} | "
        #     f"cos_sim(emb)={embedding_difference:.4f} | "
        #     f"cos_sim(sae)={sae_difference:.4f} | "
        #     f"L1(sae)={l1_sae_change:.4f} | "
        #     f"mean_delta_add={delta_add.item():+.4f} | "
        #     f"mean_delta_rem={delta_rem.item():+.4f} | "
        #     f"mean_abs_delta_stay={delta_stay.item():.4f} | "
        #     f"S={S.item():.4f}"
        # )

    # filter nan rows
    perturbation_info_df = perturbation_info_df.dropna()
    mean_delta = perturbation_info_df["combined_score"].mean()
    print("Mean combined_score after perturbations:", mean_delta)

    os.makedirs(save_path.parent, exist_ok=True)
    perturbation_info_df.to_csv(save_path, index=False)
    print("Saved perturbation results to", save_path)

    return perturbation_info_df


def get_indices_for_perturbed_attribute(
    attr_to_idx: dict[Any, Any], old_attr_name: Tensor
) -> tuple[Any, Any]:
    fam = family(old_attr_name)
    candidate_attr_names = [
        a for a in sub_attr_names if family(a) == fam and a != old_attr_name
    ]
    if len(candidate_attr_names) == 0:
        candidate_attr_names = [ATTR_FALLBACK.get(old_attr_name)]
    else:
        candidate_attr_names = candidate_attr_names[:1]  # take first only
    new_attr_name = candidate_attr_names[0]
    old_i = attr_to_idx[old_attr_name]
    new_i = attr_to_idx[new_attr_name]
    return new_i, old_i


def get_optimal_thresholds(val_loader, probe):
    pass


def as_index_tensor(ix, device):
    if isinstance(ix, (list, tuple)):
        return torch.tensor(ix, device=device, dtype=torch.long)
    if isinstance(ix, torch.Tensor):
        return ix.to(device=device, dtype=torch.long).view(-1)
    return torch.tensor([ix], device=device, dtype=torch.long)


def get_topk_matching(matching_style: str, results: dict, top_k=1) -> Any:
    matching_dict = {
        "f1": "f1_matrix",
        "jaccard": "jaccard_matrix",
        "bmp": "bmp_inv_order_matrix",
    }

    matching_matrix = results[matching_dict[matching_style]]

    if not isinstance(matching_matrix, torch.Tensor):
        matching_matrix = torch.as_tensor(matching_matrix)

    values, topk_concepts = torch.topk(matching_matrix, k=top_k, dim=1)
    if matching_dict == "bmp":
        print(
            f"Mean F1 of top-{top_k} matching concepts combined:",
            results["bmp_rec_F1_matrix"][:, top_k].mean().item(),
        )
        print(
            f"All F1 from top-1 to top-10 matching concepts combined:",
            results["bmp_rec_F1_matrix"][:, :10].mean(axis=0),
        )
    else:
        print(
            f"Mean {matching_style} score of top-{top_k} matching concepts:",
            values.mean().item(),
        )

    return topk_concepts


import torch
import numpy as np


@torch.no_grad()
def _collect_logits_and_labels(val_loader, probe, device=None, max_batches=None):
    if device is None:
        device = next(probe.parameters()).device
    probe.eval()

    logits_list, labels_list = [], []
    for b, (features, labels, _) in enumerate(val_loader):
        if max_batches is not None and b >= max_batches:
            break
        features = features.to(device).float()
        labels = labels.to(device).float()
        logits = probe(features)  # (B, C), raw logits

        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())

    logits = torch.cat(logits_list, dim=0)  # (N, C)
    labels = torch.cat(labels_list, dim=0)  # (N, C)
    return logits, labels


def _f_beta(tp, fp, fn, beta=1.0, eps=1e-12):
    b2 = beta * beta
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp + eps)


@torch.no_grad()
def get_optimal_thresholds(
    val_loader,
    probe,
    *,
    beta: float = 1.0,
    per_label: bool = True,
    grid_size: int = 1001,
    min_pos: int = 1,
    device=None,
):
    """
    Estimate thresholds on a validation set.

    Decision rule: predict y_hat = 1{sigmoid(logit) >= t}.

    Args:
        beta: optimize F_beta (beta=1 -> F1).
        per_label: if True return vector (C,). Else return scalar threshold.
        grid_size: number of candidate thresholds in [0,1].
        min_pos: only tune labels with at least this many positives in val;
                 others default to 0.5 (or could default to prevalence).
    Returns:
        thresholds: torch.Tensor, shape (C,) or ()
        info: dict with diagnostic fields
    """
    logits, y = _collect_logits_and_labels(val_loader, probe, device=device)
    probs = torch.sigmoid(logits)  # (N, C)

    N, C = probs.shape
    ts = torch.linspace(0.0, 1.0, steps=grid_size)  # candidate thresholds

    # Precompute for speed: (T, N, C) is too big; loop over thresholds.
    y_bool = y > 0.5

    if per_label:
        best_t = torch.full((C,), 0.5, dtype=torch.float32)
        best_score = torch.full((C,), -1.0, dtype=torch.float32)

        pos_counts = y_bool.sum(dim=0)  # (C,)
        tune_mask = pos_counts >= min_pos

        for t in ts:
            pred = probs >= t  # (N, C)
            tp = (pred & y_bool).sum(dim=0).float()
            fp = (pred & ~y_bool).sum(dim=0).float()
            fn = (~pred & y_bool).sum(dim=0).float()

            score = _f_beta(tp, fp, fn, beta=beta)  # (C,)

            # only update where tuning is allowed
            improved = (score > best_score) & tune_mask
            best_score[improved] = score[improved]
            best_t[improved] = float(t)

        info = {
            "best_fbeta": best_score,
            "pos_counts": pos_counts,
            "min_pos": min_pos,
            "beta": beta,
        }
        return best_t, info

    else:
        best_t = 0.5
        best_score = -1.0

        for t in ts:
            pred = probs >= t
            tp = (pred & y_bool).sum().float()
            fp = (pred & ~y_bool).sum().float()
            fn = (~pred & y_bool).sum().float()

            score = float(_f_beta(tp, fp, fn, beta=beta))
            if score > best_score:
                best_score = score
                best_t = float(t)

        info = {"best_fbeta": best_score, "beta": beta}
        return torch.tensor(best_t, dtype=torch.float32), info


from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


@torch.no_grad()
def eval_probe_on_syn(probe, cub_syn_dataset, image_encoder, batch_size=32):
    loader = DataLoader(
        cub_syn_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    device = next(probe.parameters()).device
    probe.eval()

    preds_orig_all, preds_syn_all = [], []
    gt_orig_all, gt_syn_all = [], []

    for batch in tqdm(loader, desc="Evaluating probe on synthetic dataset"):
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

        orig_img = orig_img.to(device)
        syn_img = syn_img.to(device)

        enc_orig = image_encoder.encode_image(orig_img)
        preds_orig, _ = probe.encode(
            enc_orig
        )  # (B, C), assumed binarized if cap_to_k=False

        enc_syn = image_encoder.encode_image(syn_img)
        preds_syn, _ = probe.encode(enc_syn)

        preds_orig_all.append(preds_orig.detach().cpu())
        preds_syn_all.append(preds_syn.detach().cpu())

        # ground truth attributes coming from the batch (more robust than cub_syn_dataset.attributes indexing)
        gt_orig_all.append(attrs.detach().cpu())
        gt_syn_all.append(attrs_c.detach().cpu())

    preds_orig_all = torch.cat(preds_orig_all, dim=0).numpy()
    preds_syn_all = torch.cat(preds_syn_all, dim=0).numpy()
    gt_orig_all = torch.cat(gt_orig_all, dim=0).numpy()
    gt_syn_all = torch.cat(gt_syn_all, dim=0).numpy()

    # Original vs its GT
    f1_per_attr_orig = f1_score(
        gt_orig_all, preds_orig_all, average=None, zero_division=0
    )
    f1_macro_orig = f1_score(
        gt_orig_all, preds_orig_all, average="macro", zero_division=0
    )
    f1_micro_orig = f1_score(
        gt_orig_all, preds_orig_all, average="micro", zero_division=0
    )

    # Synthetic vs its GT (attrs_c can differ for counterfactuals)
    f1_per_attr_syn = f1_score(gt_syn_all, preds_syn_all, average=None, zero_division=0)
    f1_macro_syn = f1_score(gt_syn_all, preds_syn_all, average="macro", zero_division=0)
    f1_micro_syn = f1_score(gt_syn_all, preds_syn_all, average="micro", zero_division=0)

    print("Original images syncub:")
    print(f"  Macro F1: {f1_macro_orig:.4f}")
    print(f"  Micro F1: {f1_micro_orig:.4f}")
    print(f"  Mean per-attribute F1: {float(f1_per_attr_orig.mean()):.4f}")

    print("Synthetic images:")
    print(f"  Macro F1: {f1_macro_syn:.4f}")
    print(f"  Micro F1: {f1_micro_syn:.4f}")
    print(f"  Mean per-attribute F1: {float(f1_per_attr_syn.mean()):.4f}")

    return {
        "orig": {
            "per_attr": f1_per_attr_orig,
            "macro": float(f1_macro_orig),
            "micro": float(f1_micro_orig),
        },
        "syn": {
            "per_attr": f1_per_attr_syn,
            "macro": float(f1_macro_syn),
            "micro": float(f1_micro_syn),
        },
    }


@torch.no_grad()
def eval_probe(probe, dataset, image_encoder, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = next(probe.parameters()).device
    probe.eval()

    preds_orig_all = []
    gt_orig_all = []

    for batch in tqdm(loader, desc="Evaluating probe on dataset"):
        img, label, attrs, idx = batch
        orig_img = img.to(device)
        enc_orig = image_encoder.encode_image(orig_img)
        preds_orig, _ = probe.encode(
            enc_orig
        )  # (B, C), assumed binarized if cap_to_k=False

        preds_orig_all.append(preds_orig.detach().cpu())
        gt_orig_all.append(attrs.detach().cpu())

    preds_orig_all = torch.cat(preds_orig_all, dim=0).numpy()
    gt_orig_all = torch.cat(gt_orig_all, dim=0).numpy()

    # Original vs its GT
    f1_per_attr_orig = f1_score(
        gt_orig_all, preds_orig_all, average=None, zero_division=0
    )
    f1_macro_orig = f1_score(
        gt_orig_all, preds_orig_all, average="macro", zero_division=0
    )
    f1_micro_orig = f1_score(
        gt_orig_all, preds_orig_all, average="micro", zero_division=0
    )

    print("Original images CUB:")
    print(f"  Macro F1: {f1_macro_orig:.4f}")
    print(f"  Micro F1: {f1_micro_orig:.4f}")
    print(f"  Mean per-attribute F1: {float(f1_per_attr_orig.mean()):.4f}")

    return {
        "orig": {
            "per_attr": f1_per_attr_orig,
            "macro": float(f1_macro_orig),
            "micro": float(f1_micro_orig),
        },
    }


def eval_probe_on_emb_dataset(probe, loader, batch_size=32):
    device = next(probe.parameters()).device
    probe.eval()
    preds_orig_all = []
    gt_orig_all = []

    for batch in tqdm(loader, desc="Evaluating probe on dataset"):
        embedding, attrs, key = batch
        preds_orig, _ = probe.encode(
            embedding.to(device)
        )  # (B, C), assumed binarized if cap_to_k=False

        preds_orig_all.append(preds_orig.detach().cpu())
        gt_orig_all.append(attrs.detach().cpu())

    preds_orig_all = torch.cat(preds_orig_all, dim=0).numpy()
    gt_orig_all = torch.cat(gt_orig_all, dim=0).numpy()

    # Original vs its GT
    f1_per_attr_orig = f1_score(
        gt_orig_all, preds_orig_all, average=None, zero_division=0
    )
    f1_macro_orig = f1_score(
        gt_orig_all, preds_orig_all, average="macro", zero_division=0
    )
    f1_micro_orig = f1_score(
        gt_orig_all, preds_orig_all, average="micro", zero_division=0
    )

    print("Original images embedding:")
    print(f"  Macro F1: {f1_macro_orig:.4f}")
    print(f"  Micro F1: {f1_micro_orig:.4f}")
    print(f"  Mean per-attribute F1: {float(f1_per_attr_orig.mean()):.4f}")
    return {
        "orig": {
            "per_attr": f1_per_attr_orig,
            "macro": float(f1_macro_orig),
            "micro": float(f1_micro_orig),
        },
    }


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_targeted_probe_perturbation(cfg: DictConfig):
    """
    Main function to calculate and optionally display Monosemanticity Scores.

    Args:
        cfg (DictConfig): Configuration object.
    """
    # Load data and model

    # resolve some metadata once

    if cfg.dataset.name != "CUB":
        print("Targeted probe perturbation currently only implemented for CUB dataset.")
        return

    cfg = cfg.copy()  # to avoid modifying the original config

    device = cfg.device
    matching_dir = Path(cfg.paths.metrics_dir).parent / "matching" / "linear_probe"

    # probe = LinearAttrPredictor(768, 312, cap_to_k=True).to(device)
    # if os.path.exists(matching_dir):
    #     print(f"Loading linear predictor from {matching_dir}")
    #     state = torch.load(matching_dir / "linear_attr_probe.pt", map_location=device)
    #     probe.load_state_dict(state)
    #     probe.eval()

    image_encoder, _, _ = get_image_encoder(
        cfg.model,
        device=cfg.device,
    )
    cfg.dataset.name = "CUB"
    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    probe_path = matching_dir / "linear_attr_probe.pt"
    cap_to_k = False
    if probe_path.exists():
        print(f"Loading linear predictor from {probe_path}")
        probe = LinearAttrPredictor(768, 312, cap_to_k=cap_to_k).to(device)
        state = torch.load(probe_path, map_location=device)
        probe.load_state_dict(state)
    else:
        probe = train_linear_predictor(
            train_loader,
            val_data_loader=val_loader,
            epochs=50,
            lr=0.01,
            device=device,
            cap_to_k=cap_to_k,
        )
        torch.save(probe.state_dict(), probe_path)
        print(f"Saved linear predictor to {probe_path}")

    probe.eval()
    opt_thresh, info = get_optimal_thresholds(train_loader, probe)
    # print(f"Optimal thresholds info: {info}")
    probe.set_thresholds(opt_thresh.to(device))

    preprocess = image_encoder.preprocess.transforms
    transform_img = T.Compose([*preprocess])
    cub_syn_dataset = CUBSyntheticDataset(transform=transform_img)

    cub_datamodule = load_image_datamodule(cfg.dataset, transform_img)
    cub_datamodule.setup()
    cub_dataset = cub_datamodule.train_dataset

    device = next(probe.parameters()).device
    # res = plot_embedding_distributions(
    #     cub_dataset=cub_dataset,
    #     cub_syn_dataset=cub_syn_dataset,
    #     emb_val_loader=val_loader,  # your embedding datamodule val loader
    #     image_encoder=image_encoder,
    #     device=device,
    #     batch_size=64,
    #     max_items=20000,
    # )
    # save_metric_results(results=res, metrics_dir=matching_dir)

    # res3 = eval_probe_on_emb_dataset(probe, val_loader, batch_size=32)
    # res2 = eval_probe(probe, cub_dataset, image_encoder, batch_size=32)
    # res1 = eval_probe_on_syn(probe, cub_syn_dataset, image_encoder, batch_size=32)

    # exit()
    ########################################################################
    #
    # if matching_dir.exists():
    #     results = load_results(matching_dir)
    # else:
    #     print("Calculating ground-truth matching metrics... as they do not exist yet.")
    #     results = calculate_gt_metric(cfg, probe, train_loader, compute_bmp=False)
    ########################################################################

    results = calculate_gt_metric(cfg, probe, val_loader, compute_bmp=False)
    f1_best_concepts = results["f1_best_concepts"]
    df = DataFrame(
        {
            "attr_name": cub_syn_dataset.attribute_names,
            "best_concept_id": f1_best_concepts,
            "best_concept_name": [
                cub_syn_dataset.attr_map[c.item() + 1] for c in f1_best_concepts
            ],
        }
    )
    # matching
    for matching_style in ["f1"]:  # ["f1", "jaccard", "bmp"]
        if matching_style == "bmp":
            topks = [1, 2, 3, 5, 10]
        else:
            topks = [1]  # only top-1 for f1 and jaccard

        for topk in topks:
            topk_concepts = get_topk_matching(matching_style, results, top_k=topk)

            cfg.dataset.name = "syn_cub"
            metrics_dir = matching_dir / f"pert"
            updated_df = calculate_perturbation_metric(
                cub_syn_dataset=cub_syn_dataset,
                device=device,
                topk_concepts=topk_concepts,
                image_encoder=image_encoder,
                sae=probe,
                cfg=cfg,
                metrics_dir=metrics_dir,
                file_name=f"{matching_style}_top{topk}",
            )

            print(updated_df.head())


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def _collect_embs_from_image_dataset(
    dataset, image_encoder, device, batch_size=64, max_items=20000, syn_mode=None
):
    """
    syn_mode:
      None: dataset yields (img, label, attrs, idx)
      "orig": syn dataset yields (orig_img, label, attrs, syn_img, label_c, attrs_c, old_attr_name, new_attr_name, idx)
      "syn":  same but take syn_img
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs = []

    n_seen = 0
    for batch in tqdm(loader, desc=f"Collecting embeddings (syn_mode={syn_mode})"):
        if syn_mode is None:
            img, label, attrs, idx = batch
            img = img.to(device)
        else:
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
            img = (orig_img if syn_mode == "orig" else syn_img).to(device)

        e = image_encoder.encode_image(img)

        # move to CPU now to avoid GPU memory growth
        embs.append(e.detach().cpu())
        n_seen += e.shape[0]
        if n_seen >= max_items:
            break

    embs = torch.cat(embs, dim=0)[:max_items]
    return embs


@torch.no_grad()
def _collect_embs_from_emb_loader(loader, device, max_items=20000):
    embs = []
    n_seen = 0
    for batch in tqdm(loader, desc="Collecting stored embeddings"):
        embedding, attrs, key = batch
        e = embedding.to(device)

        embs.append(e.detach().cpu())
        n_seen += e.shape[0]
        if n_seen >= max_items:
            break

    embs = torch.cat(embs, dim=0)[:max_items]
    return embs


def _summarize_embs(name, E: torch.Tensor):
    E = E.float()
    norms = torch.linalg.norm(E, dim=1)
    flat = E.flatten()

    print(f"{name}:")
    print(f"  shape: {tuple(E.shape)}")
    print(f"  dtype: {E.dtype}")
    print(
        f"  norms: mean={norms.mean().item():.6f}, std={norms.std().item():.6f}, "
        f"min={norms.min().item():.6f}, max={norms.max().item():.6f}"
    )
    print(
        f"  values: mean={flat.mean().item():.6f}, std={flat.std().item():.6f}, "
        f"min={flat.min().item():.6f}, max={flat.max().item():.6f}"
    )

    # check if embeddings look unit-normalized
    frac_close_to_1 = ((norms - 1.0).abs() < 1e-3).float().mean().item()
    print(f"  fraction(||e||≈1 within 1e-3): {frac_close_to_1:.3f}")
    print("")


def plot_embedding_distributions(
    cub_dataset,
    cub_syn_dataset,
    emb_val_loader,
    image_encoder,
    device,
    batch_size=64,
    max_items=20000,
    hist_bins=200,
):
    # collect
    E_cub = _collect_embs_from_image_dataset(
        cub_dataset, image_encoder, device, batch_size, max_items, syn_mode=None
    )
    E_syn_orig = _collect_embs_from_image_dataset(
        cub_syn_dataset, image_encoder, device, batch_size, max_items, syn_mode="orig"
    )
    E_syn_syn = _collect_embs_from_image_dataset(
        cub_syn_dataset, image_encoder, device, batch_size, max_items, syn_mode="syn"
    )
    E_stored = _collect_embs_from_emb_loader(
        emb_val_loader, device, max_items=max_items
    )

    # summaries
    _summarize_embs("CUB (images -> encoder)", E_cub)
    _summarize_embs("SynCUB orig (images -> encoder)", E_syn_orig)
    _summarize_embs("SynCUB syn (images -> encoder)", E_syn_syn)
    _summarize_embs("Stored embeddings (from datamodule)", E_stored)

    # plot: norms
    plt.figure()
    plt.hist(
        torch.linalg.norm(E_cub.float(), dim=1).numpy(),
        bins=hist_bins,
        alpha=0.5,
        label="CUB",
    )
    plt.hist(
        torch.linalg.norm(E_syn_orig.float(), dim=1).numpy(),
        bins=hist_bins,
        alpha=0.5,
        label="SynCUB-orig",
    )
    plt.hist(
        torch.linalg.norm(E_syn_syn.float(), dim=1).numpy(),
        bins=hist_bins,
        alpha=0.5,
        label="SynCUB-syn",
    )
    plt.hist(
        torch.linalg.norm(E_stored.float(), dim=1).numpy(),
        bins=hist_bins,
        alpha=0.5,
        label="stored",
    )
    plt.title("Embedding L2 norm distribution")
    plt.xlabel("||e||2")
    plt.ylabel("count")
    plt.legend()
    plt.show()

    # plot: elementwise values (flattened, subsample for speed)
    def _flat_sample(E, n=500_000):
        x = E.float().flatten()
        if x.numel() <= n:
            return x.numpy()
        idx = torch.randint(0, x.numel(), (n,))
        return x[idx].numpy()

    plt.figure()
    plt.hist(_flat_sample(E_cub), bins=hist_bins, alpha=0.5, label="CUB")
    plt.hist(_flat_sample(E_syn_orig), bins=hist_bins, alpha=0.5, label="SynCUB-orig")
    plt.hist(_flat_sample(E_syn_syn), bins=hist_bins, alpha=0.5, label="SynCUB-syn")
    plt.hist(_flat_sample(E_stored), bins=hist_bins, alpha=0.5, label="stored")
    plt.title("Embedding value distribution (flattened)")
    plt.xlabel("embedding value")
    plt.ylabel("count")
    plt.legend()
    # plt.show()
    plt.savefig("embedding_value_distribution.png")
    print("Saved embedding value distribution plot to embedding_value_distribution.png")

    return {
        "E_cub": E_cub,
        "E_syn_orig": E_syn_orig,
        "E_syn_syn": E_syn_syn,
        "E_stored": E_stored,
    }


if __name__ == "__main__":
    run_targeted_probe_perturbation()

import os
import sys

from metrics.metric_utils import extract_concept_matrix

sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")

from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm
from collections import Counter
import rootutils

# Set up project root
project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)


from metrics.calculate_metrics_gt_concept import calculate_gt_metric
from visualization.vis_utils import index_to_label_dict


from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.utils.model_load_utils import (
    load_concept_strengths_dict,
    load_concept_names,
    extract_concept_strengths,
    load_sae,
)
from utils.data_utils import (
    load_image_datamodule,
    parse_batch,
    load_embedding_datamodule,
    get_eval_emb_dataloader,
)


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def vis_topk_images_per_concept(cfg: DictConfig, show: bool = False) -> None:
    """
    For each learned concept, visualize the top 4 most activating images,
    excluding zero activations, arranged in a 2x2 grid. Labels are ignored.
    """

    # Load data and model
    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    data_loader = get_eval_emb_dataloader(
        data_module, cfg.get("embedding_dataloader_for_eval", "test")
    )
    sae = load_sae(cfg)
    sae.eval().to(cfg.device)

    if cfg.dataset.name == "CUB":
        image_data_module = load_image_datamodule(cfg.dataset)
        data_module.setup()
        image_data_loader = image_data_module.val_dataloader()
        dataset = image_data_loader.dataset
        attr_map = dataset.attr_map
    else:
        attr_map = index_to_label_dict(cfg.dataset.name)
    results = calculate_gt_metric(cfg, sae, data_loader)
    best_concept_indices = results["f1_best_concepts"]
    concept_name_map = {}
    for i in range(sae.nb_concepts):
        best_concept = best_concept_indices[i] + 1  # because attr map starts at 1

        concept_name_map[i] = attr_map[best_concept]
    print(concept_name_map)
    # data and model
    data_module = load_embedding_datamodule(cfg, load_images=True)
    data_module.setup()

    embedding_val_loader = data_module.val_dataloader()
    dataset = embedding_val_loader.dataset
    print(f"Validation samples: {len(dataset)}")

    N = len(dataset)
    D = sae.nb_concepts

    # build activations matrix with a safe row cursor
    concept_matrix, ground_truth_concept_matrix, unsparse_concept_matrix = (
        extract_concept_matrix(cfg, embedding_val_loader, sae)
    )

    # optional concept names
    if hasattr(sae, "concept_names") and sae.concept_names is not None:
        concept_names = [str(c) for c in sae.concept_names]
    else:
        concept_names = [f"concept_{i:03d}" for i in range(D)]

    def to_uint8(img_tensor):
        x = img_tensor.detach().cpu().float()
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):
            x = x[:3]
            x = x.permute(1, 2, 0)
        arr = x.numpy()
        mn = arr.min()
        mx = arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    out_base = Path(cfg.paths.vis_dir) / "top4_images_per_concept"
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Writing concept galleries to {out_base}")

    for c in tqdm(range(D), desc="Concepts"):
        activ = concept_matrix[:, c]
        nz = activ > 0
        if not torch.any(nz):
            continue

        nz_idx = torch.nonzero(nz, as_tuple=False).squeeze(1)
        k = min(4, nz_idx.numel())
        top_vals, top_pos = torch.topk(activ[nz_idx], k=k, largest=True)
        top_indices = nz_idx[top_pos].tolist()
        top_vals = top_vals.tolist()

        cols = 2
        rows = 2 if k > 2 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()

        for ax_i, (sample_idx, act_val) in enumerate(zip(top_indices, top_vals)):
            item = dataset[sample_idx]

            # parse a single sample without labels
            batch_dict = parse_batch(
                item, dataset_name=cfg.dataset.name, embedding=True
            )
            img_t = batch_dict["images"]

            img_np = to_uint8(img_t)
            axes[ax_i].imshow(img_np)
            axes[ax_i].set_title(f"idx {sample_idx}, act {act_val:.4f}", fontsize=9)
            axes[ax_i].axis("off")

        for j in range(len(top_indices), len(axes)):
            axes[j].axis("off")

        concept_title = concept_name_map[c]
        fig.suptitle(f"Top {k} images  {concept_title}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = out_base / f"{c:03d}_{concept_names[c]}.png"

        # plt.show()
        if show:
            plt.show()
        else:
            fig.savefig(out_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    vis_topk_images_per_concept()

import os
from collections import defaultdict

import torch
import sys
import numpy as np
from time import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
# Add the src directory to the Python path
src_path = Path(root) / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.overcomplete.metrics import l0, hoyer, r2_score, relative_avg_l2_loss
from src.utils.model_load_utils import  load_sae
from src.utils.data_utils import load_embedding_datamodule

from utils.data_utils import parse_batch


@torch.no_grad()
def eval_sae(model, dataloader, device, cfg):
    all_results = defaultdict(list)
    dataset_name = cfg.dataset.name
    model.eval()

    for batch in tqdm(dataloader):
        batch_dict = parse_batch(batch, dataset_name, embedding=True)
        activations = batch_dict["features"]
        targets = batch_dict["labels"]

        # Move data to the specified device
        activations = activations.to(device)
        targets = targets.to(device)
        # Ensure model is in evaluation

        _, codes1, recons1 = model(activations)
        l0_score = l0(codes1).item()
        hs = hoyer(codes1).mean().item()
        r2 = r2_score(activations, recons1).item()
        avg_l2_loss = relative_avg_l2_loss(activations, recons1)

        all_results["l0"].append(l0_score)
        all_results["hoyer"].append(hs)
        all_results["r2_score"].append(r2)
        all_results["relative_avg_l2_loss"].append(avg_l2_loss)

    # Calculate mean and std for each metric
    metrics_summary = {}
    for metric, values in all_results.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        metrics_summary[metric] = {"mean": mean_value, "std": std_value}
        print(f"{metric} - Mean: {mean_value:.4f}, Std: {std_value:.4f}")
    # Save results to a file
    results_path = Path(cfg.paths.metrics_dir) / f"eval_results_{dataset_name}.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        for metric, values in metrics_summary.items():
            f.write(
                f"{metric} - Mean: {values['mean']:.4f}, Std: {values['std']:.4f}\n"
            )


@hydra.main(
    config_path=str(root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_eval_sae(cfg: DictConfig, override=True):
    start_time = time()
    model = load_sae(cfg, pretrained=True)

    # --- DataModule setup ---
    data_module = load_embedding_datamodule(cfg)
    dataloader = data_module.val_dataloader()

    metrics_dict = {
        "l0": l0,
        "hoyer": hoyer,
        "r2_score": r2_score,
        "relative_avg_l2_loss": relative_avg_l2_loss,
    }
    logs = eval_sae(
        model=model,
        dataloader=dataloader,
        device=cfg.device,
        cfg=cfg,
    )


if __name__ == "__main__":
    run_eval_sae()

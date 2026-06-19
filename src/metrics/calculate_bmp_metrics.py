import os
import sys
from pathlib import Path

sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")

import hydra
import rootutils
from omegaconf import DictConfig

project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)
from src.utils import resolvers  # noqa: F401

from src.metrics.calculate_metrics_gt_concept import calculate_gt_metric
from src.utils.data_utils import (
    get_eval_emb_dataloader,
    load_embedding_datamodule,
    save_metric_results,
)
from src.utils.model_load_utils import load_sae


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_bmp_metrics(cfg: DictConfig):
    metrics_dir = Path(cfg.paths.metrics_dir) / "ground_truth"

    bmp_files = [f"bmp_inv_order_matrix_f{b}.pt" for b in ["0.25", "0.5", "1.0"]]
    if all((metrics_dir / f).exists() for f in bmp_files):
        print(f"BMP metrics already exist at {metrics_dir}, skipping.")
        return

    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    data_loader = get_eval_emb_dataloader(
        data_module, cfg.get("embedding_dataloader_for_eval", "test")
    )

    sae = load_sae(cfg)
    sae.eval().to(cfg.device)

    results = calculate_gt_metric(
        cfg,
        sae,
        data_loader,
        metrics_dir=metrics_dir,
        compute_bmp=True,
        compute_omp=False,
    )

    os.makedirs(metrics_dir, exist_ok=True)
    save_metric_results(metrics_dir, results)
    print(f"BMP metrics saved to {metrics_dir}")


if __name__ == "__main__":
    run_bmp_metrics()

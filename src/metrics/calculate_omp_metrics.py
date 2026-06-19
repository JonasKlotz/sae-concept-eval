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

from src.metrics.calculate_omp import compute_omp_per_class
from src.metrics.metric_utils import extract_concept_matrix
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
def run_omp_metrics(cfg: DictConfig):
    metrics_dir = Path(cfg.paths.metrics_dir) / "ground_truth"

    if metrics_dir.exists() and (metrics_dir / "nnomp_inv_order_matrix.pt").exists():
        print(f"OMP metrics already exist at {metrics_dir}, skipping.")
        return

    data_module = load_embedding_datamodule(cfg)
    data_module.setup()
    data_loader = get_eval_emb_dataloader(
        data_module, cfg.get("embedding_dataloader_for_eval", "test")
    )

    sae = load_sae(cfg)
    sae.eval().to(cfg.device)

    concept_matrix, gt_matrix, _ = extract_concept_matrix(cfg, data_loader, sae)

    _, inv_order, rec_F1 = compute_omp_per_class(gt_matrix, concept_matrix, nonneg=True)

    results = {
        "nnomp_inv_order_matrix": inv_order,
        "nnomp_rec_F1_matrix": rec_F1,
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_metric_results(metrics_dir, results)
    print(f"OMP metrics saved to {metrics_dir}")


if __name__ == "__main__":
    run_omp_metrics()

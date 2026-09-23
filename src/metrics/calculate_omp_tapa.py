from pathlib import Path


import hydra
import rootutils
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import DataLoader

project_root = Path(
    rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
)
from src.utils import resolvers  # noqa: F401

from src.datamodule.hf_syn_datasets import load_syn_dataset
from src.metrics.calculate_metrics_targeted_perturbation import (
    calculate_perturbation_metric_cub,
    calculate_perturbation_metric_coco,
)
from src.metrics.metric_utils import get_topk_matching, load_metric_results
from src.utils.data_utils import load_embedding_datamodule
from src.utils.model_load_utils import get_image_encoder, load_sae


@hydra.main(
    config_path=str(project_root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def run_omp_tapa(cfg: DictConfig):
    if cfg.dataset.name not in ["CUB", "COCO"]:
        print("TAPAScore only implemented for CUB and COCO, skipping.")
        return

    matching_dir = Path(cfg.paths.metrics_dir) / "ground_truth"

    if not matching_dir.exists() or not (matching_dir / "nnomp_inv_order_matrix.pt").exists():
        raise FileNotFoundError(
            f"OMP matching results not found at {matching_dir}. "
            "Run calculate_omp_metrics.py first."
        )

    results = load_metric_results(matching_dir)

    sae = load_sae(cfg)
    sae.eval().to(cfg.device)
    image_encoder, _, _ = get_image_encoder(cfg.model, device=cfg.device)
    transform_img = T.Compose([*image_encoder.preprocess.transforms])

    syn_dataset = load_syn_dataset(cfg.dataset.syn, cfg.dataset.name, transform=transform_img)
    if cfg.dataset.name == "CUB":
        calculate_perturbation_metric_func = calculate_perturbation_metric_cub
    else:
        calculate_perturbation_metric_func = calculate_perturbation_metric_coco

    # rename after loading of metrics so cfg.paths.metrics_dir resolves to the syn_* path
    cfg = cfg.copy()
    cfg.dataset.name = cfg.dataset.syn.name

    pert_dir = Path(cfg.paths.metrics_dir) / "pert"

    # skip if all k variants already exist
    if all((pert_dir / f"nnomp_top{k}.csv").exists() for k in [10, 5, 3, 1]):
        print(f"OMP TAPAScore results already exist at {pert_dir}, skipping.")
        return

    for k in [10, 5, 3, 1]:
        topk_concepts = get_topk_matching("nnomp", results, top_k=k)
        calculate_perturbation_metric_func(
            syn_dataset,
            cfg.device,
            topk_concepts,
            image_encoder,
            sae,
            cfg,
            pert_dir,
            f"nnomp_top{k}",
        )

    print(f"OMP TAPAScore results saved to {pert_dir}")


if __name__ == "__main__":
    run_omp_tapa()

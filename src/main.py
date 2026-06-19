import sys


sys.path.append("/home/htc/jklotz/git/rs_concepts_public")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")

from pathlib import Path
from pprint import pprint

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf


root = Path(rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False))


# lighting fix all seeds
from lightning import seed_everything

seed_everything(42, workers=True)


from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.metrics.calculate_fms import run_calculate_fms

from src.metrics.calculate_metrics_targeted_perturbation import (
    run_tapas,
)
from src.metrics.cknna import run_calculate_cknna

from src.visualization.vis_topk_images_per_concept import vis_topk_images_per_concept
from src.metrics.calculate_metrics_gt_concept import run_calculate_matching_metrics
from src.models.new_sae_lightning import train_with_lightning
from src.metrics.monosemanticity_score import run_calculate_monosemanticity
from src.scripts.calculate_embeddings_for_images import embed_images

DEBUG = False


def calculate_all_metrics(cfg):
    # calculate SOTA metrics
    run_calculate_cknna(cfg)
    run_calculate_fms(cfg)
    run_calculate_monosemanticity(cfg)


    # calculate our metrics
    run_calculate_matching_metrics(cfg)
    run_tapas(cfg)

    # # # #
    # # Visualizations
    vis_topk_images_per_concept(cfg)


@hydra.main(
    config_path=str(root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig):
    print("Running main with config:")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    embed_images(
        cfg,
    )
    print("Image embedding done")
    train_with_lightning(cfg)
    # # # #

    calculate_all_metrics(cfg)


if __name__ == "__main__":
    main()

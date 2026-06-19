import os
import sys

sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")
sys.path.append("/home/htc/jklotz/git/rs_concepts_public/src")


from pathlib import Path

import hydra
import numpy as np
import torch
import lmdb
import pandas as pd
import torchvision.transforms as T
from omegaconf import DictConfig
from safetensors.torch import save as safetensors_save
import tqdm
import pytorch_lightning as pl

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
import rootutils

root = Path(rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False))

from src.utils import resolvers  # noqa: F401 ensures resolver is registered
from src.visualization.vis_utils import plot_batch
from src.utils.io_utils import ensure_dirs
from src.utils.model_load_utils import (
    get_image_encoder,
)
from src.utils.data_utils import load_image_datamodule, parse_batch

DEBUG = False


@hydra.main(
    config_path=str(root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def embed_images(cfg: DictConfig):
    ensure_dirs(cfg.paths)
    overwrite = cfg.get("overwrite", False)

    # Set the seed for reproducibility
    pl.seed_everything(cfg.seed)

    print(f"CUDA available: {torch.cuda.is_available()} using device {cfg.device}")

    # Load the image encoder
    vlm_module, activations, handle = get_image_encoder(cfg.model, device=cfg.device)

    transform_img = T.Compose(
        [
            *vlm_module.preprocess.transforms,  # Unpack the existing Compose steps
        ]
    )
    mean = transform_img.transforms[-1].mean
    std = transform_img.transforms[-1].std

    data_module = load_image_datamodule(cfg.dataset, train_transform=transform_img, val_transform=transform_img)

    data_module.setup()
    loader_dict = {
        "train": data_module.train_dataloader(),
        "val": data_module.val_dataloader(),
        "test": data_module.test_dataloader(),
    }

    print(loader_dict)

    for loader_name, data_loader in loader_dict.items():
        # Create LMDB environment for embeddings
        embedded_data_lmdb = os.path.join(
            cfg.paths.embedded_data_dir, f"image_embeddings_{loader_name}.lmdb"
        )
        images_lmdb = os.path.join(
            cfg.paths.embedded_data_dir, f"images_{loader_name}.lmdb"
        )

        embedded_data_metadata = os.path.join(
            cfg.paths.embedded_data_dir, f"metadata_{loader_name}.parquet"
        )
        if os.path.exists(embedded_data_metadata) and not overwrite:
            print(
                f"{embedded_data_metadata} already exists. Set 'overwrite: True' to overwrite."
            )
            continue
        print(f"Start embedding {loader_name}")
        # Prepare metadata list
        metadata = []
        map_size = 1 * 1024 * 1024 * 1024 * 1024  # 1TB
        embedding_env = lmdb.open(embedded_data_lmdb, map_size=map_size)
        image_env = lmdb.open(images_lmdb, map_size=map_size)

        print(f"Number of samples: {len(data_loader.dataset)}")

        for batch_idx, batch in tqdm.tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Embedding",
            leave=False,
        ):
            batch_dict = parse_batch(batch, dataset_name=cfg.dataset.name)

            # Process the batch
            process_batch(
                vlm_module=vlm_module,
                batch_dict=batch_dict,
                embedding_env=embedding_env,
                image_env=image_env,
                metadata=metadata,
                device=cfg.device,
                mean=mean,
                std=std,
            )

        # Save metadata to Parquet
        df = pd.DataFrame(metadata)
        if cfg.dataset.name == "ade20k":
            df["label"] = df["label"].apply(
                lambda x: np.where(x == 1)[0] if isinstance(x, np.ndarray) else x
            )
            df["attrs"] = df["attrs"].apply(
                lambda x: np.where(x == 1)[0] if isinstance(x, np.ndarray) else x
            )

        for c in df.columns:
            # assert types from pandas
            df[c] = df[c].astype(str)
        df.to_parquet(embedded_data_metadata, index=False)
        print(f"Saved metadata to {embedded_data_metadata}")
        print(f"Saved {len(metadata)} embeddings to {embedded_data_lmdb}")
        embedding_env.close()


@torch.no_grad()
def process_batch(
    vlm_module,
    batch_dict,
    embedding_env,
    image_env,
    metadata,
    device,
    mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP defaults
    std=(0.26862954, 0.26130258, 0.27577711),
    cfg=None,
):
    if batch_dict is None:
        return
    images = batch_dict["features"]
    labels = batch_dict["labels"]
    keys = batch_dict["keys"]

    attrs = batch_dict.get("attrs", None)

    images = images.to(device)
    labels = labels.to(device)

    if DEBUG:
        plot_batch(images, labels)

    # Encode embeddings from normalized inputs
    extracted_activations = vlm_module.encode_image(images)  # [B, D]

    # De-normalize to [0, 1], convert to uint8, enforce RGB HWC for storage
    imgs = images.detach().cpu()
    mean_t = torch.tensor(mean, dtype=imgs.dtype, device=imgs.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=imgs.dtype, device=imgs.device).view(1, -1, 1, 1)
    imgs = imgs * std_t + mean_t
    imgs = torch.clamp(imgs, 0.0, 1.0)
    imgs_uint8_chw = (imgs * 255.0).round().to(torch.uint8)

    # Write embeddings
    with embedding_env.begin(write=True) as emb_txn:
        for i, key in enumerate(keys):
            key_str = str(key.item()) if isinstance(key, torch.Tensor) else str(key)

            emb = extracted_activations[i].unsqueeze(0).cpu()
            tensor_dict = {"embedding": emb}
            buffer = safetensors_save(tensor_dict, metadata=None)
            emb_txn.put(key_str.encode("utf-8"), buffer)

            label_np = labels[i].numpy(force=True)  #
            if attrs is not None:
                attrs_np = attrs[i].numpy(force=True)  #

                metadata.append({"key": key_str, "label": label_np, "attrs": attrs_np})
            else:
                metadata.append({"key": key_str, "label": label_np})

    embedding_env.sync()

    # Write images in RGB HWC order
    with image_env.begin(write=True) as img_txn:
        for i, key in enumerate(keys):
            key_str = str(key.item()) if isinstance(key, torch.Tensor) else str(key)

            img_chw = imgs_uint8_chw[i]  # [C, H, W], RGB
            img_hwc = img_chw.permute(1, 2, 0).contiguous()  # [H, W, C], RGB

            tensor_dict = {"image_rgb": img_hwc}
            buffer = safetensors_save(tensor_dict, metadata=None)
            img_txn.put(key_str.encode("utf-8"), buffer)

    image_env.sync()


if __name__ == "__main__":
    embed_images()

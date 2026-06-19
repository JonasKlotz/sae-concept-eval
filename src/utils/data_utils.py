import os
from pathlib import Path
from typing import Dict, Any

import torch

from datamodule.embedding_datamodule import EmbeddingDataModule


def load_image_datamodule(
    data_cfg,
    train_transform=None,
    val_transform=None,
    train_shuffle=True,
    num_workers=0,
):
    """
    Loads the data module based on the provided configuration.

    Args:
        val_transform:
        data_cfg (dict): Configuration dictionary containing data parameters.
        train_transform (callable, optional): Transformations to apply to images.

    Returns:
        pl.LightningDataModule: The data module instance.
    """
    stage = data_cfg.get("stage", "fit")

    if data_cfg["name"] == "CUB":
        from src.datamodule.cub_datamodule import CUBDataModule

        data_module = CUBDataModule(
            data_dir=data_cfg["path"],
            batch_size=data_cfg["batch_size"],
            num_workers=num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
            train_shuffle=train_shuffle,
        )

    elif data_cfg["name"] == "COCO" or data_cfg["name"] == "PartImageNet":
        from src.datamodule.coco_datamodule import COCODataModule

        data_module = COCODataModule(
            train_root=data_cfg["train_root"],
            train_ann=data_cfg["train_ann"],
            val_root=data_cfg["val_root"],
            val_ann=data_cfg["val_ann"],
            test_root=data_cfg.get("test_root"),
            test_ann=data_cfg.get("test_ann"),
            batch_size=data_cfg["batch_size"],
            num_workers=num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
        )

    else:
        raise ValueError(f"Dataset {data_cfg['name']} is not supported.")

    data_module.setup(stage=stage)
    return data_module


def parse_batch(batch, dataset_name: str, embedding: bool = False) -> Dict:
    """Parses a batch of data based on the dataset name."""
    if embedding:
        if len(batch) == 4:
            features, images, labels, keys = batch
            return {
                "features": features,
                "images": images,
                "labels": labels,
                "keys": keys,
            }
        features, labels, keys = batch
        return {"features": features, "labels": labels, "keys": keys}

    if dataset_name in [
        "Caltech256",
        "eurosat",
    ]:
        images, labels, keys = batch
        return {"features": images, "labels": labels, "keys": keys}
    elif dataset_name in [
        "flair_1_toy_dataset",
        "flair_dataset",
    ]:
        images, labels, ref_map, keys = batch
        return {"features": images, "labels": labels, "keys": keys, "masks": ref_map}

    elif dataset_name == "CUB" or dataset_name == "SUB" or dataset_name == "sub_cub":
        images, labels, attrs, keys = batch
        return {"features": images, "labels": labels, "attrs": attrs, "keys": keys}

    elif dataset_name in ["COCO", "PartImageNet"]:
        images, bboxes, masks, category_ids, mlc_vectors, anns, idx = batch
        return {
            "features": images,
            "labels": mlc_vectors,
            "bboxes": bboxes,
            "masks": masks,
            "category_ids": category_ids,
            "keys": idx,
        }
    elif dataset_name == "benv2":
        images, labels, ref_map, keys = batch
        return {"features": images, "labels": labels, "keys": keys, "masks": ref_map}
    elif dataset_name == "five-billion-pixels":
        return batch
    elif dataset_name == "ade20k":
        return {
            "features": batch["image"],
            "labels": batch["y_obj"],
            "attrs": batch["y_part"],
            # "masks": batch["mask"],
            "keys": batch["path"],
        }
    elif dataset_name == "syn_cub":
        return batch
    elif dataset_name == "syn_coco":
        return batch
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")


def load_embedding_datamodule(cfg, load_images=False):
    print(f"Load embedding module from {cfg['paths']['embedded_data_dir']}.")
    load_attrs = cfg["dataset"].get("load_attrs", False)
    print(f"Loading the attrs instead of labels: {load_attrs}")
    data_module = EmbeddingDataModule(
        cfg["paths"]["embedded_data_dir"],
        batch_size=1024,  # large batch size for embedding
        num_workers=5,
        load_attrs=load_attrs,
        load_images=load_images,
    )

    data_module.setup()
    return data_module


def save_metric_results(metrics_dir: Path, results: dict):
    os.makedirs(metrics_dir, exist_ok=True)

    for key, val in results.items():
        filename = f"{key}.pt"
        path = metrics_dir / filename
        to_save = val.cpu() if torch.is_tensor(val) else val
        torch.save(to_save, path)


def get_eval_emb_dataloader(
    data_module: EmbeddingDataModule, embedding_dataloader_for_eval
) -> Any:
    assert embedding_dataloader_for_eval in ["val", "test"], (
        f"Invalid embedding_dataloader_for_eval: {embedding_dataloader_for_eval}, must be 'val' or 'test'"
    )
    if embedding_dataloader_for_eval == "val":
        data_loader = data_module.val_dataloader()
    else:
        data_loader = data_module.test_dataloader()
    return data_loader


def load_results(matching_dir: Path):
    # get files
    files = os.listdir(matching_dir)
    results = {}
    # load files
    for file in files:
        if file.endswith(".pt"):
            metric_name = file.replace(".pt", "")
            file_path = matching_dir / file
            results[metric_name] = torch.load(
                file_path, map_location="cpu", weights_only=False
            )
    return results

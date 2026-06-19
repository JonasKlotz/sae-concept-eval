import ast
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import lmdb
import torch
import pandas as pd
from safetensors.torch import load as safetensors_load
import re


class EmbeddingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size=32,
        num_workers=0,
        load_images=False,
        load_attrs=False,
        image_as_chw_float=False,
        normalize_images=False,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ):
        super().__init__()
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)

        # Embedding LMDBs
        self.train_lmdb_path = data_dir / "image_embeddings_train.lmdb"
        self.train_metadata_path = data_dir / "metadata_train.parquet"
        self.val_lmdb_path = data_dir / "image_embeddings_val.lmdb"
        self.val_metadata_path = data_dir / "metadata_val.parquet"
        self.test_lmdb_path = data_dir / "image_embeddings_test.lmdb"
        self.test_metadata_path = data_dir / "metadata_test.parquet"

        # Image LMDBs (same pattern)
        self.train_image_lmdb_path = data_dir / "images_train.lmdb"
        self.val_image_lmdb_path = data_dir / "images_val.lmdb"
        self.test_image_lmdb_path = data_dir / "images_test.lmdb"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_attrs = load_attrs

        self.load_images = load_images
        self.image_as_chw_float = image_as_chw_float
        self.normalize_images = normalize_images
        self.mean = mean
        self.std = std

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = EmbeddingDataset(
            self.train_lmdb_path,
            self.train_metadata_path,
            load_images=self.load_images,
            image_lmdb_path=self.train_image_lmdb_path,
            image_as_chw_float=self.image_as_chw_float,
            normalize_images=self.normalize_images,
            mean=self.mean,
            std=self.std,
            load_attrs=self.load_attrs,
        )
        self.val_dataset = EmbeddingDataset(
            self.val_lmdb_path,
            self.val_metadata_path,
            load_images=self.load_images,
            image_lmdb_path=self.val_image_lmdb_path,
            image_as_chw_float=self.image_as_chw_float,
            normalize_images=self.normalize_images,
            mean=self.mean,
            std=self.std,
            load_attrs=self.load_attrs,
        )

        self.test_dataset = EmbeddingDataset(
            self.test_lmdb_path,
            self.test_metadata_path,
            load_images=self.load_images,
            image_lmdb_path=self.test_image_lmdb_path,
            image_as_chw_float=self.image_as_chw_float,
            normalize_images=self.normalize_images,
            mean=self.mean,
            std=self.std,
            load_attrs=self.load_attrs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        lmdb_path: Path,
        metadata_path: Path,
        load_images=False,
        image_lmdb_path: Path | None = None,
        image_as_chw_float=False,
        normalize_images=False,
        load_attrs=False,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ):
        self.lmdb_path = str(lmdb_path)
        self.metadata = pd.read_parquet(str(metadata_path))
        if self.metadata.empty:
            raise ValueError(f"No data found in metadata file: {metadata_path}")

        self.load_images = load_images
        self.image_lmdb_path = str(image_lmdb_path) if image_lmdb_path else None
        if self.load_images and not self.image_lmdb_path:
            raise ValueError("load_images=True but image_lmdb_path is None.")

        self.image_as_chw_float = image_as_chw_float
        self.normalize_images = normalize_images
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.load_attrs = load_attrs

    def __len__(self):
        return len(self.metadata)

    def _read_embedding(self, key: str) -> torch.Tensor:
        with lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        ) as env:
            with env.begin() as txn:
                buffer = txn.get(key.encode("utf-8"))
                if buffer is None:
                    raise KeyError(f"Key {key} not found in LMDB {self.lmdb_path}.")
                tensors = safetensors_load(buffer)
                emb = tensors["embedding"].squeeze(0)
        return torch.as_tensor(emb)

    def _read_image(self, key: str) -> torch.Tensor:
        with lmdb.open(
            self.image_lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        ) as env:
            with env.begin() as txn:
                buffer = txn.get(key.encode("utf-8"))
                if buffer is None:
                    raise KeyError(
                        f"Key {key} not found in image LMDB {self.image_lmdb_path}."
                    )
                tensors = safetensors_load(buffer)
                img_hwc = tensors.get("image_rgb", tensors.get("image", None))
                if img_hwc is None:
                    raise KeyError(f"image_rgb tensor not found for key {key}.")
        img_hwc = torch.as_tensor(img_hwc)
        if img_hwc.dtype != torch.uint8:
            img_hwc = img_hwc.to(torch.uint8)

        if not self.image_as_chw_float:
            return img_hwc  # H, W, C uint8

        img_chw = img_hwc.permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
        if self.normalize_images:
            img_chw = (img_chw - self.mean) / self.std
        return img_chw

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        key = sample["key"]
        if not self.load_attrs:
            label = sample["label"]
        else:
            label = sample["attrs"]
        emb = self._read_embedding(key)
        if isinstance(label, str):
            s_fixed = re.sub(r"\s+", " ", label)
            s_fixed = s_fixed.replace("[ ", "[").replace(" ", ",")
            label = ast.literal_eval(s_fixed)
        label = torch.tensor(label)

        if "ade20k" in self.image_lmdb_path:
            n_classes = 3688
            one_hot = torch.zeros(n_classes, dtype=torch.float32)
            one_hot[label] = 1.0
            label = one_hot

        if not self.load_images:
            return emb, label, key

        img = self._read_image(key)
        return emb, img, label, key

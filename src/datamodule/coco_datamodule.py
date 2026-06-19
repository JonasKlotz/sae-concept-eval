import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datamodule.coco_dataset import COCODataset, collate_fn
from torch.utils.data import random_split


class COCODataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_root,
        train_ann,
        val_root,
        val_ann,
        test_root=None,
        test_ann=None,
        batch_size=4,
        num_workers=4,
        train_transform=None,
        val_transform=None,
    ):
        super().__init__()
        self.train_root = train_root
        self.train_ann = train_ann
        self.val_root = val_root
        self.val_ann = val_ann
        self.test_root = test_root
        self.test_ann = test_ann
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = COCODataset(
            root_dir=self.train_root,
            annotation_file=self.train_ann,
            transform=self.train_transform,
        )
        full_val_dataset = COCODataset(
            root_dir=self.val_root,
            annotation_file=self.val_ann,
            transform=self.val_transform,
        )

        # Split validation set into validation and test sets
        val_size = int(
            0.5 * len(full_val_dataset)
        )  # 50/50 split (adjust ratio if needed)
        test_size = len(full_val_dataset) - val_size
        self.validation_dataset, self.test_dataset = random_split(
            full_val_dataset,
            [val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

import csv
import os
from typing import Literal

import pandas as pd
import rootutils
import torch
from PIL import Image
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader


# Set up project root
project_root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


class CUBDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",  # "train", "val", or "test"
        transform=None,
        return_segmentation=False,
        val_split=0.15,
        seed=42,
        only_attributes_with_high_certainty=False,
    ):
        self.root = root
        self.transform = transform
        self.return_segmentation = return_segmentation

        self.set_paths(root)

        images = pd.read_csv(self.images_txt, sep=" ", names=["img_id", "filepath"])
        split_flags = pd.read_csv(self.split_txt, sep=" ", names=["img_id", "is_train"])
        labels = pd.read_csv(self.labels_txt, sep=" ", names=["img_id", "class_id"])
        self.class_names = pd.read_csv(
            self.class_names_txt, sep=" ", names=["class_id", "class_name"]
        )
        self.class_names["class_name"] = self.class_names["class_name"].apply(
            lambda s: s.split(".", 1)[1]
        )
        self._load_attr_names()

        self.class_attributes_txt = os.path.join(
            root, "attributes", f"class_attribute_labels_continuous.txt"
        )
        self.class_attributes = pd.read_csv(
            self.class_attributes_txt, sep=" ", header=None
        )
        self.continuous_class_lbls = os.path.join(
            root, "attributes", f"class_attribute_labels_continuous.txt"
        )

        df = self._load_data(images, labels, seed, split, split_flags, val_split)

        # 3 build attribute‐presence matrix exactly as before
        attr_names_txt = os.path.join(root, "attributes", "attributes.txt")
        self.attr_map = {}
        with open(attr_names_txt, "r") as f:
            for line in f:
                aid, name = line.strip().split(" ", 1)
                self.attr_map[int(aid)] = name

        records = []
        with open(self.attr_labels_txt, "r") as f:
            reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
            for row in reader:
                if len(row) != 5:
                    continue
                img_id, a_id, present, certainty, _ = row
                if only_attributes_with_high_certainty and int(certainty) < 4:
                    present = 0
                records.append((int(img_id), self.attr_map[int(a_id)], int(present)))
        attr_df = pd.DataFrame(
            records, columns=["img_id", "attribute_name", "is_present"]
        )
        self.all_info_df = df.merge(attr_df, on="img_id")
        # drop filepath and is_train for the all_info_df
        self.all_info_df = self.all_info_df.drop(columns=["filepath", "is_train"])
        # replace class_id with class_name from class_names df
        mapping = self.class_names.set_index("class_id")["class_name"]
        self.all_info_df["class_name"] = self.all_info_df["class_id"].map(mapping)
        # check duplicates

        self.build_attr_matrix(attr_df)

    # build the matrix once (e.g., in __init__ or a loader)
    def build_attr_matrix(self, attr_df):
        # canonical attribute id order and corresponding names
        attr_ids = sorted(self.attr_map)
        self.attribute_names = [self.attr_map[i] for i in attr_ids]

        # ensure unique (img_id, attribute_name) pairs
        attr_df = attr_df.drop_duplicates(
            subset=["img_id", "attribute_name"], keep="last"
        )

        # pivot to (img_id x attribute_name) and enforce canonical column order
        self.attr_matrix = (
            attr_df.pivot(index="img_id", columns="attribute_name", values="is_present")
            .fillna(0)
            .astype(int)
            .reindex(columns=self.attribute_names, fill_value=0)
        )

    def _load_data(
        self,
        images: DataFrame,
        labels: DataFrame,
        seed: int,
        split: Literal["all"] | str,
        split_flags: DataFrame,
        val_split: float,
    ) -> Series:
        df = images.merge(split_flags, on="img_id").merge(labels, on="img_id")

        # official train or test partition
        if split in ("train", "val"):
            df = df[df["is_train"] == 1].reset_index(drop=True)
        elif split == "all":
            pass  # use all data
        else:
            df = df[df["is_train"] == 0].reset_index(drop=True)

        # 2 If we need a train/val sub‐split, do it here
        if split in ("train", "val"):
            # stratify by class_id
            train_idx, val_idx = train_test_split(
                df.index.tolist(),
                test_size=val_split,
                stratify=df["class_id"].tolist(),
                random_state=seed,
            )
            if split == "train":
                chosen = train_idx
            else:
                chosen = val_idx
        else:
            # test split: use all
            chosen = df.index.tolist()

        # store only the rows we will actually serve
        self.data = df.loc[chosen].reset_index(drop=True)
        # keep a mapping so that __getitem__ can map new idx -> original position in df
        self.indices = chosen
        return df

    def _load_attr_names(self):
        attribute_names_df = pd.read_csv(
            self.attributes_names_txt,
            sep=" ",
            header=None,
            names=["attr_id", "attr_name"],
        )
        self.attribute_names = {}
        for _, row in attribute_names_df.iterrows():
            self.attribute_names[row["attr_id"] - 1] = row["attr_name"]

    def set_paths(self, root):
        # Load images, split flags, and class labels
        self.images_txt = os.path.join(root, "images.txt")
        self.split_txt = os.path.join(root, "train_test_split.txt")
        self.labels_txt = os.path.join(root, "image_class_labels.txt")
        self.class_names_txt = os.path.join(root, "classes.txt")

        self.attributes_names_txt = os.path.join(root, "attributes", "attributes.txt")
        self.attr_labels_txt = os.path.join(
            root, "attributes", "image_attribute_labels.txt"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # support both single int and list of ints
        if isinstance(idx, (list, tuple)):
            return [self[i] for i in idx]

        # map dataset‐local idx back to the original row in self.data
        row = self.data.iloc[idx]
        img_id = row["img_id"]
        img_path = os.path.join(self.root, "images", row["filepath"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # attributes vector
        attrs = torch.tensor(self.attr_matrix.loc[img_id].values, dtype=torch.float32)

        # one‐hot class label
        label = torch.zeros(200, dtype=torch.float32)
        label[row["class_id"] - 1] = 1.0

        if self.return_segmentation:
            seg_file = row["filepath"].replace(".jpg", ".png")
            seg_path = os.path.join(self.root, "segmentations", seg_file)
            seg = Image.open(seg_path).convert("L")
            seg = transforms.ToTensor()(seg)
            return img, label, attrs, seg, idx

        return img, label, attrs, idx

    def get_image_path(self, idx):
        # map dataset‐local idx back to the original row in self.data
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root, "images", row["filepath"])
        return img_path

    def label_to_class_name(self, label_tensor):
        class_idx = torch.argmax(label_tensor).item() + 1
        class_name = self.class_names[self.class_names["class_id"] == class_idx][
            "class_name"
        ].values[0]
        return class_name

    def attributes_to_names(self, attrs_tensor, threshold=0.5):
        # attrs_tensor: shape (312,) aligned with self.attribute_names
        present_idx = (attrs_tensor >= threshold).nonzero(as_tuple=True)[0].tolist()
        return [self.attribute_names[i] for i in present_idx]

    def get_info_from_image_path(self, image_path: str):
        """
        Given an absolute image path pointing to a CUB image,
        return img_id, relative filepath, class id/name, and the full attribute vector.
        """

        image_path = os.path.abspath(image_path)
        images_root = os.path.join(self.root, "images")

        if not image_path.startswith(images_root):
            raise ValueError(
                f"Image path not inside CUB images directory: {image_path}"
            )

        rel_path = os.path.relpath(image_path, images_root)

        # Find the unique img_id + class_id for this filepath.
        # Use the per-image table self.data (unique per image), not self.all_info_df (one row per attribute).
        hits = self.data[self.data["filepath"] == rel_path]

        # If the file is not inside the current split, fall back to the full images/labels table via all_info_df.
        # all_info_df is long, so we must deduplicate by img_id/class_id.
        if len(hits) == 0:
            hits = self.all_info_df[
                self.all_info_df["img_id"].isin(
                    self.all_info_df.loc[
                        self.all_info_df["img_id"].isin(
                            self.all_info_df["img_id"].unique()
                        ),
                        "img_id",
                    ]
                )
            ]
            raise RuntimeError(
                f"Filepath not found in current dataset split: {rel_path}. "
                f"Instantiate CUBDataset with split='train' or split='test' that contains this image, "
                f"or store a global filepath->img_id map."
            )

        if len(hits) != 1:
            raise RuntimeError(
                f"Expected exactly 1 match for filepath '{rel_path}', got {len(hits)}."
            )

        row = hits.iloc[0]
        img_id = int(row["img_id"])
        class_id = int(row["class_id"])

        # class name lookup
        class_name = self.class_names.loc[
            self.class_names["class_id"] == class_id, "class_name"
        ].values[0]

        # attribute vector (unique per img_id)
        if img_id not in self.attr_matrix.index:
            raise RuntimeError(
                f"Attribute matrix has no entry for img_id={img_id} ({rel_path})."
            )

        attrs = torch.tensor(self.attr_matrix.loc[img_id].values, dtype=torch.float32)

        return {
            "img_id": img_id,
            "filepath": rel_path,
            "class_id": class_id,
            "class_name": class_name,
            "attributes": attrs,
        }


class CUBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=4,
        return_segmentation=False,
        val_split=0.15,
        seed=42,
        train_transform=None,
        val_transform=None,
        train_shuffle=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_segmentation = return_segmentation
        self.val_split = val_split
        self.seed = seed
        self.train_transform = (
            transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            if train_transform is None
            else train_transform
        )
        self.val_transform = (
            transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            if val_transform is None
            else val_transform
        )

        self.train_shuffle = train_shuffle

    def setup(self, stage=None):
        self.train_dataset = CUBDataset(
            self.data_dir,
            split="train",
            transform=self.train_transform,
            return_segmentation=self.return_segmentation,
            val_split=self.val_split,
            seed=self.seed,
        )
        self.validation_dataset = CUBDataset(
            self.data_dir,
            split="val",
            transform=self.val_transform,
            return_segmentation=self.return_segmentation,
            val_split=self.val_split,
            seed=self.seed,
        )
        self.test_dataset = CUBDataset(
            self.data_dir,
            split="test",
            transform=self.val_transform,
            return_segmentation=self.return_segmentation,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
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


if __name__ == "__main__":
    cub_dataset = CUBDataset(root="/home/jokl/data/CUB")
    print()

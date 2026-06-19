import csv
import os
from pathlib import Path
from typing import Literal, Any

import pandas as pd
import rootutils
import torch
from PIL import Image
from PIL.ImageFile import ImageFile
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from datamodule.cub_datamodule import CUBDataset

# Set up project root
project_root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
SYN_CUB_ROOT = "/scratch/htc/jklotz/data/syn_cub_dataset"


def load_replacement_attrs(replacement_attrs: Path) -> dict[Any, Any]:
    replacement_attr_dict = {}
    with open(replacement_attrs, "r") as f:
        for line in f:
            line = line.strip().split(";")
            key, value = line[0], eval(line[1])
            replacement_attr_dict[key] = value
    return replacement_attr_dict


class CUBSyntheticDataset(CUBDataset):
    def __init__(
        self,
        root=SYN_CUB_ROOT,
        split="train",  # "train", "val", or "test"
        transform=None,
        return_segmentation=False,
        val_split=0.15,
        seed=42,
    ):
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            return_segmentation=return_segmentation,
            val_split=val_split,
            seed=seed,
        )
        root = Path(root)
        # Load replacement attributes mapping
        replacement_attrs_path = root / "images" / "replacement_attrs.txt"
        self.replacement_attr_dict = load_replacement_attrs(replacement_attrs_path)
        syn_image_attribute_labels_path = (
            root / "attributes" / "syn_image_attribute_labels.txt"
        )
        syn_images_path = root / "syn_images.txt"
        syn_image_class_labels_path = root / "syn_image_class_labels.txt"
        images = pd.read_csv(
            syn_images_path, sep=" ", names=["img_id", "filepath", "complement_id"]
        )

        # quality df
        quality_df = pd.read_csv(root / "quality_df.csv")
        columns_to_keep = [ 'orig_path', 'syn_path',  'ROUND ERROR',]
        quality_df = quality_df[columns_to_keep]
        # keep only rows where ROUND ERROR is greater than 1
        # 0 and 1 are considered good quality, so we want to keep them
        quality_df = quality_df[quality_df['ROUND ERROR'] > 1]
        # go over the remaining rows and remove the corresponding images from the images dataframe
        print(f"Before filtering based on quality_df: {len(images)/2} synthetic images remain.")

        for _, row in quality_df.iterrows():
            orig_path = row['orig_path'].replace("synthetic_images/", "").replace("__", "::")
            syn_path = row['syn_path'].replace("synthetic_images/", "").replace("__", "::")

            # remove the row from the images dataframe where filepath is syn_path
            images = images[images["filepath"] != syn_path]
            # also remove the original image, since it is part of a pair with the synthetic image
            images = images[images["filepath"] != orig_path]
        print(f"After filtering based on quality_df: {len(images)/2}  synthetic images remain.")
        original_len = len(images)
        # iterate over filepaths and check if the file exists, if not, remove the row
        images = images[
            images["filepath"].apply(
                lambda x: os.path.exists(os.path.join(self.root, "synthetic_images", x))
            )
        ]
        filtered_len = len(images)
        print(
            f"Filtered synthetic images: {original_len - filtered_len} images removed because files do not exist."
        )
        labels = pd.read_csv(
            syn_image_class_labels_path, sep=" ", names=["img_id", "class_id"]
        )
        self.data = images.merge(labels, on="img_id").reset_index(drop=True)

        records = []
        with open(syn_image_attribute_labels_path, "r") as f:
            reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
            for row in reader:
                if len(row) != 5:
                    continue
                img_id, a_id, present, _, _ = row
                records.append((int(img_id), self.attr_map[int(a_id)], int(present)))
        attr_df = pd.DataFrame(
            records, columns=["img_id", "attribute_name", "is_present"]
        )

        self.build_attr_matrix(attr_df)
        self.reverse_attr_map = {v: k for k, v in self.attr_map.items()}

    def __len__(self):
        return int(len(self.data) / 2)  # each image has a complement

    def __getitem__(self, idx):
        """Get item from dataset.
        Every even index corresponds to an original image, every odd index to its complement.

        """
        # multiply idx by 2 to account for complements
        idx = idx * 2
        syn_idx = idx + 1  # index of the complement image

        filepath = self.data.iloc[idx]["filepath"]
        # split at /
        old_attr_name, new_attr_name = filepath.split("/")[1].split("_to_")
        # get attribute ids
        old_attr_id = self.reverse_attr_map[old_attr_name] - 1
        new_attr_id = self.reverse_attr_map[new_attr_name] - 1

        img, label, attrs = self.process_idx(idx)
        img_c, label_c, attrs_c = self.process_idx(syn_idx)
        attrs[old_attr_id] = 1
        attrs[new_attr_id] = 0
        attrs_c[old_attr_id] = 0
        attrs_c[new_attr_id] = 1

        # assert there is exactly one attribute difference
        diff = torch.where(attrs != attrs_c)[0].shape[0]
        assert diff == 2, (
            f"Expected exactly two attribute differences, got {diff} for idx {idx} and complement {syn_idx}"
        )

        return (
            img,
            label,
            attrs,
            img_c,
            label_c,
            attrs_c,
            old_attr_name,
            new_attr_name,
            idx,
        )

    def process_idx(self, idx: int | Any) -> tuple[ImageFile, Tensor, Tensor]:
        # map dataset‐local idx back to the original row in self.data
        row = self.data.iloc[idx]
        img_id = row["img_id"]
        img_path = os.path.join(self.root, "synthetic_images", row["filepath"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # attributes vector
        attrs = torch.tensor(self.attr_matrix.loc[img_id].values, dtype=torch.float32)

        # one‐hot class label
        label = torch.zeros(200, dtype=torch.float32)
        label[row["class_id"] - 1] = 1.0
        return img, label, attrs

    def image_paths_for_index(self, idx: int | Any):
        # ---- normalize idx to a Python list of ints ----
        if isinstance(idx, torch.Tensor):
            idx_list = idx.flatten().tolist()
        elif isinstance(idx, (list, tuple)):
            idx_list = list(idx)
        else:  # single integer
            idx_list = [int(idx)]

        results = []

        for i in idx_list:
            row = self.data.iloc[i]
            img_path = os.path.join(self.root, "synthetic_images", row["filepath"])

            complement_row = self.data.iloc[i + 1]
            complement_img_path = os.path.join(
                self.root, "synthetic_images", complement_row["filepath"]
            )

            results.append((img_path, complement_img_path))

        return results


if __name__ == "__main__":
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import numpy as np

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    dataset = CUBSyntheticDataset(
        root="/scratch/htc/jklotz/data/syn_cub_dataset",
        transform=transform,
    )
    for i in range(50):
        sample = dataset[i]
        img, label, attrs, img_c, label_c, attrs_c, idx = sample
        orig_img = np.transpose(img.numpy(), (1, 2, 0))
        syn_img = np.transpose(img_c.numpy(), (1, 2, 0))

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(orig_img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(syn_img)
        axes[1].set_title("Synthetic")
        axes[1].axis("off")
        # get indices of attributes that were changed
        changed_attrs = torch.where(attrs != attrs_c)[0].tolist()
        # add as suptitle
        plt.suptitle(f"Changed attributes: {changed_attrs}")

        plt.tight_layout()
        plt.show()

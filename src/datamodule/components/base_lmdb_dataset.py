import os
import pickle
from typing import List

import cv2
import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import utils
import albumentations as A
from safetensors.numpy import load as safetensor_load


def preprocess_segmentations(batch_dict):
    # this code is only for the xai generated segmentations

    masks: np.ndarray = batch_dict["segmentations"]
    # squeeze the masks original shape is (1, H, W, num_classes)
    masks = np.squeeze(masks, axis=0)
    # threshold the masks at 0.1
    masks = (masks > 0.1).astype(np.uint8)
    # generate a list of masks for each class
    return [masks[..., i] for i in range(masks.shape[-1])]


class LMDBBaseDataset(Dataset, ABC):
    def __init__(
        self,
        images_lmdb_path,
        split,
        labels_path=None,
        metadata_path=None,
        csv_dir_path=None,
        train_csv=None,
        val_csv=None,
        test_csv=None,
        transforms=None,
        segmentations_lmdb_path=None,
        transform_segmentations: bool = False,
        num_classes=None,
        safe_tensors: bool = False,
    ):
        """
        Parameter
        ---------
        images_lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.safe_tensors = safe_tensors
        self.lmdb_path = images_lmdb_path

        self.features_env = None

        self.segmentations_lmdb_path = segmentations_lmdb_path
        self.segmentations_env = None

        if labels_path is None and metadata_path is None:
            raise ValueError("Either labels_path or metadata_path must be provided")

        elif metadata_path is not None:
            self.patch_names, self.targets = self._parse_metadata(metadata_path, split)
        else:
            split_csv_path = self._parse_csv(
                csv_dir_path, split, test_csv, train_csv, val_csv
            )
            self.patch_names = read_csv(split_csv_path)
            self.targets = self.read_labels(labels_path)

        self.num_classes = num_classes
        # targets and patchnames are both numpy arrays, we want to shuffle them together using numpy indexing
        new_indices = np.random.permutation(len(self.patch_names))
        self.patch_names = self.patch_names[new_indices]
        self.targets = self.targets[new_indices]

        self.transform = transforms
        # todo fix the segmentations transform, its not used atm
        self.segmentations_transform = transform_segmentations

        self.loader_sanity_check = False

    @staticmethod
    def _parse_csv(csv_dir_path, split, test_csv, train_csv, val_csv):
        # read the patch names from the csv file
        assert csv_dir_path is not None or (
            train_csv is not None and val_csv is not None and test_csv is not None
        ), "Either csv_dir_path or train_csv, val_csv, test_csv must be provided"

        if split == "train" and train_csv is not None:
            split_csv_path = train_csv
        elif split == "val" and val_csv is not None:
            split_csv_path = val_csv
        elif split == "test" and test_csv is not None:
            split_csv_path = test_csv
        else:
            split_csv_path = os.path.join(csv_dir_path, f"{split}.csv")
        return split_csv_path

    ################################################ SETUP ################################################
    def read_labels(self, meta_data_path):
        df = pd.read_parquet(meta_data_path)
        patch_name_column = "name" if "name" in df.columns else "patch_name"
        df_subset = (
            df.set_index(patch_name_column)
            .loc[self.patch_names]
            .reset_index(inplace=False)
        )
        string_labels = df_subset.labels.tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels

    @abstractmethod
    def convert_to_multihot(self, labels: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def _init_lmdbs(self):
        if self.features_env is None:
            self._init_db()
        if self.segmentations_lmdb_path is not None and self.segmentations_env is None:
            self._init_seg_db()

    def _init_db(self):
        self.features_env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, meminit=False
        )

    def _init_seg_db(self):
        if self.segmentations_lmdb_path is None:
            raise ValueError("No segmentation lmdb path provided!")
        self.segmentations_env = lmdb.open(
            self.segmentations_lmdb_path, readonly=True, lock=False, meminit=False
        )

    ################################################ DATA LOADING ################################################
    def _load_data(self, idx) -> dict:
        key = self.patch_names[idx]
        batch_dict = {
            "features": self._load_patch(key),
            "labels": self.targets[idx],
            "key": key,
        }

        # we only load the segmentation mask if it is available
        if self.segmentations_lmdb_path:
            batch_dict["segmentations"] = self._load_seg_mask(key)

        return batch_dict

    def _load_patch(self, key: str) -> np.ndarray:
        # patch = None
        with self.features_env.begin(write=False) as txn:
            byte_patch = txn.get(key.encode())
            patch = self._load_bytes(
                byte_patch, self.band_names if self.safe_tensors else None
            )
        if patch is None:
            raise ValueError(f"Cannot load {key} from patch database!")

        return patch.astype(np.uint8)

    def _load_seg_mask(self, key: str) -> np.ndarray:
        # seg_mask = None
        with self.segmentations_env.begin(write=False) as txn:
            byte_patch = txn.get(key.encode("utf-8"))
            seg_mask = self._load_bytes(byte_patch, ["segmentations"])
        if seg_mask is None:
            raise ValueError(f"Cannot load {key} from segmentation database!")
        return seg_mask

    def _load_bytes(self, byte_patch, band_names=None):
        if self.safe_tensors:
            tensor_dict = safetensor_load(byte_patch)
            # stack the dict to get the tensor shape must be (H, W, C)
            tensor = np.stack([tensor_dict[band] for band in band_names], axis=-1)
            return tensor
        return pickle.loads(byte_patch)

    @property
    def band_names(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        self._init_lmdbs()
        batch_dict = self._load_data(idx)

        if not self.loader_sanity_check:
            self._loader_sanity_check(batch_dict)

        if self.segmentations_transform and self.transform:
            # We also transform the segmentations self.transform:
            mask_list = preprocess_segmentations(batch_dict)
            mask_shape = mask_list[0].shape
            if mask_shape != batch_dict["features"].shape[:2]:
                pass
            # apply the transform
            transformed = self.transform(image=batch_dict["features"], masks=mask_list)

            # get the transformed image and mask
            batch_dict["features"] = transformed["image"]
            batch_dict["masks"] = transformed["masks"]

            derive_labels = True
            if derive_labels:
                batch_dict["labels"] = self.derive_labels(batch_dict["masks"])

        # We only transform the features
        elif self.transform:
            features = batch_dict["features"]
            segmentations = batch_dict.get("segmentations", None).squeeze()

            transformed = self.transform(features)
            t_c, t_h, t_w = transformed.shape

            if segmentations is not None and not torch.all(
                torch.tensor(segmentations.squeeze().shape) == torch.tensor((t_h, t_w))
            ):
                segmentations = A.Resize(t_h, t_w, interpolation=cv2.INTER_NEAREST)(
                    image=segmentations
                )["image"]

            batch_dict["features"] = transformed
            batch_dict["masks"] = segmentations
            batch_dict["keys"] = idx
        return batch_dict

    def __len__(self):
        """Get length of Dataset."""
        return len(self.patch_names)

    def derive_labels(self, seg_mask):
        targets = torch.zeros(self.num_classes)
        for targets in np.unique(seg_mask):
            targets[int(targets)] = 1.0
        return targets

    def _loader_sanity_check(self, item_dict):
        # assert features and targets
        assert "features" in item_dict.keys()
        assert "labels" in item_dict.keys()

        # assert that the features are of shape (C, H, W)
        assert len(item_dict["features"].shape) == 3

        # assert that the targets are of shape (num_classes,)
        assert len(item_dict["labels"].shape) == 1

        # assert values of the features are between 0 and 255 with some tolerance
        assert item_dict["features"].min() >= 0 - 1e-5
        assert item_dict["features"].max() <= 255 + 1e-5

        # assert uint8
        assert item_dict["features"].dtype == np.uint8

        self.loader_sanity_check = True

    def _parse_metadata(self, metadata_path, split):
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        metadata_df = pd.read_parquet(metadata_path)
        patch_name_column = "name" if "name" in metadata_df.columns else "patch_name"

        # filter the split from the split column
        metadata_df = metadata_df[metadata_df["split"] == split]
        patch_names = np.array(metadata_df[patch_name_column].tolist())

        # get labels
        labels = metadata_df["labels"].tolist()
        targets = np.array(list(map(self.convert_to_multihot, labels)))

        return patch_names, targets


def read_csv(csv_path):
    # if file exists, read it
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path, header=None).to_numpy()[:, 0]
    raise FileNotFoundError(f"CSV file not found at {csv_path}")


def get_lmdb_keys(lmdb_path):
    # Open the LMDB environment in read-only mode
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    # Initialize an empty list to store keys
    keys = []

    # Start a read transaction
    with env.begin() as txn:
        # Get the total number of keys (size of the database)
        db_len = txn.stat()["entries"]
        print(f"Length of the LMDB: {db_len}")

        # Create a cursor to iterate over the keys
        with txn.cursor() as cursor:
            for key, _ in cursor:
                # encode the key to utf-8
                keys.append(key.decode("utf-8"))

    return keys

"""Loaders for the HuggingFace-format synthetic datasets (synCUB / synCOCO).

These read the self-contained ``images/ + metadata.csv`` export produced by
``src/scripts/hf_export/export_syncub_hf.py`` and ``export_syncoco_hf.py`` and
reproduce the exact return signatures of the original on-disk datasets
(``CUBSyntheticDataset`` / ``COCOSynDataset``), so they are drop-in replacements
for the metrics pipeline.

Versioning is handled by HuggingFace git revisions: point ``root`` at a local
export, or use :meth:`from_hub` to download a specific ``revision`` (tag, branch
or commit) of a dataset repo. The on-disk schema is identical across versions.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from datamodule.coco_dataset import COCO_IDX2NAME

NUM_CUB_ATTRS = 312
NUM_CUB_CLASSES = 200


def _parse_idx_list(value) -> list[int]:
    """Parse a label/attribute index list stored as a "[1, 5, 22]" string."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    return [int(v) for v in ast.literal_eval(str(value))]


class _HFSynBase(Dataset):
    """Shared loading helpers for the HF-format synthetic datasets."""

    def __init__(self, root, transform=None, csv_name: str = "metadata.csv"):
        self.root = Path(root)
        self.transform = transform
        csv_path = self.root / csv_name
        if not csv_path.is_file():
            raise FileNotFoundError(f"metadata CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def _load_pil(self, rel_path: str) -> Image.Image:
        return Image.open(self.root / rel_path).convert("RGB")

    def image_paths_for_index(self, idx):
        if isinstance(idx, torch.Tensor):
            idx_list = idx.flatten().tolist()
        elif isinstance(idx, (list, tuple)):
            idx_list = list(idx)
        else:
            idx_list = [int(idx)]
        rows = self.df.iloc[idx_list]
        return [
            (str(self.root / r["orig_image"]), str(self.root / r["syn_image"]))
            for _, r in rows.iterrows()
        ]

    @classmethod
    def from_hub(cls, repo_id: str, revision: str | None = None, transform=None, **kwargs):
        """Download a dataset repo revision from the Hub and open it locally."""
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=repo_id, revision=revision, repo_type="dataset"
        )
        return cls(root=local_dir, transform=transform, **kwargs)


class SynCUBHFDataset(_HFSynBase):
    """HF-format synCUB. Returns the same 9-tuple as ``CUBSyntheticDataset``::

        img, label, attrs, img_c, label_c, attrs_c, old_attr_name, new_attr_name, idx
    """

    def __init__(self, root, transform=None, csv_name: str = "metadata.csv"):
        super().__init__(root, transform=transform, csv_name=csv_name)

        attr_file = self.root / "attributes.txt"
        if not attr_file.is_file():
            raise FileNotFoundError(
                f"attributes.txt (attribute vocabulary) not found: {attr_file}"
            )
        self.attr_map: dict[int, str] = {}
        with open(attr_file) as f:
            for line in f:
                aid, name = line.strip().split(" ", 1)
                self.attr_map[int(aid)] = name
        self.reverse_attr_map = {v: k for k, v in self.attr_map.items()}
        self.attribute_names = [self.attr_map[i] for i in sorted(self.attr_map)]

    def _attr_vector(self, active_idx: list[int]) -> torch.Tensor:
        vec = torch.zeros(NUM_CUB_ATTRS, dtype=torch.float32)
        if active_idx:
            vec[active_idx] = 1.0
        return vec

    def _class_onehot(self, class_id: int) -> torch.Tensor:
        label = torch.zeros(NUM_CUB_CLASSES, dtype=torch.float32)
        label[class_id - 1] = 1.0
        return label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = self._load_pil(row["orig_image"])
        img_c = self._load_pil(row["syn_image"])
        if self.transform:
            img = self.transform(img)
            img_c = self.transform(img_c)

        attrs = self._attr_vector(_parse_idx_list(row["orig_attr_idx"]))
        attrs_c = self._attr_vector(_parse_idx_list(row["syn_attr_idx"]))
        label = self._class_onehot(int(row["class_id"]))
        label_c = self._class_onehot(int(row["class_id"]))

        return (
            img,
            label,
            attrs,
            img_c,
            label_c,
            attrs_c,
            row["old_attr"],
            row["new_attr"],
            idx,
        )


class SynCOCOHFDataset(_HFSynBase):
    """HF-format synCOCO. Returns the same 6-tuple as ``COCOSynDataset``::

        image, mlc_vector_orig, syn_image, mlc_vector_syn, removed_attr_name, coco_idx
    """

    num_classes = 80

    def __init__(
        self,
        root,
        transform=None,
        normalize: bool = True,
        csv_name: str = "metadata.csv",
        num_classes: int = 80,
    ):
        super().__init__(root, transform=transform, csv_name=csv_name)
        self.normalize = normalize
        self.num_classes = int(num_classes)

    def _mlc_vector(self, active_idx: list[int]) -> np.ndarray:
        vec = np.zeros(self.num_classes, dtype=np.float32)
        for c in active_idx:
            if 0 <= int(c) < self.num_classes:
                vec[int(c)] = 1.0
        return vec

    def _to_image_tensor(self, pil_img: Image.Image):
        if self.transform is not None:
            return self.transform(pil_img)
        img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float()
        if self.normalize:
            img = img / 255.0
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = self._to_image_tensor(self._load_pil(row["orig_image"]))
        syn_image = self._to_image_tensor(self._load_pil(row["syn_image"]))

        mlc_orig = self._mlc_vector(_parse_idx_list(row["orig_labels"]))
        mlc_syn = self._mlc_vector(_parse_idx_list(row["syn_labels"]))

        return (
            image,
            mlc_orig,
            syn_image,
            mlc_syn,
            str(row["removed_class"]),
            int(row["coco_idx"]),
        )

    def get_class_name(self, class_idx):
        return COCO_IDX2NAME.get(class_idx, f"class_{class_idx}")

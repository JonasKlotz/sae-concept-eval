from __future__ import annotations

import ast
import os
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from pycocotools.coco import COCO
from torch.utils.data import Dataset

SYN_COCO_PATH = "/scratch/htc/jklotz/data/syn_coco_dataset"

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
COCO_IDX2NAME = {i: name for i, name in enumerate(COCO_CLASSES)}
COCO_NAME2IDX = {name: i for i, name in enumerate(COCO_CLASSES)}


def collate_fn(batch):
    images, bboxes, masks, category_ids, mlc_vectors, anns, idx = zip(*batch)
    images = torch.stack(images)
    mlc_vectors = torch.stack(mlc_vectors)  # fixed-size so can stack
    return images, bboxes, masks, category_ids, mlc_vectors, anns, idx


class COCODataset(Dataset):
    num_classes = 80

    def __init__(self, root_dir, annotation_file, transform=None, normalize=True):
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # keep only images that have at least one valid annotation
        self.image_ids = [
            i for i in self.image_ids if len(self.coco.getAnnIds(imgIds=i)) > 0
        ]

        # build contiguous label space
        self.cat_ids = sorted(self.coco.getCatIds())
        self.catid2contig = {cid: i for i, cid in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        path = os.path.join(self.root_dir, image_info["file_name"])

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        bboxes, masks, category_ids = [], [], []
        mlc_vector = np.zeros(self.num_classes, dtype=np.float32)

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            segm = ann.get("segmentation", None)

            # Try segmentation first
            mask = None
            if segm is not None and len(segm) > 0:
                try:
                    mask = self.coco.annToMask(ann)
                except TypeError:
                    # Defensive fallback when pycocotools hits the bbox path
                    mask = None

            # Fallback to a rectangular mask from bbox
            if mask is None:
                x1 = max(0, int(np.floor(x)))
                y1 = max(0, int(np.floor(y)))
                x2 = min(W, int(np.ceil(x + w)))
                y2 = min(H, int(np.ceil(y + h)))
                if x2 <= x1 or y2 <= y1:
                    continue
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1

            if mask.sum() == 0:
                continue

            cat = self.catid2contig[ann["category_id"]]
            bboxes.append([x, y, x + w, y + h])
            masks.append(mask)
            category_ids.append(cat)
            mlc_vector[cat] = 1

        if self.transform is not None:
            image = torchvision.transforms.ToPILImage()(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            image = image / 255.0

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        if masks:
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32)
        else:
            # empty mask tensor with correct spatial size
            if isinstance(image, torch.Tensor):
                h_t, w_t = image.shape[1], image.shape[2]
            else:
                # if transforms produced PIL, not expected here, but guard anyway
                h_t, w_t = H, W
            masks = torch.zeros((0, h_t, w_t), dtype=torch.float32)

        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)
        mlc_vector = torch.as_tensor(mlc_vector, dtype=torch.float32)
        return image, bboxes, masks, category_ids, mlc_vector, anns, idx

    def get_class_name(self, class_idx):
        class_dict = self.coco.cats[self.cat_ids[class_idx]]
        return class_dict["name"]


def _to_list(x) -> List[int]:
    """
    Robustly parse label lists that may appear as:
    - Python-list string: "[0, 61, 62]"
    - already-a-list
    - empty / NaN
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, np.ndarray)):
                return [int(t) for t in v]
        except Exception:
            pass
    raise ValueError(f"Could not parse labels from value: {x!r}")


def _make_relative_to_root(path: str, root_dir: str) -> str:
    """
    Ensure `path` becomes relative to `root_dir`.

    If `path` is absolute and starts with `root_dir`, strip that prefix.
    If `path` is absolute but does not start with `root_dir`, fall back to the basename.
    If `path` is already relative, keep it as-is.
    """
    if not isinstance(path, str) or path.strip() == "":
        raise ValueError(f"Invalid path value: {path!r}")

    path = path.strip()
    root_dir_abs = os.path.abspath(root_dir)

    if os.path.isabs(path):
        path_abs = os.path.abspath(path)
        # best case: absolute path is under root_dir
        if os.path.commonpath([path_abs, root_dir_abs]) == root_dir_abs:
            rel = os.path.relpath(path_abs, root_dir_abs)
            return rel
        # your stated case: strip a fixed prefix that is NOT necessarily `root_dir`
        # but we cannot safely infer that prefix here, so we choose a conservative fallback.
        return os.path.join("images", os.path.basename(path_abs))
    else:
        # normalize (avoid leading "./")
        return os.path.normpath(path)


class COCOSynDataset(Dataset):
    """
    Folder structure:
      root_dir/
        images/
        metadata.csv

    Expected columns in metadata.csv:
      coco_idx, orig_path, orig_labels, syn_path, syn_labels, removed_cls_idx, removed_class

    Paths in CSV may be absolute; they are converted to paths relative to root_dir.
    """

    def __init__(
        self,
        root = SYN_COCO_PATH,
        transform=None,
        normalize: bool = True,
        csv_name: str = "metadata.csv",
        num_classes: int = 80,
        validate_images: bool = False,
    ):
        self.root_dir = os.path.abspath(root)
        self.transform = transform
        self.normalize = normalize
        self.num_classes = int(num_classes)

        csv_path = os.path.join(self.root_dir, csv_name)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"metadata CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        quality_path = os.path.join(self.root_dir, "per_pair_predictions.csv")
        quality_df = pd.read_csv(quality_path)
        columns_to_keep = ['orig_path', 'syn_path', 'both_diff_detected']
        quality_df = quality_df[columns_to_keep]

        required = {
            "coco_idx",
            "orig_path",
            "orig_labels",
            "syn_path",
            "syn_labels",
            "removed_cls_idx",
            "removed_class",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in metadata.csv: {sorted(missing)}")

        # convert paths to root-relative and store as final on-disk absolute path
        df["orig_rel"] = df["orig_path"].apply(
            lambda p: _make_relative_to_root(p, self.root_dir)
        )
        df["syn_rel"] = df["syn_path"].apply(
            lambda p: _make_relative_to_root(p, self.root_dir)
        )
        # make quality df also relative
        quality_df["orig_rel"] = quality_df["orig_path"].apply(
            lambda p: _make_relative_to_root(p, self.root_dir)
        )
        quality_df["syn_rel"] = quality_df["syn_path"].apply(
            lambda p: _make_relative_to_root(p, self.root_dir)
        )

        df = df.merge(quality_df, on=['orig_rel', 'syn_rel'], how='left')
        # len before
        #remove rows where both_diff_detected is False (i.e. no detectable difference between orig and syn)
        before_count = len(df)
        df = df[df['both_diff_detected'] == True].reset_index(drop=True)
        after_count = len(df)
        print(f"Filtered to {after_count} rows where both_diff_detected is True (removed {before_count - after_count} rows)")


        df["orig_abs"] = df["orig_rel"].apply(lambda p: os.path.join(self.root_dir, p))
        df["syn_abs"] = df["syn_rel"].apply(lambda p: os.path.join(self.root_dir, p))

        # parse label lists
        df["orig_labels_list"] = df["orig_labels"].apply(_to_list)
        df["syn_labels_list"] = df["syn_labels"].apply(_to_list)

        # optional: validate that both images are readable
        if validate_images:
            keep_mask = []
            removed_count = 0
            for _, row in df.iterrows():
                orig_img = cv2.imread(row["orig_abs"])
                syn_img = cv2.imread(row["syn_abs"])
                ok = (orig_img is not None) and (syn_img is not None)
                keep_mask.append(ok)
                if not ok:
                    removed_count += 1
            df = df.loc[keep_mask].reset_index(drop=True)
            print(f"Removed {removed_count} rows with unreadable files.")
            print(f"Remaining valid pairs: {len(df)}")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        orig_path = row["orig_abs"]
        syn_path = row["syn_abs"]

        image = cv2.imread(orig_path)
        syn_image = cv2.imread(syn_path)
        if image is None or syn_image is None:
            raise FileNotFoundError(
                f"Unreadable image(s) at idx={idx}: {orig_path}, {syn_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        syn_image = cv2.cvtColor(syn_image, cv2.COLOR_BGR2RGB)

        # multi-label vectors
        mlc_vector_orig = np.zeros(self.num_classes, dtype=np.float32)
        mlc_vector_syn = np.zeros(self.num_classes, dtype=np.float32)

        for c in row["orig_labels_list"]:
            if 0 <= int(c) < self.num_classes:
                mlc_vector_orig[int(c)] = 1.0
        for c in row["syn_labels_list"]:
            if 0 <= int(c) < self.num_classes:
                mlc_vector_syn[int(c)] = 1.0

        removed_attr_name = str(row["removed_class"])

        if self.transform is not None:
            image = torchvision.transforms.ToPILImage()(image)
            image = self.transform(image)

            syn_image = torchvision.transforms.ToPILImage()(syn_image)
            syn_image = self.transform(syn_image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            syn_image = torch.from_numpy(syn_image).permute(2, 0, 1).float()

            if self.normalize:
                image = image / 255.0
                syn_image = syn_image / 255.0

        coco_idx = int(row["coco_idx"])
        return (
            image,
            mlc_vector_orig,
            syn_image,
            mlc_vector_syn,
            removed_attr_name,
            coco_idx,
        )

    def get_class_name(self, class_idx):
        return COCO_IDX2NAME.get(class_idx, f"class_{class_idx}")

    def image_paths_for_index(self, idx):
        # normalize idx to list of integers
        if isinstance(idx, torch.Tensor):
            idx_list = idx.flatten().tolist()
        elif isinstance(idx, (list, tuple)):
            idx_list = list(idx)
        else:
            idx_list = [int(idx)]

        rows = self.df.iloc[idx_list]

        return [(row["orig_abs"], row["syn_abs"]) for _, row in rows.iterrows()]


def main():
    dataset = COCOSynDataset(
        root_dir=SYN_COCO_PATH,
        transform=None,
        normalize=True,
    )
    for i in range(5):
        sample = dataset[i]
        # plot
        image, mlc_vector_orig, syn_image, mlc_vector_syn, removed_attr_name, idx = (
            sample
        )
        mlc_orig_names = [
            dataset.get_class_name(i) for i in np.where(mlc_vector_orig >= 0.5)[0]
        ]
        mlc_syn_names = [
            dataset.get_class_name(i) for i in np.where(mlc_vector_syn >= 0.5)[0]
        ]

        print()
        # plot_coco(bboxes, category_ids, idx, img, mlc_vector)
        plot_orig_vs_syn(
            image=image,
            syn_image=syn_image,
            mlc_vector_orig=mlc_vector_orig,
            mlc_vector_syn=mlc_vector_syn,
            removed_attr_name=removed_attr_name,
            idx=idx,
            class_names=COCO_CLASSES,  # or provide list[str] matching your vector length
            threshold=0.5,
        )


def plot_coco(bboxes, category_ids, idx, img, mlc_vector):
    import matplotlib.pyplot as plt

    img_np = img.permute(1, 2, 0).numpy()
    plt.imshow(img_np)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
        )
        plt.gca().add_patch(rect)
    plt.show()
    print("Sampled image index:", idx)
    print("Number of objects:", bboxes.shape[0])
    print("Category IDs:", category_ids)
    print("MLC vector:", mlc_vector)


def decode_mlc_vector(mlc_vector, class_names=None, threshold=0.5):
    """
    Convert a multi-label one-hot / multi-hot vector into a readable label list.

    mlc_vector: array-like, shape (C,) or (C,1)
    class_names: list[str] of length C (optional)
    threshold: float, for non-binary vectors

    Returns: list[str] or list[int]
    """
    v = np.asarray(mlc_vector).reshape(-1)
    active = np.flatnonzero(v >= threshold).tolist()

    if class_names is None:
        return active

    return [class_names[i] for i in active]


def _to_hwc_uint8(img):
    """
    Robust conversion for visualization.
    Accepts:
      - torch tensor CHW or HWC
      - numpy array CHW or HWC
      - float [0,1] or uint8 [0,255]
    """
    try:
        import torch

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
    except Exception:
        pass

    x = np.asarray(img)

    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC

    if x.ndim == 2:
        x = x[..., None]

    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        if x.max() <= 1.0 + 1e-6:
            x = (x * 255.0).clip(0, 255)
        x = x.astype(np.uint8)

    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    return x


def plot_orig_vs_syn(
    image,
    syn_image,
    mlc_vector_orig,
    mlc_vector_syn,
    removed_attr_name=None,
    idx=None,
    class_names=None,
    threshold=0.5,
    max_labels_per_subtitle=18,
    figsize=(11, 5),
):
    """
    Plots original and synthetic images side by side.
    Subtitle shows decoded multi-label vectors (not one-hot).

    max_labels_per_subtitle limits clutter; remaining labels are truncated.
    """
    import matplotlib.pyplot as plt

    img0 = _to_hwc_uint8(image)
    img1 = _to_hwc_uint8(syn_image)

    labels0 = decode_mlc_vector(
        mlc_vector_orig, class_names=class_names, threshold=threshold
    )
    labels1 = decode_mlc_vector(
        mlc_vector_syn, class_names=class_names, threshold=threshold
    )

    def fmt_labels(labels):
        if len(labels) == 0:
            return "labels: (none)"
        if len(labels) > max_labels_per_subtitle:
            shown = labels[:max_labels_per_subtitle]
            return f"labels ({len(labels)}): " + ", ".join(map(str, shown)) + ", ..."
        return f"labels ({len(labels)}): " + ", ".join(map(str, labels))

    title_parts = []
    if idx is not None:
        title_parts.append(f"idx={idx}")
    if removed_attr_name is not None:
        title_parts.append(f"removed={removed_attr_name}")
    title = " | ".join(title_parts) if title_parts else None

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(img0)
    axes[0].axis("off")
    axes[0].set_title("orig")
    axes[0].text(
        0.5,
        -0.08,
        fmt_labels(labels0),
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )

    axes[1].imshow(img1)
    axes[1].axis("off")
    axes[1].set_title("syn")
    axes[1].text(
        0.5,
        -0.08,
        fmt_labels(labels1),
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )

    if title is not None:
        fig.suptitle(title, fontsize=11)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

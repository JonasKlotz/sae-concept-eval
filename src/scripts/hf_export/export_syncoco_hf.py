"""Convert the on-disk synCOCO dataset into the HuggingFace-ready format.

Output layout (mirrors the synCUB export)::

    <out_root>/
    ├── images/
    │   ├── 000000_orig.jpg   # image containing `removed_class`
    │   ├── 000000_syn.jpg    # complement with that class removed
    │   └── ...
    └── metadata.csv          # one row per pair

`metadata.csv` columns: pair_id, orig_image, syn_image, coco_idx,
removed_cls_idx, removed_class, orig_labels, syn_labels.

The script drives off `COCOSynDataset`, which already filters to pairs where the
edit was detected (`both_diff_detected == True`), so **only selected pairs are
written**. Images are copied byte-for-byte (no re-encoding).

Usage::

    python src/scripts/hf_export/export_syncoco_hf.py \
        --src-root /scratch/htc/jklotz/data/syn_coco_dataset \
        --out-root /scratch/htc/jklotz/data/hf_export/syncoco
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import rootutils
from tqdm import tqdm

_root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
# The package uses top-level imports (e.g. `from datamodule...`), so `src` must be
# on the path. Derive it from the repo root to stay portable.
sys.path.insert(0, str(Path(_root) / "src"))

from datamodule.coco_dataset import COCOSynDataset, SYN_COCO_PATH


def build(src_root: str, out_root: str, overwrite: bool = False, limit: int | None = None) -> None:
    out_dir = Path(out_root)
    images_dir = out_dir / "images"
    if images_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{images_dir} already exists. Pass --overwrite to replace it."
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    # COCOSynDataset applies the `both_diff_detected == True` quality filter on load.
    ds = COCOSynDataset(root=src_root, transform=None)

    n_pairs = len(ds) if limit is None else min(limit, len(ds))
    records: list[dict] = []

    for pair_idx in tqdm(range(n_pairs), desc="Exporting synCOCO pairs"):
        row = ds.df.iloc[pair_idx]
        orig_src = Path(row["orig_abs"])
        syn_src = Path(row["syn_abs"])

        orig_name = f"{pair_idx:06d}_orig{orig_src.suffix}"
        syn_name = f"{pair_idx:06d}_syn{syn_src.suffix}"
        shutil.copy2(orig_src, images_dir / orig_name)
        shutil.copy2(syn_src, images_dir / syn_name)

        records.append(
            {
                "pair_id": pair_idx,
                "orig_image": f"images/{orig_name}",
                "syn_image": f"images/{syn_name}",
                "coco_idx": int(row["coco_idx"]),
                "removed_cls_idx": int(row["removed_cls_idx"]),
                "removed_class": str(row["removed_class"]),
                "orig_labels": list(row["orig_labels_list"]),
                "syn_labels": list(row["syn_labels_list"]),
            }
        )

    metadata = pd.DataFrame.from_records(records)
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    print(
        f"\nWrote {len(metadata)} pairs ({2 * len(metadata)} images) to {out_dir}"
        f"\n  metadata: {out_dir / 'metadata.csv'}"
        f"\n  images:   {images_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default=SYN_COCO_PATH, help="On-disk synCOCO root.")
    parser.add_argument("--out-root", required=True, help="Destination folder for the HF export.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing images/ dir.")
    parser.add_argument("--limit", type=int, default=None, help="Export only the first N pairs (debug).")
    args = parser.parse_args()
    build(args.src_root, args.out_root, overwrite=args.overwrite, limit=args.limit)


if __name__ == "__main__":
    main()

"""Convert the on-disk synCUB dataset into the HuggingFace-ready format.

Output layout (one self-contained folder)::

    <out_root>/
    ├── images/
    │   ├── 000000_orig.jpg   # image where `old_attr` is present
    │   ├── 000000_syn.jpg    # complement where `new_attr` is present
    │   └── ...
    └── metadata.csv          # one row per pair

`metadata.csv` columns: pair_id, orig_image, syn_image, class_id, class_name,
old_attr, new_attr, old_attr_idx, new_attr_idx, orig_attr_idx, syn_attr_idx.

The script drives off `CUBSyntheticDataset`, which already applies the quality
filter (`ROUND ERROR <= 1`) and the file-existence filter, so **only selected
pairs are written**. Images are copied byte-for-byte (no re-encoding).

Usage::

    python src/scripts/hf_export/export_syncub_hf.py \
        --src-root /scratch/htc/jklotz/data/syn_cub_dataset \
        --out-root /scratch/htc/jklotz/data/hf_export/syncub
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import rootutils
import torch
from tqdm import tqdm

_root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
# The package uses top-level imports (e.g. `from datamodule...`), so `src` must be
# on the path. Derive it from the repo root to stay portable.
sys.path.insert(0, str(Path(_root) / "src"))

from datamodule.CUB_syn_dataset import CUBSyntheticDataset, SYN_CUB_ROOT


def _active_indices(attr_vec: torch.Tensor) -> list[int]:
    """Return the 0-based indices of present attributes in a 312-d 0/1 vector."""
    return torch.nonzero(attr_vec >= 0.5, as_tuple=True)[0].tolist()


def build(src_root: str, out_root: str, overwrite: bool = False, limit: int | None = None) -> None:
    out_dir = Path(out_root)
    images_dir = out_dir / "images"
    if images_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{images_dir} already exists. Pass --overwrite to replace it."
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    # transform=None -> __getitem__ returns PIL images (which we ignore); we copy
    # the raw source files instead to preserve the original bytes.
    ds = CUBSyntheticDataset(root=src_root, transform=None)
    syn_images_root = Path(ds.root) / "synthetic_images"

    # class_id -> class_name lookup (class_name already stripped of the "NNN." prefix)
    class_name_map = ds.class_names.set_index("class_id")["class_name"].to_dict()

    # ship the attribute vocabulary so the dataset is self-contained: "<1-based id> <name>"
    with open(out_dir / "attributes.txt", "w") as f:
        for attr_id in sorted(ds.attr_map):
            f.write(f"{attr_id} {ds.attr_map[attr_id]}\n")

    n_pairs = len(ds) if limit is None else min(limit, len(ds))
    records: list[dict] = []
    skipped = 0

    for pair_idx in tqdm(range(n_pairs), desc="Exporting synCUB pairs"):
        raw_idx = pair_idx * 2  # even row = original, odd row = complement
        orig_row = ds.data.iloc[raw_idx]
        syn_row = ds.data.iloc[raw_idx + 1]

        try:
            (
                _img,
                _label,
                attrs,
                _img_c,
                _label_c,
                attrs_c,
                old_attr_name,
                new_attr_name,
                _idx,
            ) = ds[pair_idx]
        except AssertionError as exc:
            # __getitem__ asserts exactly one attribute differs; skip malformed pairs.
            skipped += 1
            tqdm.write(f"[skip] pair {pair_idx}: {exc}")
            continue

        orig_src = syn_images_root / orig_row["filepath"]
        syn_src = syn_images_root / syn_row["filepath"]

        orig_name = f"{pair_idx:06d}_orig{orig_src.suffix}"
        syn_name = f"{pair_idx:06d}_syn{syn_src.suffix}"
        shutil.copy2(orig_src, images_dir / orig_name)
        shutil.copy2(syn_src, images_dir / syn_name)

        class_id = int(orig_row["class_id"])
        records.append(
            {
                "pair_id": pair_idx,
                "orig_image": f"images/{orig_name}",
                "syn_image": f"images/{syn_name}",
                "class_id": class_id,
                "class_name": class_name_map.get(class_id, ""),
                "old_attr": old_attr_name,
                "new_attr": new_attr_name,
                "old_attr_idx": ds.reverse_attr_map[old_attr_name] - 1,
                "new_attr_idx": ds.reverse_attr_map[new_attr_name] - 1,
                "orig_attr_idx": _active_indices(attrs),
                "syn_attr_idx": _active_indices(attrs_c),
            }
        )

    metadata = pd.DataFrame.from_records(records)
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    print(
        f"\nWrote {len(metadata)} pairs ({2 * len(metadata)} images) to {out_dir}"
        f"\n  metadata: {out_dir / 'metadata.csv'}"
        f"\n  images:   {images_dir}"
        + (f"\n  skipped:  {skipped} malformed pairs" if skipped else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default=SYN_CUB_ROOT, help="On-disk synCUB root.")
    parser.add_argument("--out-root", required=True, help="Destination folder for the HF export.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing images/ dir.")
    parser.add_argument("--limit", type=int, default=None, help="Export only the first N pairs (debug).")
    args = parser.parse_args()
    build(args.src_root, args.out_root, overwrite=args.overwrite, limit=args.limit)


if __name__ == "__main__":
    main()

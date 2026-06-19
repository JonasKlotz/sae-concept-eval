"""Integrity + statistics report for the HF-format synthetic exports.

Checks structural invariants (exactly-one-attribute edit per pair, valid index
ranges, every referenced image present and no orphans, no duplicate pairs) and
prints distribution statistics. Run after exporting to "really double check".

Usage::

    python src/scripts/hf_export/inspect_hf_export.py \
        --cub-hf /scratch/htc/jklotz/data/hf_export/syncub \
        --coco-hf /scratch/htc/jklotz/data/hf_export/syncoco
"""

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import rootutils

_root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
sys.path.insert(0, str(Path(_root) / "src"))

NUM_CUB_ATTRS = 312
NUM_CUB_CLASSES = 200
NUM_COCO_CLASSES = 80


def _parse(value) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in ast.literal_eval(str(value))]


def _check_images(root: Path, df: pd.DataFrame) -> list[str]:
    """Every referenced image exists and there are no orphan files."""
    errs = []
    referenced = set(df["orig_image"]) | set(df["syn_image"])
    for rel in sorted(referenced):
        if not (root / rel).is_file():
            errs.append(f"missing image file: {rel}")
    on_disk = {f"images/{p.name}" for p in (root / "images").glob("*") if p.is_file()}
    orphans = on_disk - referenced
    if orphans:
        errs.append(f"{len(orphans)} orphan image files not referenced by metadata")
    if 2 * len(df) != len(referenced):
        errs.append(
            f"image count mismatch: {len(referenced)} referenced vs {2 * len(df)} expected"
        )
    return errs


def inspect_cub(hf: str) -> bool:
    root = Path(hf)
    df = pd.read_csv(root / "metadata.csv")
    print(f"\n=== synCUB @ {hf} ===")
    print(f"pairs: {len(df)}   images: {2 * len(df)}")

    errs: list[str] = []

    # unique pair ids and filenames
    if df["pair_id"].duplicated().any():
        errs.append("duplicate pair_id values")
    if df["orig_image"].duplicated().any() or df["syn_image"].duplicated().any():
        errs.append("duplicate image filenames")

    bad_swap = 0
    bad_range = 0
    for _, row in df.iterrows():
        orig = set(_parse(row["orig_attr_idx"]))
        syn = set(_parse(row["syn_attr_idx"]))
        old_i, new_i = int(row["old_attr_idx"]), int(row["new_attr_idx"])
        # exactly the old/new attribute differ, in the right direction
        if (orig ^ syn) != {old_i, new_i} or old_i not in orig or new_i not in syn:
            bad_swap += 1
        if not (0 <= old_i < NUM_CUB_ATTRS and 0 <= new_i < NUM_CUB_ATTRS):
            bad_range += 1
        if not (1 <= int(row["class_id"]) <= NUM_CUB_CLASSES):
            bad_range += 1
    if bad_swap:
        errs.append(f"{bad_swap} pairs violate the one-attribute-swap invariant")
    if bad_range:
        errs.append(f"{bad_range} pairs have out-of-range class/attr indices")

    errs += _check_images(root, df)

    # statistics
    print(f"distinct bird classes: {df['class_id'].nunique()}/{NUM_CUB_CLASSES}")
    swapped = Counter(df["old_attr"]) + Counter(df["new_attr"])
    print(f"distinct attributes involved in swaps: {len(swapped)}")
    top = swapped.most_common(5)
    print("most frequent swapped attributes:", ", ".join(f"{k} ({v})" for k, v in top))

    if errs:
        print("INTEGRITY: FAIL")
        for e in errs:
            print(f"  - {e}")
    else:
        print("INTEGRITY: PASS (all invariants hold)")
    return not errs


def inspect_coco(hf: str) -> bool:
    from datamodule.coco_dataset import COCO_CLASSES

    root = Path(hf)
    df = pd.read_csv(root / "metadata.csv")
    print(f"\n=== synCOCO @ {hf} ===")
    print(f"pairs: {len(df)}   images: {2 * len(df)}")

    errs: list[str] = []
    if df["pair_id"].duplicated().any():
        errs.append("duplicate pair_id values")
    if df["orig_image"].duplicated().any() or df["syn_image"].duplicated().any():
        errs.append("duplicate image filenames")

    bad_removal = 0
    bad_range = 0
    bad_name = 0
    for _, row in df.iterrows():
        orig = set(_parse(row["orig_labels"]))
        syn = set(_parse(row["syn_labels"]))
        rem = int(row["removed_cls_idx"])
        # exactly the removed class differs: present in orig, absent in syn
        if (orig ^ syn) != {rem} or rem not in orig or rem in syn:
            bad_removal += 1
        if not all(0 <= c < NUM_COCO_CLASSES for c in orig | syn):
            bad_range += 1
        if COCO_CLASSES[rem] != str(row["removed_class"]):
            bad_name += 1
    if bad_removal:
        errs.append(f"{bad_removal} pairs violate the one-class-removal invariant")
    if bad_range:
        errs.append(f"{bad_range} pairs have out-of-range label indices")
    if bad_name:
        errs.append(f"{bad_name} pairs: removed_class name != COCO_CLASSES[removed_cls_idx]")

    errs += _check_images(root, df)

    print(f"distinct removed classes: {df['removed_cls_idx'].nunique()}/{NUM_COCO_CLASSES}")
    top = Counter(df["removed_class"]).most_common(5)
    print("most frequently removed classes:", ", ".join(f"{k} ({v})" for k, v in top))
    print(f"mean #labels per orig image: {df['orig_labels'].apply(lambda x: len(_parse(x))).mean():.2f}")

    if errs:
        print("INTEGRITY: FAIL")
        for e in errs:
            print(f"  - {e}")
    else:
        print("INTEGRITY: PASS (all invariants hold)")
    return not errs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cub-hf", default="/scratch/htc/jklotz/data/hf_export/syncub")
    p.add_argument("--coco-hf", default="/scratch/htc/jklotz/data/hf_export/syncoco")
    p.add_argument("--skip-cub", action="store_true")
    p.add_argument("--skip-coco", action="store_true")
    args = p.parse_args()

    ok = True
    if not args.skip_cub:
        ok &= inspect_cub(args.cub_hf)
    if not args.skip_coco:
        ok &= inspect_coco(args.coco_hf)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

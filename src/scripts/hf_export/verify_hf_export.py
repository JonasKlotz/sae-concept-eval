"""Verify the HF-format exports match the original on-disk datasets.

The HF exports are built from the original datasets' *selected* pairs in order,
so ``original[i]`` must correspond to ``hf[i]``. This script checks that the
loaders agree on labels, attributes, the changed-attribute names, and the image
pixels (CUB is pixel-exact; COCO is checked within a small decode tolerance).

Usage::

    python src/scripts/hf_export/verify_hf_export.py \
        --cub-src /scratch/htc/jklotz/data/syn_cub_dataset \
        --cub-hf  /scratch/htc/jklotz/data/hf_export/syncub \
        --coco-src /scratch/htc/jklotz/data/syn_coco_dataset \
        --coco-hf  /scratch/htc/jklotz/data/hf_export/syncoco \
        --n 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rootutils
import torch

_root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
sys.path.insert(0, str(Path(_root) / "src"))

from datamodule.CUB_syn_dataset import CUBSyntheticDataset
from datamodule.coco_dataset import COCOSynDataset
from datamodule.hf_syn_datasets import SynCUBHFDataset, SynCOCOHFDataset


def _sample_indices(n_total: int, n: int) -> list[int]:
    if n is None or n >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(0)
    return sorted(rng.choice(n_total, size=n, replace=False).tolist())


def verify_cub(src: str, hf: str, n: int) -> bool:
    old = CUBSyntheticDataset(root=src, transform=None)
    new = SynCUBHFDataset(root=hf, transform=None)
    print(f"\n[CUB] lengths: original={len(old)}  hf={len(new)}")
    assert len(old) == len(new), "length mismatch"
    assert new.attribute_names == [old.attr_map[i] for i in sorted(old.attr_map)], \
        "attribute vocabulary mismatch"

    ok = True
    for i in _sample_indices(len(old), n):
        o = old[i]
        h = new[i]
        o_img, o_lab, o_attrs, o_imgc, _, o_attrsc, o_old, o_new, _ = o
        h_img, h_lab, h_attrs, h_imgc, _, h_attrsc, h_old, h_new, _ = h

        checks = {
            "label": torch.equal(o_lab, h_lab),
            "attrs": torch.equal(o_attrs, h_attrs),
            "attrs_c": torch.equal(o_attrsc, h_attrsc),
            "old_attr": o_old == h_old,
            "new_attr": o_new == h_new,
            "img_pixels": np.array_equal(np.asarray(o_img), np.asarray(h_img)),
            "img_c_pixels": np.array_equal(np.asarray(o_imgc), np.asarray(h_imgc)),
        }
        if not all(checks.values()):
            ok = False
            failed = [k for k, v in checks.items() if not v]
            print(f"  [CUB] idx {i} MISMATCH: {failed}")
    print(f"[CUB] {'PASS' if ok else 'FAIL'}")
    return ok


def verify_coco(src: str, hf: str, n: int, tol: float = 2.0 / 255.0) -> bool:
    old = COCOSynDataset(root=src, transform=None)
    new = SynCOCOHFDataset(root=hf, transform=None)
    print(f"\n[COCO] lengths: original={len(old)}  hf={len(new)}")
    assert len(old) == len(new), "length mismatch"

    ok = True
    max_img_diff = 0.0
    for i in _sample_indices(len(old), n):
        o_img, o_mlc, o_syn, o_mlcs, o_rem, o_idx = old[i]
        h_img, h_mlc, h_syn, h_mlcs, h_rem, h_idx = new[i]

        d = max(
            float((o_img - h_img).abs().max()),
            float((o_syn - h_syn).abs().max()),
        )
        max_img_diff = max(max_img_diff, d)
        checks = {
            "mlc_orig": np.array_equal(o_mlc, h_mlc),
            "mlc_syn": np.array_equal(o_mlcs, h_mlcs),
            "removed_class": str(o_rem) == str(h_rem),
            "images_within_tol": d <= tol,
        }
        if not all(checks.values()):
            ok = False
            failed = [k for k, v in checks.items() if not v]
            print(f"  [COCO] idx {i} MISMATCH: {failed} (img diff={d:.4f})")
    print(f"[COCO] max image abs diff (cv2 vs PIL decode) = {max_img_diff:.5f}")
    print(f"[COCO] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cub-src", default="/scratch/htc/jklotz/data/syn_cub_dataset")
    p.add_argument("--cub-hf", default="/scratch/htc/jklotz/data/hf_export/syncub")
    p.add_argument("--coco-src", default="/scratch/htc/jklotz/data/syn_coco_dataset")
    p.add_argument("--coco-hf", default="/scratch/htc/jklotz/data/hf_export/syncoco")
    p.add_argument("--n", type=int, default=100, help="Number of pairs to spot-check per dataset.")
    p.add_argument("--skip-cub", action="store_true")
    p.add_argument("--skip-coco", action="store_true")
    args = p.parse_args()

    results = {}
    if not args.skip_cub:
        results["CUB"] = verify_cub(args.cub_src, args.cub_hf, args.n)
    if not args.skip_coco:
        results["COCO"] = verify_coco(args.coco_src, args.coco_hf, args.n)

    print("\n=== Summary ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

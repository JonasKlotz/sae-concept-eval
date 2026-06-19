from __future__ import annotations

import re
from typing import Optional


COLORS = {
    "metrics": {
        "f1": "#1f77b4",
        "jaccard": "#ff7f0e",
        "mi": "#2ca02c",
        "bmp": "#d62728",  # reserved for OUR method family
        "nnomp": "#2ca02c",  # green
        "probe": "#7f7f7f",

        # BMP variants -> different red tones (k-independent)
        "bmp_variants": {
            "f0.25": "#fca5a5",  # light red
            "f0.5": "#ef4444",   # medium red
            "f1.0": "#991b1b",   # dark red
        },
    },
    "sae": {
            "batchtopk": "#8ECae6",   # pastel blue
            "topk": "#FFD6A5",        # pastel orange
            "matryoshka": "#A8DADC",  # pastel teal
            "frozen": "#F1C0E8",      # pastel magenta
            "random": "#BDBDBD",      # soft grey instead of black
            "trained": "#FFB4A2",     # pastel vermillion
    },
}


def get_metric_color(metric: str) -> str:
    """
    Return a stable hex color for a metric, independent of k.

    Accepts:
      - raw keys: "bmp_f0.5_top3", "bmp_top3", "bmp_f1.0", "f1_top1", "mi_top10", ...
      - pretty labels: "FBMP F0.5 (k=3)", "FBMP (k=1)", "FBMP F1 (all k)", "F1 (k=5)", ...

    Rules:
      - BMP family uses red; F-beta BMP variants use distinct red tones.
      - Non-BMP metrics map to their family color.
      - k is ignored for color choice.
    """
    s = str(metric).strip()

    # -------------------------
    # 1) Try pretty-label forms
    # -------------------------
    sp = s.lower()

    # FBMP F<beta> (k=...) or FBMP F<beta> (all k) or similar
    m = re.search(r"\bfbmp\b", sp)
    if m:
        m_beta = re.search(r"\bf\s*([0-9]+(?:\.[0-9]+)?)\b", sp)
        if m_beta:
            beta = m_beta.group(1)
            variant_key = f"f{beta}"
            # normalize 1 -> 1.0, 0.5 -> 0.5, 0.25 -> 0.25
            if variant_key == "f1":
                variant_key = "f1.0"
            if variant_key in COLORS["metrics"]["bmp_variants"]:
                return COLORS["metrics"]["bmp_variants"][variant_key]
        return COLORS["metrics"]["bmp"]

    # Non-BMP pretty labels: "F1 (k=...)", "Jaccard (k=...)", "MI (k=...)"
    if re.search(r"\bf1\b", sp):
        return COLORS["metrics"]["f1"]
    if "jaccard" in sp:
        return COLORS["metrics"]["jaccard"]
    # match MI as a token, avoid accidental matches inside other words
    if re.search(r"\bmi\b", sp):
        return COLORS["metrics"]["mi"]

    # -------------------------
    # 2) Raw-key forms
    # -------------------------
    raw = sp.replace(" ", "")

    # nnomp_top3, nnomp
    if re.match(r"^nnomp(?:_top\d+)?$", raw):
        return COLORS["metrics"]["nnomp"]

    # bmp variants: bmp_f0.5_top3, bmp_f1.0, bmp_top3, bmp_jaccard_top2, bmp_mi_top10
    m = re.match(r"^bmp(?:_(f\d*\.?\d+|jaccard|mi))?(?:_top\d+)?$", raw)
    if m:
        variant = m.group(1)  # e.g. "f0.5" / "f1.0" / "jaccard" / "mi" / None
        if variant is None:
            return COLORS["metrics"]["bmp"]
        if variant.startswith("f"):
            if variant in COLORS["metrics"]["bmp_variants"]:
                return COLORS["metrics"]["bmp_variants"][variant]
            return COLORS["metrics"]["bmp"]
        # bmp_jaccard, bmp_mi: keep them within the BMP family (red) unless you explicitly want otherwise
        return COLORS["metrics"]["bmp"]

    # non-bmp raw keys: f1_topK, jaccard_topK, mi_topK
    m = re.match(r"^(f1|jaccard|mi)(?:_top\d+)?$", raw)
    if m:
        fam = m.group(1)
        return COLORS["metrics"][fam]

    # -------------------------
    # 3) Fallback
    # -------------------------
    return COLORS["metrics"]["probe"]

def pretty_sae_family(raw: str) -> str:
    """
    Map internal SAE family names to camera-ready titles.
    """
    key = raw.lower().strip()
    mapping = {
        "topk": "TopK",
        "batchtopk": "BatchTopK",
        "matryoshka": "Matryoshka",
        "random": "Random Activation",
        "frozen": "Untrained",
        "jumprelu": "JumpReLU",
    }
    # fallback: Title Case + "SAE"
    return mapping.get(key, f"{raw[:1].upper() + raw[1:]} SAE")


def pretty_metric_name(raw: str) -> str:
    """
    Converts internal metric identifiers to clean, camera-ready labels.

    Examples:
        bmp_top1              -> FBMP (k=1)
        bmp_f0.5_top3         -> FBMP F0.5 (k=3)
        bmp_f1.0_top5         -> FBMP F1 (k=5)
        bmp_jaccard_top2      -> FBMP Jaccard (k=2)
        bmp_mi_top10          -> FBMP MI (k=10)
        f1_top3               -> F1 (k=3)
        jaccard_top5          -> Jaccard (k=5)
        mi_top2               -> MI (k=2)
    """

    if raw == "mean_fms":
        return "FMS"
    elif raw == "monosemanticity_mean":
        return "MS"
    elif raw == "CKNNA":
        return "CKNNA"
    elif raw == "bmp_f1.0":
        return "FBMP F1 (all k)"

    s = str(raw).lower()

    # --------------------------
    # 1. BMP variants
    # --------------------------
    bmp_pattern = re.match(r"bmp(?:_(f\d*\.?\d+|jaccard|mi))?_top(\d+)", s)
    if bmp_pattern:
        variant, k = bmp_pattern.groups()
        k = int(k)

        if variant is None:
            return f"FBMP (k={k})"

        # F-beta inside BMP
        if variant.startswith("f"):
            beta = float(variant[1:])
            if beta.is_integer():
                beta = int(beta)
            return f"FBMP F{beta} (k={k})"

        if variant == "jaccard":
            return f"FBMP Jaccard (k={k})"

        if variant == "mi":
            return f"FBMP MI (k={k})"

    # --------------------------
    # 2. Non-BMP metrics
    # --------------------------
    simple_pattern = re.match(r"(f1|jaccard|mi)_top(\d+)", s)
    if simple_pattern:
        metric, k = simple_pattern.groups()
        k = int(k)

        if metric == "f1":
            return f"F1 (k={k})"
        if metric == "jaccard":
            return f"Jaccard (k={k})"
        if metric == "mi":
            return f"MI (k={k})"

    # --------------------------
    # 3. NN-OMP
    # --------------------------
    nnomp_pattern = re.match(r"nnomp_top(\d+)", s)
    if nnomp_pattern:
        k = int(nnomp_pattern.group(1))
        return f"NN-OMP (k={k})"

    # fallback
    return raw


def _parse_metric_triplet(raw: str) -> tuple[Optional[str], Optional[str], int]:
    """
    Parse metric identifiers into (family, variant, k).

    family:
      - "bmp", "f1", "jaccard", "mi"

    variant (only meaningful for bmp right now):
      - None (plain bmp)
      - "f0.25", "f0.5", "f1.0" (as string without "bmp_" prefix)
      - "jaccard"
      - "mi"

    k:
      - integer from "_top<k>", else 0

    Examples:
      bmp_top3          -> ("bmp", None, 3)
      bmp_f0.5_top10    -> ("bmp", "f0.5", 10)
      bmp_jaccard_top2  -> ("bmp", "jaccard", 2)
      f1_top5           -> ("f1", None, 5)
      mi_top1           -> ("mi", None, 1)
    """
    s = str(raw).lower()

    # bmp variants (new)
    m = re.match(r"^(bmp)(?:_(f\d*\.?\d+|jaccard|mi))?(?:_top(\d+))?$", s)
    if m:
        fam, variant, k = m.groups()
        return fam, variant, int(k) if k is not None else 0

    # non-bmp metrics (unchanged, but stricter)
    m = re.match(r"^(f1|jaccard|mi)(?:_top(\d+))?$", s)
    if m:
        fam, k = m.groups()
        return fam, None, int(k) if k is not None else 0

    return None, None, 0


def pretty_syn_dataset(name: str) -> str:
    n = str(name).lower()
    if "cub" in n:
        return "synCUB"
    if "coco" in n:
        return "synCOCO"
    return str(name)

def pretty_dataset(name: str) -> str:
    n = str(name).lower()
    if "cub" in n:
        return "CUB"
    if "coco" in n:
        return "COCO"
    return str(name)


def compute_shared_ylim(
    y_values,
    clip_unit_interval: bool = False,
):
    """Return (ymin, ymax, step) from a flat collection of y values, or None if empty."""
    import numpy as np
    arr = np.array(y_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    y_min = float(arr.min())
    y_max = float(arr.max())
    pad = 0.02 * max(1e-6, y_max - y_min)
    y_min -= pad
    y_max += pad
    if clip_unit_interval:
        y_min = max(0.0, y_min)
        y_max = min(1.0, y_max)
    span = max(1e-6, y_max - y_min)
    if span <= 0.04:
        step = 0.01
    elif span <= 0.12:
        step = 0.02
    elif span <= 0.30:
        step = 0.05
    elif span <= 0.60:
        step = 0.10
    else:
        step = 0.20
    ymin = float(np.floor(y_min / step) * step)
    ymax = float(np.ceil(y_max / step) * step)
    if clip_unit_interval:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)
    if ymax - ymin < step:
        ymax = ymin + step
    return ymin, ymax, step


def save_legend_strip(
    handles,
    labels,
    save_path,
    ncol: int | None = None,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 9,
    show: bool = True,
):
    """Save a standalone horizontal legend strip PNG (no axes, no data)."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    n = len(handles)
    if ncol is None:
        ncol = n
    if figsize is None:
        figsize = (max(2.2 * ncol, 4.0), 0.45)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.8,
        handleheight=1.0,
        columnspacing=1.2,
    )
    fig.tight_layout(pad=0.1)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved legend strip to {save_path}")

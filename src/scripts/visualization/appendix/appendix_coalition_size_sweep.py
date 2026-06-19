"""
Appendix: MATCHScore and TAPAScore as a function of coalition size k.

For each SAE family and a fixed set of dict sizes we plot how both metrics
evolve as the matching coalition grows from k=1 to k=10.

Outputs (under outputs/figures/appendix/fig_coalition_size_sweep/):
  matching/{dataset}/{model}/{sae}_matching_vs_k.png
  tapas/{syn_dataset}/{model}/{sae}_tapas_vs_k.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.visualization.plot_utils.plot_data_utils import (
    load_nested_pt_tree,
    keep_ground_truth_per_method,
    calc_matching_score_over_all_attrs_df,
    load_perturbation_dataframe,
)
from scripts.visualization.plot_utils.plot_utils import (
    pretty_sae_family,
    pretty_metric_name,
    _parse_metric_triplet,
    save_legend_strip,
)

SHOW_PLOTS = True  # set to False when running in batch/headless mode

METRICS_ROOT = "/home/jokl/PycharmProjects/rs_concepts/outputs/metrics"
OUT_DIR = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures/appendix/fig_coalition_size_sweep")

# -----------------------------------------------------------------------
# Shared style tables
# -----------------------------------------------------------------------

_COLORS = {
    "f1":      "#1f77b4",
    "mi":      "#2ca02c",
    "bmp_f1.0": "#991b1b",
    "bmp_f0.5": "#ef4444",
    "bmp_f0.25": "#fca5a5",
    "bmp_mi":  "#b22222",
}
_LABEL = {
    s: pretty_metric_name(f"{s}_top1").replace(" (k=1)", "")
    for s in _COLORS
}
# Marker and linestyle encode dict size (same channel → easier to read in legend)
_DS_MARKERS   = ["o", "s", "^", "D"]
_LINESTYLES   = ["-", "--", ":", "-."]


def _to_series(fam: str | None, var: str | None) -> str | None:
    if fam is None:
        return None
    fam_l = str(fam).lower()
    var_l = str(var).lower() if var is not None else None
    if fam_l == "f1":
        return "f1"
    if fam_l == "mi":
        return "mi"
    if fam_l == "bmp":
        if var_l in ("f1.0", "f1"):
            return "bmp_f1.0"
        if var_l == "f0.5":
            return "bmp_f0.5"
        if var_l == "f0.25":
            return "bmp_f0.25"
        if var_l == "mi":
            return "bmp_mi"
    return None


def _dedup_handles(ax):
    """Return deduplicated (handles, labels) without adding a legend to ax."""
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    return h2, l2


# -----------------------------------------------------------------------
# Matching data prep
# -----------------------------------------------------------------------

def _prepare_matching_df(
    loaded,
    dataset_name: str,
    model_name: str,
    seed: str = "42",
    sae_include: Sequence[str] = ("topk",),
    dict_sizes: Sequence[int] = (256, 2048),
    k_list: Sequence[int] = (1, 3, 5, 10),
    series: Sequence[str] = ("f1", "bmp_f1.0"),
) -> pd.DataFrame:
    data = loaded[dataset_name][model_name][seed]
    data = keep_ground_truth_per_method(data)

    df = calc_matching_score_over_all_attrs_df(data, list(k_list)).copy()
    if sae_include is not None:
        df = df[df["sae"].isin(list(sae_include) + ["frozen"])].copy()
    if dict_sizes is not None:
        df = df[df["dict_size"].isin(list(dict_sizes))].copy()

    parsed = df["metric_name"].apply(_parse_metric_triplet)
    df["family"]  = parsed.apply(lambda x: x[0])
    df["variant"] = parsed.apply(lambda x: x[1])
    df["k"]       = parsed.apply(lambda x: x[2])

    if k_list is not None:
        df = df[df["k"].isin(list(k_list))].copy()

    df["series"] = [_to_series(f, v) for f, v in zip(df["family"], df["variant"])]
    df = df[df["series"].notna() & df["series"].isin(list(series))].copy()

    if df.empty:
        raise ValueError(f"No matching rows for {dataset_name}/{model_name}.")

    # subtract frozen baseline per (dict_size, metric_name)
    frozen = df[df["sae"] == "frozen"][["dict_size", "metric_name", "matching_score"]].copy()
    df = df[df["sae"] != "frozen"].copy()
    df = df.merge(frozen, on=["dict_size", "metric_name"], how="left", suffixes=("", "_frozen"))
    df["matching_score"] = df["matching_score"] - df["matching_score_frozen"]
    # Drop rows where the frozen baseline had no counterpart (e.g. duplicate
    # bmp_rec_F1_matrix_f1.pt alongside f1.0.pt creates NaN after subtraction).
    df = df.dropna(subset=["matching_score"])
    return df


# -----------------------------------------------------------------------
# Matching plot
# -----------------------------------------------------------------------

def plot_matching_over_k(
    df: pd.DataFrame,
    title: str,
    save_path: Path,
    sae: str,
    dict_sizes: Sequence[int] = (256, 2048),
    series: Sequence[str] = ("f1", "bmp_f1.0"),
    k_list: Sequence[int] = (1, 3, 5, 10),
    figsize: tuple[float, float] = (4, 3),
):
    dff = df[df["sae"] == sae].copy()
    if dict_sizes is not None:
        dff = dff[dff["dict_size"].isin(list(dict_sizes))].copy()
    if k_list is not None:
        dff = dff[dff["k"].isin(list(k_list))].copy()
    dff = dff[dff["series"].isin(list(series))].copy()
    if dff.empty:
        print(f"  no rows for {sae}, skipping matching plot")
        return

    dict_sizes_present = [int(x) for x in sorted(dff["dict_size"].unique())]
    ds2ls     = {ds: _LINESTYLES[i % len(_LINESTYLES)]   for i, ds in enumerate(dict_sizes_present)}
    ds2marker = {ds: _DS_MARKERS[i % len(_DS_MARKERS)]   for i, ds in enumerate(dict_sizes_present)}

    ks = sorted(dff["k"].unique())
    x_pos = {int(k): i for i, k in enumerate(ks)}

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    for ds in dict_sizes_present:
        dsd = dff[dff["dict_size"] == ds]
        for s in [s for s in series if s in dff["series"].unique()]:
            rows = dsd[dsd["series"] == s].sort_values("k")
            if rows.empty:
                continue
            xs = np.array([x_pos[int(k)] for k in rows["k"].to_numpy()], dtype=float)
            ys = rows["matching_score"].to_numpy(dtype=float)
            ax.plot(xs, ys, marker=ds2marker[ds], markersize=5.0,
                    linewidth=1.6, linestyle=ds2ls[ds],
                    color=_COLORS.get(s, "black"),
                    label=f"{_LABEL.get(s, s)} | d={ds}")

    ax.axhline(0.0, alpha=0.45, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Coalition size k")
    ax.set_ylabel(r"$F_1\ \Delta\ \mathrm{MATCHScore}$")
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.grid(True, axis="y", alpha=0.25)
    h, l = _dedup_handles(ax)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return h, l


# -----------------------------------------------------------------------
# TAPAS plot
# -----------------------------------------------------------------------

def plot_tapas_over_k(
    df: pd.DataFrame,
    title: str,
    save_path: Path,
    sae: str,
    dict_sizes: Sequence[int] = (256, 2048),
    series: Sequence[str] = ("f1", "bmp_f1.0"),
    k_list: Sequence[int] = (1, 3, 5, 10),
    figsize: tuple[float, float] = (4, 3),
):
    dff = df[df["sae"] == sae].copy()
    if dict_sizes is not None:
        dff = dff[dff["dict_size"].isin(list(dict_sizes))].copy()

    parsed = dff["metric_name"].apply(_parse_metric_triplet)
    dff = dff.copy()
    dff["family"]  = parsed.apply(lambda x: x[0])
    dff["variant"] = parsed.apply(lambda x: x[1])
    dff["k"]       = parsed.apply(lambda x: x[2])

    if k_list is not None:
        dff = dff[dff["k"].isin(list(k_list))].copy()

    dff["series"] = [_to_series(f, v) for f, v in zip(dff["family"], dff["variant"])]
    dff = dff[dff["series"].notna() & dff["series"].isin(list(series))].copy()

    if dff.empty:
        print(f"  no rows for {sae}, skipping TAPAS plot")
        return

    dict_sizes_present = [int(x) for x in sorted(dff["dict_size"].unique())]
    ds2ls     = {ds: _LINESTYLES[i % len(_LINESTYLES)]   for i, ds in enumerate(dict_sizes_present)}
    ds2marker = {ds: _DS_MARKERS[i % len(_DS_MARKERS)]   for i, ds in enumerate(dict_sizes_present)}

    ks = sorted(dff["k"].unique())
    x_pos = {int(k): i for i, k in enumerate(ks)}

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    for ds in dict_sizes_present:
        dsd = dff[dff["dict_size"] == ds]
        for s in [s for s in series if s in dff["series"].unique()]:
            rows = dsd[dsd["series"] == s].sort_values("k")
            if rows.empty:
                continue
            xs = np.array([x_pos[int(k)] for k in rows["k"].to_numpy()], dtype=float)
            ys = rows["score_mean"].to_numpy(dtype=float)
            ax.plot(xs, ys, marker=ds2marker[ds], markersize=5.0,
                    linewidth=1.6, linestyle=ds2ls[ds],
                    color=_COLORS.get(s, "black"),
                    label=f"{_LABEL.get(s, s)} | d={ds}")

    ax.axhline(0.0, alpha=0.45, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Coalition size k")
    ax.set_ylabel("TAPAScore")
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.grid(True, axis="y", alpha=0.25)
    h, l = _dedup_handles(ax)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return h, l


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def fig_coalition_size_sweep():
    loaded = load_nested_pt_tree(METRICS_ROOT)

    dataset_pairs = [("CUB_attrs", "syn_cub_attrs"), ("COCO", "syn_coco")]
    model_names   = ["CLIP-ViT-L-14"]
    saes          = ("topk", "matryoshka", "batchtopk", "jumprelu")
    dict_sizes    = (256, 2048)
    k_list        = (1, 3, 5, 10)
    series        = ("f1", "bmp_f1.0")

    for dataset_name, syn_dataset in dataset_pairs:
        print(f"\n=== {dataset_name} ===")
        for model_name in model_names:
            # --- matching ---
            try:
                match_df = _prepare_matching_df(
                    loaded, dataset_name, model_name,
                    sae_include=saes, dict_sizes=dict_sizes,
                    k_list=k_list, series=series,
                )
            except (KeyError, ValueError) as e:
                print(f"  matching data unavailable: {e}")
                match_df = None

            # --- TAPAS ---
            syn_dir  = Path(METRICS_ROOT) / syn_dataset
            also_dir = Path(METRICS_ROOT) / dataset_name
            try:
                tapas_df = load_perturbation_dataframe(syn_dir, also_search_root=also_dir)
                tapas_df = tapas_df[tapas_df["model"] == model_name].copy()
            except Exception as e:
                print(f"  TAPAS data unavailable: {e}")
                tapas_df = None

            match_legend_h, match_legend_l = None, None
            tapas_legend_h, tapas_legend_l = None, None

            for sae in saes:
                sae_label = pretty_sae_family(sae)
                base = OUT_DIR / model_name / dataset_name / sae

                if match_df is not None:
                    result = plot_matching_over_k(
                        df=match_df, title=sae_label,
                        save_path=base / "matching_vs_k.png",
                        sae=sae, dict_sizes=dict_sizes,
                        series=series, k_list=k_list,
                    )
                    if result is not None and match_legend_h is None:
                        match_legend_h, match_legend_l = result

                if tapas_df is not None:
                    result = plot_tapas_over_k(
                        df=tapas_df, title=sae_label,
                        save_path=base / "tapas_vs_k.png",
                        sae=sae, dict_sizes=dict_sizes,
                        series=series, k_list=k_list,
                    )
                    if result is not None and tapas_legend_h is None:
                        tapas_legend_h, tapas_legend_l = result

            legend_dir = OUT_DIR / model_name / dataset_name
            if match_legend_h is not None:
                save_legend_strip(match_legend_h, match_legend_l,
                                  save_path=legend_dir / "matching_vs_k_legend.png",
                                  show=SHOW_PLOTS)
            if tapas_legend_h is not None:
                save_legend_strip(tapas_legend_h, tapas_legend_l,
                                  save_path=legend_dir / "tapas_vs_k_legend.png",
                                  show=SHOW_PLOTS)


if __name__ == "__main__":
    fig_coalition_size_sweep()

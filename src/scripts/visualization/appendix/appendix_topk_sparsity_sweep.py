"""
Appendix figure: SAE behaviour as a function of activation sparsity.

The k-sparsity sweep was run for TopK and JumpReLU SAEs at dict_size=1024
with top_k ∈ {8, 16, 32, 64, 128}. Folder naming convention:
    metrics/{dataset}/{model}/{seed}/{family}_1024_k{N}/...

Two line plots per (dataset, model, SAE family):
    1. ΔMATCHScore vs sparsity (matched-coalition F1, frozen-subtracted)
    2. TAPAScore   vs sparsity

NOTE: the k-sweep folders only carry F1/BMP — no NN-OMP — so we skip
nnomp_top* lines.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.visualization.fig_matching_over_dictsizes import _marker_for_metric
from scripts.visualization.plot_utils.plot_data_utils import (
    calc_matching_score_over_all_attrs_df,
    keep_ground_truth_per_method,
    load_nested_pt_tree,
    load_perturbation_dataframe,
    nonsyn_for_syn,
)
from scripts.visualization.plot_utils.plot_utils import (
    get_metric_color,
    pretty_metric_name,
    pretty_sae_family, pretty_dataset, pretty_syn_dataset,
    save_legend_strip,
)


SHOW_PLOTS = True  # set to False when running in batch/headless mode

TARGET_FAMILIES = ("topk", "jumprelu")
TARGET_DICT_SIZE = 1024


def _extract_topk(metric_name: str) -> int | None:
    m = re.search(r"_top(\d+)$", metric_name)
    return int(m.group(1)) if m else None


def _linestyle_for_topk(k: int | None) -> str:
    if k == 1:
        return "-"
    if k == 3:
        return "--"
    if k == 5:
        return ":"
    if k == 10:
        return "-."
    return "-"


def _plot_lines_over_sparsity(
    df: pd.DataFrame,
    *,
    sparsities: Sequence[int],
    metrics_sorted: Sequence[str],
    score_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
    figsize=(5,4),
):
    fig, ax = plt.subplots(figsize=figsize)

    pos_for_k = {int(k): i for i, k in enumerate(sparsities)}
    y_series: list[np.ndarray] = []

    for metric_name in metrics_sorted:
        dd = df[df["metric_name"] == metric_name].copy()
        if dd.empty:
            continue
        dd = dd.sort_values("top_k")
        x_k = dd["top_k"].to_numpy(dtype=int)
        x = np.array([pos_for_k[int(k)] for k in x_k], dtype=float)
        y = dd[score_col].to_numpy(dtype=float)

        k = _extract_topk(metric_name)
        ax.plot(
            x,
            y,
            color=get_metric_color(metric_name),
            linestyle=_linestyle_for_topk(k),
            marker=_marker_for_metric(metric_name),
            linewidth=1.8,
            markersize=6,
            label=pretty_metric_name(metric_name),
        )
        y_series.append(y)

    ax.set_xticks(np.arange(len(sparsities)))
    ax.set_xticklabels([str(int(k)) for k in sparsities])

    y_all = np.concatenate(y_series) if y_series else np.array([])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size > 0:
        y_min = float(y_all.min())
        y_max = float(y_all.max())
        pad = 0.02 * max(1e-6, (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title(title)
    ax.set_xlabel("Sparsity (K)")
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    handles_rev, labels_rev = handles[::-1], labels[::-1]
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"Saved figure to {save_path}")
    return handles_rev, labels_rev


def _matching_for_sparsity_sweep(
    tree_for_seed: dict,
    *,
    family: str,
    k_list: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (sweep_df, frozen_df) for the matching plot.

    sweep_df:  runs for `family` at TARGET_DICT_SIZE with top_k ∈ sweep values
    frozen_df: the standard frozen_{TARGET_DICT_SIZE} run used as baseline.
    """
    gt_tree = keep_ground_truth_per_method(tree_for_seed)

    df = calc_matching_score_over_all_attrs_df(
        gt_tree,
        list(k_list),
        include_k_sweep=True,
        include_probe=False,
    )
    if df.empty:
        return df, df

    sweep_df = df[
        (df["sae"] == family)
        & (df["dict_size"] == TARGET_DICT_SIZE)
        & (df["top_k"].notna())
    ].copy()

    frozen_df = df[
        (df["sae"] == "frozen") & (df["dict_size"] == TARGET_DICT_SIZE)
    ][["metric_name", "matching_score"]].copy()

    return sweep_df, frozen_df


def _tapas_for_sparsity_sweep(
    metrics_root: str,
    syn_dataset: str,
    model_name: str,
    family: str,
) -> pd.DataFrame:
    metrics_dir = Path(f"{metrics_root}/{syn_dataset}")
    also_root = Path(f"{metrics_root}/{nonsyn_for_syn(syn_dataset)}")
    df = load_perturbation_dataframe(
        metrics_dir,
        score_calculation="bin_max_then_delta",
        also_search_root=also_root,
        include_k_sweep=True,
        include_probe=False,
    )
    if df.empty:
        return df

    df = df[
        (df["model"] == model_name)
        & (df["sae"] == family)
        & (df["dict_size"] == TARGET_DICT_SIZE)
        & (df["top_k"].notna())
    ].copy()

    df = (
        df.groupby(["sae", "top_k", "metric_name"], as_index=False)
        .agg(score_mean=("score_mean", "mean"))
        .copy()
    )
    return df


def fig_sparsity_sweep():
    metrics_root = "/home/jokl/PycharmProjects/rs_concepts/outputs/metrics"
    figures_dir = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures/appendix/fig_topk_sparsity_sweep")
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_keep = [
        "f1_top1",
        "bmp_f1.0_top3",
        "bmp_f0.5_top3",
        "bmp_f0.25_top3",
    ]
    k_list = [1, 3, 5, 10]
    metrics_sorted = sorted(
        metrics_to_keep,
        key=lambda s: (_extract_topk(s) if _extract_topk(s) is not None else 10**9),
    )

    dataset_pairs = [("CUB_attrs", "syn_cub_attrs"), ("COCO", "syn_coco")]
    model_names = ["CLIP-ViT-L-14"]

    loaded = load_nested_pt_tree(metrics_root)

    for orig_dataset, syn_dataset in dataset_pairs:
        for model_name in model_names:
            legend_dir = figures_dir / orig_dataset / model_name
            match_legend_h, match_legend_l = None, None
            tapas_legend_h, tapas_legend_l = None, None

            for family in TARGET_FAMILIES:
                print(f"\n{orig_dataset} / {model_name} / {family}")

                try:
                    tree_for_seed = loaded[orig_dataset][model_name]["42"]
                except KeyError:
                    print(f"  no metrics tree for {orig_dataset}/{model_name}/42")
                    continue

                # ---- matching ----
                sweep_df, frozen_df = _matching_for_sparsity_sweep(
                    tree_for_seed, family=family, k_list=k_list,
                )
                if sweep_df.empty:
                    print(f"  no k-sweep matching rows for {family}; skipping")
                else:
                    merged = sweep_df.merge(
                        frozen_df.rename(columns={"matching_score": "matching_score_frozen"}),
                        on="metric_name",
                        how="left",
                    )
                    merged["matching_score"] = (
                        merged["matching_score"] - merged["matching_score_frozen"]
                    )
                    merged = merged[merged["metric_name"].isin(metrics_to_keep)].copy()
                    sparsities = sorted(int(k) for k in merged["top_k"].dropna().unique())

                    save_path = legend_dir / f"{family}_dict{TARGET_DICT_SIZE}_matching_vs_k.png"
                    result = _plot_lines_over_sparsity(
                        merged,
                        sparsities=sparsities,
                        metrics_sorted=metrics_sorted,
                        score_col="matching_score",
                        ylabel=r"$F_1\ \Delta\ \mathrm{MATCHScore}$",
                        title=f"{pretty_dataset(orig_dataset)} — {pretty_sae_family(family)} (d={TARGET_DICT_SIZE})",
                        save_path=save_path,
                    )
                    if result is not None and match_legend_h is None:
                        match_legend_h, match_legend_l = result

                # ---- TAPAS ----
                tapas_df = _tapas_for_sparsity_sweep(metrics_root, syn_dataset, model_name, family)
                if tapas_df.empty:
                    print(f"  no k-sweep TAPAS rows for {family}; skipping")
                    continue

                tapas_df = tapas_df[tapas_df["metric_name"].isin(metrics_to_keep)].copy()
                sparsities = sorted(int(k) for k in tapas_df["top_k"].dropna().unique())

                save_path = legend_dir / f"{family}_dict{TARGET_DICT_SIZE}_tapas_vs_k.png"
                result = _plot_lines_over_sparsity(
                    tapas_df,
                    sparsities=sparsities,
                    metrics_sorted=metrics_sorted,
                    score_col="score_mean",
                    ylabel="TAPAScore",
                    title=f"{pretty_syn_dataset(syn_dataset)} — {pretty_sae_family(family)} (d={TARGET_DICT_SIZE})",
                    save_path=save_path,
                )
                if result is not None and tapas_legend_h is None:
                    tapas_legend_h, tapas_legend_l = result

            if match_legend_h is not None:
                save_legend_strip(match_legend_h, match_legend_l,
                                  save_path=legend_dir / "matching_sparsity_legend.png",
                                  show=SHOW_PLOTS)
            if tapas_legend_h is not None:
                save_legend_strip(tapas_legend_h, tapas_legend_l,
                                  save_path=legend_dir / "tapas_sparsity_legend.png",
                                  show=SHOW_PLOTS)


if __name__ == "__main__":
    fig_sparsity_sweep()

# lighting fix all seeds

from lightning import seed_everything



seed_everything(42, workers=True)

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from utils import resolvers  # noqa: F401 ensures resolver is registered

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import torch

from scripts.visualization.plot_utils.plot_utils import COLORS, pretty_sae_family, pretty_metric_name, get_metric_color, save_legend_strip

SHOW_PLOTS = True  # set to False when running in batch/headless mode
from scripts.visualization.plot_utils.plot_data_utils import (
    load_nested_pt_tree,
    keep_ground_truth_per_method,
    extract_matching_scores, calc_matching_score_over_all_attrs_df,
)

def plot_matching_vs_dictsize_df(
    df,  # expected columns: ["sae", "dict_size", "metric_name", "matching_score"]
    sae: str,
    probe_matching,
    metrics_to_keep: Sequence[str],
    k_for_label: int = 3,  # only used for legend text of FBMP
    save_dir: str | None = None,
    fname_prefix: str = "figure3_matching",
    figsize=(6, 4),
    title: str | None = "",
):
    # filter to one SAE family and desired metrics only
    d = df[(df["sae"] == sae) & (df["metric_name"].isin(metrics_to_keep))].copy()
    if d.empty:
        print(f"Skipping {sae}: no rows after filtering.")
        return

    # Probe baseline: best 1-1 per attribute (macro-average)
    probe_matching = torch.as_tensor(probe_matching)
    probe_value = probe_matching.max(dim=1).values.mean().item()

    # ensure consistent x ordering
    dict_sizes = np.array(sorted(d["dict_size"].unique().tolist()), dtype=int)
    # marker styles (extend as needed)
    marker_map = {
        "f1_top1": "o",
        "jaccard_top1": "s",
        "mi_top1": "P",
        "probe_top1": "^",
        "bmp_f0.5_top1": "D",
        "bmp_f0.5_top2": "d",
        "bmp_f0.5_top3": "P",
        "bmp_f0.5_top5": "h",
        "bmp_f0.5_top10": "*",
        "bmp_mi_top3": "v",  # distinct triangle-down marker for MI-weighted BMP
        "bmp_f1.0_top3" : "X",
        'bmp_f0.25_top3': "h",

    }
    # metric styling: map your metric names to colors/labels
    metric_style = {
        "f1_top1": {
            "color": COLORS["metrics"]["f1"],
            "label": "F1 (k=1)",
            "lw": 1.8,
        },
        "mi_top1": {
            "color": COLORS["metrics"]["mi"],
            "label": "MI (k=1)",
            "lw": 1.8,
        },
        # BMP variants → different red tones
        "bmp_f0.25_top3": {
            "color": "#fca5a5",  # light red
            "label": rf"FBMP F0.25 (k={k_for_label})",
            "lw": 1.8,
        },
        "bmp_f0.5_top3": {
            "color": "#ef4444",  # medium red
            "label": rf"FBMP F0.5 (k={k_for_label})",
            "lw": 1.8,
        },
        "bmp_f1.0_top3": {
            "color": "#991b1b",  # dark red
            "label": rf"FBMP F1 (k={k_for_label})",
            "lw": 1.8,
        },
    }

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    y_series = []

    for metric_name in metrics_to_keep:
        dd = d[d["metric_name"] == metric_name].copy()
        if dd.empty:
            continue

        # enforce dict_size order
        dd = dd.sort_values("dict_size")
        x = dd["dict_size"].to_numpy(dtype=int)
        y = dd["matching_score"].to_numpy(dtype=float)

        style = metric_style.get(
            metric_name,
            {"label": metric_name, "lw": 1.6},
        )

        ax.plot(
            x,
            y,
            color=get_metric_color(metric_name),
            marker=marker_map[metric_name],
            linewidth=style["lw"],
            label=style["label"],
        )
        y_series.append(y)

    # probe baseline
    ax.plot(
        dict_sizes,
        np.full_like(dict_sizes, probe_value, dtype=float),
        linestyle="--",
        color=COLORS["metrics"]["probe"],
        linewidth=2.0,
        label="Probe baseline",
    )
    y_series.append(np.full(len(dict_sizes), float(probe_value), dtype=float))

    ax.set_xscale("log", base=2)
    ax.set_xticks(dict_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Dynamic y-limits and ticks (scores in [0, 1]) with adaptive step.
    y_all = np.concatenate([np.asarray(s, dtype=float) for s in y_series]) if y_series else np.array([])
    y_all = y_all[np.isfinite(y_all)]

    if y_all.size == 0:
        ymin, ymax = 0.0, 1.0
    else:
        y_min = float(y_all.min())
        y_max = float(y_all.max())

        # small padding
        pad = 0.02
        y_min = max(0.0, y_min - pad)
        y_max = min(1.0, y_max + pad)

        # choose a reasonable tick step based on span
        span = max(1e-6, y_max - y_min)
        if span <= 0.10:
            step = 0.01
        elif span <= 0.25:
            step = 0.02
        elif span <= 0.50:
            step = 0.05
        else:
            step = 0.10

        # snap to grid
        ymin = max(0.0, np.floor(y_min / step) * step)
        ymax = min(1.0, np.ceil(y_max / step) * step)

        # avoid degenerate axis
        if ymax - ymin < step:
            ymax = min(1.0, ymin + step)

    ax.set_ylim(ymin, ymax)

    # ticks follow dynamic step; include upper bound
    ticks = np.arange(ymin, ymax + 0.5 * step, step)
    ax.set_yticks(ticks)

    # optional: show 2 decimals when step < 0.1
    if step < 0.1:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    else:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    plt.title(f"{title}{pretty_sae_family(sae)}")
    ax.set_xlabel("Dictionary size")
    ax.set_ylabel("F1 Matching score")
    handles, labels = ax.get_legend_handles_labels()
    handles_rev, labels_rev = handles[::-1], labels[::-1]
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fpath = os.path.join(save_dir, f"{fname_prefix}_{sae}.png")
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {fpath}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return handles_rev, labels_rev



def fig_unnormalized_matching_dict_size():
    metrics_root = "/home/jokl/PycharmProjects/rs_concepts/outputs/metrics"
    figures_dir = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures")
    loaded = load_nested_pt_tree(metrics_root)

    dataset_names = ["CUB_attrs", "COCO"]
    model_names = ["CLIP-ViT-L-14", "DINOV2"]
    for dataset_name in dataset_names:
        for model_name in model_names:
            print(f"Processing dataset: {dataset_name}, model: {model_name}")
            cub_data = loaded[dataset_name][model_name]["42"]
            cub_data = keep_ground_truth_per_method(cub_data)

            k_list = [1, 3]
            matching_score_df = calc_matching_score_over_all_attrs_df(cub_data, k_list)

            metrics_to_keep = ["f1_top1", "bmp_f1.0_top3", "bmp_f0.5_top3", "bmp_f0.25_top3"]
            matching_score_df = matching_score_df[
                matching_score_df["metric_name"].isin(metrics_to_keep)
            ].copy()

            # probe baseline (unchanged)
            probe_path = (
                f"/home/jokl/PycharmProjects/rs_concepts/outputs/metrics/"
                f"{dataset_name}/CLIP-ViT-L-14/42/matching/linear_probe/k=32/f1_matrix.pt"
            )
            probe_matching = torch.tensor(torch.load(probe_path, map_location="cpu", weights_only=False))

            out_dir = str(figures_dir / "appendix" / "fig_unnormalized_matching_over_dictsize" / dataset_name / model_name)
            legend_handles, legend_labels = None, None

            # drive plots purely from dataframe: one figure per SAE family present in df
            for sae in sorted(matching_score_df["sae"].unique().tolist()):
                result = plot_matching_vs_dictsize_df(
                    df=matching_score_df,
                    sae=sae,
                    probe_matching=probe_matching,
                    metrics_to_keep=metrics_to_keep,
                    k_for_label=3,
                    save_dir=out_dir,
                    fname_prefix="unnormalized_matching",
                    figsize=(4, 3.2),
                )
                if result is not None and legend_handles is None:
                    legend_handles, legend_labels = result

            if legend_handles is not None:
                save_legend_strip(
                    legend_handles, legend_labels,
                    save_path=Path(out_dir) / "unnormalized_matching_legend.png",
                    show=SHOW_PLOTS,
                )



if __name__ == "__main__":
    fig_unnormalized_matching_dict_size()

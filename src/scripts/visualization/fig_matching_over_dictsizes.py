# lighting fix all seeds
from lightning import seed_everything
from pandas import DataFrame

seed_everything(42, workers=True)

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from utils import resolvers  # noqa: F401 ensures resolver is registered

SHOW_PLOTS = True  # set to False when running in batch/headless mode

import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import torch

from scripts.visualization.plot_utils.plot_utils import (
    COLORS,
    pretty_sae_family,
    pretty_metric_name,
    get_metric_color,
    pretty_dataset,
    save_legend_strip,
    compute_shared_ylim,
)
from scripts.visualization.plot_utils.plot_data_utils import (
    load_nested_pt_tree,
    keep_ground_truth_per_method,
    calc_matching_score_over_all_attrs_df,
)


# ----------------------------
# Helpers
# ----------------------------

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


def _marker_for_topk(k: int | None) -> str:
    if k == 1:
        return "o"
    if k == 3:
        return "s"
    if k == 5:
        return "D"
    if k == 10:
        return "X"
    return "o"

def _marker_for_metric(metric) -> str:
    if metric == 'f1_top1':
        return "o"
    if metric == 'f1_top3':
        return "s"
    if metric == 'bmp_f1.0_top3':
        return "X"
    if metric == 'bmp_f0.5_top3':
        return "D"
    if metric == 'bmp_f0.25_top3':
        return "h"
    if metric == 'nnomp_top3':
        return "P"
    return "o"


# ----------------------------
# Plotting
# ----------------------------

def plot_matching_vs_dictsize_df(
    df,
    sae: str,
    metrics_to_keep: Sequence[str],
    save_dir: str | None = None,
    fname_prefix: str = "figure3_matching",
    figsize=(6, 4),
    title: str | None = "",
    clip_unit_interval: bool = True,
    probe_scores: dict[str, float] | None = None,
    show_legend: bool = False,
    shared_ylim: tuple[float, float, float] | None = None,
):
    d = df[(df["sae"] == sae) & (df["metric_name"].isin(metrics_to_keep))].copy()
    if d.empty:
        print(f"Skipping {sae}: no rows after filtering.")
        return None

    dict_sizes = np.array(sorted(d["dict_size"].unique().tolist()), dtype=int)

    metrics_sorted = sorted(
        list(metrics_to_keep),
        key=lambda s: (_extract_topk(s) if _extract_topk(s) is not None else 10**9),
    )

    probe_score = None
    if probe_scores:
        ps = probe_scores.get("f1_top1")
        if ps is not None and np.isfinite(ps):
            probe_score = float(ps)

    use_break = probe_score is not None

    if use_break:
        # Broken y-axis: tall lower panel for SAE curves, narrow upper panel
        # showing only the probe line. Shared x-axis.
        fig, (ax_top, ax) = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 6], "hspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_top = None

    y_series = []
    pos_for_size = {int(s): i for i, s in enumerate(dict_sizes)}

    for metric_name in metrics_sorted:
        dd = d[d["metric_name"] == metric_name].copy()
        if dd.empty:
            continue

        dd = dd.sort_values("dict_size")
        x_sizes = dd["dict_size"].to_numpy(dtype=int)
        x = np.array([pos_for_size[int(s)] for s in x_sizes], dtype=float)
        y = dd["matching_score"].to_numpy(dtype=float)

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

    ax.set_xticks(np.arange(len(dict_sizes)))
    ax.set_xticklabels([str(int(d_)) for d_ in dict_sizes])

    # Dynamic y limits for the SAE panel
    y_all = np.concatenate([np.asarray(s, dtype=float) for s in y_series]) if y_series else np.array([])
    y_all = y_all[np.isfinite(y_all)]

    if shared_ylim is not None:
        ymin, ymax, step = shared_ylim
    else:
        result = compute_shared_ylim(y_all.tolist(), clip_unit_interval=clip_unit_interval)
        if result is None:
            ymin, ymax = (0.0, 1.0) if clip_unit_interval else (-0.5, 0.5)
            step = 0.1
        else:
            ymin, ymax, step = result

    ax.set_ylim(ymin, ymax)
    ticks = np.arange(ymin, ymax + 0.5 * step, step)
    ax.set_yticks(ticks)

    if step < 0.10:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    else:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    if use_break:
        probe_pad = max(step / 2, 0.01)
        top_lo = probe_score - probe_pad
        top_hi = probe_score + probe_pad
        ax_top.set_ylim(top_lo, top_hi)
        ax_top.axhline(
            probe_score,
            linestyle="--",
            color="black",
            alpha=0.7,
            linewidth=1.4,
            label="Probe upper bound (F1 k=1)",
        )
        ax_top.set_yticks([round(probe_score, 2)])
        # hide the shared spine + ticks between the two panels
        ax_top.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=8)

        # diagonal "break" marks
        d_mark = 0.012
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
        ax_top.plot((-d_mark, +d_mark), (-d_mark * 6, +d_mark * 6), **kwargs)
        ax_top.plot((1 - d_mark, 1 + d_mark), (-d_mark * 6, +d_mark * 6), **kwargs)
        kwargs.update(transform=ax.transAxes)
        ax.plot((-d_mark, +d_mark), (1 - d_mark, 1 + d_mark), **kwargs)
        ax.plot((1 - d_mark, 1 + d_mark), (1 - d_mark, 1 + d_mark), **kwargs)

        ax_top.set_title(f"{title}{pretty_sae_family(sae)}")
    else:
        ax.set_title(f"{title}{pretty_sae_family(sae)}")

    ax.set_xlabel("Dictionary size")
    ax.set_ylabel(r"$F_1\ \Delta\ \mathrm{MATCHScore}$")

    handles, labels = ax.get_legend_handles_labels()
    if use_break:
        ph, pl = ax_top.get_legend_handles_labels()
        handles = handles + ph
        labels = labels + pl
    handles_rev, labels_rev = handles[::-1], labels[::-1]

    if show_legend:
        ax.legend(handles_rev, labels_rev, frameon=True, loc="best")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fpath = os.path.join(save_dir, f"{fname_prefix}_{sae}.png")
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {fpath}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return handles_rev, labels_rev


# ----------------------------
# Main
# ----------------------------

def fig_matching_over_dictsizes():
    metrics_root = "/home/jokl/PycharmProjects/rs_concepts/outputs/metrics"
    figures_dir = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures")
    appendix_dir = figures_dir / "appendix" / "fig_matching_over_dictsizes_nnomp"
    loaded = load_nested_pt_tree(metrics_root)

    dataset_names = ["CUB_attrs", "COCO"]
    model_names = ["CLIP-ViT-L-14", 'DINOV2']

    # dataset_names = ["COCO", ]
    # model_names = ["CLIP-ViT-L-14",]

    for dataset_name in dataset_names:
        for model_name in model_names:
            tree = loaded[dataset_name][model_name]["42"]
            tree = keep_ground_truth_per_method(tree)

            k_list = [1, 3]
            # Trained / random / frozen rows (probe + k-sweep excluded by default).
            matching_score_df = calc_matching_score_over_all_attrs_df(tree, k_list)
            # Probe rows for the dashed upper-bound line.
            probe_df = calc_matching_score_over_all_attrs_df(
                tree, k_list, include_probe=True
            )
            probe_df = probe_df[probe_df["sae"] == "linear_probe"]

            # Main paper: FBMP + F1 metrics
            plt_metrics_to_keep(
                dataset_name,
                model_name,
                figures_dir / "fig_matching_over_dictsizes",
                matching_score_df,
                metrics_to_keep=["f1_top1", "bmp_f1.0_top3", "bmp_f0.5_top3", "bmp_f0.25_top3"],
                fname_pre="figure3_matching_minus_frozen",
                probe_df=probe_df,
            )

            # Appendix: nnomp comparison (with bmp_f0.5 for reference)
            plt_metrics_to_keep(
                dataset_name,
                model_name,
                appendix_dir,
                matching_score_df,
                metrics_to_keep=["bmp_f0.5_top3", "nnomp_top1", "nnomp_top3"],
                fname_pre="figure3_matching_minus_frozen_NN_OMP",
                probe_df=probe_df,
            )


def plt_metrics_to_keep(
    dataset_name: str,
    model_name,
    figures_dir: Path,
    matching_score_df: DataFrame,
    metrics_to_keep: list[str],
    fname_pre,
    probe_df: DataFrame | None = None,
):
    matching_score_df = matching_score_df[
        matching_score_df["metric_name"].isin(metrics_to_keep)
    ].copy()

    # Shared frozen baseline per dict_size: average frozen score over all
    # plotted metrics (incl. nnomp). Using a single shared baseline per
    # dict_size means every line in the plot is offset by the same constant
    # — line-to-line comparability is preserved.
    frozen_df = matching_score_df[matching_score_df["sae"] == "frozen"].copy()
    frozen_baseline = (
        frozen_df.groupby("dict_size", as_index=False)["matching_score"]
        .mean()
        .rename(columns={"matching_score": "matching_score_frozen"})
    )

    saes_delta = ["batchtopk", "matryoshka", "topk", "jumprelu"]

    # Probe upper bound: raw probe score (no frozen subtraction). Only the
    # f1_top1 line is drawn on the figure; we still build the dict from all
    # metrics in case other consumers want them.
    probe_scores: dict[str, float] = {}
    if probe_df is not None and not probe_df.empty:
        probe_filt = probe_df[probe_df["metric_name"].isin(metrics_to_keep)]
        for _, row in probe_filt.iterrows():
            probe_scores[row["metric_name"]] = float(row["matching_score"])

    # First pass: compute merged data for all SAEs and collect y values for shared axis
    sae_merged = {}
    all_y_values = []
    for sae in saes_delta:
        sae_df = matching_score_df[matching_score_df["sae"] == sae].copy()
        if sae_df.empty:
            continue
        merged = sae_df.merge(frozen_baseline, on=["dict_size"], how="left")
        merged["matching_score"] = merged["matching_score"] - merged["matching_score_frozen"]
        sae_merged[sae] = merged
        vals = merged[merged["metric_name"].isin(metrics_to_keep)]["matching_score"].dropna().values
        all_y_values.extend(vals.tolist())

    shared_ylim = compute_shared_ylim(all_y_values, clip_unit_interval=False)

    # Second pass: plot with shared axis
    legend_handles, legend_labels = None, None
    for sae, merged in sae_merged.items():
        result = plot_matching_vs_dictsize_df(
            df=merged,
            sae=sae,
            metrics_to_keep=metrics_to_keep,
            save_dir=str(figures_dir / dataset_name / model_name),
            fname_prefix=fname_pre,
            figsize=(4, 3.2),
            title=f"{pretty_dataset(dataset_name)} — ",
            clip_unit_interval=False,
            probe_scores=probe_scores,
            show_legend=False,
            shared_ylim=shared_ylim,
        )
        if result is not None and legend_handles is None:
            legend_handles, legend_labels = result

    if legend_handles is not None:
        save_legend_strip(
            legend_handles,
            legend_labels,
            save_path=figures_dir / dataset_name / model_name / f"{fname_pre}_legend.png",
            show=SHOW_PLOTS,
        )

    print("Done plotting matching results (delta to frozen).")


if __name__ == "__main__":
    fig_matching_over_dictsizes()
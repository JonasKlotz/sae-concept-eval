# fig_tapas_over_dictsizes.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

SHOW_PLOTS = True  # set to False when running in batch/headless mode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.visualization.fig_matching_over_dictsizes import _marker_for_metric
from scripts.visualization.plot_utils.plot_data_utils import (
    load_perturbation_dataframe,
    nonsyn_for_syn,
)
from scripts.visualization.plot_utils.plot_utils import (
    pretty_sae_family,
    pretty_metric_name,
    get_metric_color,
    pretty_syn_dataset,
    save_legend_strip,
    compute_shared_ylim,
)


# ----------------------------
# Helpers (match Fig.3 style)
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


# ----------------------------
# Plotting
# ----------------------------

def plot_tapas_vs_dictsize_df(
    df: pd.DataFrame,
    sae: str,
    metrics_to_keep: Sequence[str],
    save_dir: str | Path | None = None,
    fname_prefix: str = "appendix_tapas_vs_dictsize",
    figsize=(6, 4),
    title: str | None = "",
    probe_scores: dict[str, float] | None = None,
    show_legend: bool = False,
    shared_ylim: tuple[float, float, float] | None = None,
):
    """
    Line plot: TAPAScore vs dictionary size (log2 x-axis).
    One line per metric_name, for a fixed SAE family.

    Expected columns in df:
      - sae
      - dict_size
      - metric_name
      - score_mean   (TAPAS mean)
    """

    d = df[(df["sae"] == sae) & (df["metric_name"].isin(list(metrics_to_keep)))].copy()
    if d.empty:
        print(f"Skipping {sae}: no rows after filtering.")
        return None

    d = (
        d.groupby(["dict_size", "metric_name"], as_index=False)
        .agg(score_mean=("score_mean", "mean"))
        .copy()
    )

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
        y = dd["score_mean"].to_numpy(dtype=float)

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

    y_all = np.concatenate([np.asarray(s, dtype=float) for s in y_series]) if y_series else np.array([])
    y_all = y_all[np.isfinite(y_all)]

    if shared_ylim is not None:
        ymin, ymax, step = shared_ylim
    else:
        result = compute_shared_ylim(y_all.tolist())
        if result is None:
            ymin, ymax, step = 0.0, 1.0, 0.1
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
        sae_span = (
            float(np.nanmax(y_all)) - float(np.nanmin(y_all))
            if y_all.size > 0 else 0.1
        )
        probe_pad = max(sae_span / 6, 0.01)
        ax_top.set_ylim(probe_score - probe_pad, probe_score + probe_pad)
        ax_top.axhline(
            probe_score,
            linestyle="--",
            color="black",
            alpha=0.7,
            linewidth=1.4,
            label="Probe upper bound (F1 k=1)",
        )
        ax_top.set_yticks([round(probe_score, 2)])
        ax_top.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=8)

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
    ax.set_ylabel("TAPAScore")

    handles, labels = ax.get_legend_handles_labels()
    if use_break:
        ph, pl = ax_top.get_legend_handles_labels()
        handles = handles + ph
        labels = labels + pl
    handles_rev, labels_rev = handles[::-1], labels[::-1]

    if show_legend:
        ax.legend(handles_rev, labels_rev, frameon=True, loc="best")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fpath = save_dir / f"{sae}_{fname_prefix}.png"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {fpath}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return handles_rev, labels_rev


# ----------------------------
# Main
# ----------------------------

def _run_tapas_pass(df_full, probe_df_full, model_names, saes, metrics_to_keep, out_dir, syn_dataset, fname_prefix):
    all_metrics = df_full["metric_name"].unique().tolist()
    metrics_present = [m for m in metrics_to_keep if m in all_metrics]
    df = df_full[df_full["metric_name"].isin(metrics_present)].copy()
    probe_df = probe_df_full[probe_df_full["metric_name"].isin(metrics_present)].copy()
    if df.empty:
        print(f"Skipping {fname_prefix}: no rows for requested metrics.")
        return

    df = (
        df.groupby(["model", "sae", "dict_size", "metric_name"], as_index=False)
        .agg(score_mean=("score_mean", "mean"))
        .copy()
    )
    probe_df = (
        probe_df.groupby(["model", "metric_name"], as_index=False)
        .agg(score_mean=("score_mean", "mean"))
        .copy()
    )

    for model_name in model_names:
        dm = df[df["model"] == model_name].copy()
        probe_for_model = probe_df[probe_df["model"] == model_name]
        probe_scores = dict(zip(probe_for_model["metric_name"], probe_for_model["score_mean"]))

        # Collect all y values across SAEs for shared axis
        all_y_values = []
        for sae in saes:
            vals = dm[(dm["sae"] == sae) & (dm["metric_name"].isin(metrics_present))]["score_mean"].dropna().values
            all_y_values.extend(vals.tolist())
        shared_ylim = compute_shared_ylim(all_y_values)

        legend_handles, legend_labels = None, None
        for sae in saes:
            result = plot_tapas_vs_dictsize_df(
                df=dm,
                sae=sae,
                metrics_to_keep=metrics_present,
                save_dir=out_dir / syn_dataset / str(model_name),
                fname_prefix=fname_prefix,
                figsize=(4, 3.2),
                title=f"{pretty_syn_dataset(syn_dataset)} — ",
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
                save_path=out_dir / syn_dataset / str(model_name) / f"{fname_prefix}_legend.png",
                show=SHOW_PLOTS,
            )


def fig_tapas_over_dictsizes():
    syn_datasets = ["syn_cub_attrs", "syn_coco"]
    # syn_datasets = ["syn_coco"]

    model_names = ["CLIP-ViT-L-14", 'DINOV2']
    # model_names = ["CLIP-ViT-L-14"]

    main_metrics = [
        "f1_top1",
        "bmp_f1.0_top3",
        "bmp_f0.5_top3",
        "bmp_f0.25_top3",
    ]
    nnomp_metrics = [
        "bmp_f0.5_top3",
        "nnomp_top1",
        "nnomp_top3",
    ]
    all_metrics = main_metrics + nnomp_metrics

    saes_main = ("batchtopk", "matryoshka", "topk", "jumprelu")
    saes_all = ("batchtopk", "matryoshka", "topk", "jumprelu", "frozen", "random")

    metrics_root = "/home/jokl/PycharmProjects/rs_concepts/outputs/metrics"
    figures_dir = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures")
    out_dir = figures_dir / "fig_perturbation_over_dictsize"
    appendix_dir = figures_dir / "appendix" / "fig_perturbation_over_dictsize_nnomp"
    out_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)

    for syn_dataset in syn_datasets:
        print(f"Processing dataset: {syn_dataset}")
        metrics_dir = Path(f"{metrics_root}/{syn_dataset}")
        # nnomp + probe TAPAS csvs sit under the non-syn folder.
        also_root = Path(f"{metrics_root}/{nonsyn_for_syn(syn_dataset)}")
        score_calculation = "bin_max_then_delta"
        df_raw = load_perturbation_dataframe(
            metrics_dir,
            score_calculation=score_calculation,
            also_search_root=also_root,
            include_probe=True,
        )

        df_raw = df_raw[df_raw["metric_name"].isin(all_metrics)].copy()
        if df_raw.empty:
            print(f"Skipping {syn_dataset}: no rows for requested metrics.")
            continue

        probe_df_raw = df_raw[df_raw["sae"] == "linear_probe"].copy()
        df_raw = df_raw[df_raw["sae"] != "linear_probe"].copy()

        # Main paper: FBMP + F1 metrics, trained SAE families only
        _run_tapas_pass(
            df_raw, probe_df_raw, model_names, saes_main, main_metrics,
            out_dir, syn_dataset, fname_prefix="tapas_vs_dictsize",
        )

        # Appendix: nnomp comparison, all SAE families
        _run_tapas_pass(
            df_raw, probe_df_raw, model_names, saes_all, nnomp_metrics,
            appendix_dir, syn_dataset, fname_prefix="tapas_vs_dictsize_nnomp",
        )


if __name__ == "__main__":
    fig_tapas_over_dictsizes()
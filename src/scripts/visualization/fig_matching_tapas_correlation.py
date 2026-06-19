# lighting fix all seeds
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightning import seed_everything

seed_everything(42, workers=True)

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from utils import resolvers  # noqa: F401 ensures resolver is registered

SHOW_PLOTS = True  # set to False when running in batch/headless mode
from scripts.visualization.plot_utils.plot_data_utils import (
    load_matching_data,
    load_perturbation_dataframe, calc_matching_score_over_all_attrs_df, get_perturbed_attr_ids,
)
from scripts.visualization.plot_utils.plot_utils import (
    COLORS,
    pretty_metric_name,
    _parse_metric_triplet, pretty_syn_dataset,
)

import re


def _natural_key(s):
    # splits "BMP (k=10)" -> ["bmp (k=", 10, ")"]
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]



def plot_matching_steering_correlation_scatter2(
    df: pd.DataFrame,
    title: str = "",
    save_path: str | None = None,
    dpi: int = 300,
    fit_on: str = "all",
    per_metric_fit: bool = True,
    per_metric_fit_span: tuple[float, float] = (0.10, 0.90),
    min_points_per_metric: int = 3,
    draw_global_fit: bool = False,
    annotate_points: bool = True,
    annotate_fmt: str = "{sae}-{dict_size}",
    annotate_fontsize: int = 9,
    annotate_alpha: float = 0.9,
    figsize=(4,3),
):
    required = {"metric_name", "matching_score", "steering_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    dff = df.dropna(subset=["matching_score", "steering_score"]).copy()
    if dff.empty:
        raise ValueError("No rows with both matching_score and steering_score present.")

    # If you want this to be strict for your current use-case:
    if annotate_points:
        need = {"sae", "dict_size"}
        miss2 = need - set(dff.columns)
        if miss2:
            raise ValueError(f"annotate_points=True requires columns: {miss2}")

    marker_map = {
        "f1_top1": "o",
        "jaccard_top1": "s",
        "mi_top1": "P",
        "probe_top1": "^",
        "bmp_f0.5_top1": "D",
        "bmp_f0.5_top2": "d",
        "bmp_f0.5_top3": "X",
        "bmp_f0.5_top5": "h",
        "bmp_f0.5_top10": "*",
    }

    bmp_variant_colors = {
        "bmp_f0.5_top1": "#fb6a4a",
        "bmp_f0.5_top2": "#d95f02",
        "bmp_f0.5_top3": "red",
        "bmp_f0.5_top5": "#e7298a",
        "bmp_f0.5_top10": "#a50f15",
    }

    metric_colors = COLORS["metrics"] if isinstance(COLORS, dict) and "metrics" in COLORS else COLORS

    color_map = {}
    for m in dff["metric_name"].unique():
        if m in bmp_variant_colors:
            color_map[m] = bmp_variant_colors[m]
        else:
            family = _parse_metric_triplet(m)[0]
            color_map[m] = metric_colors.get(family, None)

    fig, ax = plt.subplots(figsize=figsize)

    preferred = [
        "bmp_f0.5_top3",
        "bmp_f0.5_top1",
        "bmp_f0.5_top2",
        "bmp_f0.5_top5",
        "bmp_f0.5_top10",
        "f1_top1",
        "jaccard_top1",
        "mi_top1",
        "probe_top1",
    ]
    metric_order = [m for m in preferred if m in set(dff["metric_name"])]
    metric_order += [m for m in sorted(dff["metric_name"].unique()) if m not in set(metric_order)]

    # deterministic small offsets to reduce label overlap
    _offsets = [
        (4, 4), (6, -4), (-6, 4), (-4, -6), (8, 2), (2, 8), (-8, 2), (2, -8)
    ]

    for m in metric_order:
        sub = dff[dff["metric_name"] == m].copy()
        x_m = sub["matching_score"].to_numpy(dtype=float)
        y_m = sub["steering_score"].to_numpy(dtype=float)

        c = color_map.get(m, None)
        mk = marker_map.get(m, "o")

        ax.scatter(x_m, y_m, s=55, alpha=0.8, marker=mk, color=c)

        # annotate each point with sae + dict_size (your case: single metric => clean)
        if annotate_points:
            # stable annotation order: sort by sae then dict_size then x
            sub = sub.sort_values(["sae", "dict_size", "matching_score", "steering_score"], kind="mergesort")
            for i, row in enumerate(sub.itertuples(index=False)):
                label = annotate_fmt.format(sae=row.sae, dict_size=int(row.dict_size))
                dx, dy = _offsets[i % len(_offsets)]
                ax.annotate(
                    label,
                    (float(row.matching_score), float(row.steering_score)),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha="left",
                    va="bottom",
                    fontsize=annotate_fontsize,
                    alpha=annotate_alpha,
                )

        # per-metric short regression segment
        label = pretty_metric_name(m)
        r_m = np.nan

        if per_metric_fit and len(x_m) >= min_points_per_metric:
            r_m = float(np.corrcoef(x_m, y_m)[0, 1])
            a_m, b_m = np.polyfit(x_m, y_m, 1)

            qlo, qhi = per_metric_fit_span
            x_lo = float(np.quantile(x_m, qlo))
            x_hi = float(np.quantile(x_m, qhi))
            if x_hi > x_lo:
                xs = np.linspace(x_lo, x_hi, 40)
                ax.plot(xs, a_m * xs + b_m, linewidth=2, alpha=0.95, color=c)

        if np.isfinite(r_m):
            label = f"{label} (r={r_m:.2f})"

        ax.scatter([], [], s=55, alpha=0.8, marker=mk, color=c, label=label)

    if draw_global_fit:
        if fit_on == "bmp_only":
            bmp_like = [m for m in dff["metric_name"].unique() if m.startswith("bmp")]
            fit_df = dff[dff["metric_name"].isin(bmp_like)] if bmp_like else dff
        else:
            fit_df = dff

        x = fit_df["matching_score"].to_numpy(dtype=float)
        y = fit_df["steering_score"].to_numpy(dtype=float)
        if len(x) >= 2:
            a, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, a * xs + b, linewidth=2)

        r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3 else np.nan
        ax.text(
            0.02, 0.98, f"Global Pearson r = {r:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
        )

    ax.set_xlabel(r"$F_1\ \Delta\ \mathrm{MATCHScore}$")
    ax.set_ylabel("TAPAScore")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: _natural_key(labels[i]))
    labels = [labels[i] for i in order]

    custom_handles = []
    for label in labels:
        metric_name = next(m for m in metric_order if pretty_metric_name(m) in label)
        custom_handles.append(
            Line2D(
                [0], [0],
                marker=marker_map.get(metric_name, "o"),
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=color_map.get(metric_name, "black"),
                markeredgewidth=1.8,
                markersize=9,
            )
        )

    ax.legend(custom_handles, labels, frameon=True)

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def fig_matching_tapas_correlation():
    syn_datasets = ["syn_cub_attrs", "syn_coco"]




    for syn_dataset in syn_datasets:
        all_attr_ids = get_perturbed_attr_ids(syn_dataset)
        metrics_root = "/home/jokl/PycharmProjects/rs_concepts/outputs/metrics"
        metrics_syn_dir = Path(f"{metrics_root}/{syn_dataset}")

        steering_score_df = load_perturbation_dataframe(metrics_syn_dir, "bin_max_then_delta")
        cub_data = load_matching_data(metrics_root)
        steering_score_df = steering_score_df.rename(columns={"score_mean": "steering_score"})

        k_list = [1, 2, 3, 5]
        matching_score_df = calc_matching_score_over_all_attrs_df(cub_data, k_list, all_attr_ids)

        # filter metrics
        metrics_to_keep = ["f1_top1", "bmp_f0.5_top3", "bmp_f1.0_top3"]
        saes_delta = ["topk", "batchtopk", "matryoshka"]
        models_to_keep = ["CLIP-ViT-L-14"]

        steering_score_df = steering_score_df[steering_score_df["metric_name"].isin(metrics_to_keep)].copy()
        matching_score_df = matching_score_df[matching_score_df["metric_name"].isin(metrics_to_keep)].copy()

        steering_score_df = steering_score_df[steering_score_df["model"].isin(models_to_keep)].copy()
        # matching_score_df = matching_score_df[matching_score_df["model"].isin(models_to_keep)].copy()

        # -------------------------------------------------
        # IMPORTANT: keep "frozen" in matching_score_df for baseline subtraction.
        # Only filter non-frozen SAE rows AFTER subtraction.
        # -------------------------------------------------

        # Build merged df for plotting (only needs delta SAEs on steering side)
        steering_plot_df = steering_score_df[steering_score_df["sae"].isin(saes_delta)].copy()

        merge_cols = ["sae", "dict_size", "metric_name"]
        df = pd.merge(
            steering_plot_df,
            matching_score_df,   # contains frozen + delta saes
            on=merge_cols,
            how="inner",
        )

        # frozen baseline for matching (computed from the full matching_score_df)
        frozen_match = (
            matching_score_df[matching_score_df["sae"] == "frozen"][
                ["dict_size", "metric_name", "matching_score"]
            ]
            .rename(columns={"matching_score": "matching_score_frozen"})
            .copy()
        )

        # subtract frozen from x-axis only
        df = df.merge(frozen_match, on=["dict_size", "metric_name"], how="left")

        missing_base = df["matching_score_frozen"].isna()
        if missing_base.any():
            print(
                "Warning: missing frozen matching baseline for some (dict_size, metric_name) pairs. "
                f"Dropping {int(missing_base.sum())} rows."
            )
            df = df[~missing_base].copy()

        df["matching_score"] = df["matching_score"] - df["matching_score_frozen"]

        # ensure frozen is not plotted (it should not be present anyway, but keep this explicit)
        df = df[df["sae"].isin(saes_delta)].copy()

        save_path = (
            f"/home/jokl/PycharmProjects/rs_concepts/outputs/figures/fig_matching_vs_tapas/"
            f"{syn_dataset}/delta_matching_score_vs_tapas.png"
        )
        plot_matching_steering_correlation_scatter2(
            df=df,
            title=pretty_syn_dataset(syn_dataset),
            save_path=save_path,
            dpi=300,
            fit_on="all",
            per_metric_fit=True,
            per_metric_fit_span=(0.1, 0.9),
            min_points_per_metric=3,
            draw_global_fit=False,
            annotate_points=False,
            figsize=(5,4),
        )


if __name__ == "__main__":
    fig_matching_tapas_correlation()

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.visualization.plot_utils.plot_data_utils import (
    load_nested_pt_tree,
    convert_dataset_metrics_to_df,
    load_perturbation_dataframe,
)
from scripts.visualization.plot_utils.plot_data_utils import nonsyn_for_syn  # noqa: F401
from scripts.visualization.plot_utils.plot_utils import (
    pretty_metric_name,
    pretty_syn_dataset,
    COLORS, pretty_dataset,
)

SHOW_PLOTS = True  # set to False when running in batch/headless mode

def plot_combined_sota_matching_tapas_grouped(
    df: pd.DataFrame,
    save_path: str | Path,
    title: str = "",
    metrics_order: list[str] | None = None,
    figsize=(7, 3.8),
    ylim: tuple[float, float] = (0.0, 1.0),
):
    """
    Three bars per metric:
      - Random Activation  (sae == "random")
      - Untrained SAE      (sae == "frozen")
      - Trained SAE        (batchtopk/matryoshka/topk)

    No frozen exclusion. Frozen IS the untrained baseline.
    Assumes non-negative scores.
    """

    trained_sae_families = ["batchtopk", "matryoshka", "topk", "jumprelu"]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["sae"] = df["sae"].astype(str)
    df["metric_name"] = df["metric_name"].astype(str)

    def is_trained(s: str) -> bool:
        sl = s.lower()
        return any(sl.startswith(f) for f in trained_sae_families)

    def group_of(s: str) -> str | None:
        sl = s.lower()
        if sl == "random":
            return "Random Activation"
        if sl == "frozen":
            return "Untrained SAE"
        if is_trained(sl):
            return "Trained SAE"
        return None  # everything else discarded

    df["group"] = df["sae"].apply(group_of)
    df = df[df["group"].notna()].copy()

    groups = ["Random Activation", "Untrained SAE", "Trained SAE"]

    df["score_mean"] = pd.to_numeric(df["score_mean"], errors="coerce")
    df["score_mean"] = df["score_mean"].clip(lower=0.0, upper=1.0)

    agg = (
        df.groupby(["metric_name", "group"], as_index=False)["score_mean"]
        .mean()
    )

    if metrics_order is not None:
        metrics = [m for m in metrics_order if m in set(agg["metric_name"].unique())]
    else:
        metrics = list(agg["metric_name"].unique())

    mean_matrix = np.full((len(metrics), len(groups)), np.nan, dtype=float)
    for i, m in enumerate(metrics):
        sub = agg[agg["metric_name"] == m]
        g2mean = dict(zip(sub["group"], sub["score_mean"]))
        for j, g in enumerate(groups):
            mean_matrix[i, j] = g2mean.get(g, np.nan)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics))
    width = 0.3
    offsets = (np.arange(len(groups)) - 1) * width

    group_color_map = {
        "Random Activation": COLORS["sae"]["random"],
        "Untrained SAE": COLORS["sae"]["frozen"],
        "Trained SAE": COLORS["sae"]["trained"],
    }

    for j, g in enumerate(groups):
        heights = mean_matrix[:, j]

        ax.bar(
            x + offsets[j],
            heights,
            width=width,
            label=g,
            color=group_color_map[g],
            edgecolor="black",
            linewidth=0.5,
        )

        for xi, h in zip(x + offsets[j], heights):
            if np.isnan(h):
                continue
            ax.text(
                xi,
                min(h + 0.02, ylim[1] - 0.01),
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title(title)
    ax.set_ylabel("Mean score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontdict={"fontsize":8})
    ax.set_ylim(*ylim)
    ax.axhline(0, linewidth=1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    print(f"Saved: {save_path}")


METRICS_MAIN = [
    ("sota",     "CKNNA"),
    ("sota",     "mean_fms"),
    ("sota",     "monosemanticity_mean"),
    ("matching", "bmp_f1.0_top3"),
    ("matching", "f1_top1"),
    ("tapas",    "bmp_f1.0_top3"),
    ("tapas",    "f1_top1"),
]

METRICS_APPENDIX_NNOMP = [
    ("matching", "bmp_f1.0_top3"),
    ("matching", "f1_top1"),
    ("matching", "nnomp_top3"),
    ("tapas",    "bmp_f1.0_top3"),
    ("tapas",    "f1_top1"),
    ("tapas",    "nnomp_top3"),
]

_PREFIX = {"sota": "", "matching": "MatchS.: ", "tapas": "TAPAS: "}


def _run_sota_pass(
    metrics_list: list[tuple[str, str]],
    out_dir: Path,
    all_metrics: dict,
    metrics_root: str,
    model_name: str,
    dataset_name_pairs: list[tuple[str, str]],
    figsize: tuple[float, float],
    fname: str,
):
    sota_raw     = [m for k, m in metrics_list if k == "sota"]
    matching_raw = [m for k, m in metrics_list if k == "matching"]
    tapas_raw    = [m for k, m in metrics_list if k == "tapas"]

    metrics_order_pretty = [
        f"{_PREFIX[kind]}{pretty_metric_name(m)}" for kind, m in metrics_list
    ]

    for orig_dataset, syn_dataset in dataset_name_pairs:
        orig_tree = all_metrics.get(orig_dataset, {})
        orig_df = convert_dataset_metrics_to_df(orig_tree)
        orig_df = orig_df[["model", "sae", "dict_size", "metric_name", "score_mean"]].copy()
        orig_df = orig_df[orig_df["model"] == model_name].drop(columns=["model"])
        orig_df["metric_name"] = orig_df["metric_name"].replace(
            {
                "bmp_rec_F1_matrix_f1.0": "bmp_f1.0_top3",
                "nnomp_rec_F1_matrix": "nnomp_top3",
                "f1_max_scores": "f1_top1",
                "mean_fms": "mean_fms",
                "monosemanticity_mean": "monosemanticity_mean",
                "CKNNA": "CKNNA",
            }
        )
        orig_df = orig_df[orig_df["metric_name"].isin(sota_raw + matching_raw)].copy()
        orig_df["metric_name"] = orig_df["metric_name"].apply(pretty_metric_name)
        matching_pretty = {pretty_metric_name(m) for m in matching_raw}
        mask = orig_df["metric_name"].isin(matching_pretty)
        orig_df.loc[mask, "metric_name"] = "MatchS.: " + orig_df.loc[mask, "metric_name"]

        syn_metrics_dir = Path(f"{metrics_root}/{syn_dataset}")
        also_root = Path(f"{metrics_root}/{orig_dataset}")
        tapas_df = load_perturbation_dataframe(syn_metrics_dir, also_search_root=also_root)
        tapas_df = tapas_df[["model", "sae", "dict_size", "metric_name", "score_mean"]].copy()
        tapas_df = tapas_df[tapas_df["model"] == model_name].drop(columns=["model"])
        tapas_df = tapas_df[tapas_df["metric_name"].isin(tapas_raw)].copy()
        tapas_df["metric_name"] = "TAPAS: " + tapas_df["metric_name"].apply(pretty_metric_name)

        combined = pd.concat([orig_df, tapas_df], ignore_index=True)

        ds_out = out_dir / f"{orig_dataset}_combined"
        ds_out.mkdir(parents=True, exist_ok=True)

        plot_combined_sota_matching_tapas_grouped(
            df=combined,
            save_path=ds_out / fname,
            title=f"{pretty_dataset(orig_dataset)} / {pretty_syn_dataset(syn_dataset)}",
            metrics_order=metrics_order_pretty,
            figsize=figsize,
            ylim=(0.0, 1.0),
        )


def fig_combined_sota_matching_tapas():
    vis_dir = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures/combined_sota_matching_tapas")
    appendix_dir = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures/appendix/combined_sota_nnomp")

    metrics_root = "/home/jokl/PycharmProjects/rs_concepts/outputs/metrics"
    all_metrics = load_nested_pt_tree(metrics_root)

    model_name = "CLIP-ViT-L-14"
    dataset_name_pairs = [("CUB_attrs", "syn_cub_attrs"), ("COCO", "syn_coco")]

    # Main paper: SOTA baselines + our methods (no nnomp)
    _run_sota_pass(
        METRICS_MAIN,
        out_dir=vis_dir,
        all_metrics=all_metrics,
        metrics_root=metrics_root,
        model_name=model_name,
        dataset_name_pairs=dataset_name_pairs,
        figsize=(5, 4),
        fname="combined_sota_matching_tapas_aggregated.png",
    )

    # Appendix: our methods vs NN-OMP (no SOTA baselines)
    _run_sota_pass(
        METRICS_APPENDIX_NNOMP,
        out_dir=appendix_dir,
        all_metrics=all_metrics,
        metrics_root=metrics_root,
        model_name=model_name,
        dataset_name_pairs=dataset_name_pairs,
        figsize=(5, 4),
        fname="combined_nnomp_comparison_aggregated.png",
    )


if __name__ == "__main__":
    fig_combined_sota_matching_tapas()
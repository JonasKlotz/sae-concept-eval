import pandas as pd
import re
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.visualization.plot_utils.plot_utils import pretty_dataset, pretty_sae_family, COLORS


def load_sae_metric_results(metrics_dir):
    """
    Load SAE evaluation result txt files into a tidy dataframe.

    Expected directory structure:
    metrics/<dataset>/<model>/<seed>/<sae>_<dict_size>/eval_results_<dataset>.txt

    Returns
    -------
    pd.DataFrame
        columns:
        dataset, model, sae, dict_size, metric, mean, std
    """

    metrics_dir = Path(metrics_dir)

    rows = []

    # find all result files
    result_files = metrics_dir.rglob("eval_results_*.txt")

    metric_pattern = re.compile(
        r"(.+?) - Mean:\s*([0-9.eE+-]+), Std:\s*([0-9.eE+-]+)"
    )

    for file in result_files:

        parts = file.parts

        dataset = parts[-5]
        model = parts[-4]

        sae_dict = parts[-2]
        sae, dict_size = sae_dict.rsplit("_", 1)
        dict_size = int(dict_size)

        with open(file, "r") as f:
            for line in f:

                match = metric_pattern.match(line.strip())
                if not match:
                    continue

                metric, mean, std = match.groups()

                rows.append(
                    {
                        "dataset": dataset.lower(),
                        "model": model.lower(),
                        "sae": sae.lower(),
                        "dict_size": dict_size,
                        "metric": metric,
                        "mean": float(mean),
                        "std": float(std),
                    }
                )

    df = pd.DataFrame(rows)

    return df



def pretty_model(name: str) -> str:
    n = str(name).lower()
    if "clip" in n:
        return "CLIP-ViT-L-14"
    if "dinov2" in n:
        return "DINOv2"
    return str(name)


from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sae_loss_scaling(df, save_path=None):

    required_cols = {"dataset", "model", "sae", "dict_size", "metric", "mean"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"

    loss_df = df[df["metric"] == "loss"].copy()
    loss_df["loss"] = loss_df["mean"]

    loss_df["dataset_pretty"] = loss_df["dataset"].map(pretty_dataset)
    loss_df["sae_pretty"] = loss_df["sae"].map(pretty_sae_family)
    loss_df["model_pretty"] = loss_df["model"].map(pretty_model)

    sae_order_raw = ["topk", "batchtopk", "matryoshka", "random", "frozen"]
    present_raw = [s for s in sae_order_raw if s in loss_df["sae"].unique()]

    hue_order = [pretty_sae_family(s) for s in present_raw]
    palette = {pretty_sae_family(s): COLORS["sae"][s] for s in present_raw}

    # different markers per SAE
    marker_map = {
        pretty_sae_family("topk"): "o",
        pretty_sae_family("batchtopk"): "s",
        pretty_sae_family("matryoshka"): "D",
        pretty_sae_family("random"): "X",
        pretty_sae_family("frozen"): "^",
    }

    dataset_order = ["CUB", "COCO"]
    model_order = ["CLIP-ViT-L-14", "DINOv2"]

    sns.set_style("whitegrid")

    g = sns.relplot(
        data=loss_df,
        x="dict_size",
        y="loss",
        hue="sae_pretty",
        hue_order=hue_order,
        palette=palette,
        style="sae_pretty",
        markers=marker_map,
        row="dataset_pretty",
        row_order=dataset_order,
        col="model_pretty",
        col_order=model_order,
        kind="line",
        dashes=False,
        linewidth=2,
        height=2.2,
        aspect=1.35,
        facet_kws={"sharex": True, "sharey": False},
        legend=True
    )

    # log x scale
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks([128,256,512,1024,2048,4096])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # black axes
    for ax in g.axes.flat:

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1)

        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=max(0, ymin), top=ymax * 1.08)

    g.set_axis_labels("Dictionary size", "Reconstruction loss")

    g.fig.subplots_adjust(right=0.80)

    sns.move_legend(
        g,
        "center right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
        title="SAE"
    )

    for i, row_name in enumerate(g.row_names):
        for j, col_name in enumerate(g.col_names):
            g.axes[i, j].set_title(f"{row_name} | {col_name}")

    legend = g._legend
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(1.0)

    plt.tight_layout(rect=[0, 0, 0.80, 1])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()



def fig_sae_training_stats():
    metrics_dir = "/home/jokl/Downloads/pluto3/metrics"
    figures_dir = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures/appendix")
    df = load_sae_metric_results(metrics_dir)
    # save df to figures dir for later use
    df = df.sort_values(["dataset", "model", "sae", "dict_size"])
    df.to_csv(figures_dir / "sae_metric_results.csv", index=False)

    plot_sae_loss_scaling(
        df,
        save_path=figures_dir / "sae_loss_scaling.pdf"
    )


if __name__ == "__main__":
    fig_sae_training_stats()
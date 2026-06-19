from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_in_one(
    metrics_csv: Path, out_dir: Path, filename: str = "training_curves.png"
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)

    if "epoch" not in df.columns:
        raise RuntimeError(f"'epoch' column not found in {metrics_csv}")

    # epoch aggregation (robust to multiple rows per epoch)
    df_epoch = df.groupby("epoch", as_index=False).mean(numeric_only=True)

    cols = ["train/loss", "val/loss", "val/mAP", "val/F1_micro"]
    cols = [c for c in cols if c in df_epoch.columns]
    if len(cols) == 0:
        raise RuntimeError(
            f"None of {cols} found in {metrics_csv}. Available: {list(df_epoch.columns)}"
        )

    fig, ax = plt.subplots()
    for c in cols:
        ax.plot(df_epoch["epoch"], df_epoch[c], label=c)

    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)

    print("Saved:", out_path)


if __name__ == "__main__":
    dataset_name = "cub"
    out_root = Path(
        f"/scratch/htc/jklotz/outputs/concept_xai/classifier/{dataset_name}"
    )
    metrics_csv = out_root / "lightning_csv" / "version_1" / "metrics.csv"
    plots_dir = Path(
        f"/home/htc/jklotz/git/rs_concepts/plots/classifier/{dataset_name}"
    )

    plot_all_in_one(metrics_csv, plots_dir)

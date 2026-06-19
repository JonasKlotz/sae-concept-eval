"""
Appendix: Δ_stay leakage analysis.

For each (dataset, model, seed, sae, dict_size, matching_style, k) configuration
we walk the per-pair perturbation csv and compute:

    TAPAScore = mean(delta_add) - mean(delta_rem)
    Δ_stay    = sum(delta_stay_attr * n_stay_attrs) / sum(n_stay_attrs)

For synCOCO (removal only) `delta_add` is absent and TAPAScore reduces to
-mean(delta_rem).

Outputs:
    1. Stdout summary string with Δ_stay = mean ± std and Pearson r vs TAPAScore
    2. PNG figure: Δ_stay vs dict_size, line per SAE family, panel per dataset
       (uses a single fixed matching_style + k for visual cleanliness)
    3. CSV with the full per-config table for reproducibility
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from scripts.visualization.plot_utils.plot_data_utils import parse_sae_run_name
from scripts.visualization.plot_utils.plot_utils import (
    COLORS,
    pretty_sae_family,
    pretty_syn_dataset,
)

SHOW_PLOTS = False  # batch/headless mode — figures saved to disk only


METRICS_ROOT = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/metrics")
OUT_DIR = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures/appendix/fig_delta_stay")

SYN_DATASETS = ("syn_cub_attrs", "syn_coco")
MODEL_NAMES = ("CLIP-ViT-L-14",)
TRAINED_FAMILIES = ("topk", "batchtopk", "matryoshka", "jumprelu")

# Each (matching_style, top_k) entry produces one PNG. Add or remove tuples
# to control which figures are generated. The headline statistic prints over
# ALL (style, k) configs regardless.
PLOT_CONFIGS: tuple[tuple[str, int], ...] = (
    ("bmp_f0.5", 3),
    ("f1", 1),
)

CSV_NAME_RE = re.compile(r"^(?P<style>.+)_top(?P<k>\d+)\.csv$")

# Fall back when COLORS["sae"] does not have an entry for a family
_SAE_COLOR_FALLBACK = {
    "jumprelu": "#C9B1FF",
}


def _sae_color(family: str) -> str:
    if family in COLORS["sae"]:
        return COLORS["sae"][family]
    return _SAE_COLOR_FALLBACK.get(family, "#444444")


def _per_config_metrics(df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    """Return (tapas, delta_stay) for one csv, or (None, None) on bad data."""
    if "delta_stay_attr" not in df.columns or "n_stay_attrs" not in df.columns:
        return None, None
    if "delta_rem" not in df.columns:
        return None, None

    n = df["n_stay_attrs"].astype(float)
    n_sum = float(n.sum())
    if n_sum <= 0:
        return None, None

    delta_stay = float((df["delta_stay_attr"].astype(float) * n).sum() / n_sum)

    if "delta_add" in df.columns:
        delta_add_mean = float(df["delta_add"].astype(float).mean())
    else:
        delta_add_mean = 0.0
    delta_rem_mean = float(df["delta_rem"].astype(float).mean())
    tapas = delta_add_mean - delta_rem_mean

    return tapas, delta_stay


def collect_records() -> pd.DataFrame:
    rows: list[dict] = []
    for syn_dataset in SYN_DATASETS:
        ds_root = METRICS_ROOT / syn_dataset
        if not ds_root.is_dir():
            continue
        for model_name in MODEL_NAMES:
            model_root = ds_root / model_name
            if not model_root.is_dir():
                continue
            for seed_dir in sorted(model_root.iterdir()):
                if not seed_dir.is_dir():
                    continue
                seed = seed_dir.name
                for sae_dir in sorted(seed_dir.iterdir()):
                    if not sae_dir.is_dir():
                        continue
                    parsed = parse_sae_run_name(sae_dir.name)
                    if parsed is None or parsed.is_probe or parsed.is_k_sweep:
                        continue
                    if parsed.family not in TRAINED_FAMILIES + ("frozen", "random"):
                        continue

                    pert_dir = sae_dir / "pert"
                    if not pert_dir.is_dir():
                        continue

                    for csv_path in sorted(pert_dir.glob("*.csv")):
                        m = CSV_NAME_RE.match(csv_path.name)
                        if m is None:
                            continue
                        style = m.group("style")
                        top_k = int(m.group("k"))
                        try:
                            df = pd.read_csv(csv_path)
                        except Exception as e:
                            print(f"  skipping {csv_path.name}: {e}")
                            continue
                        if df.empty:
                            continue
                        tapas, delta_stay = _per_config_metrics(df)
                        if tapas is None:
                            continue
                        rows.append(
                            {
                                "dataset": syn_dataset,
                                "model": model_name,
                                "seed": seed,
                                "sae": parsed.family,
                                "dict_size": parsed.dict_size,
                                "matching_style": style,
                                "top_k": top_k,
                                "tapas": tapas,
                                "delta_stay": delta_stay,
                                "n_pairs": int(len(df)),
                            }
                        )
    return pd.DataFrame(rows)


def headline_summary(records: pd.DataFrame) -> str:
    if records.empty:
        return "No records — cannot compute summary."
    mean = float(records["delta_stay"].mean())
    std = float(records["delta_stay"].std(ddof=1))
    if len(records) >= 3 and records["delta_stay"].std() > 0 and records["tapas"].std() > 0:
        r, _ = pearsonr(records["delta_stay"].to_numpy(), records["tapas"].to_numpy())
    else:
        r = float("nan")
    return f"Δ_stay = {mean:.2f} ± {std:.2f}, r = {r:.2f}  (N={len(records)} configs)"


def plot_delta_stay_vs_dictsize(
    records: pd.DataFrame,
    save_dir: Path,
    matching_style: str,
    top_k: int,
):
    fig_records = records[
        (records["matching_style"] == matching_style)
        & (records["top_k"] == top_k)
        & (records["sae"].isin(TRAINED_FAMILIES))
    ].copy()
    if fig_records.empty:
        print(
            f"No rows for matching_style={matching_style} top_k={top_k}; "
            "skipping figure."
        )
        return

    datasets_present = [d for d in SYN_DATASETS if d in fig_records["dataset"].unique()]
    n_panels = len(datasets_present)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.6 * n_panels, 3.4), sharey=False
    )
    if n_panels == 1:
        axes = [axes]

    for ax, syn_dataset in zip(axes, datasets_present):
        d = fig_records[fig_records["dataset"] == syn_dataset]
        dict_sizes = sorted(d["dict_size"].dropna().unique().astype(int).tolist())
        pos_for_size = {int(s): i for i, s in enumerate(dict_sizes)}

        dataset_mean = float(
            fig_records[fig_records["dataset"] == syn_dataset]["delta_stay"].mean()
        )

        for family in TRAINED_FAMILIES:
            dd = d[d["sae"] == family].copy()
            if dd.empty:
                continue
            dd = (
                dd.groupby("dict_size", as_index=False)
                .agg(delta_stay=("delta_stay", "mean"))
                .sort_values("dict_size")
            )
            x = np.array(
                [pos_for_size[int(s)] for s in dd["dict_size"].to_numpy(int)],
                dtype=float,
            )
            ax.plot(
                x,
                dd["delta_stay"].to_numpy(float),
                color=_sae_color(family),
                marker="o",
                linewidth=1.8,
                markersize=6,
                label=pretty_sae_family(family),
            )

        ax.axhline(
            dataset_mean,
            linestyle="--",
            color="black",
            alpha=0.55,
            linewidth=1.2,
            label="Dataset mean",
        )
        ax.set_xticks(np.arange(len(dict_sizes)))
        ax.set_xticklabels([str(s) for s in dict_sizes])
        ax.set_title(pretty_syn_dataset(syn_dataset))
        ax.set_xlabel("Dictionary size")
        ax.set_ylabel(r"$\Delta_{\mathrm{stay}}$")
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[-1].get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    fig.legend(
        h2,
        l2,
        loc="lower center",
        ncol=min(len(l2), 5),
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        f"Δ_stay vs dict size (matching = {matching_style}, k={top_k})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"delta_stay_vs_dictsize_{matching_style}_top{top_k}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"Saved figure to {save_path}")


# Matching config used for the LaTeX table and scatter CSV
TABLE_STYLE = "bmp_f0.5"
TABLE_K = 3

_DATASET_PRETTY = {
    "syn_cub_attrs": "synCUB",
    "syn_coco": "synCOCO",
}


def plot_delta_stay_vs_tapas(records: pd.DataFrame, save_dir: Path) -> None:
    """Scatter: Δ_stay (x) vs TAPAScore (y), one panel per dataset, coloured by SAE family."""
    sub = records[
        (records["matching_style"] == TABLE_STYLE)
        & (records["top_k"] == TABLE_K)
        & (records["sae"].isin(TRAINED_FAMILIES))
    ].copy()
    if sub.empty:
        print(f"No rows for {TABLE_STYLE} k={TABLE_K} — skipping scatter plot.")
        return

    datasets_present = [d for d in SYN_DATASETS if d in sub["dataset"].unique()]
    n_panels = len(datasets_present)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 3.8), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, syn_dataset in zip(axes, datasets_present):
        d = sub[sub["dataset"] == syn_dataset]
        for family in TRAINED_FAMILIES:
            dd = d[d["sae"] == family]
            if dd.empty:
                continue
            ax.scatter(
                dd["delta_stay"].to_numpy(float),
                dd["tapas"].to_numpy(float),
                color=_sae_color(family),
                label=pretty_sae_family(family),
                s=45,
                alpha=0.85,
                zorder=3,
            )

        # Pearson r annotation
        xs = d["delta_stay"].to_numpy(float)
        ys = d["tapas"].to_numpy(float)
        if len(xs) >= 3 and xs.std() > 0 and ys.std() > 0:
            r, _ = pearsonr(xs, ys)
            ax.text(
                0.97, 0.96, f"$r = {r:.2f}$",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
            )

        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(_DATASET_PRETTY.get(syn_dataset, syn_dataset))
        ax.set_xlabel(r"$\Delta_{\mathrm{stay}}$")
        ax.set_ylabel("TAPAScore")
        ax.grid(True, alpha=0.2)

    handles, labels = axes[-1].get_legend_handles_labels()
    seen: set[str] = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    fig.legend(h2, l2, loc="lower center", ncol=min(len(l2), 4),
               frameon=True, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        rf"$\Delta_{{\mathrm{{stay}}}}$ vs TAPAScore "
        f"(matching = {TABLE_STYLE}, k={TABLE_K})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"delta_stay_vs_tapas_{TABLE_STYLE}_top{TABLE_K}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"Saved scatter to {save_path}")


def write_latex_table(records: pd.DataFrame, save_dir: Path) -> None:
    """Booktabs table: rows=SAE variant (+All), cols=synCUB / synCOCO."""
    sub = records[
        (records["matching_style"] == TABLE_STYLE)
        & (records["top_k"] == TABLE_K)
        & (records["sae"].isin(TRAINED_FAMILIES))
    ].copy()
    if sub.empty:
        print(f"No rows for {TABLE_STYLE} k={TABLE_K} — skipping LaTeX table.")
        return

    # One Δ_stay value per (sae, dict_size, dataset) — aggregate over dict sizes
    grp = sub.groupby(["sae", "dataset"])["delta_stay"]
    table = grp.agg(["mean", "std"]).reset_index()
    table["cell"] = table.apply(
        lambda r: f"${r['mean']:.2f} \\pm {r['std']:.2f}$", axis=1
    )
    pivot = table.pivot(index="sae", columns="dataset", values="cell")

    # Row order: paper's canonical family order
    family_order = [f for f in TRAINED_FAMILIES if f in pivot.index]
    pivot = pivot.reindex(family_order)

    # "All" row: mean ± std across all families × dict_sizes per dataset
    all_row = {}
    for ds in SYN_DATASETS:
        vals = sub[sub["dataset"] == ds]["delta_stay"]
        if not vals.empty:
            all_row[ds] = f"${vals.mean():.2f} \\pm {vals.std(ddof=1):.2f}$"
        else:
            all_row[ds] = "---"

    col_order = [ds for ds in SYN_DATASETS if ds in pivot.columns]
    col_headers = " & ".join(_DATASET_PRETTY.get(ds, ds) for ds in col_order)

    lines = [
        "\\begin{table}[ht]",
        "  \\centering",
        f"  \\caption{{$\\Delta_{{\\mathrm{{stay}}}}$ (mean$\\pm$std over dict sizes) "
        f"for matching style \\texttt{{{TABLE_STYLE}}}, $k={TABLE_K}$.}}",
        "  \\label{tab:delta_stay}",
        "  \\begin{tabular}{l" + "c" * len(col_order) + "}",
        "    \\toprule",
        f"    SAE Variant & {col_headers} \\\\",
        "    \\midrule",
    ]
    for fam in family_order:
        row_cells = " & ".join(
            pivot.at[fam, ds] if ds in pivot.columns else "---" for ds in col_order
        )
        lines.append(f"    {pretty_sae_family(fam)} & {row_cells} \\\\")
    lines += [
        "    \\midrule",
        "    All & "
        + " & ".join(all_row.get(ds, "---") for ds in col_order)
        + " \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    tex_path = save_dir / "delta_stay_table.tex"
    tex_path.write_text(tex)
    print(f"Wrote LaTeX table to {tex_path}")
    print(tex)


def write_scatter_csv(records: pd.DataFrame, save_dir: Path) -> None:
    """Per-config CSV for scatter plot: (variant, dict_size, dataset, delta_stay, tapas)."""
    sub = records[
        (records["matching_style"] == TABLE_STYLE)
        & (records["top_k"] == TABLE_K)
        & (records["sae"].isin(TRAINED_FAMILIES))
    ][["sae", "dict_size", "dataset", "delta_stay", "tapas"]].copy()
    sub = sub.rename(columns={"sae": "variant", "tapas": "TAPAScore"})
    sub["dataset"] = sub["dataset"].map(lambda d: _DATASET_PRETTY.get(d, d))
    csv_path = save_dir / "delta_stay_scatter.csv"
    sub.to_csv(csv_path, index=False)
    print(f"Wrote scatter CSV to {csv_path}")


def main():
    records = collect_records()
    if records.empty:
        print("No perturbation csvs were parsed. Check METRICS_ROOT.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "delta_stay_records.csv"
    records.to_csv(csv_path, index=False)
    print(f"Wrote per-config table to {csv_path}")

    print("\n=== Headline summary (all configs) ===")
    print(headline_summary(records))

    print("\n=== Per-dataset summary ===")
    for ds, sub in records.groupby("dataset"):
        print(f"  {ds:14s}  {headline_summary(sub)}")

    print("\n=== Per-(matching_style, k) summary ===")
    for (style, k), sub in records.groupby(["matching_style", "top_k"]):
        print(f"  {style:14s} k={k:<3d} {headline_summary(sub)}")

    print("\n=== Sanity check: rebuttal targets ===")
    for ds, target_mean, target_std, target_r in [
        ("syn_coco", 0.14, 0.12, -0.53),
        ("syn_cub_attrs", 0.15, 0.10, -0.40),
    ]:
        sub = records[records["dataset"] == ds]
        if sub.empty:
            print(f"  {ds}: no data")
            continue
        m = sub["delta_stay"].mean()
        s = sub["delta_stay"].std(ddof=1)
        r, _ = pearsonr(sub["delta_stay"].to_numpy(), sub["tapas"].to_numpy())
        flag_m = " *** DEVIATED" if abs(m - target_mean) > 0.03 else ""
        flag_r = " *** DEVIATED" if abs(r - target_r) > 0.10 else ""
        print(f"  {ds}: Δ_stay={m:.2f}±{s:.2f} (target {target_mean}±{target_std}){flag_m}"
              f"  r={r:.2f} (target {target_r}){flag_r}")

    write_latex_table(records, OUT_DIR)
    write_scatter_csv(records, OUT_DIR)
    plot_delta_stay_vs_tapas(records, OUT_DIR)

    for matching_style, top_k in PLOT_CONFIGS:
        plot_delta_stay_vs_dictsize(
            records, OUT_DIR, matching_style=matching_style, top_k=top_k
        )


if __name__ == "__main__":
    main()
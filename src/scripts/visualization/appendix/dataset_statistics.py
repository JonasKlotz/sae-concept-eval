from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datamodule.CUB_syn_dataset import CUBSyntheticDataset
from datamodule.coco_dataset import COCOSynDataset

OUT_DIR = Path("/home/jokl/PycharmProjects/rs_concepts/outputs/figures/appendix/dataset_statistics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_cub_attribute_counts(dataset):
    attribute_counts = defaultdict(int)

    for sample in dataset:
        removed_attr_string = sample[6]
        added_attr_string = sample[7]
        attribute_counts[removed_attr_string] += 1
        attribute_counts[added_attr_string] += 1

    return dict(attribute_counts)


def collect_coco_attribute_counts(dataset):
    attribute_counts = defaultdict(int)

    for sample in dataset:
        attr_string = sample[4]
        attribute_counts[attr_string] += 1

    return dict(attribute_counts)


def print_dataset_stats(dataset_name, num_samples, attribute_counts):
    counts = list(attribute_counts.values())
    avg_pairs_per_attribute = np.mean(counts)

    print(f"\n{dataset_name}:")
    print(f"Number of samples: {num_samples}")
    print(f"Number of unique attributes: {len(attribute_counts)}")
    print(f"Average number of pairs per attribute: {avg_pairs_per_attribute:.2f}")


def plot_attribute_histogram(attribute_counts, dataset_name, bins=20):
    counts = list(attribute_counts.values())
    avg_pairs_per_attribute = np.mean(counts)

    plt.figure(figsize=(6, 4))
    sns.histplot(counts, bins=bins, kde=True)

    plt.xlabel("Number of image pairs per attribute")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Attribute Pairs in {dataset_name}")
    plt.axvline(
        float(avg_pairs_per_attribute),
        linestyle="--",
        label=f"Mean = {avg_pairs_per_attribute:.2f}"
    )
    plt.legend()
    plt.tight_layout()
    slug = dataset_name.lower().replace(" ", "_")
    plt.savefig(OUT_DIR / f"{slug}_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# def plot_sorted_attribute_barplot(attribute_counts, dataset_name, top_k=None):
#     sorted_items = sorted(attribute_counts.items(), key=lambda x: x[1], reverse=True)
#
#     if top_k is not None:
#         sorted_items = sorted_items[:top_k]
#
#     attrs, counts = zip(*sorted_items)
#
#     plt.figure(figsize=(max(10, len(attrs) * 0.3), 5))
#     sns.barplot(x=list(attrs), y=list(counts))
#
#     plt.xticks(rotation=90)
#     plt.xlabel("Attribute")
#     plt.ylabel("Number of image pairs")
#     if top_k is None:
#         plt.title(f"{dataset_name} Attribute Distribution (sorted)")
#     else:
#         plt.title(f"Top {top_k} Attributes in {dataset_name}")
#
#     plt.tight_layout()
#     plt.show()
def plot_sorted_attribute_barplot(attribute_counts, dataset_name, top_k=None, split_threshold=45):
    sorted_items = sorted(attribute_counts.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        sorted_items = sorted_items[:top_k]

    attrs, counts = zip(*sorted_items)

    n_attrs = len(attrs)

    # If many attributes (e.g. COCO), split into two plots
    if n_attrs > split_threshold:
        midpoint = n_attrs // 2

        attrs_1, counts_1 = attrs[:midpoint], counts[:midpoint]
        attrs_2, counts_2 = attrs[midpoint:], counts[midpoint:]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

        sns.barplot(x=list(attrs_1), y=list(counts_1), ax=axes[0])
        axes[0].set_xticklabels(attrs_1, rotation=90)
        axes[0].set_xlabel("Attribute")
        axes[0].set_ylabel("Number of image pairs")
        axes[0].set_title(f"{dataset_name} Attribute Distribution (1/2)")

        sns.barplot(x=list(attrs_2), y=list(counts_2), ax=axes[1])
        axes[1].set_xticklabels(attrs_2, rotation=90)
        axes[1].set_xlabel("Attribute")
        axes[1].set_ylabel("Number of image pairs")
        axes[1].set_title(f"{dataset_name} Attribute Distribution (2/2)")

        plt.tight_layout()
        slug = dataset_name.lower().replace(" ", "_")
        fig.savefig(OUT_DIR / f"{slug}_barplot.pdf", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    else:
        fig = plt.figure(figsize=(max(10, n_attrs * 0.3), 5))
        sns.barplot(x=list(attrs), y=list(counts))

        plt.xticks(rotation=90)
        plt.xlabel("Attribute")
        plt.ylabel("Number of image pairs")
        plt.title(f"{dataset_name} Attribute Distribution")

        plt.tight_layout()
        slug = dataset_name.lower().replace(" ", "_")
        fig.savefig(OUT_DIR / f"{slug}_barplot.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

def main():
    data_root = Path("/home/jokl/data/")

    # syn_cub = CUBSyntheticDataset(
    #     root_dir=str(data_root / "syn_cub_dataset"),
    #     transform=None,
    # )
    syn_coco = COCOSynDataset(
        root=str(data_root / "syn_coco_dataset"),
        transform=None,
    )

    # attributes_manipulated_cub = collect_cub_attribute_counts(syn_cub)
    attributes_manipulated_coco = collect_coco_attribute_counts(syn_coco)

    # print_dataset_stats(
    #     dataset_name="CUB Synthetic Dataset",
    #     num_samples=len(syn_cub),
    #     attribute_counts=attributes_manipulated_cub,
    # )
    print_dataset_stats(
        dataset_name="COCO Synthetic Dataset",
        num_samples=len(syn_coco),
        attribute_counts=attributes_manipulated_coco,
    )

    # plot_attribute_histogram(
    #     attribute_counts=attributes_manipulated_cub,
    #     dataset_name="CUB Synthetic Dataset",
    # )
    # plot_sorted_attribute_barplot(
    #     attribute_counts=attributes_manipulated_cub,
    #     dataset_name="CUB Synthetic Dataset",
    # )

    plot_attribute_histogram(
        attribute_counts=attributes_manipulated_coco,
        dataset_name="COCO Synthetic Dataset",
    )
    plot_sorted_attribute_barplot(
        attribute_counts=attributes_manipulated_coco,
        dataset_name="COCO Synthetic Dataset",
    )


if __name__ == "__main__":
    main()
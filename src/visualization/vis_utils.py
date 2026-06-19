import numpy as np
import pandas as pd
import rootutils
from matplotlib import pyplot as plt

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


def index_to_label_dict(dataset_name, data_path=(root / "data")) -> dict:
    if dataset_name == "CUB":
        path = f"{data_path}/classes.txt"
        df = pd.read_csv(path, sep=" ", names=["img_id", "class_name"])
        # drop image id
        df = df.drop(columns=["img_id"])

        # convert to dict
        label_dict = df.to_dict()["class_name"]
    elif dataset_name == "COCO":
        from datamodule.coco_dataset import COCO_IDX2NAME

        return COCO_IDX2NAME
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return label_dict


if __name__ == "__main__":
    print(index_to_label_dict("CUB"))


def plot_batch(images, labels):
    """
    Plots a batch of images and their corresponding labels.
    Applies per-image min-max scaling for visualization.
    """
    images = images.detach().cpu().numpy()  # shape: (B, C, H, W)
    labels = labels.detach().cpu().numpy()  # shape: (B, H, W)
    batch_size = images.shape[0]

    fig, axes = plt.subplots(batch_size, 2, figsize=(8, 4 * batch_size))
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(batch_size):
        # Min-max scale the image
        img = images[i]
        img_min, img_max = img.min(), img.max()
        img_scaled = (img - img_min) / (img_max - img_min + 1e-8)
        # Convert from CHW to HWC for plotting
        img_scaled = img_scaled.transpose(1, 2, 0)
        axes[i, 0].imshow(img_scaled)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(labels[i], cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Label")
    plt.tight_layout()
    plt.show()

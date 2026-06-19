from sklearn.tree import DecisionTreeClassifier
import joblib
from typing import Dict
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import logging
import os
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_root_node(
    model_file: str = None, tree_model: DecisionTreeClassifier = None
) -> Dict[str, int]:
    """
    Load a trained DecisionTreeClassifier from a file and extract the root node information.

    Parameters:
        model_file (str): Path to the file containing the trained DecisionTreeClassifier.

    Returns:
        dict: Information about the root node, including feature index, threshold,
              and impurity.
    """
    if tree_model is None:
        # Load the model from the file
        tree_model = joblib.load(model_file)

    # Check if the model is fitted
    if not hasattr(tree_model, "tree_"):
        raise ValueError("The tree model is not fitted yet.")

    # Access the underlying tree structure
    tree = tree_model.tree_

    # Extract root node details
    root_index = 0
    root_feature = tree.feature[root_index]  # Feature index used for splitting
    root_threshold = tree.threshold[root_index]  # Threshold for the split
    root_impurity = tree.impurity[root_index]  # Impurity at the root node

    return {
        "feature_index": root_feature.item(),
        "threshold": root_threshold.item(),
        "impurity": root_impurity.item(),
    }


def get_tree_stats(
    file: str = None, clf: DecisionTreeClassifier = None
) -> pd.DataFrame:
    # clf: tree._classes.DecisionTreeClassifier = pickle.load(file)
    assert (file is None and clf is not None) or (file is not None and clf is None), (
        "Both file and clf are None, one needs to be set!"
    )

    if clf is None:
        with open(file, "rb") as f:
            clf: DecisionTreeClassifier = pickle.load(f)

    res = defaultdict(dict)
    for i in range(1, clf.tree_.max_depth + 2):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        ind_root = np.where(clf.tree_.compute_node_depths() == 1)
        values_root = (
            clf.tree_.weighted_n_node_samples[ind_root][:, None]
            * clf.tree_.value[ind_root][:, 0, :]
        ).sum(0)
        ind = np.where(clf.tree_.compute_node_depths() == i)
        values = (
            clf.tree_.weighted_n_node_samples[ind][:, None]
            * clf.tree_.value[ind][:, 0, :]
        ).sum(0)
        diff = values_root - values
        tn += diff[0]
        tp += diff[1]
        for pair in (
            clf.tree_.weighted_n_node_samples[ind][:, None]
            * clf.tree_.value[ind][:, 0, :]
        ):
            if pair[0] >= pair[1]:  # Predict negative (0)
                tn += pair[0]  # ✅ True Negatives (correctly classified as 0)
                fn += pair[1]  # ❌ False Negatives (misclassified as 0, should be 1)
            else:  # Predict positive (1)
                fp += pair[0]  # ❌ False Positives (misclassified as 1, should be 0)
                tp += pair[1]  # ✅ True Positives (correctly classified as 1)

        y_pred = (
            [1 for _ in range(int(tp))]
            + [0 for _ in range(int(tn))]
            + [1 for _ in range(int(fp))]
            + [0 for _ in range(int(fn))]
        )
        y_true = (
            [1 for _ in range(int(tp))]
            + [0 for _ in range(int(tn))]
            + [0 for _ in range(int(fp))]
            + [1 for _ in range(int(fn))]
        )

        F1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        F1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        F1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = 0
        used_feats = set(
            clf.tree_.feature[np.where(clf.tree_.compute_node_depths() < i)].tolist()
        )
        used_feats.discard(-2)

        res[i - 1] = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "auc": auc,
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1_macro": F1_macro,
            "Recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "Precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "F1_micro": F1_micro,
            "Recall_micro": recall_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "Precision_micro": precision_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "F1_binary": F1_binary,
            "Recall_binary": recall_score(
                y_true, y_pred, average="binary", zero_division=0
            ),
            "Precision_binary": precision_score(
                y_true, y_pred, average="binary", zero_division=0
            ),
            "Nodes": np.where(clf.tree_.compute_node_depths() < i)[0].size,
            "Leafs": (
                clf.tree_.children_right[np.where(clf.tree_.compute_node_depths() < i)]
                == -1
            ).sum()
            + len(np.where(clf.tree_.compute_node_depths() == i)[0]),
            "used_feats_num": len(used_feats),
            "used_feats": used_feats,
        }
    df = pd.DataFrame(res).T
    df.index.rename("depth", inplace=True)
    return df


if __name__ == "__main__":
    print("Starting")
    load_from_ckpt = True
    model_name = "llama3-l24576-b11-k2048"

    ckpt_dirs = {
        "Checkpoint Path": "Result Dir",
    }
    for ckpt_dir, res_path in ckpt_dirs.items():
        for i in [0, 5]:
            logger.info(f"{model_name}_s{i}")
            k = int(model_name.split("-")[-1][1:])

            skip = False

            # RTP Tree Stats
            file = f"./llama3_SAE/SAE_eval/{res_path}/rtp_tree_{model_name}_s{i}"
            if os.path.isfile(file + "_stats.csv"):
                logger.info(f"{res_path}/rtp_tree_{model_name}_s{i}_stats.csv exists")
            elif not os.path.isfile(file + ".pkl"):
                logger.info(
                    f"{res_path}/rtp_tree_{model_name}_s{i}.pkl does not exists"
                )
            else:
                df = get_tree_stats(file=f"{file}.pkl")
                df.to_csv(f"{file}_stats.csv")

            # RTP Tree ALL
            file = f"./llama3_SAE/SAE_eval/{res_path}/rtp_tree_test_{model_name}_s{i}"
            if os.path.isfile(file + "_stats.csv"):
                logger.info(
                    f"{res_path}/rtp_tree_test_{model_name}_s{i}_stats.csv exists"
                )
            elif not os.path.isfile(file + ".pkl"):
                logger.info(
                    f"{res_path}/rtp_tree_test_{model_name}_s{i}.pkl does not exists"
                )
            else:
                df = get_tree_stats(file=f"{file}.pkl")
                df.to_csv(f"{file}_stats.csv")

            # SP Tree valid
            file = f"./llama3_SAE/SAE_eval/{res_path}/sp_tree_valid_{model_name}_s{i}"
            if os.path.isfile(file + "_stats.csv"):
                logger.info(
                    f"{res_path}/sp_tree_valid_{model_name}_s{i}_stats.csv exists"
                )
            elif not os.path.isfile(file + ".pkl"):
                logger.info(
                    f"{res_path}/sp_tree_valid_{model_name}_s{i}.pkl does not exists"
                )
            else:
                df = get_tree_stats(file=f"{file}.pkl")
                df.to_csv(f"{file}_stats.csv")

            # SP Tree test
            file = f"./llama3_SAE/SAE_eval/{res_path}/sp_tree_test_{model_name}_s{i}"
            if os.path.isfile(file + "_stats.csv"):
                logger.info(
                    f"{res_path}/sp_tree_test_{model_name}_s{i}_stats.csv exists"
                )
            elif not os.path.isfile(file + ".pkl"):
                logger.info(
                    f"{res_path}/sp_tree_test_{model_name}_s{i}.pkl does not exists"
                )
            else:
                df = get_tree_stats(file=f"{file}.pkl")
                df.to_csv(f"{file}_stats.csv")

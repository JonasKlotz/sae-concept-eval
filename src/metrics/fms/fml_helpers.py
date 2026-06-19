import pandas as pd
import re
from glob import glob
import pandas as pd
from glob import glob
from typing import List, Dict, Optional


def load_local_tree_stats_data(df, concept) -> pd.DataFrame:
    df["concept"] = concept
    # Compute MS_local
    df["MS_local"] = df.apply(
        lambda x: 2
        * (
            df[(df["Nodes"] == 1) & (df["num_cuts"] == 0)]["Accuracy"].mean()
            - x["Accuracy"]
        )
        if x["num_cuts"] != 0
        else None,
        axis=1,
    )

    return df[df["Nodes"] == 1][
        [
            "concept",
            "num_cuts",
            "MS_local",
        ]
    ]


def load_tree_stats_data(
    df,
    concept,
) -> pd.DataFrame:
    df["concept"] = concept

    df["MS_global"] = df.apply(
        lambda x: 1
        - (
            sum(
                df[(df["Nodes"] != 1) & (df["concept"] == x["concept"])]["Accuracy"]
                - df[(df["Nodes"] == 1) & (df["concept"] == x["concept"])][
                    "Accuracy"
                ].item()
            )
            / len(df[(df["Nodes"] != 1) & (df["concept"] == x["concept"])]["Accuracy"])
        )
        if x["Nodes"] == 1
        else None,
        axis=1,
    )

    return df[df["Nodes"] == 1][
        [
            "Accuracy",
            "concept",
            "MS_global",
        ]
    ]

import json
import re
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import rootutils
from pathlib import Path

from numpy import ndarray, dtype
from pandas import DataFrame, Series

from metrics.metric_utils import get_topk_matching

root = Path(rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False))


# ---------------------------------------------------------------------------
# SAE run-name parsing
# ---------------------------------------------------------------------------
# {sae_run} folder under metrics/{dataset}/{model}/{seed}/ has these forms:
#   {family}_{dict_size}              standard, e.g. "topk_256"
#   {family}_{dict_size}_k{top_k}     K-sparsity sweep, e.g. "topk_1024_k32"
#   "linear_probe"                    probe baseline (no dict size)
#   "matching"                        legacy stale layout — ignored
#
# All discovery / aggregation flows through parse_sae_run_name. Callers can
# opt in to k-sweep / probe runs via flags; default keeps main figures clean.

@dataclass
class SAERunId:
    family: str
    dict_size: Optional[int]
    top_k: Optional[int]
    is_k_sweep: bool
    is_probe: bool
    raw: str


_K_SWEEP_RE = re.compile(r"^(?P<fam>[a-z]+)_(?P<size>\d+)_k(?P<k>\d+)$")
_STANDARD_RE = re.compile(r"^(?P<fam>[a-z]+)_(?P<size>\d+)$")


def parse_sae_run_name(name: str) -> Optional[SAERunId]:
    if name == "linear_probe":
        return SAERunId("linear_probe", None, None, False, True, name)
    if name == "matching" or name.endswith(".py"):
        return None
    m = _K_SWEEP_RE.match(name)
    if m:
        return SAERunId(m["fam"], int(m["size"]), int(m["k"]), True, False, name)
    m = _STANDARD_RE.match(name)
    if m:
        return SAERunId(m["fam"], int(m["size"]), None, False, False, name)
    return None


# nnomp + probe TAPAS csvs sit under the non-syn dataset folder by accident
# (calculate_omp_tapa.py writes to cfg.paths.metrics_dir before the cfg.dataset
# rename). These mappings let us merge them into the syn-side dataframe.
_SYN_FOR_NONSYN = {"CUB_attrs": "syn_cub_attrs", "COCO": "syn_coco"}
_NONSYN_FOR_SYN = {v: k for k, v in _SYN_FOR_NONSYN.items()}


def nonsyn_for_syn(name: str) -> str:
    return _NONSYN_FOR_SYN[name]


def syn_for_nonsyn(name: str) -> str:
    return _SYN_FOR_NONSYN[name]


@dataclass
class RunKey:
    dataset: str
    model: str
    seed: str
    sae: str
    dict_size: Optional[int]
    path: Path
    metric_name: str
    top_k: Optional[int] = None
    is_probe: bool = False


def parse_run_dir(
    run_dir: Path,
    *,
    include_k_sweep: bool = False,
    include_probe: bool = False,
) -> Optional[RunKey]:
    """
    metrics_root / dataset / model / seed / saeName / metric
    """
    try:
        metric_name = run_dir.name
        sae_part = run_dir.parent.name
        parsed = parse_sae_run_name(sae_part)
        if parsed is None:
            return None
        if parsed.is_k_sweep and not include_k_sweep:
            return None
        if parsed.is_probe and not include_probe:
            return None

        seed = run_dir.parent.parent.name
        model = run_dir.parent.parent.parent.name
        dataset = run_dir.parent.parent.parent.parent.name
        return RunKey(
            dataset=dataset,
            model=model,
            seed=seed,
            sae=parsed.family,
            dict_size=parsed.dict_size,
            path=run_dir,
            metric_name=metric_name,
            top_k=parsed.top_k,
            is_probe=parsed.is_probe,
        )
    except Exception:
        return None


def load_csv(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    df = pd.read_csv(path)

    result_dict = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # ensure a concrete numeric dtype and handle NaNs if needed
            values = df[col].to_numpy(dtype=np.float32)
            result_dict[col] = torch.from_numpy(values)
        else:
            # non-numeric columns cannot be converted directly
            # options: drop, keep as list, or encode
            result_dict[col] = df[col].tolist()

    return result_dict


def load_pt(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(e)
        return None


def load_all_metrics(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for fname in list(run_dir.glob("*")):
        if fname.suffix == ".pt":
            out[fname.stem] = load_pt(fname)
        elif fname.suffix == ".csv":
            out[fname.stem] = load_csv(fname)
    return out


def load_nested_pt_tree(root_path):
    """
    Recursively traverse a directory and build a nested dictionary:

    - If an entry is a `.pt` file: load it with torch.load() and store under
      key = filename (without extension).
    - If an entry is a directory: create a nested dictionary using the directory
      name as key and recurse into it.

    Parameters
    ----------
    root_path : str or Path
        Root directory to traverse.

    Returns
    -------
    dict
        Nested dictionary containing loaded `.pt` files.
    """
    root_path = Path(root_path)
    result = {}

    for item in sorted(root_path.iterdir()):
        if item.is_dir():
            result[item.name] = load_nested_pt_tree(item)
        elif item.is_file() and item.suffix == ".pt":
            key = item.stem
            try:
                result[key] = torch.load(item, map_location="cpu", weights_only=False)
            except:
                print(f"Failed to load {item} with torch.load, skipping.")

    return result


def keep_ground_truth_per_method(d):
    out = {}
    for method, sub in d.items():
        if isinstance(sub, dict) and "ground_truth" in sub:
            out[method] = sub["ground_truth"]
    return out


def extract_matching_scores(
    data: dict, k: int, sae_names
) -> tuple[
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
]:
    dict_sizes = []
    f1_scores = []
    jac_scores = []
    bmp_scores = []
    mi_scores = []

    for name in sae_names:
        dict_size = int(re.findall(r"\d+", name)[0])
        # if dict_size == 4096:
        #     print("PLOTDATAUTILS WARNING REMOVE AS WE DONT HAVE ALL DATA PLEASE CHANGE LATER")
        #     continue
        dict_sizes.append(dict_size)

        sae_dict = data[name]

        f1_sae = torch.as_tensor(sae_dict["f1_max_scores"]).float().mean().item()
        jac_sae = torch.as_tensor(sae_dict["jac_max_scores"]).float().mean().item()
        mi_sae = torch.as_tensor(sae_dict["mi_max_scores"]).float().mean().item()

        bmp_sae_matrix = torch.as_tensor(
            sae_dict.get("bmp_rec_F1_matrix_f0.5")
        ).float()  # (A, steps)
        if bmp_sae_matrix.ndim != 2:
            raise ValueError(
                f"bmp_rec_F1_matrix for {name} must be 2D, got shape {tuple(bmp_sae_matrix.shape)}"
            )
        if bmp_sae_matrix.shape[1] < k:
            raise ValueError(
                f"bmp_rec_F1_matrix for {name} has only {bmp_sae_matrix.shape[1]} steps < k={k}"
            )

        # Enforce reconstruction budget k
        bmp_per_attr = torch.max(bmp_sae_matrix[:, :k], dim=1)[0]  # (A,)
        bmp_sae = bmp_per_attr.mean().item()

        f1_scores.append(f1_sae)
        jac_scores.append(jac_sae)
        mi_scores.append(mi_sae)
        bmp_scores.append(bmp_sae)

    # sort by dict size
    order = np.argsort(dict_sizes)
    dict_sizes = np.array(dict_sizes)[order]
    f1_scores = np.array(f1_scores)[order]
    jac_scores = np.array(jac_scores)[order]
    mi_scores = np.array(mi_scores)[order]
    bmp_scores = np.array(bmp_scores)[order]
    return bmp_scores, dict_sizes, f1_scores, jac_scores, mi_scores


def load_matching_data(metrics_root: str, dataset="CUB_attrs", model="CLIP-ViT-L-14", seed="42") -> dict[Any, Any]:
    loaded = load_nested_pt_tree(metrics_root)
    cub_data = loaded[dataset][model][seed]
    # remove all subsubdicts that are not 'ground_truth'
    cub_data = keep_ground_truth_per_method(cub_data)
    return cub_data


def discover_runs_cub_pert(
    metrics_root: Path,
    *,
    include_k_sweep: bool = False,
    include_probe: bool = False,
    rewrite_dataset: Optional[str] = None,
):
    """
    Walk depth to run directories. Returns a dict of loaded artifacts keyed by
    (dataset, model, seed, sae, dict_size, metric_name).

    rewrite_dataset: if given, override the parsed dataset name in the keys.
        Useful when nnomp/probe TAPAS csvs live under the non-syn dataset folder
        but should be merged into the syn-side dataframe — pass the syn name.
    """
    run_dirs = metrics_root.glob("*/*/*/*")
    loaded = {}
    for rd in run_dirs:
        if not rd.is_dir():
            continue
        rk = parse_run_dir(
            rd,
            include_k_sweep=include_k_sweep,
            include_probe=include_probe,
        )
        if rk is None:
            continue
        dataset = rewrite_dataset if rewrite_dataset is not None else rk.dataset
        key = (dataset, rk.model, rk.seed, rk.sae, rk.dict_size, rk.metric_name, rk.top_k)
        artifacts = load_all_metrics(rd)
        if len(artifacts) == 0:
            continue
        loaded[key] = artifacts

    return loaded


def parse_df_pert(loaded: dict[Any, Any], score_calculation) -> DataFrame:
    rows = []
    for config_key, subdict in loaded.items():
        for metric_name, inner in subdict.items():
            dataset_name = config_key[0]
            eps = 1e-8  # small constant to prevent division by zero if needed
            if dataset_name == "syn_cub_attrs":
                # if 'delta_add_max' not in inner or len(inner['delta_add_max']) == 0:
                #     continue  # skip if any required metric is missing delta_add_max_first
                if 'delta_add_max_first_bin' not in inner or 'delta_rem_max_first_bin' not in inner:
                    # if score_calculation == "bin_max_then_delta":
                    #     print(f"{config_key} failed: "
                    #                      f"we need 'delta_add_max_first_bin' to calculate the bin_max_then_delta score, "
                    #                      "but it is missing in the metrics for this run.")
                    continue  # skip if any required metric is missing delta_add_max_first
                if  config_key[3] =="linear_probe":
                    print()
                delta_add_mean = (
                    inner["delta_add"].mean().item()
                )  # should be positive
                delta_rem_mean = (
                    inner["delta_rem"].mean().item()
                )  # should be negative
                delta_stay_mean = inner["delta_stay"].mean().item()



                delta_stay_max = inner["delta_stay_max"].mean().item()
                delta_add_max = inner["delta_add_max"].mean().item()
                delta_rem_min = inner["delta_rem_min"].mean().item()

                delta_add_max_first = inner["delta_add_max_first"].mean().item()
                delta_rem_min_first = inner["delta_rem_min_first"].mean().item()
                delta_stay_max_first = inner["delta_stay_max_first"].mean().item()

                delta_add_max_first_bin = inner["delta_add_max_first_bin"].mean().item()
                delta_rem_min_first_bin = inner["delta_rem_min_first_bin"].mean().item()
                delta_rem_max_first_bin = inner["delta_rem_max_first_bin"].mean().item()
                #- "max_then_delta": for each run, take the max first then calculate delta_add = max(z') - max(z) ..
                #- "delta_then_max": for each run, take the delta first then the max = max(z' - z) ...
                #- "delta_then_mean": for each run, take the delta first then the mean = mean(z' - z) ...
                #- "bin_max_then_delta": for each run, binarize the z by > 0,  then delta_add = max(z' at add_idx) - max(z at add_idx)) ...

                if score_calculation == "delta_then_mean":
                    mean_val = delta_add_mean - delta_rem_mean
                # mean_val = (0 - delta_rem_mean) / delta_stay_mean + eps
                elif score_calculation == "delta_then_max":
                    mean_val = delta_add_max - delta_rem_min
                elif score_calculation == "max_then_delta":
                    mean_val = delta_add_max_first - delta_rem_min_first
                elif score_calculation == "bin_max_then_delta":
                    mean_val = delta_add_max_first_bin - delta_rem_max_first_bin
                else:
                    raise ValueError(f"Unknown score_calculation method: {score_calculation}")

                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": config_key[1],
                        "seed": config_key[2],
                        "sae": config_key[3],
                        "dict_size": config_key[4],
                        "top_k": config_key[6],
                        "metric_name": metric_name,
                        "score_mean": mean_val,
                        "delta_add_mean": delta_add_mean,
                        "delta_rem_mean": delta_rem_mean,
                        "delta_stay_mean": delta_stay_mean,
                        "delta_stay_max": delta_stay_max,
                        "delta_add_max": delta_add_max,
                        "delta_rem_min": delta_rem_min,
                        "delta_add_max_first": delta_add_max_first,
                        "delta_rem_min_first": delta_rem_min_first,
                        "delta_stay_max_first": delta_stay_max_first,
                    }
                )
            elif dataset_name == "syn_coco":  # we dont have delta add
                if 'delta_rem_max_first_bin' not in inner or len(inner['delta_rem_max_first_bin']) == 0:
                    continue  # skip if any required metric is missing delta_add_max_first
                delta_rem_mean = (
                    inner["delta_rem"].mean().item()
                )  # should be negative
                delta_stay_mean = inner["delta_stay"].mean().item()

                delta_stay_max = inner["delta_stay_max"].mean().item()
                delta_rem_min = inner["delta_rem_min"].mean().item()

                delta_rem_min_first = inner["delta_rem_min_first"].mean().item()
                delta_stay_max_first = inner["delta_stay_max_first"].mean().item()

                delta_rem_min_first_bin = inner["delta_rem_min_first_bin"].mean().item()
                delta_rem_max_first_bin = inner["delta_rem_max_first_bin"].mean().item()
                if score_calculation == "delta_then_mean":
                    mean_val = - delta_rem_mean
                elif score_calculation == "delta_then_max":
                    mean_val = - delta_rem_min
                elif score_calculation == "max_then_delta":
                    mean_val = - delta_rem_min_first
                elif score_calculation == "bin_max_then_delta":
                    mean_val = - delta_rem_max_first_bin
                else:
                    raise ValueError(f"Unknown score_calculation method: {score_calculation}")

                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": config_key[1],
                        "seed": config_key[2],
                        "sae": config_key[3],
                        "dict_size": config_key[4],
                        "top_k": config_key[6],
                        "metric_name": metric_name,
                        "score_mean": mean_val,
                        "delta_rem_mean": delta_rem_mean,
                        "delta_stay_mean": delta_stay_mean,

                    }
                )

    df = pd.DataFrame(rows)
    cols_to_drop = ["seed", "dataset"]
    df_final = df.drop(columns=cols_to_drop)

    # ensure types and ordering
    df = df_final.copy()
    df["dict_size"] = pd.to_numeric(df["dict_size"])
    df = df.sort_values(["sae", "dict_size"])
    return df


def load_perturbation_dataframe(
    metrics_dir: Path,
    score_calculation: str = "bin_max_then_delta",
    *,
    also_search_root: Optional[Path] = None,
    include_k_sweep: bool = False,
    include_probe: bool = False,
) -> Series | DataFrame | Any:
    """
    Load perturbation (TAPAS) metrics from one or two paths.

    metrics_dir: root for the syn dataset, e.g. .../metrics/syn_cub_attrs
    also_search_root: optional non-syn root, e.g. .../metrics/CUB_attrs.
        Discovered runs there are re-tagged with the syn-equivalent dataset
        name so parse_df_pert's syn-name dispatch keeps working. This is how
        we pull in nnomp pert csvs and probe pert csvs that the pipeline
        accidentally writes under the non-syn folder.
    """
    loaded = discover_runs_cub_pert(
        metrics_dir,
        include_k_sweep=include_k_sweep,
        include_probe=include_probe,
    )
    if also_search_root is not None and also_search_root.exists():
        nonsyn_name = also_search_root.name
        try:
            syn_name = syn_for_nonsyn(nonsyn_name)
        except KeyError:
            syn_name = metrics_dir.name
        extra = discover_runs_cub_pert(
            also_search_root,
            include_k_sweep=include_k_sweep,
            include_probe=include_probe,
            rewrite_dataset=syn_name,
        )
        # The non-syn root also contains ground_truth/ etc. subdirs whose
        # contents are not pert csvs; only merge the pert-family entries.
        # When the same (dataset, model, seed, sae, dict_size, "pert") key
        # exists on both sides, MERGE the inner artifact dicts — the non-syn
        # side typically only has nnomp_top*.csv while the syn side has bmp/f1.
        for key, val in extra.items():
            if key[5] != "pert":
                continue
            if key in loaded and isinstance(loaded[key], dict) and isinstance(val, dict):
                merged = dict(loaded[key])
                merged.update(val)
                loaded[key] = merged
            else:
                loaded[key] = val

    df = parse_df_pert(loaded, score_calculation)

    metric_names = df["metric_name"].unique()
    metric_names = [mn for mn in metric_names if "jaccard" not in mn]
    df = df[df["metric_name"].isin(metric_names)]
    return df


def _split_sae_and_dict_size(sae_id: str) -> Tuple[str, Optional[int]]:
    """
    Expected SAE id format: "{sae_name}_{dictsize}", e.g. "batchtopk_1024".
    Linear probe: returns ("linear_probe", None). K-sweep names parse to
    (family, dict_size) and lose the top_k suffix — callers that need it
    should use parse_sae_run_name directly.
    """
    parsed = parse_sae_run_name(str(sae_id))
    if parsed is None:
        return str(sae_id), None
    return parsed.family, parsed.dict_size


def convert_dataset_metrics_to_df(
    all_metrics_for_dataset: Dict[str, Any],
    metrics_to_extract: Dict[str, str] | None = None,
    *,
    include_k_sweep: bool = False,
    include_probe: bool = False,
) -> pd.DataFrame:
    """
    Input nesting:
      model -> seed -> sae -> metric_family -> metric_name -> value

    metrics_to_extract:
      list of (metric_family, metric_name) pairs to extract.

    K-sweep / probe runs are skipped by default (set the flags to True to keep
    them).

    Output columns:
      model, seed, sae, dict_size, metric_family, metric_name, score_mean
    """
    if metrics_to_extract is None:
        metrics_to_extract = [
            ("CKNNA", "CKNNA"),
            ("fms", "mean_fms"),
            ("monosemanticity", "monosemanticity_mean"),
            ("ground_truth", 'f1_max_scores'),
            ("ground_truth", 'bmp_rec_F1_matrix_f1.0'),
            ("ground_truth", 'nnomp_rec_F1_matrix'),
        ]

    rows: List[dict] = []

    for model, seeds_dict in (all_metrics_for_dataset or {}).items():
        if not isinstance(seeds_dict, dict):
            continue

        for seed, saes_dict in seeds_dict.items():
            if not isinstance(saes_dict, dict):
                continue

            for sae_id, fams_dict in saes_dict.items():
                if not isinstance(fams_dict, dict):
                    continue

                parsed = parse_sae_run_name(str(sae_id))
                if parsed is None:
                    continue
                if parsed.is_k_sweep and not include_k_sweep:
                    continue
                if parsed.is_probe and not include_probe:
                    continue

                sae_name = parsed.family
                dict_size = parsed.dict_size

                for fam, wanted_metric_name in metrics_to_extract:
                    fam_dict = fams_dict.get(fam, None)
                    if not isinstance(fam_dict, dict):
                        continue

                    if wanted_metric_name not in fam_dict:
                        continue

                    value = fam_dict[wanted_metric_name]
                    if wanted_metric_name in ('bmp_rec_F1_matrix_f1.0', 'nnomp_rec_F1_matrix'):
                        # (A, steps) -> top-3 column slice, max per attribute, mean over A
                        if isinstance(value, torch.Tensor):
                            value = value[:, :3].max(dim=1).values.mean().item()
                        else:
                            value = value[:, :3].max(axis=1).mean().item()
                    elif wanted_metric_name == 'f1_max_scores':
                        value = value.mean().item()
                    if isinstance(value, torch.Tensor):
                        value = value.item()

                    rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "sae": sae_name,
                            "dict_size": dict_size,
                            "metric_family": fam,
                            "metric_name": wanted_metric_name,
                            "score_mean": value,
                        }
                    )

    df = pd.DataFrame(
        rows,
        columns=[
            "model",
            "seed",
            "sae",
            "dict_size",
            "metric_family",
            "metric_name",
            "score_mean",
        ],
    )

    # Optional: make types a bit more predictable
    if not df.empty:
        df["dict_size"] = pd.to_numeric(df["dict_size"], errors="coerce").astype(
            "Int64"
        )

    return df


def calc_matching_score_over_all_attrs_df(
    cub_data: dict,
    k_list: list[int],
    all_attr_ids=None,
    *,
    include_k_sweep: bool = False,
    include_probe: bool = False,
) -> pd.DataFrame:
    """
    cub_data: {sae_run: {metric_name: tensor_or_array}} (already filtered to ground_truth).

    Default: skips K-sweep folders and the linear_probe folder. Pass the
    matching include_* flag to keep them.

    Linear probe rows get sae="linear_probe", dict_size=-1, top_k=-1.
    K-sweep rows get sae=parsed.family, dict_size=parsed.dict_size, top_k=parsed.top_k.
    Standard rows get sae=parsed.family, dict_size=parsed.dict_size, top_k=None.
    """
    rows = []
    if all_attr_ids is not None:
        all_attr_ids = np.array(list(sorted(all_attr_ids)))

    for sae_name, result_dict in cub_data.items():
        parsed = parse_sae_run_name(str(sae_name))
        if parsed is None:
            continue
        if parsed.is_k_sweep and not include_k_sweep:
            continue
        if parsed.is_probe and not include_probe:
            continue

        sae_family = parsed.family
        if parsed.is_probe:
            dict_size = -1
            top_k = -1
        else:
            dict_size = parsed.dict_size
            top_k = parsed.top_k  # None for standard, int for k-sweep

        out_dict = {}
        if all_attr_ids is None:
            all_attr_ids = np.arange(result_dict["f1_max_scores"].shape[0])  # (A,)

        if 'f1_rec_F1_matrix' in result_dict:
            f1_matrix = torch.as_tensor(result_dict["f1_rec_F1_matrix"]).float()  # (A, steps)
            cols_f1 = f1_matrix.shape[1]

            out_dict["f1_top1"] = float(f1_matrix[all_attr_ids, 0].mean().item())
            out_dict["f1_top3"] = float(f1_matrix[all_attr_ids, 2].mean().item())
            out_dict["f1_top5"] = float(f1_matrix[all_attr_ids, 4].mean().item())
            if cols_f1 >= 10:
                out_dict["f1_top10"] = float(f1_matrix[all_attr_ids, 9].mean().item())


        # bmp_rec_F1_matrix_* (one per beta variant, plus jaccard/mi if present)
        for key, matrix in result_dict.items():
            if not key.startswith("bmp_rec_F1_matrix_"):
                continue
            suffix = key.replace("bmp_rec_F1_matrix_", "")
            Kmax = matrix.shape[1]
            for k in k_list:
                if 1 <= k <= Kmax:
                    metric_name = f"bmp_{suffix}_top{k}"
                    out_dict[metric_name] = float(matrix[all_attr_ids, k - 1].mean().item())

        # nnomp_rec_F1_matrix
        nnomp_matrix = result_dict.get("nnomp_rec_F1_matrix")
        if nnomp_matrix is not None:
            nnomp_t = torch.as_tensor(nnomp_matrix).float()
            Kmax = nnomp_t.shape[1]
            for k in k_list:
                if 1 <= k <= Kmax:
                    metric_name = f"nnomp_top{k}"
                    out_dict[metric_name] = float(nnomp_t[all_attr_ids, k - 1].mean().item())

        for metric_name, matching_score in out_dict.items():
            rows.append(
                {
                    "sae": sae_family,
                    "dict_size": dict_size,
                    "top_k": top_k,
                    "metric_name": metric_name,
                    "matching_score": matching_score,
                }
            )

    return pd.DataFrame(rows)


def get_perturbed_attr_ids(dataset: str) -> set[int]:
    if "cub" in dataset.lower():
        syn_dataset = "syn_cub_attrs"
        with open(
                f"/home/jokl/PycharmProjects/rs_concepts/outputs/metrics/{syn_dataset}/all_attr_ids.txt",
                "r",
        ) as f:
            return {int(line.strip()) for line in f if line.strip()}
    elif "coco" in dataset.lower():
        syn_dataset = "syn_coco"
        return set(range(0, 80))  # COCO has 80 attributes by design
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


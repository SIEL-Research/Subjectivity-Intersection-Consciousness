#!/usr/bin/env python3
"""Run DPA-SCOUT-007: H1-zero source placement and Lambda self-selection prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


EXPECTED = {
    "qpu": "e345ad16d12baf94d76864a3cce5112890bfaf8e4463eb895abbcb4f07048b3e",
    "geometry": "e455445d9b07b798da49a3031c28e83adc7ff71a6384046b513f79b83f5865ca",
    "topology": "8618b6de54d6752950588f3159282c4095ebb01d8485ad4b31be8e9a16a2f3a5",
    "kernels": "f33dbb7be3b7f9902761a8c99d7a66b059d66d3499af89b6ed117120d56d9757",
    "proxy": "7792473d63b2869030bd1a310d1ae0fb7f0a589753c2508ab1f5f5a77bd50eb9",
    "signature": "19894740a2c5351b1f1b207e9a8b5556094bdb1a4f0adf5a89134b8a0143af49",
}
QCOLS = ["p00", "p01", "p10", "p11"]
QUALITY_COLS = ["cq_overall", "eq_overall", "finite_fraction", "noninterpolated_fraction"]
BRAID_COLS = [f"braid_{index}" for index in range(8)]
GENERAL_COLS = [f"general_{index}" for index in range(10)]
TASK_SETS = {
    "full": list(range(26)),
    "odd": list(range(1, 26, 2)),
    "even": list(range(0, 26, 2)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def natural_label(value: str) -> int:
    return int(value[1:])


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return (values - np.mean(values)) / scale


def coherence_vectors(probabilities: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    root = np.sqrt(probabilities)
    outputs = []
    for kernel in kernels:
        rho = root[:, :, None] * np.asarray(kernel, dtype=float)[None, :, :] * root[:, None, :]
        diagonal = np.zeros_like(rho)
        indices = np.arange(4)
        diagonal[:, indices, indices] = probabilities
        outputs.append(np.linalg.norm(rho - diagonal, axis=(1, 2)))
    return np.column_stack(outputs)


def general_probability_gram(probabilities: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.asarray(probabilities, dtype=float))
    gram = root[:, :, None] * root[:, None, :]
    upper = np.triu_indices(4)
    return gram[:, upper[0], upper[1]]


def center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return matrix - matrix.mean(axis=0)[None, :] - matrix.mean(axis=1)[:, None] + matrix.mean()


def normalize_kernel(matrix: np.ndarray) -> np.ndarray:
    matrix = center(np.asarray(matrix, dtype=float))
    norm = float(np.linalg.norm(matrix))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("degenerate kernel")
    return matrix / norm


def linear_kernel(values: np.ndarray, standardize_columns: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if np.any(~np.isfinite(values)):
        raise RuntimeError("nonfinite feature")
    values = values - values.mean(axis=0, keepdims=True)
    if standardize_columns:
        scale = values.std(axis=0)
        keep = scale > 1e-12
        if not np.any(keep):
            raise RuntimeError("all standardized columns degenerate")
        values = values[:, keep] / scale[keep]
    return normalize_kernel(values @ values.T)


def optional_kernel(values: np.ndarray, standardize_columns: bool = False) -> np.ndarray | None:
    try:
        return linear_kernel(values, standardize_columns)
    except RuntimeError as error:
        if "degenerate" in str(error):
            return None
        raise


def residualize(matrix: np.ndarray, controls: list[np.ndarray | None]) -> np.ndarray:
    target = np.asarray(matrix, dtype=float).reshape(-1)
    usable = []
    for control in controls:
        if control is None:
            continue
        vector = np.asarray(control, dtype=float).reshape(-1)
        if np.all(np.isfinite(vector)) and np.linalg.norm(vector) > 1e-12:
            usable.append(vector)
    if not usable:
        return target
    design = np.column_stack(usable)
    return target - design @ np.linalg.lstsq(design, target, rcond=None)[0]


def alignment(left: np.ndarray, target: np.ndarray, controls: list[np.ndarray | None]) -> float:
    left_residual = residualize(left, controls)
    target_residual = residualize(target, controls)
    denominator = float(np.linalg.norm(left_residual) * np.linalg.norm(target_residual))
    if denominator <= 1e-12:
        raise RuntimeError("degenerate residual alignment")
    return float(left_residual @ target_residual / denominator)


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    x = rankdata(np.asarray(x, dtype=float), method="average")
    y = rankdata(np.asarray(y, dtype=float), method="average")
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    ranked_controls = np.column_stack([
        rankdata(controls[:, index], method="average") for index in range(controls.shape[1])
    ])
    design = np.column_stack([np.ones(len(x)), ranked_controls])
    keep = np.std(design, axis=0) > 1e-12
    keep[0] = True
    design = design[:, keep]
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator <= 1e-12:
        raise RuntimeError("degenerate partial Spearman")
    return float(rx @ ry / denominator)


def load_inputs(args):
    paths = {name: getattr(args, name) for name in EXPECTED}
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED:
        raise RuntimeError(f"input hash mismatch: {hashes}")
    qpu = pd.read_csv(args.qpu)
    geometry = pd.read_csv(args.geometry)
    topology = pd.read_csv(args.topology)
    proxy = pd.read_csv(args.proxy)
    proxy["noninterpolated_fraction"] = 1.0 - proxy.interpolated_fraction
    joined = (qpu
              .merge(geometry, on=["label", "task_idx"], validate="one_to_one")
              .merge(topology, on=["label", "task_idx"], validate="one_to_one")
              .merge(proxy[["label", "task_idx", *QUALITY_COLS]], on=["label", "task_idx"], validate="one_to_one"))
    joined["_label_order"] = joined.label.map(natural_label)
    joined = joined.sort_values(["_label_order", "task_idx"]).drop(columns="_label_order").reset_index(drop=True)
    labels = [f"P{index}" for index in range(1, 27)]
    if len(joined) != 676 or sorted(joined.label.unique(), key=natural_label) != labels:
        raise RuntimeError("26x26 join failed")
    if not all(joined[joined.label.eq(label)].task_idx.tolist() == list(range(26)) for label in labels):
        raise RuntimeError("task identity/order failed")
    records = json.loads(args.kernels.read_text())
    records = sorted(records, key=lambda item: int(item["mask"]))
    kernels = [np.asarray(record["correlation"], dtype=float) for record in records]
    if len(kernels) != 8 or any(kernel.shape != (4, 4) for kernel in kernels):
        raise RuntimeError("fixed Braid kernels invalid")
    return paths, hashes, joined, labels, kernels


def build_tasks(joined: pd.DataFrame, labels: list[str], kernels: list[np.ndarray]):
    source_rows = []
    receiver_rows = []
    participant_rows = []
    for label in labels:
        block = joined[joined.label.eq(label)].sort_values("task_idx").reset_index(drop=True)
        probability = block[QCOLS].to_numpy(float)
        braid = coherence_vectors(probability, kernels)
        general = general_probability_gram(probability)
        radius = np.sqrt(block.I2.to_numpy(float))
        h1 = block.h1_total_persistence.to_numpy(float)
        lam = block[["lambda2_4ch", "lambda3_4ch", "lambda4_4ch"]].sum(axis=1).to_numpy(float)
        quality = block[QUALITY_COLS].to_numpy(float)
        h1_zero = block.h1_zero.to_numpy(float)
        topology_lambda2 = block.lambda2.to_numpy(float)
        aspl = block.aspl.to_numpy(float)
        topology_switch = (
            zscore(h1_zero) + zscore(topology_lambda2) - zscore(aspl)
        ) / np.sqrt(3.0)
        for position, target_task in enumerate(range(26)):
            source = {
                "label": label,
                "target_task": target_task,
                "radius": float(radius[position]),
            }
            source.update({column: float(braid[position, index]) for index, column in enumerate(BRAID_COLS)})
            source.update({column: float(general[position, index]) for index, column in enumerate(GENERAL_COLS)})
            source_rows.append(source)
            receiver = {
                "label": label,
                "target_task": target_task,
                "h1_zero": float(h1_zero[position]),
                "h1_persistence": float(h1[position]),
                "topology_lambda2": float(topology_lambda2[position]),
                "aspl": float(aspl[position]),
                "topology_switch": float(topology_switch[position]),
            }
            receiver.update({column: float(quality[position, index]) for index, column in enumerate(QUALITY_COLS)})
            receiver_rows.append(receiver)
        quality_level = float(np.mean([
            np.mean(block.cq_overall.to_numpy(float)) / 100.0,
            np.mean(block.eq_overall.to_numpy(float)) / 100.0,
            np.mean(block.finite_fraction.to_numpy(float)),
            np.mean(block.noninterpolated_fraction.to_numpy(float)),
        ]))
        participant_rows.append({
            "label": label,
            "Lambda_mean": float(np.mean(lam)),
            "Lambda_peak": float(np.max(lam)),
            "Lambda_stability": float(-np.std(lam)),
            "h1_zero_rate": float(np.mean(h1_zero)),
            "quality_level": quality_level,
            "radius_mean": float(np.mean(radius)),
        })
    sources = pd.DataFrame(source_rows)
    receivers = pd.DataFrame(receiver_rows)
    participants = pd.DataFrame(participant_rows)
    if len(sources) != 676 or len(receivers) != 676 or len(participants) != 26:
        raise RuntimeError("task or participant row count failed")
    for frame in (sources, receivers, participants):
        if not np.isfinite(frame.select_dtypes(include=[np.number])).all().all():
            raise RuntimeError("nonfinite derived feature")
    return sources, receivers, participants


def select_tasks(frame: pd.DataFrame, label: str, tasks: list[int]) -> pd.DataFrame:
    block = frame[(frame.label.eq(label)) & (frame.target_task.isin(tasks))].sort_values("target_task").reset_index(drop=True)
    if len(block) != len(tasks) or block.target_task.tolist() != tasks:
        raise RuntimeError(f"task selection failed: {label}")
    return block


def calculate_scores(source: pd.DataFrame, receiver: pd.DataFrame) -> dict[str, float]:
    braid_kernel = linear_kernel(source[BRAID_COLS].to_numpy(float))
    general_kernel = linear_kernel(source[GENERAL_COLS].to_numpy(float))
    radius_kernel = optional_kernel(source[["radius"]].to_numpy(float))
    quality_kernel = optional_kernel(receiver[QUALITY_COLS].to_numpy(float), standardize_columns=True)
    controls_main = [radius_kernel, quality_kernel, general_kernel]
    controls_general = [radius_kernel, quality_kernel, braid_kernel]
    targets = {
        name: optional_kernel(receiver[[column]].to_numpy(float))
        for name, column in {
            "switch": "topology_switch",
            "zero": "h1_zero",
            "persistence": "h1_persistence",
            "topology_lambda2": "topology_lambda2",
            "aspl": "aspl",
        }.items()
    }

    def score(left: np.ndarray, target: np.ndarray | None, controls) -> float:
        if target is None:
            return float("nan")
        return alignment(left, target, controls)

    return {
        "switch_score": score(braid_kernel, targets["switch"], controls_main),
        "zero_score": score(braid_kernel, targets["zero"], controls_main),
        "persistence_score": score(braid_kernel, targets["persistence"], controls_main),
        "topology_lambda2_score": score(braid_kernel, targets["topology_lambda2"], controls_main),
        "aspl_score": score(braid_kernel, targets["aspl"], controls_main),
        "general_switch_score": score(general_kernel, targets["switch"], controls_general),
        "general_zero_score": score(general_kernel, targets["zero"], controls_general),
    }


def participant_summary(scores: pd.DataFrame, receiver: str, task_set: str) -> dict:
    block = scores[(scores.receiver.eq(receiver)) & (scores.task_set.eq(task_set))]
    own = block[block.source.eq(receiver)].iloc[0]
    false = block[~block.source.eq(receiver)]
    output = {"receiver": receiver, "task_set": task_set}
    for stem in ("switch", "zero", "persistence", "topology_lambda2", "aspl", "general_switch", "general_zero"):
        column = f"{stem}_score"
        valid = block[np.isfinite(block[column])]
        own_value = float(own[column])
        if not np.isfinite(own_value) or valid.empty:
            output[f"{stem}_own"] = float("nan")
            output[f"{stem}_false_max"] = float("nan")
            output[f"{stem}_false_label"] = "NONE"
            output[f"{stem}_margin"] = float("nan")
            output[f"{stem}_own_rank"] = 0
            continue
        ranked = valid.sort_values(column, ascending=False).reset_index(drop=True)
        false_valid = false[np.isfinite(false[column])]
        false_best = false_valid.loc[false_valid[column].idxmax()]
        output[f"{stem}_own"] = own_value
        output[f"{stem}_false_max"] = float(false_best[column])
        output[f"{stem}_false_label"] = str(false_best.source)
        output[f"{stem}_margin"] = float(own_value - false_best[column])
        output[f"{stem}_own_rank"] = int(ranked.index[ranked.source.eq(receiver)][0] + 1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpu", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--null-count", type=int, default=10000)
    args = parser.parse_args()

    paths, hashes, joined, labels, kernels = load_inputs(args)
    sources, receivers, lambda_features = build_tasks(joined, labels, kernels)
    signature = pd.read_csv(args.signature)
    if len(signature) != 26 or sorted(signature.label.unique(), key=natural_label) != labels:
        raise RuntimeError("historical signature identity failed")

    score_rows = []
    for task_set, tasks in TASK_SETS.items():
        for receiver_label in labels:
            receiver = select_tasks(receivers, receiver_label, tasks)
            for source_label in labels:
                source = select_tasks(sources, source_label, tasks)
                score_rows.append({
                    "task_set": task_set,
                    "receiver": receiver_label,
                    "source": source_label,
                    **calculate_scores(source, receiver),
                })
    scores = pd.DataFrame(score_rows)
    summaries = pd.DataFrame([
        participant_summary(scores, receiver, task_set)
        for task_set in TASK_SETS for receiver in labels
    ])

    p1_source = select_tasks(sources, "P1", TASK_SETS["full"])
    p1_receiver = select_tasks(receivers, "P1", TASK_SETS["full"])
    full_summary = summaries[summaries.task_set.eq("full")].sort_values("receiver", key=lambda col: col.map(natural_label))
    p1_observed = float(full_summary[full_summary.receiver.eq("P1")].switch_own.iloc[0])

    shift_rows = []
    moving = [*BRAID_COLS, *GENERAL_COLS, "radius"]
    for shift in range(1, 26):
        shifted = p1_source.copy()
        shifted.loc[:, moving] = np.roll(p1_source[moving].to_numpy(float), shift, axis=0)
        value = calculate_scores(shifted, p1_receiver)["switch_score"]
        shift_rows.append({"shift": shift, "switch_score": value})
    shift_max = float(max(row["switch_score"] for row in shift_rows))
    shift_gap = p1_observed - shift_max

    rng = np.random.default_rng(args.seed)
    task_null_rows = []
    for iteration in range(args.null_count):
        permuted = p1_source.copy()
        permuted.loc[:, moving] = p1_source[moving].to_numpy(float)[rng.permutation(26)]
        value = calculate_scores(permuted, p1_receiver)["switch_score"]
        task_null_rows.append({"iteration": iteration, "switch_score": value})
    task_null_values = np.asarray([row["switch_score"] for row in task_null_rows])
    task_null_p = float((int(np.sum(task_null_values >= p1_observed)) + 1) / (args.null_count + 1))

    lambda_frame = lambda_features.merge(
        full_summary[["receiver", "switch_margin"]].rename(columns={"receiver": "label", "switch_margin": "self_selection_margin"}),
        on="label",
        validate="one_to_one",
    )
    lambda_frame["_label_order"] = lambda_frame.label.map(natural_label)
    lambda_frame = lambda_frame.sort_values("_label_order").drop(columns="_label_order").reset_index(drop=True)
    predictors = ["Lambda_mean", "Lambda_peak", "Lambda_stability"]
    controls = lambda_frame[["h1_zero_rate", "quality_level", "radius_mean"]].to_numpy(float)
    target = lambda_frame.self_selection_margin.to_numpy(float)
    observed_predictors = {
        predictor: partial_spearman(lambda_frame[predictor].to_numpy(float), target, controls)
        for predictor in predictors
    }
    winner = max(predictors, key=lambda predictor: abs(observed_predictors[predictor]))
    observed_max_abs = abs(observed_predictors[winner])
    max_t_rows = []
    for iteration in range(args.null_count):
        order = rng.permutation(26)
        scores_null = {
            predictor: partial_spearman(
                lambda_frame[predictor].to_numpy(float), target[order], controls[order]
            ) for predictor in predictors
        }
        maximum = max(abs(value) for value in scores_null.values())
        max_t_rows.append({"iteration": iteration, "max_abs_rho": maximum, **scores_null})
    max_t_values = np.asarray([row["max_abs_rho"] for row in max_t_rows])
    lambda_max_t_p = float((int(np.sum(max_t_values >= observed_max_abs)) + 1) / (args.null_count + 1))
    winner_rho = observed_predictors[winner]
    p1_feature = float(lambda_frame[lambda_frame.label.eq("P1")][winner].iloc[0])
    p9_feature = float(lambda_frame[lambda_frame.label.eq("P9")][winner].iloc[0])
    p1_predicted_above_p9 = bool((p1_feature - p9_feature) * winner_rho > 0)
    predictor_rows = [{
        "predictor": predictor,
        "partial_spearman": observed_predictors[predictor],
        "is_max_abs_winner": predictor == winner,
        "familywise_maxT_p": lambda_max_t_p if predictor == winner else np.nan,
    } for predictor in predictors]

    indexed = summaries.set_index(["task_set", "receiver"])
    p1 = indexed.loc[("full", "P1")]
    p9 = indexed.loc[("full", "P9")]
    odd_score = float(indexed.loc[("odd", "P1")].switch_own)
    even_score = float(indexed.loc[("even", "P1")].switch_own)
    zero_rates = lambda_features.set_index("label").h1_zero_rate
    p1_zero_rate = float(zero_rates.loc["P1"])
    p1_zero_rate_rank = int(zero_rates.rank(method="min", ascending=False).loc["P1"])
    historical_p1 = signature[signature.label.eq("P1")].iloc[0]
    historical_direction_pass = bool(
        historical_p1.delta_h1_zero > 0
        and historical_p1.delta_lambda2 > 0
        and historical_p1.delta_aspl < 0
    )
    checks = {
        "P1_switch_score_positive": bool(p1.switch_own > 0),
        "P1_switch_rank_1": bool(p1.switch_own_rank == 1),
        "P1_switch_margin_positive": bool(p1.switch_margin > 0),
        "P1_H1_zero_score_positive": bool(p1.zero_own > 0),
        "P1_H1_zero_rank_1": bool(p1.zero_own_rank == 1),
        "P1_H1_zero_margin_positive": bool(p1.zero_margin > 0),
        "P1_source_shift_gap_positive": bool(shift_gap > 0),
        "P1_task_permutation_p_le_0_05": bool(task_null_p <= 0.05),
        "P1_switch_margin_gt_general": bool(p1.switch_margin > p1.general_switch_margin),
        "P1_odd_switch_positive": bool(odd_score > 0),
        "P1_even_switch_positive": bool(even_score > 0),
        "P9_switch_margin_nonpositive": bool(p9.switch_margin <= 0),
        "P1_switch_margin_gt_P9": bool(p1.switch_margin > p9.switch_margin),
        "P1_H1_zero_rate_rank_1": bool(p1_zero_rate_rank == 1),
        "historical_P1_direction_signature": historical_direction_pass,
        "Lambda_maxT_p_le_0_05": bool(lambda_max_t_p <= 0.05),
        "Lambda_winner_places_P1_above_P9": p1_predicted_above_p9,
    }
    core = (
        checks["P1_switch_margin_positive"]
        and checks["P1_H1_zero_margin_positive"]
        and checks["P1_source_shift_gap_positive"]
    )
    fold_stable = checks["P1_odd_switch_positive"] and checks["P1_even_switch_positive"]
    if not core:
        decision = "NO-GO"
        reason = "P1_14CH_SWITCH_OR_H1_ZERO_FAILED_OWN_SOURCE_SPECIFICITY"
    elif not fold_stable:
        decision = "OPEN"
        reason = "FULL_14CH_SWITCH_PATH_SURVIVED_BUT_ODD_EVEN_SPLIT"
    elif all(checks.values()):
        decision = "SCOPED EXPLORATORY PASS"
        reason = "14CH_SWITCH_H1_ZERO_AND_LAMBDA_SELECTOR_CLEARED_ALL_FIXED_CONTROLS"
    else:
        decision = "CLOSED_SCOPED"
        reason = "14CH_SWITCH_CORE_SURVIVED_BUT_DIRECTION_OR_LAMBDA_SELECTOR_FAILED"

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    sources.to_csv(output / "source_task_features.csv", index=False)
    receivers.to_csv(output / "receiver_task_features.csv", index=False)
    lambda_frame.to_csv(output / "lambda_participant_features.csv", index=False)
    scores.to_csv(output / "source_receiver_scores.csv", index=False)
    summaries.to_csv(output / "participant_taskset_summary.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(output / "p1_source_shift_scores.csv", index=False)
    pd.DataFrame(task_null_rows).to_csv(output / "p1_task_permutation_null.csv", index=False)
    pd.DataFrame(predictor_rows).to_csv(output / "lambda_predictor_scores.csv", index=False)
    pd.DataFrame(max_t_rows).to_csv(output / "lambda_maxT_null.csv", index=False)
    manifest = {
        name: {"path": str(path.resolve()), "sha256": hashes[name], "size": path.stat().st_size}
        for name, path in paths.items()
    }
    write_json(output / "INPUT_MANIFEST.json", manifest)
    result = {
        "scout_id": "DPA-SCOUT-007",
        "parent": "Q-012-X19",
        "decision": decision,
        "reason": reason,
        "evidence_class": "Exploratory result",
        "checks": checks,
        "p1_full": {key: (value.item() if hasattr(value, "item") else value) for key, value in p1.to_dict().items()},
        "p9_full": {key: (value.item() if hasattr(value, "item") else value) for key, value in p9.to_dict().items()},
        "p1_odd_switch_score": odd_score,
        "p1_even_switch_score": even_score,
        "p1_h1_zero_rate": p1_zero_rate,
        "p1_h1_zero_rate_rank": p1_zero_rate_rank,
        "historical_p1_signature": {
            "delta_h1_zero": float(historical_p1.delta_h1_zero),
            "delta_lambda2": float(historical_p1.delta_lambda2),
            "delta_aspl": float(historical_p1.delta_aspl),
        },
        "p1_shift_max": shift_max,
        "p1_shift_gap": float(shift_gap),
        "p1_task_permutation_p": task_null_p,
        "lambda_predictors": observed_predictors,
        "lambda_winner": winner,
        "lambda_winner_rho": winner_rho,
        "lambda_maxT_p": lambda_max_t_p,
        "lambda_p1_value": p1_feature,
        "lambda_p9_value": p9_feature,
        "lambda_winner_places_P1_above_P9": p1_predicted_above_p9,
        "claim_ceiling": "Same-archive Braid correspondence with an oriented 14-channel topology switch and four-electrode Lambda participant prediction only; no causal control, ontological self, HMB, consciousness, or confirmation.",
    }
    write_json(output / "SUMMARY.json", result)
    write_json(output / "EXECUTION_LOG.json", {
        "status": "COMPLETE",
        "scout_id": "DPA-SCOUT-007",
        "seed": args.seed,
        "null_count_each": args.null_count,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "run_scout_sha256": sha256(Path(__file__)),
        "input_manifest_sha256": sha256(output / "INPUT_MANIFEST.json"),
        "output_hashes": {
            path.name: sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "EXECUTION_LOG.json"
        },
    })
    print(json.dumps({
        "decision": decision,
        "reason": reason,
        "checks": checks,
        "p1_switch_own": float(p1.switch_own),
        "p1_switch_rank": int(p1.switch_own_rank),
        "p1_switch_margin": float(p1.switch_margin),
        "p1_zero_own": float(p1.zero_own),
        "p1_zero_rank": int(p1.zero_own_rank),
        "p1_zero_margin": float(p1.zero_margin),
        "p9_switch_margin": float(p9.switch_margin),
        "shift_gap": float(shift_gap),
        "task_null_p": task_null_p,
        "odd_score": odd_score,
        "even_score": even_score,
        "lambda_predictors": observed_predictors,
        "lambda_winner": winner,
        "lambda_maxT_p": lambda_max_t_p,
        "lambda_p1_value": p1_feature,
        "lambda_p9_value": p9_feature,
    }, indent=2))


if __name__ == "__main__":
    main()

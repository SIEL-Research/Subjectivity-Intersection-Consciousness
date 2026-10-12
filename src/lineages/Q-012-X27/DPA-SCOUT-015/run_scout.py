#!/usr/bin/env python3
"""Resolve pre-lock versus post-response Braid/14ch-EEG topology phenotypes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


BASE_SHA256 = "17810cef9b6bc5cfa8df8447bbbba53988889f5d6924964889e7865523ff123e"
SIGNED_SHA256 = "7e876a60ed752fb53f56d3eec10d5bb4b43beafbc604fceddf82d973704507b8"
MOVEMENT_SHA256 = "3d9c9b66696d89c1a0a823979be72fc75a2fa2835deb1af4ff6a68f8e792e610"
TOPOLOGY_SCRIPT_SHA256 = "ef957658a10fdc46ca6b7e211ec36d62e67b42aa44405773a8641f9ffa6327e8"
TOPOLOGY_REFERENCE_SHA256 = "8618b6de54d6752950588f3159282c4095ebb01d8485ad4b31be8e9a16a2f3a5"
EXPECTED_PRIOR = {
    "P1_score": 0.8823166589393112,
    "P1_rank": 2,
    "P21_score": 1.7310028292651864,
    "P21_rank": 1,
}
BASELINE_TOL = 1e-10
EVALUABLE = ["P1", "P2", "P10", "P14", "P17", "P18", "P21", "P25"]
PHASES = ["pre", "post", "delta"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_module(path: Path, expected: str, name: str):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{name} hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def segment_metrics(topology, segment: np.ndarray) -> dict[str, float | int]:
    if len(segment) < topology.MIN_SEG_SAMPLES:
        return {
            "n_samples": int(len(segment)), "n_valid_channels": 0,
            "h1_total_persistence": np.nan, "h1_max_persistence": np.nan,
            "h1_count": np.nan, "h1_zero": np.nan, "lambda2": np.nan, "aspl": np.nan,
        }
    cmat, valid = topology.corr_from_segment(segment)
    if cmat is None:
        return {
            "n_samples": int(len(segment)),
            "n_valid_channels": int(valid.sum()) if valid is not None else 0,
            "h1_total_persistence": np.nan, "h1_max_persistence": np.nan,
            "h1_count": np.nan, "h1_zero": np.nan, "lambda2": np.nan, "aspl": np.nan,
        }
    dmat = 1.0 - cmat
    np.fill_diagonal(dmat, 0.0)
    total_p, max_p, count = topology.h1_persistence_summary_from_distance(dmat)
    return {
        "n_samples": int(len(segment)), "n_valid_channels": int(cmat.shape[0]),
        "h1_total_persistence": total_p, "h1_max_persistence": max_p,
        "h1_count": count, "h1_zero": int(total_p <= topology.H1_ZERO_TOL),
        "lambda2": topology.weighted_lambda2(cmat), "aspl": topology.weighted_aspl(cmat),
    }


def raw_task_metrics(label: str, topology) -> tuple[pd.DataFrame, pd.DataFrame, Path, float]:
    raw_path = topology.raw_file_for(label)
    q = topology.load_quantum(label).sort_values("task_idx").reset_index(drop=True)
    if q.task_idx.astype(int).tolist() != list(range(26)):
        raise RuntimeError(f"{label} task identity mismatch")
    clean, eeg_cols, fs = topology.minimal_clean_eeg(raw_path)
    power = topology.fullband_power_matrix(clean, eeg_cols, fs)
    times = clean["TimeStamp"].to_numpy(dtype="datetime64[ns]")

    full_rows = []
    for (ia, ib), task in zip(topology.task_segment_bounds(times, q), q.task_idx.astype(int)):
        row = segment_metrics(topology, power[ia:ib, :])
        row.update({"label": label, "task_idx": int(task)})
        full_rows.append(row)

    phase_rows = []
    for _, qrow in q.iterrows():
        mid = qrow["mid"]
        bounds = {
            "pre": (
                int(np.searchsorted(times, np.datetime64((mid - pd.Timedelta(seconds=1)).to_datetime64()), side="left")),
                int(np.searchsorted(times, np.datetime64(mid.to_datetime64()), side="left")),
            ),
            "post": (
                int(np.searchsorted(times, np.datetime64(mid.to_datetime64()), side="left")),
                int(np.searchsorted(times, np.datetime64((mid + pd.Timedelta(seconds=1)).to_datetime64()), side="right")),
            ),
        }
        for phase, (ia, ib) in bounds.items():
            row = segment_metrics(topology, power[ia:ib, :])
            row.update({"label": label, "task_idx": int(qrow.task_idx), "phase": phase})
            phase_rows.append(row)
    return pd.DataFrame(full_rows), pd.DataFrame(phase_rows), raw_path, fs


def check_raw_baseline(full: pd.DataFrame, reference: pd.DataFrame) -> dict[str, object]:
    keys = ["label", "task_idx"]
    left = full.sort_values(keys).reset_index(drop=True)
    right = reference.sort_values(keys).reset_index(drop=True)
    identity = left[keys].equals(right[keys])
    exact_columns = ["n_valid_channels", "h1_count", "h1_zero"]
    continuous = ["h1_total_persistence", "h1_max_persistence", "lambda2", "aspl"]
    exact = {}
    for col in exact_columns:
        a = left[col].to_numpy(float)
        b = right[col].to_numpy(float)
        exact[col] = bool(np.array_equal(a, b, equal_nan=True))
    differences = {}
    nan_patterns = {}
    for col in continuous:
        a = left[col].to_numpy(float)
        b = right[col].to_numpy(float)
        nan_patterns[col] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        finite = np.isfinite(a) & np.isfinite(b)
        differences[col] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    passed = bool(
        identity and all(exact.values()) and all(nan_patterns.values())
        and all(value <= BASELINE_TOL for value in differences.values())
    )
    return {
        "status": "PASS" if passed else "FAILED_IMPLEMENTATION",
        "task_identity": identity, "exact_columns": exact,
        "nan_patterns": nan_patterns, "max_absolute_differences": differences,
    }


def add_phase_axes(phase_metrics: pd.DataFrame, full_receivers: pd.DataFrame, movement) -> pd.DataFrame:
    wide = phase_metrics.pivot(index=["label", "task_idx"], columns="phase", values=["lambda2", "aspl"])
    wide.columns = [f"{metric}_{phase}" for metric, phase in wide.columns]
    wide = wide.reset_index()
    gate = full_receivers[["label", "target_task", "h1_zero"]].rename(columns={"target_task": "task_idx"})
    wide = wide.merge(gate, on=["label", "task_idx"], how="left", validate="one_to_one")
    rows = []
    for label in EVALUABLE:
        block = wide[wide.label.eq(label)].sort_values("task_idx").reset_index(drop=True)
        if block.task_idx.astype(int).tolist() != list(range(26)) or block.isna().any().any():
            raise RuntimeError(f"{label} phase axis input invalid")
        axes = {
            "pre": (movement.zscore(block.lambda2_pre.to_numpy(float)) - movement.zscore(block.aspl_pre.to_numpy(float))) / np.sqrt(2.0),
            "post": (movement.zscore(block.lambda2_post.to_numpy(float)) - movement.zscore(block.aspl_post.to_numpy(float))) / np.sqrt(2.0),
            "delta": (
                movement.zscore((block.lambda2_post - block.lambda2_pre).to_numpy(float))
                - movement.zscore((block.aspl_post - block.aspl_pre).to_numpy(float))
            ) / np.sqrt(2.0),
        }
        for phase, values in axes.items():
            for index, value in enumerate(values):
                rows.append({
                    "label": label, "target_task": int(block.loc[index, "task_idx"]),
                    "h1_zero": int(block.loc[index, "h1_zero"]), "phase": phase,
                    "select_axis": float(value),
                })
    return pd.DataFrame(rows)


def summarize_source(frame: pd.DataFrame, own: str, column: str) -> dict[str, object]:
    own_value = float(frame.loc[frame.source.eq(own), column].iloc[0])
    false = frame[~frame.source.eq(own)]
    best = false[column].idxmax()
    return {
        "score": own_value,
        "rank": int(frame[column].rank(method="min", ascending=False)[frame.source.eq(own)].iloc[0]),
        "best_false_source": str(false.loc[best, "source"]),
        "best_false_score": float(false.loc[best, column]),
        "margin": own_value - float(false.loc[best, column]),
    }


def prior_baseline(sources: pd.DataFrame, receivers: pd.DataFrame, labels: list[str], movement, base) -> dict[str, object]:
    result = {}
    for receiver_label in ("P1", "P21"):
        receiver = receivers[receivers.label.eq(receiver_label)].sort_values("target_task").reset_index(drop=True).copy()
        receiver["select_axis"] = (
            base.zscore(receiver.topology_lambda2.to_numpy(float))
            - base.zscore(receiver.aspl.to_numpy(float))
        ) / np.sqrt(2.0)
        rows = []
        for source_label in labels:
            source = sources[sources.label.eq(source_label)].sort_values("target_task").reset_index(drop=True)
            rows.append({"source": source_label, "score": movement.interaction_score(source, receiver)["score"]})
        summary = summarize_source(pd.DataFrame(rows), receiver_label, "score")
        result[receiver_label] = summary
    passed = bool(
        abs(result["P1"]["score"] - EXPECTED_PRIOR["P1_score"]) <= 1e-12
        and result["P1"]["rank"] == EXPECTED_PRIOR["P1_rank"]
        and abs(result["P21"]["score"] - EXPECTED_PRIOR["P21_score"]) <= 1e-12
        and result["P21"]["rank"] == EXPECTED_PRIOR["P21_rank"]
    )
    return {"status": "PASS" if passed else "FAILED_IMPLEMENTATION", **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-evaluator", type=Path, required=True)
    parser.add_argument("--signed-evaluator", type=Path, required=True)
    parser.add_argument("--movement-evaluator", type=Path, required=True)
    parser.add_argument("--topology-script", type=Path, required=True)
    parser.add_argument("--topology-reference", type=Path, required=True)
    for name in ("qpu", "geometry", "topology", "kernels", "proxy", "signature"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--null-count", type=int, default=5000)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if sha256(args.topology_reference) != TOPOLOGY_REFERENCE_SHA256:
        raise RuntimeError("topology reference hash mismatch")
    base = load_module(args.base_evaluator, BASE_SHA256, "base_evaluator_015")
    signed = load_module(args.signed_evaluator, SIGNED_SHA256, "signed_evaluator_015")
    movement = load_module(args.movement_evaluator, MOVEMENT_SHA256, "movement_evaluator_015")
    topology = load_module(args.topology_script, TOPOLOGY_SCRIPT_SHA256, "topology_evaluator_015")

    paths, hashes, joined, labels, fixed_kernels = base.load_inputs(args)
    sources_raw, receivers, _ = base.build_tasks(joined, labels, fixed_kernels)
    sources = signed.add_signed_features(sources_raw, base)
    gate_counts = receivers.groupby("label").h1_zero.sum().astype(int)
    actual_evaluable = [label for label in labels if gate_counts[label] >= 4]
    if actual_evaluable != EVALUABLE:
        raise RuntimeError(f"evaluable identity changed: {actual_evaluable}")

    reference = pd.read_csv(args.topology_reference)
    full_blocks = []
    phase_blocks = []
    raw_paths = {}
    sampling_rates = {}
    for label in EVALUABLE:
        full, phases, raw_path, fs = raw_task_metrics(label, topology)
        full_blocks.append(full)
        phase_blocks.append(phases)
        raw_paths[label] = raw_path
        sampling_rates[label] = fs
    full_raw = pd.concat(full_blocks, ignore_index=True)
    phase_metrics = pd.concat(phase_blocks, ignore_index=True)
    baseline = check_raw_baseline(full_raw, reference[reference.label.isin(EVALUABLE)].copy())
    prior = prior_baseline(sources, receivers, labels, movement, base)
    baseline["prior_endpoint"] = prior
    write_json(output / "BASELINE_GATE.json", baseline)
    full_raw.to_csv(output / "RAW_FULL_WINDOW_RECOMPUTATION.csv", index=False)
    if baseline["status"] != "PASS" or prior["status"] != "PASS":
        raise RuntimeError("baseline reproduction failed")

    validity = phase_metrics.groupby(["label", "phase"]).agg(
        n_tasks=("task_idx", "size"), min_samples=("n_samples", "min"),
        min_valid_channels=("n_valid_channels", "min"), valid_metrics=("lambda2", lambda x: int(x.notna().sum())),
        h1_zero_count=("h1_zero", "sum"),
    ).reset_index()
    valid_windows = bool(
        (validity.n_tasks == 26).all() and (validity.min_samples >= topology.MIN_SEG_SAMPLES).all()
        and (validity.min_valid_channels >= topology.MIN_VALID_CHANNELS).all()
        and (validity.valid_metrics == 26).all()
    )
    phase_metrics.to_csv(output / "RAW_PRE_POST_TASK_TOPOLOGY.csv", index=False)
    validity.to_csv(output / "WINDOW_VALIDITY.csv", index=False)
    if not valid_windows:
        write_json(output / "SUMMARY.json", {
            "scout_id": "DPA-SCOUT-015", "parent": "Q-012-X27", "decision": "INCONCLUSIVE",
            "classification": "RAW_PHASE_WINDOWS_INVALID", "evidence_class": "Exploratory result",
        })
        return

    phase_axes = add_phase_axes(phase_metrics, receivers, movement)
    phase_axes.to_csv(output / "PHASE_SELECTION_AXES.csv", index=False)
    rng = np.random.default_rng(args.seed)
    source_rows = []
    null_rows = []
    participant_rows = []
    moving_columns = [*base.BRAID_COLS, *base.GENERAL_COLS, "radius", "coherence_mean", "general_dispersion"]

    for receiver_label in EVALUABLE:
        receiver_summaries = {}
        for phase in PHASES:
            receiver = phase_axes[
                phase_axes.label.eq(receiver_label) & phase_axes.phase.eq(phase)
            ].sort_values("target_task").reset_index(drop=True)
            rows = []
            for source_label in labels:
                source = sources[sources.label.eq(source_label)].sort_values("target_task").reset_index(drop=True)
                main_score = movement.interaction_score(source, receiver, general_only=False)
                general_score = movement.interaction_score(source, receiver, general_only=True)
                row = {
                    "receiver": receiver_label, "phase": phase, "source": source_label,
                    "interaction_score": main_score["score"], "gate_slope": main_score["gate_slope"],
                    "nongate_slope": main_score["nongate_slope"],
                    "interaction_change": main_score["interaction_change"],
                    "condition_number": main_score["condition_number"],
                    "general_interaction_score": general_score["score"],
                }
                rows.append(row)
                source_rows.append(row)
            source_frame = pd.DataFrame(rows)
            main_summary = summarize_source(source_frame, receiver_label, "interaction_score")
            general_summary = summarize_source(source_frame, receiver_label, "general_interaction_score")

            own_source = sources[sources.label.eq(receiver_label)].sort_values("target_task").reset_index(drop=True)
            exceed = 0
            for iteration in range(args.null_count):
                permuted = own_source.copy()
                order = rng.permutation(26)
                permuted.loc[:, moving_columns] = own_source[moving_columns].to_numpy(float)[order]
                value = movement.interaction_score(permuted, receiver, general_only=False)["score"]
                exceed += int(value >= main_summary["score"])
                null_rows.append({
                    "receiver": receiver_label, "phase": phase,
                    "iteration": iteration, "interaction_score": value,
                })
            permutation_p = float((exceed + 1) / (args.null_count + 1))
            own_row = source_frame[source_frame.source.eq(receiver_label)].iloc[0]
            receiver_summaries[phase] = {
                **main_summary, "permutation_p": permutation_p,
                "general_rank": general_summary["rank"], "general_margin": general_summary["margin"],
                "gate_slope": float(own_row.gate_slope), "nongate_slope": float(own_row.nongate_slope),
                "interaction_change": float(own_row.interaction_change),
            }
        participant_rows.append({
            "receiver": receiver_label, "full_h1_zero_count": int(gate_counts[receiver_label]),
            **{
                f"{phase}_{key}": value
                for phase, summary in receiver_summaries.items()
                for key, value in summary.items()
            },
        })

    participants = pd.DataFrame(participant_rows)
    source_scores = pd.DataFrame(source_rows)
    nulls = pd.DataFrame(null_rows)
    participants.to_csv(output / "PARTICIPANT_PHASE_COMPARISON.csv", index=False)
    source_scores.to_csv(output / "ALL_RECEIVER_PHASE_SOURCE_SCORES.csv", index=False)
    nulls.to_csv(output / "PHASE_TASK_PERMUTATION_NULLS.csv", index=False)

    p1 = participants[participants.receiver.eq("P1")].iloc[0]
    p21 = participants[participants.receiver.eq("P21")].iloc[0]
    p1_checks = {
        "pre_score_positive": bool(p1.pre_score > 0),
        "pre_source_rank_1": bool(p1.pre_rank == 1),
        "pre_margin_positive": bool(p1.pre_margin > 0),
        "pre_permutation_p_le_0_05": bool(p1.pre_permutation_p <= 0.05),
        "pre_score_gt_post_score": bool(p1.pre_score > p1.post_score),
        "pre_braid_margin_gt_general": bool(p1.pre_margin > p1.pre_general_margin),
    }
    p21_checks = {
        "delta_score_positive": bool(p21.delta_score > 0),
        "delta_source_rank_1": bool(p21.delta_rank == 1),
        "delta_margin_positive": bool(p21.delta_margin > 0),
        "delta_permutation_p_le_0_05": bool(p21.delta_permutation_p <= 0.05),
        "delta_score_gt_pre_score": bool(p21.delta_score > p21.pre_score),
        "delta_braid_margin_gt_general": bool(p21.delta_margin > p21.delta_general_margin),
    }
    p1_pass = all(p1_checks.values())
    p21_pass = all(p21_checks.values())
    decision = "SCOPED EXPLORATORY PASS" if p1_pass and p21_pass else "NO-GO"
    classification = (
        "P1_PRE_LOCK_AND_P21_POST_RESPONSE" if p1_pass and p21_pass
        else "TEMPORAL_PHENOTYPE_NOT_JOINTLY_RESOLVED"
    )

    input_paths = {
        **paths, "base_evaluator": args.base_evaluator, "signed_evaluator": args.signed_evaluator,
        "movement_evaluator": args.movement_evaluator, "topology_script": args.topology_script,
        "topology_reference": args.topology_reference, **{f"raw_{k}": v for k, v in raw_paths.items()},
    }
    input_hashes = {
        **hashes, "base_evaluator": sha256(args.base_evaluator),
        "signed_evaluator": sha256(args.signed_evaluator), "movement_evaluator": sha256(args.movement_evaluator),
        "topology_script": sha256(args.topology_script), "topology_reference": sha256(args.topology_reference),
        **{f"raw_{k}": sha256(v) for k, v in raw_paths.items()},
    }
    write_json(output / "INPUT_MANIFEST.json", {
        name: {"path": str(path.resolve()), "sha256": input_hashes[name], "size": path.stat().st_size}
        for name, path in input_paths.items()
    })
    summary = {
        "scout_id": "DPA-SCOUT-015", "parent": "Q-012-X27", "decision": decision,
        "classification": classification, "evidence_class": "Exploratory result", "result_informed": True,
        "evaluable_participants": EVALUABLE, "p1_pre_lock_pass": p1_pass,
        "p21_post_response_pass": p21_pass, "p1_checks": p1_checks, "p21_checks": p21_checks,
        "sampling_rates_hz": sampling_rates,
        "p1": {phase: {
            key: int(p1[f"{phase}_{key}"]) if key == "rank" else float(p1[f"{phase}_{key}"])
            for key in ("score", "rank", "margin", "permutation_p", "general_margin", "gate_slope", "nongate_slope")
        } for phase in PHASES},
        "p21": {phase: {
            key: int(p21[f"{phase}_{key}"]) if key == "rank" else float(p21[f"{phase}_{key}"])
            for key in ("score", "rank", "margin", "permutation_p", "general_margin", "gate_slope", "nongate_slope")
        } for phase in PHASES},
        "claim_ceiling": "Same-archive result-informed exploratory comparison of pre/post-midpoint 14ch EEG topology and source-specific Braid contraction for eight evaluable participants; no causal control, physical mask, Body/perception, consciousness, population generality, or confirmation.",
    }
    write_json(output / "SUMMARY.json", summary)
    write_json(output / "EXECUTION_LOG.json", {
        "status": "COMPLETE", "scout_id": "DPA-SCOUT-015", "seed": args.seed,
        "null_count_per_participant_phase": args.null_count, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "run_scout_sha256": sha256(Path(__file__)),
        "output_hashes": {
            p.name: sha256(p) for p in sorted(output.iterdir())
            if p.is_file() and p.name != "EXECUTION_LOG.json"
        },
    })
    print(json.dumps(summary, indent=2))
    preview = participants[[
        "receiver", "pre_score", "pre_rank", "pre_permutation_p",
        "post_score", "post_rank", "post_permutation_p",
        "delta_score", "delta_rank", "delta_permutation_p",
    ]]
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()

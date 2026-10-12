#!/usr/bin/env python3
"""Compare signed Braid-EEG movement phenotypes across participants."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


BASE_EVALUATOR_SHA256 = "17810cef9b6bc5cfa8df8447bbbba53988889f5d6924964889e7865523ff123e"
SIGNED_EVALUATOR_SHA256 = "7e876a60ed752fb53f56d3eec10d5bb4b43beafbc604fceddf82d973704507b8"
EXPECTED_P1_SIGNED = 0.899490014676022
EXPECTED_P1_SIGNED_P = 0.03888888888888889


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_module(path: Path, expected: str, name: str):
    if sha256(path) != expected:
        raise RuntimeError(f"{name} hash mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(values.std())
    if scale <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


def interaction_score(source: pd.DataFrame, receiver: pd.DataFrame, general_only: bool = False):
    gate = receiver.h1_zero.to_numpy(float)
    y = zscore(receiver.select_axis.to_numpy(float))
    radius = zscore(source.radius.to_numpy(float))
    if general_only:
        x = zscore(source.general_dispersion.to_numpy(float))
        design = np.column_stack([np.ones(len(x)), x, gate, x * gate, radius])
    else:
        x = zscore(source.coherence_mean.to_numpy(float))
        dispersion = zscore(source.general_dispersion.to_numpy(float))
        design = np.column_stack([np.ones(len(x)), x, gate, x * gate, radius, dispersion])
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise RuntimeError("interaction design rank deficient")
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    gate_slope = float(beta[1] + beta[3])
    nongate_slope = float(beta[1])
    return {
        "score": -gate_slope,
        "gate_slope": gate_slope,
        "nongate_slope": nongate_slope,
        "interaction_change": float(-beta[3]),
        "design_rank": rank,
        "condition_number": float(np.linalg.cond(design)),
    }


def source_summary(frame: pd.DataFrame, own_label: str, column: str):
    own = float(frame.loc[frame.source.eq(own_label), column].iloc[0])
    false = frame[~frame.source.eq(own_label)]
    index = false[column].idxmax()
    return {
        "score": own,
        "rank": int(frame[column].rank(method="min", ascending=False)[frame.source.eq(own_label)].iloc[0]),
        "best_false_source": str(false.loc[index, "source"]),
        "best_false_score": float(false.loc[index, column]),
        "margin": own - float(false.loc[index, column]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-evaluator", type=Path, required=True)
    parser.add_argument("--signed-evaluator", type=Path, required=True)
    for name in ("qpu", "geometry", "topology", "kernels", "proxy", "signature"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--null-count", type=int, default=5000)
    args = parser.parse_args()

    base = load_module(args.base_evaluator, BASE_EVALUATOR_SHA256, "base_evaluator")
    signed = load_module(args.signed_evaluator, SIGNED_EVALUATOR_SHA256, "signed_evaluator")
    paths, hashes, joined, labels, fixed_kernels = base.load_inputs(args)
    sources_raw, receivers, _ = base.build_tasks(joined, labels, fixed_kernels)
    sources = signed.add_signed_features(sources_raw, base)
    receivers = receivers.sort_values(["label", "target_task"]).reset_index(drop=True).copy()
    receivers["select_axis"] = np.nan
    for label in labels:
        mask = receivers.label.eq(label)
        block = receivers.loc[mask].sort_values("target_task")
        receivers.loc[block.index, "select_axis"] = (
            base.zscore(block.topology_lambda2.to_numpy(float))
            - base.zscore(block.aspl.to_numpy(float))
        ) / np.sqrt(2.0)

    gate_counts = receivers.groupby("label").h1_zero.sum().astype(int).reindex(labels)
    evaluable = [label for label in labels if gate_counts[label] >= 4]
    rng = np.random.default_rng(args.seed)
    participant_rows = []
    source_rows = []
    null_rows = []

    for receiver_label in evaluable:
        receiver = receivers[receivers.label.eq(receiver_label)].sort_values("target_task").reset_index(drop=True)
        tasks = receiver.loc[receiver.h1_zero.eq(1), "target_task"].astype(int).tolist()
        gate_count = len(tasks)
        within_status = "NONDISCRIMINATING_SMALL_SUBSET"
        within_score = np.nan
        within_p = np.nan
        if gate_count >= 6:
            source = sources[sources.label.eq(receiver_label)].sort_values("target_task").reset_index(drop=True)
            gate_source = signed.subset(source, tasks)
            gate_receiver = signed.subset(receiver, tasks)
            within_score, _ = signed.signed_score(gate_source, gate_receiver)
            permutation_values = []
            moving = [*base.BRAID_COLS, *base.GENERAL_COLS, "radius", "coherence_mean", "general_dispersion"]
            for order in itertools.permutations(range(gate_count)):
                permuted = gate_source.copy()
                permuted.loc[:, moving] = gate_source[moving].to_numpy(float)[list(order)]
                value, _ = signed.signed_score(permuted, gate_receiver)
                permutation_values.append(value)
            within_p = float(np.mean(np.asarray(permutation_values) >= within_score - 1e-15))
            within_status = "IDENTIFIABLE"

        receiver_source_rows = []
        for source_label in labels:
            source = sources[sources.label.eq(source_label)].sort_values("target_task").reset_index(drop=True)
            main = interaction_score(source, receiver, general_only=False)
            general = interaction_score(source, receiver, general_only=True)
            receiver_source_rows.append({
                "receiver": receiver_label,
                "source": source_label,
                "interaction_score": main["score"],
                "gate_slope": main["gate_slope"],
                "nongate_slope": main["nongate_slope"],
                "interaction_change": main["interaction_change"],
                "condition_number": main["condition_number"],
                "general_interaction_score": general["score"],
            })
        source_frame = pd.DataFrame(receiver_source_rows)
        source_rows.extend(receiver_source_rows)
        main_summary = source_summary(source_frame, receiver_label, "interaction_score")
        general_summary = source_summary(source_frame, receiver_label, "general_interaction_score")

        own_source = sources[sources.label.eq(receiver_label)].sort_values("target_task").reset_index(drop=True)
        moving = [*base.BRAID_COLS, *base.GENERAL_COLS, "radius", "coherence_mean", "general_dispersion"]
        exceed = 0
        for iteration in range(args.null_count):
            permuted = own_source.copy()
            permuted.loc[:, moving] = own_source[moving].to_numpy(float)[rng.permutation(26)]
            value = interaction_score(permuted, receiver)["score"]
            exceed += int(value >= main_summary["score"])
            null_rows.append({"receiver": receiver_label, "iteration": iteration, "interaction_score": value})
        permutation_p = float((exceed + 1) / (args.null_count + 1))
        core_pass = bool(
            main_summary["score"] > 0 and main_summary["rank"] == 1 and main_summary["margin"] > 0
            and permutation_p <= 0.05 and main_summary["margin"] > general_summary["margin"]
        )
        participant_rows.append({
            "receiver": receiver_label,
            "gate_count": gate_count,
            "within_gate_status": within_status,
            "within_gate_signed_score": within_score,
            "within_gate_exact_p": within_p,
            "interaction_score": main_summary["score"],
            "interaction_source_rank": main_summary["rank"],
            "interaction_false_source": main_summary["best_false_source"],
            "interaction_false_margin": main_summary["margin"],
            "interaction_permutation_p": permutation_p,
            "nongate_slope": float(source_frame.loc[source_frame.source.eq(receiver_label), "nongate_slope"].iloc[0]),
            "gate_slope": float(source_frame.loc[source_frame.source.eq(receiver_label), "gate_slope"].iloc[0]),
            "interaction_change": float(source_frame.loc[source_frame.source.eq(receiver_label), "interaction_change"].iloc[0]),
            "general_interaction_score": general_summary["score"],
            "general_source_rank": general_summary["rank"],
            "general_false_margin": general_summary["margin"],
            "condition_number": float(source_frame.loc[source_frame.source.eq(receiver_label), "condition_number"].iloc[0]),
            "core_movement_selector": core_pass,
        })

    participants = pd.DataFrame(participant_rows)
    p1 = participants[participants.receiver.eq("P1")].iloc[0]
    p10 = participants[participants.receiver.eq("P10")].iloc[0]
    other_core = participants[(~participants.receiver.eq("P1")) & participants.core_movement_selector].receiver.tolist()
    checks = {
        "P1_within_gate_gt_P10": bool(p1.within_gate_signed_score > p10.within_gate_signed_score),
        "P1_within_gate_p_le_0_05": bool(p1.within_gate_exact_p <= 0.05),
        "P1_interaction_positive": bool(p1.interaction_score > 0),
        "P1_interaction_source_rank_1": bool(p1.interaction_source_rank == 1),
        "P1_interaction_margin_positive": bool(p1.interaction_false_margin > 0),
        "P1_interaction_p_le_0_05": bool(p1.interaction_permutation_p <= 0.05),
        "P1_braid_margin_gt_general": bool(p1.interaction_false_margin > p1.general_false_margin),
        "no_other_core_movement_selector": len(other_core) == 0,
    }
    if all(checks.values()):
        decision = "SCOPED EXPLORATORY PASS"
        classification = "P1_UNIQUE_MOVEMENT_PHENOTYPE"
    elif bool(p1.core_movement_selector) and other_core:
        decision = "OPEN"
        classification = "SHARED_MOVEMENT_PHENOTYPE"
    else:
        decision = "NO-GO"
        classification = "P1_NOT_UNIQUE_MOVEMENT_PHENOTYPE"

    output = args.output_dir
    participants.to_csv(output / "participant_movement_comparison.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output / "all_receiver_source_interaction_scores.csv", index=False)
    pd.DataFrame(null_rows).to_csv(output / "participant_interaction_nulls.csv", index=False)
    manifest_paths = {**paths, "base_evaluator": args.base_evaluator, "signed_evaluator": args.signed_evaluator}
    manifest_hashes = {**hashes, "base_evaluator": sha256(args.base_evaluator), "signed_evaluator": sha256(args.signed_evaluator)}
    write_json(output / "INPUT_MANIFEST.json", {
        name: {"path": str(path.resolve()), "sha256": manifest_hashes[name], "size": path.stat().st_size}
        for name, path in manifest_paths.items()
    })
    baseline_pass = bool(
        abs(p1.within_gate_signed_score - EXPECTED_P1_SIGNED) <= 1e-12
        and abs(p1.within_gate_exact_p - EXPECTED_P1_SIGNED_P) <= 1e-12
    )
    write_json(output / "BASELINE_GATE.json", {
        "status": "PASS" if baseline_pass else "FAILED_IMPLEMENTATION",
        "P1_within_gate_signed_score": float(p1.within_gate_signed_score),
        "P1_within_gate_exact_p": float(p1.within_gate_exact_p),
    })
    if not baseline_pass:
        raise RuntimeError("signed baseline reproduction failed")
    summary = {
        "scout_id": "DPA-SCOUT-013",
        "parent": "Q-012-X25",
        "decision": decision,
        "classification": classification,
        "evidence_class": "Exploratory result",
        "checks": checks,
        "other_core_movement_selectors": other_core,
        "evaluable_participants": evaluable,
        "direct_within_gate_comparable": ["P1", "P10"],
        "claim_ceiling": "Same-archive comparison of participant-specific signed Braid-contraction/EEG-integration movement phenotypes; no claim for participants with fewer than four gate events, causality, HMB, consciousness, or confirmation.",
    }
    write_json(output / "SUMMARY.json", summary)
    write_json(output / "EXECUTION_LOG.json", {
        "status": "COMPLETE", "scout_id": "DPA-SCOUT-013", "seed": args.seed,
        "null_count_per_evaluable_participant": args.null_count,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "run_scout_sha256": sha256(Path(__file__)),
        "output_hashes": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "EXECUTION_LOG.json"},
    })
    print(json.dumps(summary, indent=2))
    print(participants.to_string(index=False))


if __name__ == "__main__":
    main()

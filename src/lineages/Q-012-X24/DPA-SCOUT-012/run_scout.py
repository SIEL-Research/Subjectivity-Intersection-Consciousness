#!/usr/bin/env python3
"""Test signed global Braid contraction against gated EEG integration direction."""

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


BASE_SHA256 = "17810cef9b6bc5cfa8df8447bbbba53988889f5d6924964889e7865523ff123e"
GATE_TASKS = [1, 6, 10, 11, 13, 17, 18]
EXPECTED = {"score": 0.7164166478029015, "rank": 1, "margin": 0.04756231665727784}
OFFDIAGONAL_GENERAL = ["general_1", "general_2", "general_3", "general_5", "general_6", "general_8"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_base(path: Path):
    if sha256(path) != BASE_SHA256:
        raise RuntimeError("base evaluator hash mismatch")
    spec = importlib.util.spec_from_file_location("dpa_scout_007_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def subset(frame: pd.DataFrame, tasks: list[int]) -> pd.DataFrame:
    block = frame[frame.target_task.isin(tasks)].sort_values("target_task").reset_index(drop=True)
    if block.target_task.tolist() != sorted(tasks):
        raise RuntimeError("task subset identity failed")
    return block


def add_signed_features(source: pd.DataFrame, base) -> pd.DataFrame:
    output = source.copy()
    output["coherence_mean"] = output[base.BRAID_COLS].mean(axis=1)
    offdiag = output[OFFDIAGONAL_GENERAL].to_numpy(float)
    output["general_dispersion"] = np.sqrt(2.0 * np.sum(offdiag ** 2, axis=1))
    return output


def residual(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones(len(values)), controls])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def signed_score(source: pd.DataFrame, receiver: pd.DataFrame, feature: str = "coherence_mean") -> tuple[float, float]:
    controls = source[["radius", "general_dispersion"]].to_numpy(float)
    left = residual(source[feature].to_numpy(float), controls)
    right = residual(receiver.select_axis.to_numpy(float), controls)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        raise RuntimeError("degenerate signed endpoint")
    rho = float(left @ right / denominator)
    unique_fraction = float(np.linalg.norm(left) / np.linalg.norm(source[feature].to_numpy(float) - source[feature].mean()))
    return -rho, unique_fraction


def general_score(source: pd.DataFrame, receiver: pd.DataFrame) -> float:
    controls = source[["radius"]].to_numpy(float)
    left = residual(source.general_dispersion.to_numpy(float), controls)
    right = residual(receiver.select_axis.to_numpy(float), controls)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        raise RuntimeError("degenerate general competitor")
    return float(-(left @ right / denominator))


def full_kernel_score(base, source: pd.DataFrame, receiver: pd.DataFrame) -> float:
    braid = base.linear_kernel(source[base.BRAID_COLS].to_numpy(float))
    general = base.linear_kernel(source[base.GENERAL_COLS].to_numpy(float))
    radius = base.optional_kernel(source[["radius"]].to_numpy(float))
    target = base.linear_kernel(receiver[["select_axis"]].to_numpy(float))
    return base.alignment(braid, target, [radius, general])


def profile(base, sources, receiver, labels, tasks):
    rows = []
    for label in labels:
        source = subset(sources[sources.label.eq(label)], tasks)
        value, unique = signed_score(source, receiver)
        rows.append({
            "source": label,
            "signed_braid_score": value,
            "unique_fraction_after_controls": unique,
            "signed_general_score": general_score(source, receiver),
        })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, column: str):
    own = float(frame.loc[frame.source.eq("P1"), column].iloc[0])
    false = frame[~frame.source.eq("P1")]
    index = false[column].idxmax()
    return {
        "score": own,
        "rank": int(frame[column].rank(method="min", ascending=False)[frame.source.eq("P1")].iloc[0]),
        "best_false_source": str(false.loc[index, "source"]),
        "best_false_score": float(false.loc[index, column]),
        "margin": own - float(false.loc[index, column]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-evaluator", type=Path, required=True)
    for name in ("qpu", "geometry", "topology", "kernels", "proxy", "signature"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = load_base(args.base_evaluator)
    paths, hashes, joined, labels, fixed_kernels = base.load_inputs(args)
    sources_raw, receivers, _ = base.build_tasks(joined, labels, fixed_kernels)
    sources = add_signed_features(sources_raw, base)
    receiver_all = receivers[receivers.label.eq("P1")].sort_values("target_task").reset_index(drop=True).copy()
    receiver_all["select_axis"] = (
        base.zscore(receiver_all.topology_lambda2.to_numpy(float))
        - base.zscore(receiver_all.aspl.to_numpy(float))
    ) / np.sqrt(2.0)
    receiver_gate = subset(receiver_all, GATE_TASKS)
    baseline_rows = []
    for label in labels:
        source = subset(sources[sources.label.eq(label)], GATE_TASKS)
        baseline_rows.append({"source": label, "score": full_kernel_score(base, source, receiver_gate)})
    baseline_frame = pd.DataFrame(baseline_rows)
    own = float(baseline_frame.loc[baseline_frame.source.eq("P1"), "score"].iloc[0])
    false = baseline_frame[~baseline_frame.source.eq("P1")]
    baseline = {
        "score": own,
        "rank": int(baseline_frame.score.rank(method="min", ascending=False)[baseline_frame.source.eq("P1")].iloc[0]),
        "margin": own - float(false.score.max()),
    }
    output = args.output_dir
    baseline_pass = bool(
        abs(baseline["score"] - EXPECTED["score"]) <= 1e-12
        and baseline["rank"] == EXPECTED["rank"]
        and abs(baseline["margin"] - EXPECTED["margin"]) <= 1e-12
    )
    write_json(output / "BASELINE_GATE.json", {"status": "PASS" if baseline_pass else "FAILED_IMPLEMENTATION", **baseline})
    if not baseline_pass:
        raise RuntimeError(f"baseline gate failed: {baseline}")

    gate_profile = profile(base, sources, receiver_gate, labels, GATE_TASKS)
    braid_summary = summarize(gate_profile, "signed_braid_score")
    general_summary = summarize(gate_profile, "signed_general_score")
    own_source_all = sources[sources.label.eq("P1")].sort_values("target_task").reset_index(drop=True)
    nongate_tasks = [task for task in range(26) if task not in GATE_TASKS]
    nongate_score, nongate_unique = signed_score(
        subset(own_source_all, nongate_tasks), subset(receiver_all, nongate_tasks)
    )
    braid_summary["nongate_score"] = nongate_score
    braid_summary["gate_minus_nongate"] = braid_summary["score"] - nongate_score
    braid_summary["unique_fraction_after_controls"] = float(
        gate_profile.loc[gate_profile.source.eq("P1"), "unique_fraction_after_controls"].iloc[0]
    )
    braid_summary["nongate_unique_fraction_after_controls"] = nongate_unique

    own_gate = subset(own_source_all, GATE_TASKS)
    moving = [*base.BRAID_COLS, *base.GENERAL_COLS, "radius", "coherence_mean", "general_dispersion"]
    permutation_rows = []
    for index, order in enumerate(itertools.permutations(range(7))):
        permuted = own_gate.copy()
        permuted.loc[:, moving] = own_gate[moving].to_numpy(float)[list(order)]
        value, unique = signed_score(permuted, receiver_gate)
        permutation_rows.append({"permutation": index, "signed_braid_score": value, "unique_fraction": unique})
    permutations = pd.DataFrame(permutation_rows)
    exact_p = float(np.mean(permutations.signed_braid_score >= braid_summary["score"] - 1e-15))
    braid_summary["exact_permutation_p"] = exact_p

    jackknife_rows = []
    for omitted in GATE_TASKS:
        tasks = [task for task in GATE_TASKS if task != omitted]
        value, unique = signed_score(subset(own_source_all, tasks), subset(receiver_all, tasks))
        jackknife_rows.append({"omitted_task": omitted, "score": value, "unique_fraction": unique, "score_positive": value > 0})
    jackknife = pd.DataFrame(jackknife_rows)
    positive_jackknife = int(jackknife.score_positive.sum())

    p1_indexed = own_source_all.set_index("target_task")
    receiver_indexed = receiver_all.set_index("target_task")
    dominant_pair = {
        "delta_coherence_mean_13_minus_1": float(p1_indexed.loc[13, "coherence_mean"] - p1_indexed.loc[1, "coherence_mean"]),
        "delta_select_axis_13_minus_1": float(receiver_indexed.loc[13, "select_axis"] - receiver_indexed.loc[1, "select_axis"]),
    }
    component_rows = []
    for column in base.BRAID_COLS:
        value, unique = signed_score(own_gate, receiver_gate, feature=column)
        component_rows.append({"component": column, "signed_score": value, "unique_fraction": unique})
    components = pd.DataFrame(component_rows)

    checks = {
        "P1_score_positive": braid_summary["score"] > 0,
        "P1_source_rank_1": braid_summary["rank"] == 1,
        "P1_false_margin_positive": braid_summary["margin"] > 0,
        "exact_permutation_p_le_0_05": exact_p <= 0.05,
        "gate_score_gt_nongate": braid_summary["gate_minus_nongate"] > 0,
        "jackknife_positive_at_least_5_of_7": positive_jackknife >= 5,
        "braid_margin_gt_general_margin": braid_summary["margin"] > general_summary["margin"],
        "dominant_pair_inverse_polarity": (
            dominant_pair["delta_coherence_mean_13_minus_1"] < 0
            and dominant_pair["delta_select_axis_13_minus_1"] > 0
        ),
    }
    decision = "SCOPED EXPLORATORY PASS" if all(checks.values()) else "NO-GO"

    gate_profile.to_csv(output / "source_profile.csv", index=False)
    baseline_frame.to_csv(output / "full_kernel_baseline_sources.csv", index=False)
    permutations.to_csv(output / "exact_task_permutations.csv", index=False)
    jackknife.to_csv(output / "leave_one_task_out.csv", index=False)
    components.to_csv(output / "p1_component_signed_scores.csv", index=False)
    manifest_paths = {**paths, "base_evaluator": args.base_evaluator}
    manifest_hashes = {**hashes, "base_evaluator": sha256(args.base_evaluator)}
    write_json(output / "INPUT_MANIFEST.json", {
        name: {"path": str(path.resolve()), "sha256": manifest_hashes[name], "size": path.stat().st_size}
        for name, path in manifest_paths.items()
    })
    summary = {
        "scout_id": "DPA-SCOUT-012",
        "parent": "Q-012-X24",
        "decision": decision,
        "evidence_class": "Exploratory result",
        "result_informed": True,
        "checks": checks,
        "braid": braid_summary,
        "general_competitor": general_summary,
        "jackknife_positive_count": positive_jackknife,
        "dominant_pair": dominant_pair,
        "claim_ceiling": "Same-archive result-informed signed global Braid coherence relation to P1 H1-zero-gated 14ch integration direction only; no physical mask, generator transition, causal transfer, HMB, consciousness, or confirmation.",
    }
    write_json(output / "SUMMARY.json", summary)
    write_json(output / "EXECUTION_LOG.json", {
        "status": "COMPLETE", "scout_id": "DPA-SCOUT-012",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "run_scout_sha256": sha256(Path(__file__)),
        "output_hashes": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "EXECUTION_LOG.json"},
    })
    print(json.dumps(summary, indent=2))
    print(gate_profile.sort_values("signed_braid_score", ascending=False).head(10).to_string(index=False))
    print(jackknife.to_string(index=False))
    print(components.to_string(index=False))


if __name__ == "__main__":
    main()

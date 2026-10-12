#!/usr/bin/env python3
"""Adaptive direction-sensitive Braid-time to Beta-change scout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "run_band_participant_scout.py"
SPEC = importlib.util.spec_from_file_location("band_scout", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def residualize(values: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    nuisance = np.asarray(nuisance, dtype=float)
    design = np.column_stack([np.ones(len(values)), nuisance])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 1e-12 else 0.0


def selected_correlations(source: np.ndarray, radius: np.ndarray, frame: pd.DataFrame):
    source_residual = residualize(source, radius)
    rows = {}
    for metric in BASE.METRICS:
        beta_delta = np.diff(frame[f"Beta_{metric}"].to_numpy(float))
        target = residualize(beta_delta, radius)
        correlations = np.asarray([correlation(source_residual[:, component], target) for component in range(source.shape[1])])
        selected = int(np.argmax(np.abs(correlations)))
        rows[metric] = {"component": selected, "correlation": float(correlations[selected]),
                        "score": float(abs(correlations[selected]))}
    return rows


def composite(rows) -> float:
    return float(np.median([rows[metric]["score"] for metric in BASE.METRICS]))


def half_consistency(source, radius, frame, selected):
    halves = (np.arange(0, 12), np.arange(13, 25))
    output = {}
    for metric in BASE.METRICS:
        component = selected[metric]["component"]
        full_sign = np.sign(selected[metric]["correlation"])
        beta_delta = np.diff(frame[f"Beta_{metric}"].to_numpy(float))
        half_corrs = []
        for indices in halves:
            sx = residualize(source[indices, component], radius[indices])
            sy = residualize(beta_delta[indices], radius[indices])
            half_corrs.append(correlation(sx, sy))
        consistent = bool(full_sign != 0 and np.sign(half_corrs[0]) == full_sign and np.sign(half_corrs[1]) == full_sign)
        output[metric] = {"half_1": half_corrs[0], "half_2": half_corrs[1], "consistent": consistent}
    return output


def preflight() -> dict:
    rng = np.random.default_rng(411)
    source = rng.normal(size=(25, 8))
    radius = rng.normal(size=(25, 3))
    target = source[:, 4] * 1.2 + radius[:, 0] * 0.6 + rng.normal(scale=0.08, size=25)
    sr = residualize(source, radius)
    tr = residualize(target, radius)
    corrs = [correlation(sr[:, i], tr) for i in range(8)]
    recovered = int(np.argmax(np.abs(corrs)))
    constant = correlation(np.ones(25), tr)
    permuted = max(abs(correlation(sr[:, i][rng.permutation(25)], tr)) for i in range(8))
    checks = {"recovers_component_4": recovered == 4, "positive_gt_0_9": abs(corrs[recovered]) > 0.9,
              "positive_gt_permuted": abs(corrs[recovered]) > permuted, "constant_zero": constant == 0.0}
    return {"status": "PASS_LOCAL_PREFLIGHT_NOT_INDEPENDENT_E0" if all(checks.values()) else "FAIL_LOCAL_PREFLIGHT",
            "checks": checks, "values": {"recovered": recovered, "positive": abs(corrs[recovered]),
                                            "permuted": permuted, "constant": constant}}


def add_one(exceedances, count):
    return float((exceedances + 1) / (count + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "focal"))
    parser.add_argument("--eeg", type=Path, required=True)
    parser.add_argument("--source-kernel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()
    destination = args.output_dir / "iteration_002_time_direction"
    labels, frames, matrices = BASE.load_inputs(args.eeg, args.source_kernel)
    packets, basis = BASE.build_features(labels, frames, matrices)
    if args.mode == "preflight":
        baseline = BASE.baseline_check(frames)
        capacity = preflight()
        status = "PASS_LOCAL_PREFLIGHT_NOT_INDEPENDENT_E0" if baseline["passed"] and capacity["status"].startswith("PASS") else "FAIL_LOCAL_PREFLIGHT"
        write_json(destination / "preflight.json", {"status": status, "baseline": baseline, "capacity": capacity,
                   "script_sha256": sha256(Path(__file__)), "base_script_sha256": sha256(BASE_PATH),
                   "inputs": {"eeg": sha256(args.eeg), "source_kernel": sha256(args.source_kernel)}})
        if not status.startswith("PASS"):
            raise RuntimeError("directional preflight failed")
        return

    preflight_path = destination / "preflight.json"
    gate = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not gate["status"].startswith("PASS") or gate["script_sha256"] != sha256(Path(__file__)) or gate["base_script_sha256"] != sha256(BASE_PATH):
        raise RuntimeError("preflight revision mismatch")

    observed_rows, source_rows, null_rows = [], [], []
    rng = np.random.default_rng(args.seed)
    for receiver in labels:
        selected = selected_correlations(packets[receiver]["Time"], packets[receiver]["TimeRadius"], frames[receiver])
        halves = half_consistency(packets[receiver]["Time"], packets[receiver]["TimeRadius"], frames[receiver], selected)
        observed_rows.append({"participant": receiver, "composite": composite(selected),
                              "consistent_metrics": sum(halves[m]["consistent"] for m in BASE.METRICS),
                              **{f"score_{m}": selected[m]["score"] for m in BASE.METRICS},
                              **{f"corr_{m}": selected[m]["correlation"] for m in BASE.METRICS},
                              **{f"component_{m}": selected[m]["component"] for m in BASE.METRICS},
                              **{f"half1_{m}": halves[m]["half_1"] for m in BASE.METRICS},
                              **{f"half2_{m}": halves[m]["half_2"] for m in BASE.METRICS}})
        for donor in labels:
            donor_selected = selected_correlations(packets[donor]["Time"], packets[donor]["TimeRadius"], frames[receiver])
            source_rows.append({"receiver": receiver, "source": donor, "composite": composite(donor_selected)})

    for iteration in range(args.n_perm):
        scores = []
        for participant in labels:
            source, radius = packets[participant]["Time"], packets[participant]["TimeRadius"]
            order = rng.permutation(25)
            selected = selected_correlations(source[order], radius[order], frames[participant])
            value = composite(selected)
            scores.append(value)
            null_rows.append({"iteration": iteration, "participant": participant, "composite": value})
        maximum = max(scores)
        for row in null_rows[-26:]:
            row["iteration_max"] = maximum

    observed, sources, nulls = pd.DataFrame(observed_rows), pd.DataFrame(source_rows), pd.DataFrame(null_rows)
    for index, row in observed.iterrows():
        participant, value = row["participant"], float(row["composite"])
        own_null = nulls[nulls["participant"] == participant]["composite"].to_numpy(float)
        max_null = nulls.drop_duplicates("iteration")["iteration_max"].to_numpy(float)
        observed.loc[index, "p_task_raw"] = add_one(int(np.sum(own_null >= value)), len(own_null))
        observed.loc[index, "p_participant_fwer"] = add_one(int(np.sum(max_null >= value)), len(max_null))
        donor = sources[sources["receiver"] == participant].copy()
        donor["rank"] = donor["composite"].rank(method="min", ascending=False).astype(int)
        own_rank = int(donor.loc[donor["source"] == participant, "rank"].iloc[0])
        observed.loc[index, "own_source_rank"] = own_rank
        strict = observed.loc[index, "p_participant_fwer"] <= .05 and own_rank <= 3 and row["consistent_metrics"] >= 2
        lead = observed.loc[index, "p_task_raw"] <= .05 and own_rank <= 5 and row["consistent_metrics"] >= 2
        observed.loc[index, "tier"] = "TIER_A_TIME_DIRECTION" if strict else ("TIER_B_TIME_DIRECTION" if lead else "NONE")
    observed["rank"] = observed["composite"].rank(method="min", ascending=False).astype(int)
    observed = observed.sort_values("rank")
    destination.mkdir(parents=True, exist_ok=True)
    observed.to_csv(destination / "participant_scores.csv", index=False)
    sources.to_csv(destination / "source_scores.csv", index=False)
    nulls.to_csv(destination / "task_nulls.csv", index=False)
    candidates = observed[observed["tier"] != "NONE"]
    summary = {"status": "Exploratory result", "decision": "TIME_DIRECTION_SIGNAL" if len(candidates) else "NO_TIME_DIRECTION_SIGNAL",
               "mapping": "Braid signed order response -> signed Beta change", "n_permutations": args.n_perm,
               "candidates": candidates.to_dict("records"), "top_five": observed.head(5).to_dict("records"),
               "time_basis": basis, "claim_ceiling": "Adaptive same-archive participant-selection scout only."}
    write_json(destination / "summary.json", summary)
    write_json(destination / "execution_log.json", {"status": "COMPLETE", "decision": summary["decision"],
               "seed": args.seed, "n_permutations": args.n_perm, "preflight_sha256": sha256(preflight_path),
               "output_hashes": {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name != "execution_log.json"},
               "script_sha256": sha256(Path(__file__))})


if __name__ == "__main__":
    main()

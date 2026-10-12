#!/usr/bin/env python3
"""Population-level EEG-native Mind feature replacement scout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
METRICS = ("mean", "peak", "stability")
LAGS = (4, 8, 16)
NEW_CANDIDATES = ("lag_flow", "conditional_gain", "reentry_spectrum")
ALL_CANDIDATES = NEW_CANDIDATES + ("third_order_odd",)
BAND_TABLE_SHA = "b012c0d63708389b4242a1f659300303dba8b5cb548571caccc978ea7ceb3ea0"
BASELINE_BETA_POPULATION = 0.0389597806356118
BASELINE_BAND_POPULATION = {
    "Delta": 0.010694723278990821,
    "Theta": 0.0427551525455008,
    "Alpha": 0.06626239146918708,
    "Beta": 0.03895978063561179,
    "Gamma": 0.02709895765534598,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_one(exceedances: int, count: int) -> float:
    return float((exceedances + 1) / (count + 1))


HERE = Path(__file__).resolve().parent
S004_PATH = HERE.parent / "DPA-SCOUT-004" / "run_eeg_native_dcs_scout.py"
SPEC = importlib.util.spec_from_file_location("dpa_scout_004", S004_PATH)
S004 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(S004)
BODY = S004.BODY


def lag_flow(segment: np.ndarray) -> np.ndarray:
    x = S004.zcols(segment)
    features = []
    upper = np.triu_indices(x.shape[1], 1)
    for lag in LAGS:
        cross = x[:-lag].T @ x[lag:] / (len(x) - lag)
        features.extend((cross - cross.T)[upper].tolist())
    return np.asarray(features, float)


def residual_against(vector: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(control)), control])
    return vector - design @ np.linalg.lstsq(design, vector, rcond=None)[0]


def squared_partial_gain(source: np.ndarray, destination_past: np.ndarray, destination_future: np.ndarray) -> float:
    source_residual = residual_against(source, destination_past)
    future_residual = residual_against(destination_future, destination_past)
    denominator = float(np.linalg.norm(source_residual) * np.linalg.norm(future_residual))
    if denominator <= 1e-12:
        return 0.0
    correlation = float(source_residual @ future_residual / denominator)
    return correlation * correlation


def conditional_gain(segment: np.ndarray) -> np.ndarray:
    x = S004.zcols(segment)
    features = []
    for lag in LAGS:
        past, future = x[:-lag], x[lag:]
        for left in range(x.shape[1]):
            for right in range(left + 1, x.shape[1]):
                left_to_right = squared_partial_gain(past[:, left], past[:, right], future[:, right])
                right_to_left = squared_partial_gain(past[:, right], past[:, left], future[:, left])
                features.append(left_to_right - right_to_left)
    return np.asarray(features, float)


def reentry_spectrum(segment: np.ndarray) -> np.ndarray:
    x = S004.zcols(segment)
    features = []
    for lag in LAGS:
        past, future = x[:-lag], x[lag:]
        alpha = 1e-3 * len(past)
        transition = np.linalg.solve(past.T @ past + alpha * np.eye(x.shape[1]), past.T @ future)
        features.extend(np.linalg.svd(transition, compute_uv=False).tolist())
    return np.asarray(features, float)


def feature_family(segment: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "lag_flow": lag_flow(segment),
        "conditional_gain": conditional_gain(segment),
        "reentry_spectrum": reentry_spectrum(segment),
        "third_order_odd": S004.causality(segment),
    }


def pooled_projection(raw_features: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict]]:
    projected = {label: {} for label in raw_features}
    audit = {}
    for candidate in ALL_CANDIDATES:
        pooled = np.vstack([raw_features[label][candidate] for label in sorted(raw_features, key=lambda x: int(x[1:]))])
        center = pooled.mean(axis=0, keepdims=True)
        scale = pooled.std(axis=0, keepdims=True)
        scale[scale <= 1e-12] = 1.0
        standardized = (pooled - center) / scale
        _, singular, vh = np.linalg.svd(standardized, full_matrices=False)
        rank = int(np.sum(singular > max(float(singular[0]), 1.0) * 1e-12))
        dimension = min(8, rank)
        basis = vh[:dimension].T
        offset = 0
        for label in sorted(raw_features, key=lambda x: int(x[1:])):
            count = len(raw_features[label][candidate])
            projected[label][candidate] = standardized[offset:offset + count] @ basis
            offset += count
        audit[candidate] = {
            "raw_dimension": int(pooled.shape[1]),
            "rank": rank,
            "retained_dimension": dimension,
            "leading_singular_values": singular[:8].tolist(),
        }
    return projected, audit


def load_inputs(args):
    if sha256(args.bands) != BAND_TABLE_SHA:
        raise RuntimeError("band table hash mismatch")
    bands = pd.read_csv(args.bands)
    baseline_log = json.loads(args.baseline_log.read_text(encoding="utf-8"))
    qdir = args.raw_root / "IDPC_Reproduction"
    eegdir = args.raw_root / "IDPC_EEG"
    raw_features, energy, frames = {}, {}, {}
    input_hashes = {"raw_eeg": {}, "windows": {}}
    for participant in range(1, 27):
        label = f"P{participant}"
        raw_path = BODY.raw_file_for(eegdir, label)
        window_path = qdir / f"{label}_eeg_timeseries.csv"
        raw_hash = sha256(raw_path)
        window_hash = sha256(window_path)
        if raw_hash != baseline_log["raw_manifest"]["raw_eeg"].get(raw_path.name):
            raise RuntimeError(f"raw EEG baseline mismatch {label}")
        if window_hash != baseline_log["raw_manifest"]["windows"].get(window_path.name):
            raise RuntimeError(f"window baseline mismatch {label}")
        raw = BODY.load_raw(raw_path)
        times, values, _ = S004.preprocess(raw)
        windows = S004.task_windows(qdir, label)
        candidate_rows = {candidate: [] for candidate in ALL_CANDIDATES}
        energy_rows = []
        for task in range(26):
            row = windows[windows["task_idx"].astype(int).eq(task)].iloc[0]
            start = np.datetime64(row["start"].to_datetime64())
            end = np.datetime64(row["end"].to_datetime64())
            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")
            segment = values[left:right]
            if len(segment) < 103:
                raise RuntimeError(f"short segment {label} {task}")
            family = feature_family(segment)
            for candidate, vector in family.items():
                candidate_rows[candidate].append(vector)
            energy_rows.append([float(np.mean(segment * segment))])
        raw_features[label] = {candidate: np.asarray(rows) for candidate, rows in candidate_rows.items()}
        energy[label] = np.asarray(energy_rows)
        frames[label] = bands[bands["label"].eq(label)].sort_values("task_idx").reset_index(drop=True)
        input_hashes["raw_eeg"][raw_path.name] = raw_hash
        input_hashes["windows"][window_path.name] = window_hash
    projected, pca_audit = pooled_projection(raw_features)
    return projected, raw_features, energy, frames, input_hashes, pca_audit


def participant_band_score(task_feature: np.ndarray, energy: np.ndarray, frame: pd.DataFrame, band: str) -> tuple[dict[str, float], float]:
    receiver = S004.time_receiver(task_feature)
    nuisance = S004.time_receiver(energy)
    scores = {
        metric: S004.association(receiver, S004.band_features(frame, (band,), metric, True), [nuisance])
        for metric in METRICS
    }
    return scores, float(np.median(list(scores.values())))


def half_scores(task_feature: np.ndarray, energy: np.ndarray, frame: pd.DataFrame) -> tuple[float, float]:
    receiver = S004.time_receiver(task_feature)
    nuisance = S004.time_receiver(energy)
    band = S004.band_features(frame, ("Beta",), "mean", True)
    indices = (np.arange(0, 12), np.arange(13, 25))
    return tuple(float(S004.association(receiver[ix], band[ix], [nuisance[ix]])) for ix in indices)


def signflip_family(differences: dict[str, np.ndarray], rng: np.random.Generator, count: int) -> tuple[dict[str, float], dict[str, float]]:
    observed = {candidate: float(values.mean()) for candidate, values in differences.items()}
    raw_exceed = {candidate: 0 for candidate in NEW_CANDIDATES}
    fwer_exceed = {candidate: 0 for candidate in NEW_CANDIDATES}
    for _ in range(count):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=26)
        values = {candidate: float(np.mean(differences[candidate] * signs)) for candidate in NEW_CANDIDATES}
        maximum = max(values.values())
        for candidate in NEW_CANDIDATES:
            raw_exceed[candidate] += values[candidate] >= observed[candidate]
            fwer_exceed[candidate] += maximum >= observed[candidate]
    return (
        {candidate: add_one(raw_exceed[candidate], count) for candidate in NEW_CANDIDATES},
        {candidate: add_one(fwer_exceed[candidate], count) for candidate in NEW_CANDIDATES},
    )


def preflight() -> dict:
    rng = np.random.default_rng(7007)
    signal = rng.normal(size=(320, 14))
    flow = lag_flow(signal)
    reverse = lag_flow(signal[::-1])
    family = feature_family(signal)
    checks = {
        "lag_flow_reversal_odd": bool(np.allclose(flow, -reverse, atol=1e-12, rtol=1e-10)),
        "lag_flow_dimension_273": len(family["lag_flow"]) == 273,
        "conditional_gain_dimension_273": len(family["conditional_gain"]) == 273,
        "reentry_spectrum_dimension_42": len(family["reentry_spectrum"]) == 42,
        "baseline_dimension_42": len(family["third_order_odd"]) == 42,
        "all_finite": all(np.all(np.isfinite(value)) for value in family.values()),
        "all_nonconstant": all(float(np.std(value)) > 1e-12 for value in family.values()),
    }
    return {"status": "PASS_LOCAL_PREFLIGHT_NOT_INDEPENDENT_E0" if all(checks.values()) else "FAIL_LOCAL_PREFLIGHT",
            "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-signflip", type=int, default=100000)
    parser.add_argument("--n-bootstrap", type=int, default=50000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pre = preflight()
    write_json(args.output_dir / "preflight.json", pre)
    if not pre["status"].startswith("PASS"):
        raise RuntimeError("local endpoint preflight failed")

    features, raw_features, energy, frames, input_hashes, pca_audit = load_inputs(args)
    labels = [f"P{i}" for i in range(1, 27)]
    observed_rows = []
    half_rows = []
    for label in labels:
        for candidate in ALL_CANDIDATES:
            for band in BANDS:
                metric_scores, composite = participant_band_score(features[label][candidate], energy[label], frames[label], band)
                observed_rows.append({"participant": label, "candidate": candidate, "band": band,
                                      **metric_scores, "composite": composite})
            first, second = half_scores(features[label][candidate], energy[label], frames[label])
            half_rows.append({"participant": label, "candidate": candidate, "half1": first, "half2": second})
    observed = pd.DataFrame(observed_rows)
    halves = pd.DataFrame(half_rows)

    # Exact-path/result baseline: reproduce the prior third-order population band vector.
    baseline_rows = []
    for label in labels:
        for band in BANDS:
            _, composite = participant_band_score(raw_features[label]["third_order_odd"], energy[label], frames[label], band)
            baseline_rows.append({"participant": label, "band": band, "composite": composite})
    baseline = pd.DataFrame(baseline_rows).groupby("band").composite.mean().to_dict()
    baseline_errors = {band: abs(float(baseline[band]) - expected) for band, expected in BASELINE_BAND_POPULATION.items()}
    baseline_gate = {
        "status": "PASS" if max(baseline_errors.values()) <= 1e-12 else "FAIL",
        "expected": BASELINE_BAND_POPULATION,
        "observed": {band: float(value) for band, value in baseline.items()},
        "absolute_errors": baseline_errors,
    }
    write_json(args.output_dir / "baseline_gate.json", baseline_gate)
    if baseline_gate["status"] != "PASS":
        raise RuntimeError("DPA-SCOUT-004 baseline reproduction failed")

    rng = np.random.default_rng(args.seed)
    null_rows = []
    for iteration in range(args.n_perm):
        participant_scores = {candidate: [] for candidate in NEW_CANDIDATES}
        for label in labels:
            order = rng.permutation(26)
            for candidate in NEW_CANDIDATES:
                _, composite = participant_band_score(features[label][candidate][order], energy[label][order], frames[label], "Beta")
                participant_scores[candidate].append(composite)
        population = {candidate: float(np.mean(values)) for candidate, values in participant_scores.items()}
        maximum = max(population.values())
        for candidate in NEW_CANDIDATES:
            null_rows.append({"iteration": iteration, "candidate": candidate,
                              "population_beta": population[candidate], "iteration_max": maximum})
    nulls = pd.DataFrame(null_rows)

    wide = observed.pivot_table(index=["participant", "candidate"], columns="band", values="composite")
    differences = {
        candidate: (wide.loc[(slice(None), candidate), "Beta"].to_numpy()
                    - wide.loc[(slice(None), candidate), [band for band in BANDS if band != "Beta"]].mean(axis=1).to_numpy())
        for candidate in NEW_CANDIDATES
    }
    sign_raw, sign_fwer = signflip_family(differences, rng, args.n_signflip)

    summary_rows = []
    bootstrap_rows = []
    for candidate in ALL_CANDIDATES:
        group = observed[observed.candidate.eq(candidate)]
        population = group.groupby("band").composite.mean().sort_values(ascending=False)
        beta = float(population["Beta"])
        beta_rank = int(population.rank(ascending=False, method="min")["Beta"])
        beta_values = group[group.band.eq("Beta")].sort_values("participant", key=lambda s: s.str[1:].astype(int)).composite.to_numpy()
        boot = beta_values[rng.integers(0, 26, size=(args.n_bootstrap, 26))].mean(axis=1)
        ci = np.quantile(boot, [0.025, 0.5, 0.975])
        for index, value in enumerate(boot):
            bootstrap_rows.append({"iteration": index, "candidate": candidate, "population_beta": float(value)})
        h = halves[halves.candidate.eq(candidate)]
        half1, half2 = float(h.half1.mean()), float(h.half2.mean())
        if candidate in NEW_CANDIDATES:
            candidate_null = nulls[nulls.candidate.eq(candidate)]
            p_raw = add_one(int(np.sum(candidate_null.population_beta.to_numpy() >= beta)), args.n_perm)
            maxima = candidate_null.iteration_max.to_numpy()
            p_fwer = add_one(int(np.sum(maxima >= beta)), args.n_perm)
            diff_mean = float(differences[candidate].mean())
            s_raw, s_fwer = sign_raw[candidate], sign_fwer[candidate]
            passed = bool(beta > 0 and beta_rank == 1 and p_fwer <= .05 and diff_mean > 0 and s_fwer <= .05
                          and ci[0] > 0 and half1 > 0 and half2 > 0)
        else:
            p_raw = p_fwer = s_raw = s_fwer = float("nan")
            diff_mean = float(beta_values.mean() - group[~group.band.eq("Beta")].groupby("participant").composite.mean().mean())
            passed = False
        summary_rows.append({
            "candidate": candidate,
            "Beta_population": beta,
            "Beta_rank": beta_rank,
            "best_band": str(population.index[0]),
            "best_band_score": float(population.iloc[0]),
            "Beta_task_p_raw": p_raw,
            "Beta_task_p_fwer_candidates": p_fwer,
            "Beta_minus_other_mean": diff_mean,
            "specificity_signflip_p_raw": s_raw,
            "specificity_signflip_p_fwer_candidates": s_fwer,
            "Beta_bootstrap_ci_low": float(ci[0]),
            "Beta_bootstrap_median": float(ci[1]),
            "Beta_bootstrap_ci_high": float(ci[2]),
            "half1_population": half1,
            "half2_population": half2,
            "pass": passed,
            **{f"population_{band}": float(population[band]) for band in BANDS},
        })
    summary_frame = pd.DataFrame(summary_rows)
    passed_candidates = summary_frame.loc[summary_frame["pass"], "candidate"].tolist()
    decision = "SCOPED_EXPLORATORY_PASS_MIND_BETA" if passed_candidates else "NO_GO_BOUNDED_MIND_FEATURE_FAMILY"

    observed.to_csv(args.output_dir / "participant_candidate_band_scores.csv", index=False)
    halves.to_csv(args.output_dir / "participant_beta_half_scores.csv", index=False)
    nulls.to_csv(args.output_dir / "task_permutation_nulls.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "participant_bootstrap.csv", index=False)
    summary_frame.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    result = {
        "scout_id": "DPA-SCOUT-007",
        "operational_lineage": "Q-014-X3",
        "evidence_status": "Exploratory result",
        "decision": decision,
        "passed_candidates": passed_candidates,
        "population_scope": "26 participant blocks; no participant selection",
        "baseline_gate": baseline_gate,
        "pca_audit": pca_audit,
        "input_hashes": input_hashes,
        "seed": args.seed,
        "n_permutations": args.n_perm,
        "n_signflips": args.n_signflip,
        "n_bootstrap": args.n_bootstrap,
        "claim_ceiling": "Same-archive population-level EEG temporal-feature to band-summary geometry only; no thought content, neural causation, Braid Time, subjectivity, consciousness, or whole-HMB validation.",
    }
    write_json(args.output_dir / "summary.json", result)
    output_hashes = {path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}
    write_json(args.output_dir / "execution_log.json", {
        "status": "COMPLETE",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "script_sha256": sha256(Path(__file__)),
        "band_table_sha256": sha256(args.bands),
        "baseline_log_sha256": sha256(args.baseline_log),
        "output_hashes_before_execution_log": output_hashes,
    })
    print(json.dumps({"decision": decision, "passed_candidates": passed_candidates,
                      "candidates": summary_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

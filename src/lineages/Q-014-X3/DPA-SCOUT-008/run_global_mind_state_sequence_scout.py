#!/usr/bin/env python3
"""Population Mind scout using whole-brain state sequence features."""

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


BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
METRICS = ("mean", "peak", "stability")
LAGS = (4, 8, 16)
CANDIDATES = ("microstate_transition", "ordinal_vocabulary", "innovation_spectrum")
BAND_TABLE_SHA = "b012c0d63708389b4242a1f659300303dba8b5cb548571caccc978ea7ceb3ea0"
BASELINE_BAND_POPULATION = {
    "Delta": 0.010694723278990821,
    "Theta": 0.0427551525455008,
    "Alpha": 0.06626239146918708,
    "Beta": 0.03895978063561179,
    "Gamma": 0.02709895765534598,
}
PERMUTATIONS3 = list(itertools.permutations(range(3)))
PERMUTATION_CODE = {value: index for index, value in enumerate(PERMUTATIONS3)}


HERE = Path(__file__).resolve().parent
S007_PATH = HERE.parent / "DPA-SCOUT-007" / "run_global_mind_feature_scout.py"
SPEC = importlib.util.spec_from_file_location("dpa_scout_007", S007_PATH)
S007 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(S007)
S004, BODY = S007.S004, S007.BODY


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


def signflip_family(differences: dict[str, np.ndarray], rng: np.random.Generator, count: int):
    observed = {candidate: float(values.mean()) for candidate, values in differences.items()}
    raw_exceed = {candidate: 0 for candidate in CANDIDATES}
    fwer_exceed = {candidate: 0 for candidate in CANDIDATES}
    for _ in range(count):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=26)
        values = {candidate: float(np.mean(differences[candidate] * signs)) for candidate in CANDIDATES}
        maximum = max(values.values())
        for candidate in CANDIDATES:
            raw_exceed[candidate] += values[candidate] >= observed[candidate]
            fwer_exceed[candidate] += maximum >= observed[candidate]
    return ({candidate: add_one(raw_exceed[candidate], count) for candidate in CANDIDATES},
            {candidate: add_one(fwer_exceed[candidate], count) for candidate in CANDIDATES})


def session_spatial_basis(values: np.ndarray, dimension: int = 4) -> np.ndarray:
    covariance = values.T @ values / len(values)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order[:dimension]]


def microstate_transition(segment: np.ndarray, basis: np.ndarray) -> np.ndarray:
    scores = segment @ basis
    dominant = np.argmax(np.abs(scores), axis=1)
    selected = scores[np.arange(len(scores)), dominant]
    states = 2 * dominant + (selected >= 0).astype(int)
    features = []
    for lag in LAGS:
        table = np.zeros((8, 8), float)
        np.add.at(table, (states[:-lag], states[lag:]), 1.0)
        table /= max(float(table.sum()), 1.0)
        features.extend(table.ravel().tolist())
    return np.asarray(features)


def ordinal_vocabulary(segment: np.ndarray) -> np.ndarray:
    x = S004.zcols(segment)
    features = []
    for lag in LAGS:
        for channel in range(x.shape[1]):
            triples = np.column_stack([x[:-2 * lag, channel], x[lag:-lag, channel], x[2 * lag:, channel]])
            orders = np.argsort(triples, axis=1, kind="stable")
            counts = np.zeros(6, float)
            for order in orders:
                counts[PERMUTATION_CODE[tuple(int(v) for v in order)]] += 1.0
            counts /= max(float(counts.sum()), 1.0)
            features.extend(counts.tolist())
    return np.asarray(features)


def innovation_spectrum(segment: np.ndarray) -> np.ndarray:
    x = S004.zcols(segment)
    features = []
    for lag in LAGS:
        past, future = x[:-lag], x[lag:]
        alpha = 1e-3 * len(past)
        transition = np.linalg.solve(past.T @ past + alpha * np.eye(x.shape[1]), past.T @ future)
        residual = future - past @ transition
        covariance = residual.T @ residual / len(residual)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        features.extend(eigenvalues.tolist())
    return np.asarray(features)


def project_features(raw_features: dict[str, dict[str, np.ndarray]]):
    labels = [f"P{i}" for i in range(1, 27)]
    output = {label: {} for label in labels}
    audit = {}
    for candidate in CANDIDATES:
        pooled = np.vstack([raw_features[label][candidate] for label in labels])
        center, scale = pooled.mean(axis=0), pooled.std(axis=0)
        scale[scale <= 1e-12] = 1.0
        z = (pooled - center) / scale
        _, singular, vh = np.linalg.svd(z, full_matrices=False)
        rank = int(np.sum(singular > max(float(singular[0]), 1.0) * 1e-12))
        dimension = min(8, rank)
        projected = z @ vh[:dimension].T
        for index, label in enumerate(labels):
            output[label][candidate] = projected[index * 26:(index + 1) * 26]
        audit[candidate] = {"raw_dimension": int(pooled.shape[1]), "rank": rank,
                            "retained_dimension": dimension,
                            "leading_singular_values": singular[:8].tolist()}
    return output, audit


def load_inputs(args):
    if sha256(args.bands) != BAND_TABLE_SHA:
        raise RuntimeError("band table hash mismatch")
    bands = pd.read_csv(args.bands)
    baseline_log = json.loads(args.baseline_log.read_text(encoding="utf-8"))
    qdir, eegdir = args.raw_root / "IDPC_Reproduction", args.raw_root / "IDPC_EEG"
    labels = [f"P{i}" for i in range(1, 27)]
    raw_features, old_feature, energy, frames = {}, {}, {}, {}
    hashes = {"raw_eeg": {}, "windows": {}}
    for label in labels:
        raw_path = BODY.raw_file_for(eegdir, label)
        window_path = qdir / f"{label}_eeg_timeseries.csv"
        raw_hash, window_hash = sha256(raw_path), sha256(window_path)
        if raw_hash != baseline_log["raw_manifest"]["raw_eeg"].get(raw_path.name):
            raise RuntimeError(f"raw EEG mismatch {label}")
        if window_hash != baseline_log["raw_manifest"]["windows"].get(window_path.name):
            raise RuntimeError(f"window mismatch {label}")
        raw = BODY.load_raw(raw_path)
        times, values, _ = S004.preprocess(raw)
        windows = S004.task_windows(qdir, label)
        spatial_basis = session_spatial_basis(values)
        rows = {candidate: [] for candidate in CANDIDATES}
        old_rows, energy_rows = [], []
        for task in range(26):
            row = windows[windows["task_idx"].astype(int).eq(task)].iloc[0]
            start, end = np.datetime64(row["start"].to_datetime64()), np.datetime64(row["end"].to_datetime64())
            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")
            segment = values[left:right]
            if len(segment) < 103:
                raise RuntimeError(f"short segment {label} {task}")
            rows["microstate_transition"].append(microstate_transition(segment, spatial_basis))
            rows["ordinal_vocabulary"].append(ordinal_vocabulary(segment))
            rows["innovation_spectrum"].append(innovation_spectrum(segment))
            old_rows.append(S004.causality(segment))
            energy_rows.append([float(np.mean(segment * segment))])
        raw_features[label] = {candidate: np.asarray(value) for candidate, value in rows.items()}
        old_feature[label] = np.asarray(old_rows)
        energy[label] = np.asarray(energy_rows)
        frames[label] = bands[bands.label.eq(label)].sort_values("task_idx").reset_index(drop=True)
        hashes["raw_eeg"][raw_path.name] = raw_hash
        hashes["windows"][window_path.name] = window_hash
    projected, audit = project_features(raw_features)
    return projected, old_feature, energy, frames, hashes, audit


def half_scores(feature, energy, frame):
    receiver = S004.time_receiver(feature)
    nuisance = S004.time_receiver(energy)
    beta = S004.band_features(frame, ("Beta",), "mean", True)
    indices = (np.arange(0, 12), np.arange(13, 25))
    return tuple(float(S004.association(receiver[ix], beta[ix], [nuisance[ix]])) for ix in indices)


def preflight() -> dict:
    rng = np.random.default_rng(8008)
    signal = rng.normal(size=(320, 14))
    basis = session_spatial_basis(signal)
    values = {
        "microstate_transition": microstate_transition(signal, basis),
        "ordinal_vocabulary": ordinal_vocabulary(signal),
        "innovation_spectrum": innovation_spectrum(signal),
    }
    micro_sums = values["microstate_transition"].reshape(3, 64).sum(axis=1)
    ordinal_sums = values["ordinal_vocabulary"].reshape(3, 14, 6).sum(axis=2)
    checks = {
        "dimensions": [len(values[name]) for name in CANDIDATES] == [192, 252, 42],
        "all_finite": all(np.all(np.isfinite(value)) for value in values.values()),
        "all_nonconstant": all(float(np.std(value)) > 1e-12 for value in values.values()),
        "microstate_tables_normalized": bool(np.allclose(micro_sums, 1.0)),
        "ordinal_distributions_normalized": bool(np.allclose(ordinal_sums, 1.0)),
        "innovation_nonnegative": bool(np.min(values["innovation_spectrum"]) >= -1e-12),
    }
    return {"status": "PASS_LOCAL_PREFLIGHT_NOT_INDEPENDENT_E0" if all(checks.values()) else "FAIL_LOCAL_PREFLIGHT",
            "checks": checks}


def main():
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
        raise RuntimeError("preflight failed")

    features, old_feature, energy, frames, input_hashes, pca_audit = load_inputs(args)
    labels = [f"P{i}" for i in range(1, 27)]

    baseline_rows = []
    for label in labels:
        for band in BANDS:
            _, composite = S007.participant_band_score(old_feature[label], energy[label], frames[label], band)
            baseline_rows.append({"participant": label, "band": band, "composite": composite})
    baseline = pd.DataFrame(baseline_rows).groupby("band").composite.mean().to_dict()
    errors = {band: abs(float(baseline[band]) - expected) for band, expected in BASELINE_BAND_POPULATION.items()}
    baseline_gate = {"status": "PASS" if max(errors.values()) <= 1e-12 else "FAIL",
                     "expected": BASELINE_BAND_POPULATION,
                     "observed": {band: float(value) for band, value in baseline.items()},
                     "absolute_errors": errors}
    write_json(args.output_dir / "baseline_gate.json", baseline_gate)
    if baseline_gate["status"] != "PASS":
        raise RuntimeError("baseline reproduction failed")

    observed_rows, half_rows = [], []
    for label in labels:
        for candidate in CANDIDATES:
            for band in BANDS:
                scores, composite = S007.participant_band_score(features[label][candidate], energy[label], frames[label], band)
                observed_rows.append({"participant": label, "candidate": candidate, "band": band,
                                      **scores, "composite": composite})
            first, second = half_scores(features[label][candidate], energy[label], frames[label])
            half_rows.append({"participant": label, "candidate": candidate, "half1": first, "half2": second})
    observed, halves = pd.DataFrame(observed_rows), pd.DataFrame(half_rows)

    rng = np.random.default_rng(args.seed)
    null_rows = []
    for iteration in range(args.n_perm):
        participant_scores = {candidate: [] for candidate in CANDIDATES}
        for label in labels:
            order = rng.permutation(26)
            for candidate in CANDIDATES:
                _, composite = S007.participant_band_score(features[label][candidate][order], energy[label][order], frames[label], "Beta")
                participant_scores[candidate].append(composite)
        population = {candidate: float(np.mean(values)) for candidate, values in participant_scores.items()}
        maximum = max(population.values())
        for candidate in CANDIDATES:
            null_rows.append({"iteration": iteration, "candidate": candidate,
                              "population_beta": population[candidate], "iteration_max": maximum})
    nulls = pd.DataFrame(null_rows)

    wide = observed.pivot_table(index=["participant", "candidate"], columns="band", values="composite")
    differences = {
        candidate: wide.loc[(slice(None), candidate), "Beta"].to_numpy()
        - wide.loc[(slice(None), candidate), [band for band in BANDS if band != "Beta"]].mean(axis=1).to_numpy()
        for candidate in CANDIDATES
    }
    sign_raw, sign_fwer = signflip_family(differences, rng, args.n_signflip)

    summary_rows, bootstrap_rows = [], []
    for candidate in CANDIDATES:
        group = observed[observed.candidate.eq(candidate)]
        population = group.groupby("band").composite.mean().sort_values(ascending=False)
        beta = float(population["Beta"])
        rank = int(population.rank(ascending=False, method="min")["Beta"])
        beta_values = group[group.band.eq("Beta")].sort_values("participant", key=lambda s: s.str[1:].astype(int)).composite.to_numpy()
        boot = beta_values[rng.integers(0, 26, size=(args.n_bootstrap, 26))].mean(axis=1)
        ci = np.quantile(boot, [0.025, 0.5, 0.975])
        for index, value in enumerate(boot):
            bootstrap_rows.append({"iteration": index, "candidate": candidate, "population_beta": float(value)})
        candidate_null = nulls[nulls.candidate.eq(candidate)]
        p_raw = add_one(int(np.sum(candidate_null.population_beta.to_numpy() >= beta)), args.n_perm)
        p_fwer = add_one(int(np.sum(candidate_null.iteration_max.to_numpy() >= beta)), args.n_perm)
        h = halves[halves.candidate.eq(candidate)]
        half1, half2 = float(h.half1.mean()), float(h.half2.mean())
        difference = float(differences[candidate].mean())
        passed = bool(beta > 0 and rank == 1 and p_fwer <= .05 and difference > 0
                      and sign_fwer[candidate] <= .05 and ci[0] > 0 and half1 > 0 and half2 > 0)
        summary_rows.append({
            "candidate": candidate,
            "Beta_population": beta,
            "Beta_rank": rank,
            "best_band": str(population.index[0]),
            "best_band_score": float(population.iloc[0]),
            "Beta_task_p_raw": p_raw,
            "Beta_task_p_fwer_candidates": p_fwer,
            "Beta_minus_other_mean": difference,
            "specificity_signflip_p_raw": sign_raw[candidate],
            "specificity_signflip_p_fwer_candidates": sign_fwer[candidate],
            "Beta_bootstrap_ci_low": float(ci[0]),
            "Beta_bootstrap_median": float(ci[1]),
            "Beta_bootstrap_ci_high": float(ci[2]),
            "half1_population": half1,
            "half2_population": half2,
            "pass": passed,
            **{f"population_{band}": float(population[band]) for band in BANDS},
        })
    summary_frame = pd.DataFrame(summary_rows)
    passed = summary_frame.loc[summary_frame["pass"], "candidate"].tolist()
    decision = "SCOPED_EXPLORATORY_PASS_MIND_BETA" if passed else "NO_GO_MIND_STATE_SEQUENCE_FAMILY"

    observed.to_csv(args.output_dir / "participant_candidate_band_scores.csv", index=False)
    halves.to_csv(args.output_dir / "participant_beta_half_scores.csv", index=False)
    nulls.to_csv(args.output_dir / "task_permutation_nulls.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "participant_bootstrap.csv", index=False)
    summary_frame.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    write_json(args.output_dir / "summary.json", {
        "scout_id": "DPA-SCOUT-008", "operational_lineage": "Q-014-X3",
        "evidence_status": "Exploratory result", "decision": decision,
        "passed_candidates": passed, "population_scope": "26 participant blocks; no participant selection",
        "baseline_gate": baseline_gate, "pca_audit": pca_audit, "input_hashes": input_hashes,
        "seed": args.seed, "n_permutations": args.n_perm,
        "n_signflips": args.n_signflip, "n_bootstrap": args.n_bootstrap,
        "claim_ceiling": "Same-archive population-level EEG state-sequence to band-summary geometry only; no thought content, causation, Braid Time, subjectivity, consciousness, or whole-HMB validation."
    })
    output_hashes = {path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}
    write_json(args.output_dir / "execution_log.json", {
        "status": "COMPLETE", "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "script_sha256": sha256(Path(__file__)),
        "band_table_sha256": sha256(args.bands), "baseline_log_sha256": sha256(args.baseline_log),
        "output_hashes_before_execution_log": output_hashes,
    })
    print(json.dumps({"decision": decision, "passed_candidates": passed,
                      "candidates": summary_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

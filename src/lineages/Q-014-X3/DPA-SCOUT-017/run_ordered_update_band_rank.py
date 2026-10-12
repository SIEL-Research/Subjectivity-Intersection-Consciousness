#!/usr/bin/env python3
"""Forward-minus-reverse innovation test against five EEG band means."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
BAND_SHA = "b012c0d63708389b4242a1f659300303dba8b5cb548571caccc978ea7ceb3ea0"
BASELINE_LOG_SHA = "5173d94477a1b9b041888e302c3e1e7731f7ee08e3652ec50377acd5b5c564da"

HERE = Path(__file__).resolve().parent
S008_PATH = HERE.parent / "DPA-SCOUT-008" / "run_global_mind_state_sequence_scout.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


S008 = load_module("dpa_s008_ordered_update", S008_PATH)
S004, BODY = S008.S004, S008.BODY


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(buffer.getvalue().encode("utf-8"))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, float) - np.mean(left)
    right = np.asarray(right, float) - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 1e-12 else 0.0


def residual_z(values: np.ndarray, energy: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), energy])
    residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    scale = float(residual.std())
    return residual / scale if scale > 1e-12 else np.zeros_like(residual)


def innovation_load(segment: np.ndarray) -> float:
    return float(np.mean(S008.innovation_spectrum(segment)))


def ordered_update(segment: np.ndarray) -> tuple[float, float, float]:
    forward = innovation_load(segment)
    reverse = innovation_load(segment[::-1])
    return forward - reverse, forward, reverse


def preflight() -> dict:
    rng = np.random.default_rng(1717)
    n, dimension = 2500, 14
    transition = np.zeros((dimension, dimension))
    transition[np.arange(dimension), np.arange(dimension)] = .5
    transition[np.arange(dimension - 1), np.arange(1, dimension)] = .8
    values = np.zeros((n, dimension))
    for index in range(n - 1):
        values[index + 1] = values[index] @ transition + rng.normal(scale=.2, size=dimension)
    score, forward, reverse = ordered_update(values[300:])
    reversed_score, _, _ = ordered_update(values[300:][::-1])
    checks = {"asymmetric_witness_nonzero": abs(score) > .05,
              "reversal_negates": abs(score + reversed_score) < 1e-12,
              "finite": bool(np.isfinite([score, forward, reverse]).all())}
    return {"status": "PASS_LOCAL_PREFLIGHT" if all(checks.values()) else "FAIL_LOCAL_PREFLIGHT",
            "checks": checks, "values": {"ordered_update": score, "forward": forward, "reverse": reverse}}


def load_rows(args):
    if sha256(args.bands) != BAND_SHA or sha256(args.baseline_log) != BASELINE_LOG_SHA:
        raise RuntimeError("input hash mismatch")
    bands = pd.read_csv(args.bands)
    baseline = json.loads(args.baseline_log.read_text(encoding="utf-8"))["raw_manifest"]
    qdir, eegdir = args.raw_root / "IDPC_Reproduction", args.raw_root / "IDPC_EEG"
    rows, hashes = [], {"raw_eeg": {}, "windows": {}}
    for participant in range(1, 27):
        label = f"P{participant}"
        raw_path = BODY.raw_file_for(eegdir, label)
        window_path = qdir / f"{label}_eeg_timeseries.csv"
        if sha256(raw_path) != baseline["raw_eeg"].get(raw_path.name):
            raise RuntimeError(f"raw EEG mismatch {label}")
        if sha256(window_path) != baseline["windows"].get(window_path.name):
            raise RuntimeError(f"window mismatch {label}")
        raw = BODY.load_raw(raw_path)
        times, values, _ = S004.preprocess(raw)
        windows = S004.task_windows(qdir, label)
        frame = bands[bands.label.eq(label)].sort_values("task_idx").reset_index(drop=True)
        if len(frame) != 26:
            raise RuntimeError(f"band rows mismatch {label}")
        for task in range(26):
            window = windows[windows.task_idx.astype(int).eq(task)].iloc[0]
            start, end = np.datetime64(window.start.to_datetime64()), np.datetime64(window.end.to_datetime64())
            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, end, side="right")
            segment = values[left:right]
            if len(segment) < 103:
                raise RuntimeError(f"short task segment {label} {task}")
            score, forward, reverse = ordered_update(segment)
            rows.append({"participant": label, "task_idx": task,
                         "ordered_update": score, "forward_innovation": forward,
                         "reverse_innovation": reverse,
                         "broadband_energy": float(np.mean(segment * segment)),
                         **{f"{band}_mean": float(frame.iloc[task][f"{band}_mean"]) for band in BANDS}})
        hashes["raw_eeg"][raw_path.name] = sha256(raw_path)
        hashes["windows"][window_path.name] = sha256(window_path)
    return pd.DataFrame(rows), hashes


def build_arrays(rows: pd.DataFrame):
    update, outcomes, retained = [], {band: [] for band in BANDS}, []
    for participant in range(1, 27):
        label = f"P{participant}"
        frame = rows[rows.participant.eq(label)].sort_values("task_idx").reset_index(drop=True)
        energy = frame.broadband_energy.to_numpy(float)
        current_update = residual_z(frame.ordered_update.to_numpy(float), energy)
        current_bands = {band: residual_z(frame[f"{band}_mean"].to_numpy(float), energy) for band in BANDS}
        update.append(current_update)
        for band in BANDS:
            outcomes[band].append(current_bands[band])
        for task in range(26):
            retained.append({**frame.iloc[task].to_dict(), "ordered_update_residual_z": current_update[task],
                             **{f"{band}_residual_z": current_bands[band][task] for band in BANDS}})
    return np.asarray(update), {band: np.asarray(value) for band, value in outcomes.items()}, pd.DataFrame(retained)


def participant_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray([correlation(x[i], y[i]) for i in range(26)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=50000)
    parser.add_argument("--n-signflip", type=int, default=100000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pre = preflight()
    write_json(args.output_dir / "preflight.json", pre)
    if not pre["status"].startswith("PASS"):
        raise RuntimeError("preflight failed")

    rows, hashes = load_rows(args)
    update, bands, retained = build_arrays(rows)
    participant = {band: participant_correlations(update, bands[band]) for band in BANDS}
    observed = {band: float(value.mean()) for band, value in participant.items()}
    rank = 1 + sorted(observed.values(), reverse=True).index(observed["Beta"])
    halves = {}
    for name, indices in (("first", np.arange(13)), ("second", np.arange(13, 26))):
        halves[name] = {band: float(np.mean([correlation(update[p, indices], bands[band][p, indices])
                                             for p in range(26)])) for band in BANDS}

    rng = np.random.default_rng(args.seed)
    perm_rows, exceed = [], 0
    for iteration in range(args.n_perm):
        order = np.argsort(rng.random((26, 26)), axis=1)
        permuted = np.take_along_axis(update, order, axis=1)
        values = {band: float(participant_correlations(permuted, bands[band]).mean()) for band in BANDS}
        maximum = max(abs(value) for value in values.values())
        exceed += maximum >= abs(observed["Beta"])
        perm_rows.append({"iteration": iteration, **values, "max_abs": maximum})
    permutation_p = (exceed + 1) / (args.n_perm + 1)

    boot_indices = rng.integers(0, 26, size=(args.n_bootstrap, 26))
    beta_boot = participant["Beta"][boot_indices].mean(axis=1)
    contrast_participant = participant["Beta"] - participant["Alpha"]
    contrast_boot = contrast_participant[boot_indices].mean(axis=1)
    beta_ci = np.quantile(beta_boot, [.025, .5, .975]).tolist()
    contrast_ci = np.quantile(contrast_boot, [.025, .5, .975]).tolist()
    contrast_observed = float(contrast_participant.mean())

    signflip_rows, signflip_exceed = [], 0
    for iteration in range(args.n_signflip):
        value = float(np.mean(contrast_participant * rng.choice(np.asarray([-1.0, 1.0]), size=26)))
        signflip_exceed += value >= contrast_observed
        signflip_rows.append({"iteration": iteration, "Beta_minus_Alpha": value})
    contrast_p = (signflip_exceed + 1) / (args.n_signflip + 1)

    core_gates = {"A_beta_positive": observed["Beta"] > 0,
                  "B_beta_rank1": rank == 1,
                  "C_five_band_max_permutation": permutation_p <= .05,
                  "D_beta_both_halves_positive": halves["first"]["Beta"] > 0 and halves["second"]["Beta"] > 0,
                  "E_beta_bootstrap_lower_positive": beta_ci[0] > 0,
                  "F_provenance_completeness_direction_retention": len(rows) == 676 and
                                                                    len(hashes["raw_eeg"]) == 26 and
                                                                    len(hashes["windows"]) == 26 and
                                                                    float(np.std(update)) > 1e-12 and
                                                                    len(perm_rows) == args.n_perm and
                                                                    len(signflip_rows) == args.n_signflip}
    specificity_gates = {"S_beta_exceeds_alpha": contrast_observed > 0,
                         "S_signflip_p": contrast_p <= .05,
                         "S_bootstrap_lower_positive": contrast_ci[0] > 0}
    provenance = core_gates["F_provenance_completeness_direction_retention"]
    core_pass = all(core_gates.values())
    specific = all(specificity_gates.values())
    if not provenance:
        decision = "FAILED_IMPLEMENTATION_OR_PROVENANCE"
    elif core_pass and specific:
        decision = "SCOPED_EXPLORATORY_PASS_ORDERED_UPDATE_BETA_RANK1_AND_SPECIFIC"
    elif core_pass:
        decision = "SCOPED_EXPLORATORY_PASS_ORDERED_UPDATE_BETA_RANK1_NOT_SPECIFIC"
    else:
        decision = "NO_GO_ORDERED_UPDATE_BETA_RANK1_ON_FIXED_ENDPOINT"

    retained.to_csv(args.output_dir / "per_task_ordered_update.csv", index=False)
    pd.DataFrame([{"participant": f"P{i + 1}", **{band: participant[band][i] for band in BANDS},
                   "Beta_minus_Alpha": contrast_participant[i]} for i in range(26)]).to_csv(
                       args.output_dir / "participant_correlations.csv", index=False)
    write_csv_gz(pd.DataFrame(perm_rows), args.output_dir / "permutation_nulls.csv.gz")
    write_csv_gz(pd.DataFrame({"iteration": np.arange(args.n_bootstrap), "Beta": beta_boot,
                               "Beta_minus_Alpha": contrast_boot}), args.output_dir / "participant_bootstrap.csv.gz")
    write_csv_gz(pd.DataFrame(signflip_rows), args.output_dir / "specificity_signflips.csv.gz")
    summary = {"scout_id": "DPA-SCOUT-017", "evidence_status": "Exploratory result",
               "decision": decision, "endpoint": "forward_minus_reverse_innovation_load",
               "population_correlations": observed, "beta_rank_of_5": rank,
               "five_band_max_permutation_p": permutation_p, "half_scores": halves,
               "beta_bootstrap_ci": beta_ci, "beta_minus_alpha": contrast_observed,
               "beta_minus_alpha_signflip_p": contrast_p,
               "beta_minus_alpha_bootstrap_ci": contrast_ci,
               "ordered_update_descriptives": {"mean": float(rows.ordered_update.mean()),
                                                "std": float(rows.ordered_update.std()),
                                                "positive_tasks": int((rows.ordered_update > 0).sum())},
               "core_gates": core_gates, "specificity_gates": specificity_gates,
               "n_permutations": args.n_perm, "n_bootstrap": args.n_bootstrap,
               "n_signflips": args.n_signflip, "seed": args.seed,
               "input_hashes": {"baseline_log": sha256(args.baseline_log), "band_table": sha256(args.bands), **hashes},
               "claim_ceiling": "Same-archive exploratory forward-minus-reverse EEG update to band-mean association; no thought, subjective Mind, interventionist causality, physical or Braid Time, consciousness, confirmation or independent replication."}
    write_json(args.output_dir / "summary.json", summary)
    output_hashes = {path.name: sha256(path) for path in args.output_dir.iterdir() if path.is_file()}
    write_json(args.output_dir / "execution_log.json", {"status": "COMPLETE", "python": platform.python_version(),
               "numpy": np.__version__, "pandas": pd.__version__, "script_sha256": sha256(Path(__file__)),
               "output_hashes_before_execution_log": output_hashes})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

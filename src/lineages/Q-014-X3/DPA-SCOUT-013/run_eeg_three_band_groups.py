#!/usr/bin/env python3
"""EEG-only population test of Delta+Theta | Beta | Alpha+Gamma."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
AXES = ("Heart", "Mind", "Body")
FIXED = (("Delta", "Theta"), ("Beta",), ("Alpha", "Gamma"))
TASK_SHA = "fe23efab633a31046c9b9c25a9f2ef02133d8c0f702c2db35968f478cc74e698"
BAND_SHA = "b012c0d63708389b4242a1f659300303dba8b5cb548571caccc978ea7ceb3ea0"
BASELINE_SHA = "949440d9ed8987c33fbc525c1a68c3bda705c0024fc6543107eb426ef98dc3b1"

HERE = Path(__file__).resolve().parent
S004_PATH = HERE.parent / "DPA-SCOUT-004" / "run_eeg_native_dcs_scout.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


S004 = load_module("dpa_s004_three_groups", S004_PATH)
BODY = S004.BODY


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


def centered_distance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    scale = values.std(axis=0, keepdims=True)
    scale[scale <= 1e-12] = 1.0
    z = (values - values.mean(axis=0, keepdims=True)) / scale
    delta = z[:, None, :] - z[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    return distance - distance.mean(axis=0, keepdims=True) - distance.mean(axis=1, keepdims=True) + distance.mean()


def upper(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[np.triu_indices(len(matrix), 1)], float)


def residual(vector: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(vector)), nuisance])
    return vector - design @ np.linalg.lstsq(design, vector, rcond=None)[0]


def residual_distance(values: np.ndarray, energy: np.ndarray) -> np.ndarray:
    n = len(values)
    value_distance = upper(centered_distance(values))
    energy_distance = upper(centered_distance(energy))
    vector = residual(value_distance, energy_distance)
    matrix = np.zeros((n, n), float)
    indices = np.triu_indices(n, 1)
    matrix[indices] = vector
    matrix[(indices[1], indices[0])] = vector
    return matrix


def vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 1e-12 else 0.0


def group_key(group: tuple[str, ...]) -> str:
    return "+".join(group)


def canonical(group) -> tuple[str, ...]:
    return tuple(sorted(group, key=BANDS.index))


def assignments():
    rows = []
    for mind in BANDS:
        remaining = [band for band in BANDS if band != mind]
        for heart in itertools.combinations(remaining, 2):
            body = tuple(band for band in remaining if band not in heart)
            rows.append((canonical(heart), (mind,), canonical(body)))
    return rows


def preflight() -> dict:
    rng = np.random.default_rng(1313)
    latent = rng.normal(size=(26, 2))
    energy = rng.normal(size=(26, 1))
    observed = vector_correlation(upper(residual_distance(latent, energy)),
                                  upper(residual_distance(latent + rng.normal(scale=.01, size=latent.shape), energy)))
    permuted = vector_correlation(upper(residual_distance(latent, energy)),
                                  upper(residual_distance(latent[rng.permutation(26)], energy)))
    checks = {"positive_witness": observed > .9, "positive_exceeds_permuted": observed > permuted,
              "assignment_count_30": len(assignments()) == 30,
              "fixed_assignment_present": FIXED in assignments()}
    return {"status": "PASS_LOCAL_PREFLIGHT" if all(checks.values()) else "FAIL_LOCAL_PREFLIGHT",
            "checks": checks, "values": {"positive": observed, "permuted": permuted}}


def load_inputs(args):
    if sha256(args.task_table) != TASK_SHA or sha256(args.bands) != BAND_SHA or sha256(args.baseline) != BASELINE_SHA:
        raise RuntimeError("input hash mismatch")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["source_manifest"]
    task = pd.read_csv(args.task_table)
    bands = pd.read_csv(args.bands)
    qdir, eegdir = args.raw_root / "IDPC_Reproduction", args.raw_root / "IDPC_EEG"
    features, band_values, energies = {axis: [] for axis in AXES}, [], []
    hashes = {"raw_eeg": {}, "task_windows": {}}
    labels = [f"P{i}" for i in range(1, 27)]
    for label in labels:
        raw_path = BODY.raw_file_for(eegdir, label)
        if sha256(raw_path) != baseline["raw_eeg"].get(raw_path.name):
            raise RuntimeError(f"raw hash mismatch {label}")
        raw = BODY.load_raw(raw_path)
        windows = S004.task_windows(qdir, label)
        duality, _, separation, energy = S004.extract_eeg_features(raw, windows)
        mind_frame = task[task.participant.eq(label)].sort_values("task_idx")
        band_frame = bands[bands.label.eq(label)].sort_values("task_idx")
        if len(mind_frame) != 26 or len(band_frame) != 26:
            raise RuntimeError(f"incomplete task block {label}")
        mind = mind_frame[["microstate_switch_rate", "innovation_load"]].to_numpy(float)
        band = band_frame[[f"{name}_mean" for name in BANDS]].to_numpy(float)
        if any(np.any(~np.isfinite(value)) for value in (duality, mind, separation, energy, band)):
            raise RuntimeError(f"nonfinite input {label}")
        features["Heart"].append(duality)
        features["Mind"].append(mind)
        features["Body"].append(separation)
        band_values.append(band)
        energies.append(energy)
        window_path = qdir / f"{label}_eeg_timeseries.csv"
        hashes["raw_eeg"][raw_path.name] = sha256(raw_path)
        hashes["task_windows"][window_path.name] = sha256(window_path)
    return ({axis: np.asarray(value) for axis, value in features.items()}, np.asarray(band_values), np.asarray(energies), hashes)


def residual_blocks(features, bands, energies, indices=None):
    if indices is None:
        indices = np.arange(26)
    two_groups = [tuple(group) for group in itertools.combinations(BANDS, 2)]
    one_groups = [(band,) for band in BANDS]
    band_index = {band: i for i, band in enumerate(BANDS)}
    feature_vectors = {axis: [] for axis in AXES}
    group_matrices = []
    for participant in range(26):
        energy = energies[participant, indices]
        feature_vectors["Heart"].append(upper(residual_distance(features["Heart"][participant, indices], energy)))
        feature_vectors["Mind"].append(upper(residual_distance(features["Mind"][participant, indices], energy)))
        feature_vectors["Body"].append(upper(residual_distance(features["Body"][participant, indices], energy)))
        current = {}
        for group in [*two_groups, *one_groups]:
            columns = [band_index[band] for band in group]
            current[group] = residual_distance(bands[participant, indices][:, columns], energy)
        group_matrices.append(current)
    return feature_vectors, group_matrices, two_groups, one_groups


def group_scores(feature_vectors, group_matrices, order_by_participant=None):
    two_groups = [tuple(group) for group in itertools.combinations(BANDS, 2)]
    one_groups = [(band,) for band in BANDS]
    participant_scores = {axis: {} for axis in AXES}
    for axis, groups in (("Heart", two_groups), ("Mind", one_groups), ("Body", two_groups)):
        for group in groups:
            scores = []
            for participant in range(26):
                matrix = group_matrices[participant][group]
                if order_by_participant is not None:
                    order = order_by_participant[participant]
                    matrix = matrix[np.ix_(order, order)]
                scores.append(vector_correlation(feature_vectors[axis][participant], upper(matrix)))
            participant_scores[axis][group] = np.asarray(scores)
    population = {axis: {group: float(values.mean()) for group, values in groups.items()}
                  for axis, groups in participant_scores.items()}
    return population, participant_scores


def assignment_table(population):
    rows = []
    for heart, mind, body in assignments():
        scores = [population["Heart"][heart], population["Mind"][mind], population["Body"][body]]
        rows.append({"heart_group": group_key(heart), "mind_group": group_key(mind), "body_group": group_key(body),
                     "Heart_score": scores[0], "Mind_score": scores[1], "Body_score": scores[2],
                     "joint_min": min(scores)})
    return pd.DataFrame(rows).sort_values("joint_min", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--task-table", type=Path, required=True)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=50000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pre = preflight()
    write_json(args.output_dir / "preflight.json", pre)
    if not pre["status"].startswith("PASS"):
        raise RuntimeError("preflight failed")
    features, bands, energies, hashes = load_inputs(args)
    feature_vectors, group_matrices, _, _ = residual_blocks(features, bands, energies)
    population, participant_scores = group_scores(feature_vectors, group_matrices)
    table = assignment_table(population)
    fixed_groups = dict(zip(AXES, FIXED))
    fixed_scores = {axis: population[axis][fixed_groups[axis]] for axis in AXES}
    fixed_joint = min(fixed_scores.values())
    fixed_mask = (table.heart_group.eq(group_key(FIXED[0])) & table.mind_group.eq(group_key(FIXED[1])) & table.body_group.eq(group_key(FIXED[2])))
    fixed_rank = 1 + int(np.where(table.index.to_numpy() == table[fixed_mask].index[0])[0][0])
    axis_ranks = {axis: 1 + sorted(population[axis].values(), reverse=True).index(fixed_scores[axis]) for axis in AXES}

    halves = {}
    for name, indices in (("first", np.arange(13)), ("second", np.arange(13, 26))):
        fv, gm, _, _ = residual_blocks(features, bands, energies, indices)
        pop, _ = group_scores(fv, gm)
        halves[name] = {axis: pop[axis][fixed_groups[axis]] for axis in AXES}

    rng = np.random.default_rng(args.seed)
    null_rows = []
    axis_exceed = {axis: 0 for axis in AXES}
    joint_exceed = 0
    for iteration in range(args.n_perm):
        orders = [rng.permutation(26) for _ in range(26)]
        null_population, _ = group_scores(feature_vectors, group_matrices, orders)
        axis_max = {axis: max(null_population[axis].values()) for axis in AXES}
        null_table = assignment_table(null_population)
        joint_max = float(null_table.joint_min.max())
        for axis in AXES:
            axis_exceed[axis] += axis_max[axis] >= fixed_scores[axis]
        joint_exceed += joint_max >= fixed_joint
        null_rows.append({"iteration": iteration, **{f"{axis}_fixed": null_population[axis][fixed_groups[axis]] for axis in AXES},
                          **{f"{axis}_max": axis_max[axis] for axis in AXES}, "assignment_joint_max": joint_max})
    axis_p = {axis: (axis_exceed[axis] + 1) / (args.n_perm + 1) for axis in AXES}
    joint_p = (joint_exceed + 1) / (args.n_perm + 1)

    fixed_participant = np.column_stack([participant_scores[axis][fixed_groups[axis]] for axis in AXES])
    bootstrap_indices = rng.integers(0, 26, size=(args.n_bootstrap, 26))
    bootstrap_axis = fixed_participant[bootstrap_indices].mean(axis=1)
    bootstrap_joint = bootstrap_axis.min(axis=1)
    bootstrap_ci = {axis: np.quantile(bootstrap_axis[:, i], [0.025, 0.5, 0.975]).tolist() for i, axis in enumerate(AXES)}
    bootstrap_ci["joint_min"] = np.quantile(bootstrap_joint, [0.025, 0.5, 0.975]).tolist()
    bootstrap_rows = pd.DataFrame({"iteration": np.arange(args.n_bootstrap),
                                   **{axis: bootstrap_axis[:, i] for i, axis in enumerate(AXES)}, "joint_min": bootstrap_joint})

    gates = {"A_all_positive": min(fixed_scores.values()) > 0,
             "B_all_axis_rank1_and_assignment_rank1": all(rank == 1 for rank in axis_ranks.values()) and fixed_rank == 1,
             "C_axis_and_joint_permutation": all(value <= .05 for value in axis_p.values()) and joint_p <= .05,
             "D_all_halves_positive": all(halves[half][axis] > 0 for half in halves for axis in AXES),
             "E_all_bootstrap_lower_positive": all(bootstrap_ci[key][0] > 0 for key in [*AXES, "joint_min"]),
             "F_provenance_non_degeneracy_retention": len(hashes["raw_eeg"]) == 26 and len(hashes["task_windows"]) == 26 and
                                                     len(null_rows) == args.n_perm and fixed_participant.shape == (26, 3) and
                                                     all(np.std(features[axis]) > 1e-12 for axis in AXES)}
    science_pass = all(value for key, value in gates.items() if not key.startswith("F_"))
    decision = ("SCOPED_EXPLORATORY_PASS_EEG_THREE_BAND_GROUPS" if gates["F_provenance_non_degeneracy_retention"] and science_pass
                else "NO_GO_EEG_THREE_BAND_GROUPS_ON_FIXED_ENDPOINT" if gates["F_provenance_non_degeneracy_retention"]
                else "FAILED_IMPLEMENTATION_OR_PROVENANCE")

    axis_rows = []
    for axis in AXES:
        for group, value in population[axis].items():
            axis_rows.append({"axis": axis, "band_group": group_key(group), "population_score": value,
                              "rank": 1 + sorted(population[axis].values(), reverse=True).index(value),
                              "is_fixed": group == fixed_groups[axis]})
    participant_rows = []
    for participant in range(26):
        participant_rows.append({"participant": f"P{participant + 1}",
                                 **{axis: fixed_participant[participant, i] for i, axis in enumerate(AXES)}})
    pd.DataFrame(axis_rows).to_csv(args.output_dir / "axis_group_scores.csv", index=False)
    table.to_csv(args.output_dir / "assignment_scores.csv", index=False)
    pd.DataFrame(participant_rows).to_csv(args.output_dir / "per_participant_fixed_scores.csv", index=False)
    write_csv_gz(pd.DataFrame(null_rows), args.output_dir / "permutation_nulls.csv.gz")
    write_csv_gz(bootstrap_rows, args.output_dir / "participant_bootstrap.csv.gz")
    summary = {"scout_id": "DPA-SCOUT-013", "evidence_status": "Exploratory result", "decision": decision,
               "fixed_mapping": {"Heart": "Delta+Theta", "Mind": "Beta", "Body": "Alpha+Gamma"},
               "fixed_population_scores": fixed_scores, "fixed_joint_min": fixed_joint,
               "axis_group_ranks": axis_ranks, "assignment_rank_of_30": fixed_rank,
               "axis_max_family_permutation_p": axis_p, "joint_assignment_max_permutation_p": joint_p,
               "half_scores": halves, "participant_bootstrap_ci": bootstrap_ci, "gates": gates,
               "n_permutations": args.n_perm, "n_bootstrap": args.n_bootstrap, "seed": args.seed,
               "input_hashes": {"task_table": sha256(args.task_table), "band_table": sha256(args.bands),
                                "baseline": sha256(args.baseline), **hashes},
               "claim_ceiling": "Same-archive EEG-only result-informed exploratory feature/band correspondence; no subjective contents, Braid, light/time/space, causality, consciousness, confirmation or independent replication."}
    write_json(args.output_dir / "summary.json", summary)
    output_hashes = {path.name: sha256(path) for path in args.output_dir.iterdir() if path.is_file()}
    write_json(args.output_dir / "execution_log.json", {"status": "COMPLETE", "python": platform.python_version(),
               "numpy": np.__version__, "pandas": pd.__version__, "script_sha256": sha256(Path(__file__)),
               "output_hashes_before_execution_log": output_hashes})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

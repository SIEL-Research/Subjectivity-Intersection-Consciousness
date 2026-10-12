#!/usr/bin/env python3
"""EEG-native Duality/Causality/Separation correspondence scout."""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


MASKS = (0, 5, 8, 13, 16, 21, 24, 29)
PAIRS = ((5, 8), (5, 16), (8, 16))
QCOLS = ("p00", "p01", "p10", "p11")
BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
METRICS = ("mean", "peak", "stability")
AXES = ("Duality", "Causality", "Separation")
CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")
LR_PAIRS = (("AF3", "AF4"), ("F7", "F8"), ("F3", "F4"), ("FC5", "FC6"), ("T7", "T8"), ("P7", "P8"), ("O1", "O2"))
SOURCE_SHA = "7ba1c987a9389f7426becdde136986be7fbc7b17b92dac938cc2a43cc39b373e"
BAND_TABLE_SHA = "b012c0d63708389b4242a1f659300303dba8b5cb548571caccc978ea7ceb3ea0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


HERE = Path(__file__).resolve().parent
BODY_SCRIPT = HERE.parent / "DPA-SCOUT-002" / "run_body_raw_scout.py"
SPEC = importlib.util.spec_from_file_location("body_raw", BODY_SCRIPT)
BODY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BODY)


def zcols(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    scale = values.std(axis=0, keepdims=True)
    scale[scale <= 1e-12] = 1.0
    return (values - values.mean(axis=0, keepdims=True)) / scale


def centered_distance(values: np.ndarray) -> np.ndarray:
    x = zcols(np.asarray(values, float))
    delta = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(delta * delta, axis=2))
    return d - d.mean(axis=0, keepdims=True) - d.mean(axis=1, keepdims=True) + d.mean()


def upper(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[np.triu_indices(len(matrix), 1)], float)


def residual(vector: np.ndarray, nuisances: list[np.ndarray]) -> np.ndarray:
    usable = [np.asarray(x, float).reshape(-1) for x in nuisances if np.linalg.norm(x) > 1e-12]
    design = np.column_stack([np.ones(len(vector)), *usable])
    return vector - design @ np.linalg.lstsq(design, vector, rcond=None)[0]


def association(left: np.ndarray, right: np.ndarray, nuisance_features: list[np.ndarray]) -> float:
    dl, dr = upper(centered_distance(left)), upper(centered_distance(right))
    dns = [upper(centered_distance(item)) for item in nuisance_features]
    rl, rr = residual(dl, dns), residual(dr, dns)
    denom = float(np.linalg.norm(rl) * np.linalg.norm(rr))
    return float(rl @ rr / denom) if denom > 1e-12 else 0.0


def source_matrices(source_path: Path) -> dict[int, np.ndarray]:
    if sha256(source_path) != SOURCE_SHA:
        raise RuntimeError("source hash mismatch")
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    records = {int(item["mask"]): item for item in raw["carrier_records"]}
    matrices = {}
    for mask in MASKS:
        gram = [[F(value) for value in row] for row in records[mask]["ordered_edge_gram"]]
        matrices[mask] = np.asarray([[float(gram[i][j] ** 2 / (gram[i][i] * gram[j][j]) - int(i == j))
                                     for j in range(4)] for i in range(4)])
    return matrices


def generic_matrices() -> dict[int, np.ndarray]:
    output = {}
    for ordinal, mask in enumerate(MASKS):
        vectors = np.asarray([[int(i == j) + (ordinal + 1) * (i + 1) for j in range(4)] for i in range(4)], float)
        gram = vectors @ vectors.T
        corr = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
        output[mask] = corr * corr - np.eye(4)
    return output


def commutator(values: np.ndarray, direction: int) -> np.ndarray:
    index = {mask: i for i, mask in enumerate(MASKS)}
    out = np.zeros((8, 8), float)
    for mask, column in index.items():
        row = index[mask ^ direction]
        out[row, column] = values[row] - values[column]
    return out


def mixed(oa, ea, ob, eb, d, e):
    a, b = commutator(oa, d), commutator(eb, e)
    c, f = commutator(ob, d), commutator(ea, e)
    return (a @ b - b @ a) - (c @ f - f @ c)


def source_packet(probability: np.ndarray, matrices: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    centered = probability - 0.25
    e0 = np.full(4, 0.5)
    odd = np.column_stack([centered @ matrices[m].T @ e0 for m in MASKS])
    even = np.column_stack([np.einsum("ni,ij,nj->n", centered, matrices[m], centered) for m in MASKS])
    radius = np.linalg.norm(centered, axis=1)[:, None]
    time = []
    for task in range(25):
        time.append(np.concatenate([mixed(odd[task], even[task], odd[task + 1], even[task + 1], d, e).reshape(-1)
                                    for d, e in PAIRS]))
    return {"Light": odd, "Space": even, "time_raw": np.asarray(time), "radius": radius}


def preprocess(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    columns = [f"EEG.{channel}" for channel in CHANNELS]
    values = raw[columns].to_numpy(float)
    for column in range(values.shape[1]):
        values[:, column] = BODY.winsorize(values[:, column])
        good = np.isfinite(values[:, column])
        indices = np.arange(len(values))
        values[~good, column] = np.interp(indices[~good], indices[good], values[good, column])
    values -= np.median(values, axis=0, keepdims=True)
    times = raw["TimeStampUTC"].to_numpy(dtype="datetime64[ns]")
    fs = BODY.estimate_fs(times)
    filtered = np.column_stack([BODY.bandpass(values[:, column], fs) for column in range(values.shape[1])])
    filtered = zcols(filtered)
    return times, filtered, fs


def duality(segment: np.ndarray) -> np.ndarray:
    index = {channel: i for i, channel in enumerate(CHANNELS)}
    asymmetry = []
    for left, right in LR_PAIRS:
        x, y = segment[:, index[left]], segment[:, index[right]]
        left_rms, right_rms = float(np.sqrt(np.mean(x * x))), float(np.sqrt(np.mean(y * y)))
        asymmetry.append((left_rms - right_rms) / max(left_rms + right_rms, 1e-12))
    return np.asarray(asymmetry)


def causality(segment: np.ndarray) -> np.ndarray:
    x = zcols(segment)
    features = []
    for lag in (4, 8, 16):
        if len(x) <= lag + 4:
            raise RuntimeError("short segment for causality")
        past, future = x[:-lag], x[lag:]
        odd = np.mean(past * past * future - past * future * future, axis=0)
        scale = np.maximum(np.mean((past * past + future * future) / 2.0, axis=0) ** 1.5, 1e-12)
        features.extend((odd / scale).tolist())
    return np.asarray(features)


def separation(segment: np.ndarray) -> np.ndarray:
    return np.asarray([float(np.sqrt(np.mean((segment[:, i] - segment[:, j]) ** 2)))
                       for i in range(len(CHANNELS)) for j in range(i + 1, len(CHANNELS))])


def task_windows(qdir: Path, label: str) -> pd.DataFrame:
    path = qdir / f"{label}_eeg_timeseries.csv"
    windows = pd.read_csv(path, usecols=["task_idx", "bin_start_utc", "bin_end_utc"])
    windows["bin_start_utc"] = pd.to_datetime(windows["bin_start_utc"], utc=True, format="mixed")
    windows["bin_end_utc"] = pd.to_datetime(windows["bin_end_utc"], utc=True, format="mixed")
    return windows.groupby("task_idx", as_index=False).agg(start=("bin_start_utc", "min"), end=("bin_end_utc", "max"))


def extract_eeg_features(raw: pd.DataFrame, windows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times, values, _ = preprocess(raw)
    dual, causal, separate, energy = [], [], [], []
    for task in range(26):
        row = windows[windows["task_idx"].astype(int).eq(task)].iloc[0]
        start, end = np.datetime64(row["start"].to_datetime64()), np.datetime64(row["end"].to_datetime64())
        left, right = np.searchsorted(times, start, side="left"), np.searchsorted(times, end, side="right")
        segment = values[left:right]
        if len(segment) < 103:
            raise RuntimeError(f"short task segment {task}")
        dual.append(duality(segment))
        causal.append(causality(segment))
        separate.append(separation(segment))
        energy.append([float(np.mean(segment * segment))])
    return np.asarray(dual), np.asarray(causal), np.asarray(separate), np.asarray(energy)


def time_receiver(causal: np.ndarray) -> np.ndarray:
    return np.column_stack([causal[:-1], causal[1:], np.diff(causal, axis=0)])


def source_basis(packets: dict[str, dict[str, np.ndarray]], key: str, dimension: int = 8) -> np.ndarray:
    pooled = np.vstack([packet[key] for packet in packets.values()])
    _, singular, vh = np.linalg.svd(pooled, full_matrices=False)
    rank = int(np.sum(singular > max(float(singular[0]), 1.0) * 1e-12))
    return vh[: min(dimension, rank)].T


def band_features(frame: pd.DataFrame, bands: tuple[str, ...], metric: str, transition: bool = False) -> np.ndarray:
    values = frame[[f"{band}_{metric}" for band in bands]].to_numpy(float)
    return np.column_stack([values[:-1], values[1:], np.diff(values, axis=0)]) if transition else values


def halves(axis: str) -> tuple[np.ndarray, np.ndarray]:
    return (np.arange(0, 12), np.arange(13, 25)) if axis == "Causality" else (np.arange(0, 13), np.arange(13, 26))


def all_assignments():
    rows = []
    for causal in BANDS:
        remaining = [band for band in BANDS if band != causal]
        for dual in itertools.combinations(remaining, 2):
            separate = tuple(band for band in remaining if band not in dual)
            rows.append((tuple(sorted(dual)), causal, tuple(sorted(separate))))
    return rows


def add_one(exceedances: int, count: int) -> float:
    return float((exceedances + 1) / (count + 1))


def preflight() -> dict:
    rng = np.random.default_rng(944)
    latent = rng.normal(size=(26, 3))
    nuisance = rng.normal(size=(26, 1))
    target = latent + rng.normal(scale=.01, size=latent.shape)
    positive = association(latent, target, [nuisance])
    permuted = association(latent[rng.permutation(26)], target, [nuisance])
    constant = association(np.ones((26, 2)), target, [nuisance])
    # The retained third-order statistic is odd under exact time reversal.
    signal = rng.normal(size=(240, len(CHANNELS)))
    forward, reverse = causality(signal), causality(signal[::-1])
    checks = {
        "positive_gt_0_9": positive > .9,
        "positive_gt_permuted": positive > permuted,
        "constant_zero": abs(constant) <= 1e-12,
        "reversal_negates_all_coordinates": np.allclose(forward, -reverse, atol=1e-12, rtol=1e-10),
        "heart_dimension_7": len(duality(signal)) == 7,
        "mind_dimension_42": len(forward) == 42,
        "body_dimension_91": len(separation(signal)) == 91,
    }
    return {"status": "PASS_LOCAL_PREFLIGHT_NOT_INDEPENDENT_E0" if all(checks.values()) else "FAIL_LOCAL_PREFLIGHT",
            "checks": {key: bool(value) for key, value in checks.items()},
            "values": {"positive": positive, "permuted": permuted, "constant": constant,
                       "forward": forward.tolist(), "reverse": reverse.tolist()}}


def load_all(args):
    if sha256(args.bands) != BAND_TABLE_SHA:
        raise RuntimeError("band table hash mismatch")
    bands = pd.read_csv(args.bands)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    source_path = Path(__file__).resolve().parents[6] / "outputs/journal-submission-management/tsc-braid-dual-output-consciousness-20260928/SOURCE_PACKET_CP_HYP_033_X21R1_RAW.json"
    true_matrices, generic = source_matrices(source_path), generic_matrices()
    qdir, eegdir = args.raw_root / "IDPC_Reproduction", args.raw_root / "IDPC_EEG"
    labels = [f"P{i}" for i in range(1, 27)]
    eeg, source, generic_source, frames = {}, {}, {}, {}
    hashes = {"raw_eeg": {}, "quantum": {}, "windows": {}}
    for label in labels:
        raw_path = BODY.raw_file_for(eegdir, label)
        q_path = qdir / f"{label}_quantum_timeseries.csv"
        if sha256(raw_path) != baseline["source_manifest"]["raw_eeg"][raw_path.name]:
            raise RuntimeError(f"raw hash mismatch {label}")
        if sha256(q_path) != baseline["source_manifest"]["quantum_timing"][q_path.name]:
            raise RuntimeError(f"quantum hash mismatch {label}")
        q = BODY.load_quantum(qdir, label)
        raw = BODY.load_raw(raw_path)
        window_path = qdir / f"{label}_eeg_timeseries.csv"
        dual, causal, separate, energy = extract_eeg_features(raw, task_windows(qdir, label))
        if any(array.shape[0] != 26 or np.any(~np.isfinite(array)) for array in (dual, causal, energy, separate)):
            raise RuntimeError(f"bad EEG feature block {label}")
        eeg[label] = {"Duality": dual, "Causality": time_receiver(causal), "Separation": separate,
                      "energy": energy, "time_energy": time_receiver(energy)}
        probability = q[list(QCOLS)].to_numpy(float)
        source[label], generic_source[label] = source_packet(probability, true_matrices), source_packet(probability, generic)
        frames[label] = bands[bands["label"].eq(label)].sort_values("task_idx").reset_index(drop=True)
        hashes["raw_eeg"][raw_path.name] = sha256(raw_path)
        hashes["quantum"][q_path.name] = sha256(q_path)
        hashes["windows"][window_path.name] = sha256(window_path)
    true_basis, generic_basis = source_basis(source, "time_raw"), source_basis(generic_source, "time_raw")
    for label in labels:
        source[label]["Time"] = source[label]["time_raw"] @ true_basis
        generic_source[label]["Time"] = generic_source[label]["time_raw"] @ generic_basis
    return labels, eeg, source, generic_source, frames, hashes


def axis_band_score(eeg, frame, assignment, metric, axis):
    dual, causal, separate = assignment
    if axis == "Duality":
        return association(eeg[axis], band_features(frame, dual, metric), [eeg["energy"]])
    if axis == "Causality":
        return association(eeg[axis], band_features(frame, (causal,), metric, True), [eeg["time_energy"]])
    return association(eeg[axis], band_features(frame, separate, metric), [eeg["energy"]])


def braid_arrays(packet, generic, eeg, axis):
    if axis == "Causality":
        radius = time_receiver(packet["radius"])
        return packet["Time"], generic["Time"], eeg[axis], [radius, eeg["time_energy"]]
    sector = "Light" if axis == "Duality" else "Space"
    return packet[sector], generic[sector], eeg[axis], [packet["radius"], eeg["energy"]]


def run(args):
    labels, eeg, source, generic, frames, hashes = load_all(args)
    rng = np.random.default_rng(args.seed)
    assignments = all_assignments()
    focal = (tuple(sorted(("Delta", "Theta"))), "Beta", tuple(sorted(("Alpha", "Gamma"))))
    band_rows, assignment_rows, band_null_rows = [], [], []
    braid_rows, source_rows, braid_null_rows = [], [], []

    # Five-band family.
    band_observed = {}
    for label in labels:
        per_assignment = []
        for assignment in assignments:
            axis_composites, row = {}, {"participant": label, "duality_bands": "+".join(assignment[0]),
                                        "causality_band": assignment[1], "separation_bands": "+".join(assignment[2])}
            for axis in AXES:
                scores = {metric: axis_band_score(eeg[label], frames[label], assignment, metric, axis) for metric in METRICS}
                row.update({f"{axis}_{metric}": scores[metric] for metric in METRICS})
                axis_composites[axis] = float(np.median(list(scores.values())))
                row[f"{axis}_composite"] = axis_composites[axis]
            row["triad_min"] = min(axis_composites.values())
            assignment_rows.append(row)
            per_assignment.append(row)
        ranked = sorted(per_assignment, key=lambda item: item["triad_min"], reverse=True)
        observed = next(item for item in per_assignment if (tuple(item["duality_bands"].split("+")), item["causality_band"], tuple(item["separation_bands"].split("+"))) == focal)
        observed = dict(observed)
        observed["assignment_rank"] = 1 + ranked.index(next(item for item in ranked if item is next(x for x in per_assignment if x is not None and x["duality_bands"] == "+".join(focal[0]) and x["causality_band"] == focal[1] and x["separation_bands"] == "+".join(focal[2]))))
        for axis in AXES:
            half_scores = []
            for indices in halves(axis):
                assignment = focal
                if axis == "Duality":
                    score = association(eeg[label][axis][indices], band_features(frames[label], assignment[0], "mean")[indices], [eeg[label]["energy"][indices]])
                elif axis == "Causality":
                    score = association(eeg[label][axis][indices], band_features(frames[label], (assignment[1],), "mean", True)[indices], [eeg[label]["time_energy"][indices]])
                else:
                    score = association(eeg[label][axis][indices], band_features(frames[label], assignment[2], "mean")[indices], [eeg[label]["energy"][indices]])
                half_scores.append(score)
            observed[f"{axis}_half1"], observed[f"{axis}_half2"] = half_scores
        band_observed[label] = observed

    for iteration in range(args.n_perm):
        iteration_values = {axis: [] for axis in AXES}
        triads = []
        for label in labels:
            axis_values = {}
            for axis in AXES:
                length = len(eeg[label][axis])
                order = rng.permutation(length)
                if axis == "Duality": bands_tuple, transition, nuisance = focal[0], False, eeg[label]["energy"][order]
                elif axis == "Causality": bands_tuple, transition, nuisance = (focal[1],), True, eeg[label]["time_energy"][order]
                else: bands_tuple, transition, nuisance = focal[2], False, eeg[label]["energy"][order]
                scores = [association(eeg[label][axis][order], band_features(frames[label], bands_tuple, metric, transition), [nuisance]) for metric in METRICS]
                axis_values[axis] = float(np.median(scores))
                iteration_values[axis].append(axis_values[axis])
                band_null_rows.append({"iteration": iteration, "participant": label, "axis": axis, "score": axis_values[axis]})
            triads.append(min(axis_values.values()))
        maxima = {axis: max(values) for axis, values in iteration_values.items()}
        triad_max = max(triads)
        for row in band_null_rows[-26 * 3:]:
            row["iteration_max"] = maxima[row["axis"]]
            row["iteration_triad_max"] = triad_max

    band_null = pd.DataFrame(band_null_rows)
    for label in labels:
        row = band_observed[label]
        axis_passes = []
        for axis in AXES:
            value = row[f"{axis}_composite"]
            own = band_null[(band_null.participant == label) & (band_null.axis == axis)].score.to_numpy(float)
            maximum = band_null[band_null.axis == axis].drop_duplicates("iteration").iteration_max.to_numpy(float)
            row[f"{axis}_p_raw"] = add_one(int(np.sum(own >= value)), len(own))
            row[f"{axis}_p_fwer"] = add_one(int(np.sum(maximum >= value)), len(maximum))
            passed = value > 0 and row[f"{axis}_p_raw"] <= .05 and row[f"{axis}_p_fwer"] <= .05 and row[f"{axis}_half1"] > 0 and row[f"{axis}_half2"] > 0 and row["assignment_rank"] == 1
            row[f"{axis}_pass"] = bool(passed)
            axis_passes.append(passed)
        row["triad_pass"] = bool(all(axis_passes))
        band_rows.append(row)

    # Braid family.
    braid_observed = {}
    for label in labels:
        row = {"participant": label}
        for axis in AXES:
            true_values, generic_values, receiver, nuisances = braid_arrays(source[label], generic[label], eeg[label], axis)
            score = association(true_values, receiver, [*nuisances, generic_values])
            generic_score = association(generic_values, receiver, nuisances)
            hs = []
            for indices in halves(axis):
                hs.append(association(true_values[indices], receiver[indices], [*[x[indices] for x in nuisances], generic_values[indices]]))
            row.update({f"{axis}_score": score, f"{axis}_generic": generic_score,
                        f"{axis}_half1": hs[0], f"{axis}_half2": hs[1]})
            for donor in labels:
                donor_true, donor_generic, _, donor_nuisances = braid_arrays(source[donor], generic[donor], eeg[label], axis)
                source_rows.append({"receiver": label, "source": donor, "axis": axis,
                                    "score": association(donor_true, receiver, [*donor_nuisances, donor_generic])})
        row["triad_min"] = min(row[f"{axis}_score"] for axis in AXES)
        braid_observed[label] = row

    for iteration in range(args.n_perm):
        values = {axis: [] for axis in AXES}
        triads = []
        for label in labels:
            axis_values = {}
            for axis in AXES:
                true_values, generic_values, receiver, nuisances = braid_arrays(source[label], generic[label], eeg[label], axis)
                order = rng.permutation(len(true_values))
                value = association(true_values[order], receiver, [*[x[order] for x in nuisances], generic_values[order]])
                axis_values[axis] = value
                values[axis].append(value)
                braid_null_rows.append({"iteration": iteration, "participant": label, "axis": axis, "score": value})
            triads.append(min(axis_values.values()))
        maxima = {axis: max(items) for axis, items in values.items()}
        triad_max = max(triads)
        for row in braid_null_rows[-26 * 3:]:
            row["iteration_max"] = maxima[row["axis"]]
            row["iteration_triad_max"] = triad_max

    sources, braid_null = pd.DataFrame(source_rows), pd.DataFrame(braid_null_rows)
    for label in labels:
        row = braid_observed[label]
        passes = []
        for axis in AXES:
            value = row[f"{axis}_score"]
            own = braid_null[(braid_null.participant == label) & (braid_null.axis == axis)].score.to_numpy(float)
            maximum = braid_null[braid_null.axis == axis].drop_duplicates("iteration").iteration_max.to_numpy(float)
            donors = sources[(sources.receiver == label) & (sources.axis == axis)].copy()
            donors["rank"] = donors.score.rank(method="min", ascending=False).astype(int)
            rank = int(donors[donors.source == label].iloc[0]["rank"])
            row[f"{axis}_p_raw"] = add_one(int(np.sum(own >= value)), len(own))
            row[f"{axis}_p_fwer"] = add_one(int(np.sum(maximum >= value)), len(maximum))
            row[f"{axis}_source_rank"] = rank
            passed = value > 0 and row[f"{axis}_p_raw"] <= .05 and row[f"{axis}_p_fwer"] <= .05 and row[f"{axis}_half1"] > 0 and row[f"{axis}_half2"] > 0 and rank <= 3 and value > row[f"{axis}_generic"]
            row[f"{axis}_pass"] = bool(passed)
            passes.append(passed)
        row["triad_pass"] = bool(all(passes))
        braid_rows.append(row)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(band_rows).to_csv(out / "band_participant_scores.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(out / "band_assignment_scores.csv", index=False)
    band_null.to_csv(out / "band_task_nulls.csv", index=False)
    pd.DataFrame(braid_rows).to_csv(out / "braid_participant_scores.csv", index=False)
    sources.to_csv(out / "braid_source_scores.csv", index=False)
    braid_null.to_csv(out / "braid_task_nulls.csv", index=False)
    summary = {
        "status": "Exploratory result",
        "band_triad_passes": [row["participant"] for row in band_rows if row["triad_pass"]],
        "braid_triad_passes": [row["participant"] for row in braid_rows if row["triad_pass"]],
        "band_axis_passes": {axis: [row["participant"] for row in band_rows if row[f"{axis}_pass"]] for axis in AXES},
        "braid_axis_passes": {axis: [row["participant"] for row in braid_rows if row[f"{axis}_pass"]] for axis in AXES},
        "band_top": sorted(band_rows, key=lambda row: row["triad_min"], reverse=True)[:8],
        "braid_top": sorted(braid_rows, key=lambda row: row["triad_min"], reverse=True)[:8],
        "n_permutations": args.n_perm,
        "seed": args.seed,
        "claim_ceiling": "Same-archive exploratory EEG-native structural correspondence only.",
    }
    write_json(out / "summary.json", summary)
    write_json(out / "execution_log.json", {
        "status": "COMPLETE", "input_hashes": {"bands": sha256(args.bands), "baseline": sha256(args.baseline)},
        "source_hash": SOURCE_SHA, "raw_manifest": hashes, "seed": args.seed, "n_permutations": args.n_perm,
        "script_sha256": sha256(Path(__file__)), "body_helper_sha256": sha256(BODY_SCRIPT),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "output_hashes": {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "execution_log.json"},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "focal"))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()
    if args.mode == "preflight":
        result = preflight()
        result.update({"script_sha256": sha256(Path(__file__)), "body_helper_sha256": sha256(BODY_SCRIPT)})
        write_json(args.output_dir / "preflight.json", result)
        if not result["status"].startswith("PASS"):
            raise RuntimeError("preflight failed")
        return
    gate = json.loads((args.output_dir / "preflight.json").read_text(encoding="utf-8"))
    if not gate["status"].startswith("PASS") or gate["script_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("missing or stale preflight")
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Body-only raw archive scout for DPA-SCOUT-002 R001-BODY."""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy.sparse.csgraph import shortest_path


MASKS = (0, 5, 8, 13, 16, 21, 24, 29)
DIRECTIONS = (5, 8, 16)
PAIRS = ((5, 8), (5, 16), (8, 16))
QCOLS = ("p00", "p01", "p10", "p11")
CHANNELS4 = ("AF3", "F7", "FC5", "FC6")
CHANNELS14 = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")
SOURCE_SHA = "7ba1c987a9389f7426becdde136986be7fbc7b17b92dac938cc2a43cc39b373e"
PUBLIC_TOPOLOGY_SHA = "9e441741a4b8d42120f5fdbcd38e85394886697c651ee8e08791c78783e0cf35"
ZERO = F(0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def add_one(exceedances: int, count: int) -> float:
    return (exceedances + 1) / (count + 1)


def source_matrices(source_path: Path) -> dict[int, np.ndarray]:
    if sha256(source_path) != SOURCE_SHA:
        raise RuntimeError("source mirror hash mismatch")
    raw = json.loads(source_path.read_text())
    records = {int(item["mask"]): item for item in raw["carrier_records"]}
    if tuple(sorted(records)) != MASKS:
        raise RuntimeError("source mask mismatch")
    matrices = {}
    for mask in MASKS:
        gram = [[F(value) for value in row] for row in records[mask]["ordered_edge_gram"]]
        matrices[mask] = np.asarray(
            [
                [float(gram[i][j] ** 2 / (gram[i][i] * gram[j][j]) - int(i == j)) for j in range(4)]
                for i in range(4)
            ],
            dtype=float,
        )
    return matrices


def generic_matrices() -> dict[int, np.ndarray]:
    matrices = {}
    for ordinal, mask in enumerate(MASKS):
        vectors = np.asarray(
            [[int(i == j) + (ordinal + 1) * (i + 1) for j in range(4)] for i in range(4)],
            dtype=float,
        )
        gram = vectors @ vectors.T
        denom = np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
        corr = gram / denom
        matrices[mask] = corr * corr - np.eye(4)
    return matrices


def source_records(probabilities: np.ndarray, matrices: dict[int, np.ndarray]):
    delta = np.asarray(probabilities, dtype=float) - 0.25
    e0 = np.full(4, 0.5)
    odd = np.asarray([e0 @ matrices[mask] @ delta for mask in MASKS])
    body = np.asarray([delta @ matrices[mask] @ delta for mask in MASKS])
    radius = float(np.linalg.norm(delta))
    return odd, body, radius


def commutator(values: np.ndarray, direction: int) -> np.ndarray:
    index = {mask: i for i, mask in enumerate(MASKS)}
    out = np.zeros((8, 8), dtype=float)
    for mask, col in index.items():
        row = index[mask ^ direction]
        out[row, col] = values[row] - values[col]
    return out


def mixed(odd_a, even_a, odd_b, even_b, d, e):
    oa, eb = commutator(odd_a, d), commutator(even_b, e)
    ob, ea = commutator(odd_b, d), commutator(even_a, e)
    return (oa @ eb - eb @ oa) - (ob @ ea - ea @ ob)


def incident_mind_features(odd: np.ndarray, body: np.ndarray) -> np.ndarray:
    count = len(odd)
    out = np.zeros((count, 6), dtype=float)
    for task in range(count - 1):
        norms = np.asarray(
            [np.linalg.norm(mixed(odd[task], body[task], odd[task + 1], body[task + 1], d, e)) for d, e in PAIRS]
        )
        out[task, 3:] = norms
        out[task + 1, :3] = norms
    return out


def raw_file_for(eeg_dir: Path, label: str) -> Path:
    matches = sorted(eeg_dir.glob(f"{label}_EPOCX_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one raw EEG file for {label}, found {len(matches)}")
    return matches[0]


def load_quantum(qdir: Path, label: str) -> pd.DataFrame:
    q = pd.read_csv(qdir / f"{label}_quantum_timeseries.csv")
    q = q.iloc[:-4].copy().reset_index(drop=True)
    if len(q) != 26 or q["task_idx"].astype(int).tolist() != list(range(26)):
        raise RuntimeError(f"bad retained tasks for {label}")
    values = q[list(QCOLS)].to_numpy(float)
    if np.max(np.abs(values.sum(axis=1) - 1.0)) > 1e-10 or np.min(values) < -1e-12:
        raise RuntimeError(f"bad probabilities for {label}")
    q["start"] = pd.to_datetime(q["start_utc"], utc=True, format="mixed")
    q["end"] = pd.to_datetime(q["end_utc"], utc=True, format="mixed")
    q["mid"] = q["start"] + (q["end"] - q["start"]) / 2
    if not q["mid"].is_monotonic_increasing:
        raise RuntimeError(f"nonmonotonic quantum time for {label}")
    return q


def load_raw(eeg_file: Path) -> pd.DataFrame:
    raw = pd.read_csv(eeg_file, skiprows=1, low_memory=False)
    needed = ["Timestamp"] + [f"EEG.{channel}" for channel in CHANNELS14]
    missing = [column for column in needed if column not in raw.columns]
    if missing:
        raise RuntimeError(f"missing raw columns {missing}")
    for column in needed:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    ts_col = "OriginalTimestamp" if "OriginalTimestamp" in raw.columns else "Timestamp"
    raw["TimeStampUTC"] = pd.to_datetime(pd.to_numeric(raw[ts_col], errors="coerce"), unit="s", utc=True)
    raw = raw.dropna(subset=["Timestamp", "TimeStampUTC"]).sort_values("Timestamp").reset_index(drop=True)
    return raw


def laplacian_spectrum(segment: np.ndarray) -> np.ndarray:
    if len(segment) < 103 or segment.shape[1] != 4:
        raise RuntimeError("invalid four-channel segment")
    correlation = np.corrcoef(segment, rowvar=False)
    weights = np.abs((correlation + correlation.T) / 2.0)
    np.fill_diagonal(weights, 0.0)
    laplacian = np.diag(weights.sum(axis=1)) - weights
    eigenvalues = np.linalg.eigvalsh(laplacian)
    if abs(eigenvalues[0]) > 1e-10 or np.min(eigenvalues[1:]) <= 0:
        raise RuntimeError("invalid Laplacian")
    return eigenvalues[1:]


def four_channel_geometry(raw: pd.DataFrame, qdir: Path, label: str) -> tuple[np.ndarray, list[int], str]:
    window_path = qdir / f"{label}_eeg_timeseries.csv"
    windows = pd.read_csv(window_path, usecols=["task_idx", "bin_start_utc", "bin_end_utc"])
    windows["bin_start_utc"] = pd.to_datetime(windows["bin_start_utc"], utc=True, format="mixed")
    windows["bin_end_utc"] = pd.to_datetime(windows["bin_end_utc"], utc=True, format="mixed")
    windows = windows.groupby("task_idx", as_index=False).agg(start=("bin_start_utc", "min"), end=("bin_end_utc", "max"))
    columns = [f"EEG.{channel}" for channel in CHANNELS4]
    spectra, counts = [], []
    for task in range(26):
        row = windows[windows["task_idx"].astype(int).eq(task)]
        if len(row) != 1:
            raise RuntimeError(f"missing four-channel window {label} {task}")
        start, end = row.iloc[0]["start"].timestamp(), row.iloc[0]["end"].timestamp()
        segment = raw[raw["Timestamp"].between(start, end)][columns].dropna().to_numpy(float)
        counts.append(len(segment))
        spectra.append(laplacian_spectrum(segment))
    return np.asarray(spectra), counts, sha256(window_path)


def winsorize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    good = np.isfinite(x)
    if good.sum() < 10:
        return x
    lo, hi = np.quantile(x[good], [0.005, 0.995])
    return np.clip(x, lo, hi)


def estimate_fs(times: np.ndarray) -> float:
    dt = np.median(np.diff(times).astype("timedelta64[ns]").astype(np.int64)) / 1e9
    return float(1.0 / dt)


def bandpass(x: np.ndarray, fs: float, lo=0.5, hi=45.0) -> np.ndarray:
    nyq = fs / 2.0
    upper = min(hi, nyq * 0.999)
    lower = max(lo, 0.01)
    b, a = butter(4, [lower / nyq, upper / nyq], btype="band")
    return filtfilt(b, a, x)


def clean_fullband_power(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    columns = [f"EEG.{channel}" for channel in CHANNELS14]
    values = raw[columns].to_numpy(float)
    values[~np.isfinite(values)] = np.nan
    for column in range(values.shape[1]):
        values[:, column] = winsorize(values[:, column])
        good = np.isfinite(values[:, column])
        if good.sum() < 3:
            raise RuntimeError("insufficient EEG samples")
        indices = np.arange(len(values))
        values[~good, column] = np.interp(indices[~good], indices[good], values[good, column])
    values -= np.median(values, axis=0, keepdims=True)
    times = raw["TimeStampUTC"].to_numpy(dtype="datetime64[ns]")
    fs = estimate_fs(times)
    for column in range(values.shape[1]):
        values[:, column] = bandpass(values[:, column], fs)
    smooth = max(1, int(round(0.25 * fs)))
    kernel = np.ones(smooth) / smooth
    power = np.zeros_like(values)
    for column in range(values.shape[1]):
        filtered = bandpass(values[:, column], fs)
        envelope = np.abs(hilbert(filtered))
        power[:, column] = np.convolve(envelope * envelope, kernel, mode="same")
    return times, power, fs


def h1_persistence(distance: np.ndarray) -> float:
    n = len(distance)
    simplices = [((i,), 0, 0.0) for i in range(n)]
    simplices += [((i, j), 1, float(distance[i, j])) for i in range(n) for j in range(i + 1, n)]
    simplices += [
        ((i, j, k), 2, float(max(distance[i, j], distance[i, k], distance[j, k])))
        for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)
    ]
    simplices.sort(key=lambda item: (item[2], item[1], item[0]))
    index = {verts: idx for idx, (verts, _, _) in enumerate(simplices)}
    boundaries = []
    for verts, dim, _ in simplices:
        if dim == 0:
            boundaries.append(set())
        elif dim == 1:
            boundaries.append({index[(verts[0],)], index[(verts[1],)]})
        else:
            i, j, k = verts
            boundaries.append({index[(i, j)], index[(i, k)], index[(j, k)]})
    reduced, low_map, pairs = {}, {}, []
    for j, (_, dim, filt) in enumerate(simplices):
        column = set(boundaries[j])
        while column and max(column) in low_map:
            column ^= reduced[low_map[max(column)]]
        reduced[j] = set(column)
        if column:
            low = max(column)
            low_map[low] = j
            if dim == 2 and simplices[low][1] == 1:
                pairs.append((simplices[low][2], filt))
    return float(sum(max(0.0, death - birth) for birth, death in pairs))


def topology_geometry(raw: pd.DataFrame, q: pd.DataFrame) -> np.ndarray:
    times, power, _ = clean_fullband_power(raw)
    output = []
    for _, row in q.iterrows():
        start = np.datetime64((row["mid"] - pd.Timedelta(seconds=1)).to_datetime64())
        end = np.datetime64((row["mid"] + pd.Timedelta(seconds=1)).to_datetime64())
        left = int(np.searchsorted(times, start, side="left"))
        right = int(np.searchsorted(times, end, side="right"))
        segment = power[left:right]
        if len(segment) < 16:
            raise RuntimeError("short topology segment")
        variance = np.nanvar(segment, axis=0)
        valid = np.isfinite(variance) & (variance > 1e-10)
        if valid.sum() < 6:
            raise RuntimeError("too few topology channels")
        corr = np.corrcoef(segment[:, valid], rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.abs((corr + corr.T) / 2.0)
        np.fill_diagonal(corr, 0.0)
        degree = corr.sum(axis=1)
        lap = np.diag(degree) - corr
        lambda2 = float(np.sort(np.linalg.eigvalsh(lap))[1])
        dist = np.full_like(corr, np.inf)
        np.fill_diagonal(dist, 0.0)
        positive = corr > 0
        dist[positive] = 1.0 / corr[positive]
        paths = shortest_path(dist, directed=False, unweighted=False)
        if not np.isfinite(paths).all():
            raise RuntimeError("disconnected topology graph")
        aspl = float(np.mean(paths[~np.eye(len(paths), dtype=bool)]))
        vr_distance = 1.0 - corr
        np.fill_diagonal(vr_distance, 0.0)
        output.append((h1_persistence(vr_distance), aspl, lambda2))
    result = np.asarray(output)
    if result.shape != (26, 3) or np.any(~np.isfinite(result)):
        raise RuntimeError("bad topology output")
    return result


def centered_distance(values: np.ndarray, zscore=False):
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    if zscore:
        scale = values.std(axis=0)
        if np.any(scale <= 1e-14):
            return None
        values = (values - values.mean(axis=0)) / scale
    delta = values[:, None, :] - values[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    return distance - distance.mean(axis=0)[None, :] - distance.mean(axis=1)[:, None] + distance.mean()


def residual(target: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    vector = target.reshape(-1)
    usable = [control.reshape(-1) for control in controls if control is not None and np.linalg.norm(control) > 1e-12]
    if not usable:
        return vector
    design = np.column_stack(usable)
    return vector - design @ np.linalg.lstsq(design, vector, rcond=None)[0]


def correlation(left: np.ndarray, right: np.ndarray):
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    return 0.0 if min(left_norm, right_norm) <= 1e-10 else float(left @ right / (left_norm * right_norm))


def unique_scores(primary, competitor, laplacian, topology, radius, odd, mind):
    pdist = centered_distance(primary)
    cdist = centered_distance(competitor)
    ldist = centered_distance(laplacian)
    tdist = centered_distance(topology, zscore=True)
    rdist = centered_distance(radius)
    odist = centered_distance(odd, zscore=True)
    mdist = centered_distance(mind, zscore=True)
    if any(item is None for item in (pdist, cdist, ldist, tdist, rdist, odist, mdist)):
        return {"topology_unique": 0.0, "laplacian_unique": 0.0, "joint_min": 0.0}
    base = [rdist, odist, mdist, cdist]
    source_top = residual(pdist, base + [ldist])
    target_top = residual(tdist, base + [ldist])
    source_lap = residual(pdist, base + [tdist])
    target_lap = residual(ldist, base + [tdist])
    topology_unique = correlation(source_top, target_top)
    laplacian_unique = correlation(source_lap, target_lap)
    return {
        "topology_unique": topology_unique,
        "laplacian_unique": laplacian_unique,
        "joint_min": min(topology_unique, laplacian_unique),
        "topology_source_norm": float(np.linalg.norm(source_top)),
        "topology_receiver_norm": float(np.linalg.norm(target_top)),
        "laplacian_source_norm": float(np.linalg.norm(source_lap)),
        "laplacian_receiver_norm": float(np.linalg.norm(target_lap)),
    }


def rank_of(rows, label):
    ordered = sorted(rows, key=lambda row: row["joint_min"], reverse=True)
    return 1 + next(index for index, row in enumerate(ordered) if row["label"] == label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--n-perm", type=int, default=2000)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[6]
    source_path = repo / "outputs/journal-submission-management/tsc-braid-dual-output-consciousness-20260928/SOURCE_PACKET_CP_HYP_033_X21R1_RAW.json"
    public_script = Path("/Users/satoru/Documents/Codex/2026-04-23-zenodo-eeg-raw-github-af7-8/analyze_all14_20corr30_task_topology_link.py")
    if sha256(public_script) != PUBLIC_TOPOLOGY_SHA:
        raise RuntimeError("public topology method hash mismatch")
    baseline = json.loads(args.baseline.read_text())
    eeg_dir = args.raw_root / "IDPC_EEG"
    qdir = args.raw_root / "IDPC_Reproduction"
    braid = source_matrices(source_path)
    generic = generic_matrices()

    features, per_task_rows = {}, []
    actual_hashes = {"raw_eeg": {}, "quantum_timing": {}, "pipeline_windows": {}}
    for index in range(1, 27):
        label = f"P{index}"
        qpath = qdir / f"{label}_quantum_timeseries.csv"
        eegpath = raw_file_for(eeg_dir, label)
        qhash, ehash = sha256(qpath), sha256(eegpath)
        expected_q = baseline["source_manifest"]["quantum_timing"].get(qpath.name)
        expected_e = baseline["source_manifest"]["raw_eeg"].get(eegpath.name)
        if qhash != expected_q or ehash != expected_e:
            raise RuntimeError(f"frozen raw hash mismatch for {label}")
        actual_hashes["quantum_timing"][qpath.name] = qhash
        actual_hashes["raw_eeg"][eegpath.name] = ehash

        q = load_quantum(qdir, label)
        raw = load_raw(eegpath)
        lap, segment_counts, window_hash = four_channel_geometry(raw, qdir, label)
        topo = topology_geometry(raw, q)
        actual_hashes["pipeline_windows"][f"{label}_eeg_timeseries.csv"] = window_hash
        probabilities = q[list(QCOLS)].to_numpy(float)
        odd, body, radius, generic_body = [], [], [], []
        for probability in probabilities:
            o, b, r = source_records(probability, braid)
            _, gb, _ = source_records(probability, generic)
            odd.append(o); body.append(b); radius.append(r); generic_body.append(gb)
        odd = np.asarray(odd); body = np.asarray(body); radius = np.asarray(radius)[:, None]; generic_body = np.asarray(generic_body)
        mind = incident_mind_features(odd, body)
        features[label] = {"odd": odd, "body": body, "radius": radius, "generic": generic_body, "mind": mind, "lap": lap, "topo": topo}
        for task in range(26):
            row = {"label": label, "task_idx": task, "segment_rows_4ch": segment_counts[task], "radius": radius[task, 0]}
            row.update({f"body_{mask}": body[task, i] for i, mask in enumerate(MASKS)})
            row.update({f"odd_{mask}": odd[task, i] for i, mask in enumerate(MASKS)})
            row.update({f"mind_incident_{i}": mind[task, i] for i in range(6)})
            row.update({f"laplacian_{i}": lap[task, i] for i in range(3)})
            row.update({f"topology_{name}": topo[task, i] for i, name in enumerate(("h1", "aspl", "lambda2"))})
            per_task_rows.append(row)

    matched, source = [], []
    for label, value in features.items():
        braid_score = unique_scores(value["body"], value["generic"], value["lap"], value["topo"], value["radius"], value["odd"], value["mind"])
        generic_score = unique_scores(value["generic"], value["body"], value["lap"], value["topo"], value["radius"], value["odd"], value["mind"])
        matched.append({"label": label, **braid_score, "generic_joint_min": generic_score["joint_min"]})
        p1 = features["P1"]
        braid_source = unique_scores(value["body"], value["generic"], p1["lap"], p1["topo"], value["radius"], value["odd"], value["mind"])
        generic_source = unique_scores(value["generic"], value["body"], p1["lap"], p1["topo"], value["radius"], value["odd"], value["mind"])
        source.append({"label": label, **braid_source, "generic_joint_min": generic_source["joint_min"]})

    observed = next(row for row in matched if row["label"] == "P1")
    matched_rank, source_rank = rank_of(matched, "P1"), rank_of(source, "P1")
    rng = np.random.default_rng(args.seed)
    nulls, exceedances = [], 0
    p1 = features["P1"]
    for iteration in range(args.n_perm):
        order = rng.permutation(26)
        score = unique_scores(p1["body"][order], p1["generic"], p1["lap"], p1["topo"], p1["radius"], p1["odd"], p1["mind"])
        nulls.append({"iteration": iteration, **score})
        exceedances += score["joint_min"] >= observed["joint_min"]
    task_p = add_one(exceedances, args.n_perm)

    gates = {
        "A1_both_views_positive": observed["topology_unique"] > 0 and observed["laplacian_unique"] > 0,
        "A2_joint_positive": observed["joint_min"] > 0,
        "A3_p1_matched_apex": matched_rank == 1,
        "A4_p1_source_apex": source_rank == 1,
        "A5_task_specificity": task_p <= 0.05,
        "A6_braid_exceeds_generic": observed["joint_min"] > observed["generic_joint_min"],
        "B1_frozen_raw_hashes": len(actual_hashes["raw_eeg"]) == len(actual_hashes["quantum_timing"]) == 26,
        "B2_retention": len(per_task_rows) == 676 and len(matched) == len(source) == 26 and len(nulls) == args.n_perm,
        "B3_finite_nonconstant": all(np.all(np.isfinite(value[key])) and np.max(np.std(value[key], axis=0)) > 1e-12 for value in features.values() for key in ("body", "lap", "topo")),
    }
    b_pass = all(value for key, value in gates.items() if key.startswith("B"))
    a_pass = all(value for key, value in gates.items() if key.startswith("A"))
    decision = "PASS_BODY_RAW_SELECTIVITY" if b_pass and a_pass else "NO-GO_BODY_RAW_SELECTIVITY" if b_pass else "OPEN_IMPLEMENTATION_OR_PROVENANCE"
    summary = {
        "scout_id": "DPA-SCOUT-002",
        "iteration": "R001-BODY",
        "operational_lineage": "Q-014-X3",
        "evidence_status": "Exploratory result",
        "decision": decision,
        "gates": gates,
        "p1": {**observed, "matched_rank": matched_rank, "source_rank": source_rank, "task_exceedances": exceedances, "task_p": task_p},
        "best_false_matched": sorted((row for row in matched if row["label"] != "P1"), key=lambda row: row["joint_min"], reverse=True)[0],
        "best_false_source": sorted((row for row in source if row["label"] != "P1"), key=lambda row: row["joint_min"], reverse=True)[0],
        "claim_ceiling": "P1-local same-archive exploratory objective Body precursor only; no perceptual breadth, physical space, causal transfer, consciousness, subjectivity, confirmation, or whole Heart/Mind/Body result.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_task_rows).to_csv(args.output_dir / "per_task_features.csv", index=False)
    pd.DataFrame(matched).to_csv(args.output_dir / "matched_scores.csv", index=False)
    pd.DataFrame(source).to_csv(args.output_dir / "p1_source_scores.csv", index=False)
    pd.DataFrame(nulls).to_csv(args.output_dir / "task_nulls.csv", index=False)
    write_json(args.output_dir / "input_manifest.json", actual_hashes)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "execution_log.json", {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "seed": args.seed, "n_perm": args.n_perm, "baseline_sha256": sha256(args.baseline),
        "source_sha256": sha256(source_path), "public_topology_method_sha256": sha256(public_script),
        "evaluator_sha256": sha256(Path(__file__)), "raw_root": str(args.raw_root), "result_informed": False,
    })


if __name__ == "__main__":
    main()

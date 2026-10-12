#!/usr/bin/env python3
"""Source-defined Braid light/time/space scout against fixed EEG bands."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


MASKS = (0, 5, 8, 13, 16, 21, 24, 29)
PAIRS = ((5, 8), (5, 16), (8, 16))
METRICS = ("mean", "peak", "stability")
SECTORS = ("Light", "Time", "Space")
QCOLS = ("p00", "p01", "p10", "p11")
EEG_SHA = "e345ad16d12baf94d76864a3cce5112890bfaf8e4463eb895abbcb4f07048b3e"
KERNEL_SHA = "f33dbb7be3b7f9902761a8c99d7a66b059d66d3499af89b6ed117120d56d9757"
EXPECTED_P1_CORRS = np.asarray([0.5838572568879881, 0.7624412235653151, 0.7431428894584844])
EXPECTED_P1_SCORE = 0.6964804566372624
WALSH = np.asarray(
    [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5],
     [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]], dtype=float
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def natural_label(label: str) -> int:
    return int(label[1:])


def zcols(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale <= 1e-12] = 1.0
    return (values - mean) / scale


def centered_distance(values: np.ndarray, standardize: bool = False) -> np.ndarray:
    x = zcols(values) if standardize else np.asarray(values, dtype=float)
    delta = x[:, None, :] - x[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    return distance - distance.mean(axis=0, keepdims=True) - distance.mean(axis=1, keepdims=True) + distance.mean()


def upper(matrix: np.ndarray) -> np.ndarray:
    index = np.triu_indices(len(matrix), 1)
    return np.asarray(matrix[index], dtype=float)


def residual(vector: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(vector)), nuisance])
    beta = np.linalg.lstsq(design, vector, rcond=None)[0]
    return vector - design @ beta


def signed_partial_distance_score(source: np.ndarray, eeg: np.ndarray, radius: np.ndarray) -> float:
    ds = upper(centered_distance(source, standardize=False))
    de = upper(centered_distance(eeg, standardize=True))
    dr = upper(centered_distance(radius, standardize=False))
    rs, re = residual(ds, dr[:, None]), residual(de, dr[:, None])
    denom = float(np.linalg.norm(rs) * np.linalg.norm(re))
    return float(rs @ re / denom) if denom > 1e-12 else 0.0


def commutator(values: np.ndarray, direction: int) -> np.ndarray:
    index = {mask: i for i, mask in enumerate(MASKS)}
    output = np.zeros((8, 8), dtype=float)
    for mask, column in index.items():
        row = index[mask ^ direction]
        output[row, column] = values[row] - values[column]
    return output


def mixed(odd_a, even_a, odd_b, even_b, d, e):
    oa, eb = commutator(odd_a, d), commutator(even_b, e)
    ob, ea = commutator(odd_b, d), commutator(even_a, e)
    return (oa @ eb - eb @ oa) - (ob @ ea - ea @ ob)


def mind_flat(odd: np.ndarray, even: np.ndarray) -> np.ndarray:
    rows = []
    for task in range(len(odd) - 1):
        rows.append(np.concatenate([
            mixed(odd[task], even[task], odd[task + 1], even[task + 1], d, e).reshape(-1)
            for d, e in PAIRS
        ]))
    return np.asarray(rows)


def source_features(probability: np.ndarray, matrices: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    centered = probability - 0.25
    e0 = np.full(4, 0.5)
    odd = np.column_stack([centered @ matrices[mask].T @ e0 for mask in MASKS])
    even = np.column_stack([
        np.einsum("ni,ij,nj->n", centered, matrices[mask], centered) for mask in MASKS
    ])
    radius = np.linalg.norm(centered @ WALSH[1:].T, axis=1)[:, None]
    return {"Light": odd, "Space": even, "radius": radius, "mind_raw": mind_flat(odd, even)}


def load_inputs(eeg_path: Path, kernel_path: Path):
    if sha256(eeg_path) != EEG_SHA:
        raise RuntimeError("EEG table hash mismatch")
    if sha256(kernel_path) != KERNEL_SHA:
        raise RuntimeError("source-kernel hash mismatch")
    table = pd.read_csv(eeg_path)
    required = {"label", "task_idx", *QCOLS}
    required |= {f"{band}_{metric}" for band in ("Delta", "Theta", "Alpha", "Beta", "Gamma") for metric in METRICS}
    if len(table) != 676 or not required.issubset(table.columns):
        raise RuntimeError("EEG table dimensions or columns mismatch")
    labels = sorted(table["label"].unique(), key=natural_label)
    if labels != [f"P{i}" for i in range(1, 27)]:
        raise RuntimeError("participant set mismatch")
    frames = {}
    for label in labels:
        frame = table[table["label"].eq(label)].sort_values("task_idx").reset_index(drop=True)
        if frame["task_idx"].astype(int).tolist() != list(range(26)):
            raise RuntimeError(f"task mismatch: {label}")
        frames[label] = frame
    raw = json.loads(kernel_path.read_text(encoding="utf-8"))
    if tuple(sorted(int(item["mask"]) for item in raw)) != MASKS:
        raise RuntimeError("kernel mask mismatch")
    matrices = {}
    for item in raw:
        correlation = np.asarray(item["correlation"], dtype=float)
        matrices[int(item["mask"])] = correlation * correlation - np.eye(4)
    return labels, frames, matrices


def baseline_check(frames: dict[str, pd.DataFrame]) -> dict:
    frame = frames["P1"]
    q = frame["p00"].to_numpy(float)
    corrs = np.asarray([np.corrcoef(q, frame[f"Delta_{metric}"].to_numpy(float))[0, 1] for metric in METRICS])
    score = float(np.mean(np.abs(corrs)))
    passed = bool(np.allclose(corrs, EXPECTED_P1_CORRS, atol=1e-12, rtol=0) and abs(score - EXPECTED_P1_SCORE) <= 1e-12)
    return {"passed": passed, "p1_delta_p00_corrs": corrs.tolist(), "p1_score": score}


def build_features(labels, frames, matrices):
    raw = {}
    mind_blocks = []
    for label in labels:
        packet = source_features(frames[label][list(QCOLS)].to_numpy(float), matrices)
        raw[label] = packet
        mind_blocks.append(packet["mind_raw"])
    pooled = np.vstack(mind_blocks)
    _, singular, vh = np.linalg.svd(pooled, full_matrices=False)
    rank = int(np.sum(singular > max(float(singular[0]), 1.0) * 1e-12))
    dimension = min(8, rank)
    basis = vh[:dimension].T
    for label in labels:
        raw[label]["Time"] = raw[label]["mind_raw"] @ basis
        radius = raw[label]["radius"]
        raw[label]["TimeRadius"] = np.column_stack([radius[:-1, 0], radius[1:, 0], np.diff(radius[:, 0])])
    return raw, {"rank": rank, "dimension": dimension, "singular_values": singular[:12].tolist()}


def eeg_feature(frame: pd.DataFrame, sector: str, metric: str) -> np.ndarray:
    if sector == "Light":
        return frame[[f"Delta_{metric}", f"Theta_{metric}"]].to_numpy(float)
    if sector == "Space":
        return frame[[f"Gamma_{metric}", f"Alpha_{metric}"]].to_numpy(float)
    beta = frame[f"Beta_{metric}"].to_numpy(float)
    return np.column_stack([beta[:-1], beta[1:], np.diff(beta)])


def sector_arrays(packet: dict[str, np.ndarray], sector: str):
    if sector == "Time":
        return packet["Time"], packet["TimeRadius"]
    return packet[sector], packet["radius"]


def metric_scores(source: np.ndarray, radius: np.ndarray, frame: pd.DataFrame, sector: str) -> dict[str, float]:
    return {metric: signed_partial_distance_score(source, eeg_feature(frame, sector, metric), radius) for metric in METRICS}


def composite(scores: dict[str, float]) -> float:
    return float(np.median([scores[metric] for metric in METRICS]))


def half_composites(source, radius, frame, sector):
    if sector == "Time":
        halves = (np.arange(0, 12), np.arange(13, 25))
    else:
        halves = (np.arange(0, 13), np.arange(13, 26))
    output = []
    for indices in halves:
        scores = {
            metric: signed_partial_distance_score(source[indices], eeg_feature(frame, sector, metric)[indices], radius[indices])
            for metric in METRICS
        }
        output.append(composite(scores))
    return output


def capacity_preflight() -> dict:
    rng = np.random.default_rng(719)
    latent = np.linspace(-2.0, 2.0, 26)[:, None]
    source = np.column_stack([latent[:, 0], np.sin(latent[:, 0] * 2.0)])
    radius = rng.normal(size=(26, 1))
    eeg = source @ np.asarray([[1.0, -0.4], [0.3, 0.8]]) + rng.normal(scale=0.03, size=(26, 2))
    positive = signed_partial_distance_score(source, eeg, radius)
    permutation = signed_partial_distance_score(source[rng.permutation(26)], eeg, radius)
    unrelated = signed_partial_distance_score(rng.normal(size=(26, 2)), eeg, radius)
    constant = signed_partial_distance_score(np.ones((26, 2)), eeg, radius)
    radius_only = signed_partial_distance_score(np.column_stack([radius[:, 0], radius[:, 0] ** 2]), eeg, radius)
    replay = signed_partial_distance_score(source, eeg, radius)
    receiver = eeg
    candidates = [rng.normal(size=(26, 2)) for _ in range(5)]
    candidates.insert(3, source)
    ranks = [signed_partial_distance_score(candidate, receiver, radius) for candidate in candidates]
    true_rank = int(pd.Series(ranks).rank(method="min", ascending=False).iloc[3])
    checks = {
        "positive_gt_0_9": positive > 0.9,
        "positive_gt_permuted": positive > permutation,
        "positive_gt_unrelated": positive > unrelated,
        "constant_zero": abs(constant) <= 1e-12,
        "radius_only_not_positive_witness": radius_only < 0.5,
        "deterministic_replay": positive == replay,
        "true_source_rank_1": true_rank == 1,
    }
    return {
        "status": "PASS_LOCAL_PREFLIGHT_NOT_INDEPENDENT_E0" if all(checks.values()) else "FAIL_LOCAL_PREFLIGHT",
        "checks": checks,
        "values": {"positive": positive, "permutation": permutation, "unrelated": unrelated,
                   "constant": constant, "radius_only": radius_only, "replay": replay,
                   "true_source_rank": true_rank},
    }


def run_preflight(args) -> None:
    labels, frames, matrices = load_inputs(args.eeg, args.source_kernel)
    baseline = baseline_check(frames)
    capacity = capacity_preflight()
    passed = baseline["passed"] and capacity["status"].startswith("PASS")
    output = {
        "status": "PASS_LOCAL_PREFLIGHT_NOT_INDEPENDENT_E0" if passed else "FAIL_LOCAL_PREFLIGHT",
        "input_hashes": {"eeg": sha256(args.eeg), "source_kernel": sha256(args.source_kernel)},
        "shape": {"rows": 676, "participants": len(labels), "tasks_each": 26},
        "baseline": baseline,
        "capacity": capacity,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "script_sha256": sha256(Path(__file__)),
    }
    write_json(args.output_dir / "preflight" / "preflight.json", output)
    if not passed:
        raise RuntimeError("local preflight failed")


def add_one(exceedances: int, count: int) -> float:
    return float((exceedances + 1) / (count + 1))


def run_focal(args) -> None:
    preflight_path = args.output_dir / "preflight" / "preflight.json"
    if not preflight_path.exists():
        raise RuntimeError("missing preflight")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight["status"].startswith("PASS") or preflight["script_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("preflight/script revision mismatch")
    labels, frames, matrices = load_inputs(args.eeg, args.source_kernel)
    packets, time_basis = build_features(labels, frames, matrices)
    observed_rows = []
    source_rows = []
    null_rows = []
    rng = np.random.default_rng(args.seed)

    observed_lookup = {}
    for sector in SECTORS:
        for receiver in labels:
            source, radius = sector_arrays(packets[receiver], sector)
            scores = metric_scores(source, radius, frames[receiver], sector)
            halves = half_composites(source, radius, frames[receiver], sector)
            observed_lookup[(sector, receiver)] = composite(scores)
            observed_rows.append({
                "sector": sector, "participant": receiver,
                **{f"score_{metric}": scores[metric] for metric in METRICS},
                "composite": composite(scores), "score_min": min(scores.values()),
                "score_max": max(scores.values()), "positive_metrics": sum(value > 0 for value in scores.values()),
                "half_1": halves[0], "half_2": halves[1],
            })
            for donor in labels:
                donor_source, donor_radius = sector_arrays(packets[donor], sector)
                donor_scores = metric_scores(donor_source, donor_radius, frames[receiver], sector)
                source_rows.append({"sector": sector, "receiver": receiver, "source": donor,
                                    "composite": composite(donor_scores),
                                    **{f"score_{metric}": donor_scores[metric] for metric in METRICS}})

        for iteration in range(args.n_perm):
            iteration_scores = []
            for participant in labels:
                source, radius = sector_arrays(packets[participant], sector)
                order = rng.permutation(len(source))
                perm_scores = metric_scores(source[order], radius[order], frames[participant], sector)
                value = composite(perm_scores)
                iteration_scores.append(value)
                null_rows.append({"sector": sector, "iteration": iteration, "participant": participant,
                                  "composite": value,
                                  **{f"score_{metric}": perm_scores[metric] for metric in METRICS}})
            current_max = max(iteration_scores)
            for row in null_rows[-len(labels):]:
                row["iteration_max"] = current_max

    observed = pd.DataFrame(observed_rows)
    source_table = pd.DataFrame(source_rows)
    nulls = pd.DataFrame(null_rows)
    for index, row in observed.iterrows():
        sector, participant, value = row["sector"], row["participant"], float(row["composite"])
        own_null = nulls[(nulls["sector"] == sector) & (nulls["participant"] == participant)]["composite"].to_numpy(float)
        max_null = nulls[nulls["sector"] == sector].drop_duplicates("iteration")["iteration_max"].to_numpy(float)
        observed.loc[index, "p_task_raw"] = add_one(int(np.sum(own_null >= value)), len(own_null))
        observed.loc[index, "p_participant_fwer"] = add_one(int(np.sum(max_null >= value)), len(max_null))
        donor = source_table[(source_table["sector"] == sector) & (source_table["receiver"] == participant)].copy()
        donor["rank"] = donor["composite"].rank(method="min", ascending=False).astype(int)
        own_rank = int(donor.loc[donor["source"] == participant, "rank"].iloc[0])
        observed.loc[index, "own_source_rank"] = own_rank
        strict = (observed.loc[index, "p_participant_fwer"] <= 0.05 and own_rank <= 3 and
                  row["half_1"] > 0 and row["half_2"] > 0 and row["positive_metrics"] >= 2)
        lead = (observed.loc[index, "p_task_raw"] <= 0.05 and own_rank <= 5 and
                row["half_1"] > 0 and row["half_2"] > 0 and row["positive_metrics"] >= 2)
        observed.loc[index, "tier"] = "TIER_A_STRICT" if strict else ("TIER_B_LEAD" if lead else "NONE")

    observed["sector_rank"] = observed.groupby("sector")["composite"].rank(method="min", ascending=False).astype(int)
    observed["sector_percentile"] = observed.groupby("sector")["composite"].rank(method="average", pct=True)
    wide = observed.pivot(index="participant", columns="sector", values="sector_percentile").reset_index()
    wide["triple_min_percentile"] = wide[list(SECTORS)].min(axis=1)
    wide["triple_mean_percentile"] = wide[list(SECTORS)].mean(axis=1)
    wide["triple_rank"] = wide["triple_min_percentile"].rank(method="min", ascending=False).astype(int)
    wide = wide.sort_values(["triple_rank", "participant"], key=lambda s: s.map(natural_label) if s.name == "participant" else s)

    results = args.output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    observed.sort_values(["sector", "sector_rank"]).to_csv(results / "participant_sector_scores.csv", index=False)
    source_table.to_csv(results / "source_scores.csv", index=False)
    nulls.to_csv(results / "task_nulls.csv", index=False)
    wide.to_csv(results / "triple_rankings.csv", index=False)

    tier_a = observed[observed["tier"] == "TIER_A_STRICT"]
    tier_b = observed[observed["tier"] == "TIER_B_LEAD"]
    all_three = all((tier_a["sector"] == sector).any() for sector in SECTORS)
    decision = "SCOUT_SIGNAL_ALL_THREE_SECTORS" if all_three else (
        "SCOUT_PARTIAL_SIGNAL" if len(tier_a) + len(tier_b) > 0 else "NO_USEFUL_SIGNAL"
    )
    top_five = {}
    for sector in SECTORS:
        rows = observed[observed["sector"] == sector].sort_values("sector_rank").head(5)
        top_five[sector] = rows.to_dict("records")
    summary = {
        "scout_id": "Q-014-X3/DPA-SCOUT-003", "evidence_status": "Exploratory result",
        "decision": decision, "n_permutations": args.n_perm, "seed": args.seed,
        "mapping": {"Light": "Delta+Theta", "Time": "Beta", "Space": "Gamma+Alpha"},
        "tier_a": tier_a.to_dict("records"), "tier_b": tier_b.to_dict("records"),
        "top_five": top_five, "triple_top_five": wide.head(5).to_dict("records"),
        "time_basis": time_basis,
        "claim_ceiling": "Same-archive participant-selecting exploratory correspondence only; not confirmation or causal evidence.",
    }
    write_json(results / "summary.json", summary)
    write_json(args.output_dir / "execution_log.json", {
        "status": "COMPLETE", "decision": decision, "seed": args.seed, "n_permutations": args.n_perm,
        "input_hashes": {"eeg": sha256(args.eeg), "source_kernel": sha256(args.source_kernel),
                         "preflight": sha256(preflight_path)},
        "output_hashes": {path.name: sha256(path) for path in sorted(results.iterdir()) if path.is_file()},
        "script_sha256": sha256(Path(__file__)), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "focal"))
    parser.add_argument("--eeg", type=Path, required=True)
    parser.add_argument("--source-kernel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()
    if args.mode == "preflight":
        run_preflight(args)
    else:
        run_focal(args)


if __name__ == "__main__":
    main()

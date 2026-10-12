"""Frozen complementary multi-view distance endpoint for Q-012-X12."""

from __future__ import annotations

import numpy as np


def centered_distance(values: np.ndarray, column_zscore: bool = False):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if column_zscore:
        scales = values.std(axis=0)
        if np.any(~np.isfinite(scales)) or np.any(scales <= 1e-14):
            return None
        values = (values - values.mean(axis=0)) / scales
    differences = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    return (
        distances
        - distances.mean(axis=0)[None, :]
        - distances.mean(axis=1)[:, None]
        + distances.mean()
    )


def project_residual(target: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    vector = np.asarray(target, dtype=float).reshape(-1)
    usable = [np.asarray(control, dtype=float).reshape(-1) for control in controls]
    usable = [control for control in usable if np.linalg.norm(control) > 1e-12]
    if not usable:
        return vector
    design = np.column_stack(usable)
    coefficients = np.linalg.lstsq(design, vector, rcond=None)[0]
    return vector - design @ coefficients


def signed_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    score = (
        0.0
        if min(left_norm, right_norm) <= 1e-10
        else float(np.dot(left, right) / (left_norm * right_norm))
    )
    return score, left_norm, right_norm


def complementary_scores(
    coherence: np.ndarray,
    laplacian: np.ndarray,
    topology: np.ndarray,
    radius: np.ndarray,
) -> dict[str, float]:
    source_distance = centered_distance(coherence, False)
    laplacian_distance = centered_distance(laplacian, False)
    topology_distance = centered_distance(topology, True)
    radius_distance = centered_distance(radius, False)
    if any(
        item is None
        for item in (
            source_distance,
            laplacian_distance,
            topology_distance,
            radius_distance,
        )
    ):
        return {
            "topology_unique": 0.0,
            "laplacian_unique": 0.0,
            "joint_min": 0.0,
            "topology_source_norm": 0.0,
            "topology_receiver_norm": 0.0,
            "laplacian_source_norm": 0.0,
            "laplacian_receiver_norm": 0.0,
        }

    source_for_topology = project_residual(
        source_distance, [radius_distance, laplacian_distance]
    )
    topology_residual = project_residual(
        topology_distance, [radius_distance, laplacian_distance]
    )
    topology_score, topology_source_norm, topology_receiver_norm = signed_correlation(
        source_for_topology, topology_residual
    )

    source_for_laplacian = project_residual(
        source_distance, [radius_distance, topology_distance]
    )
    laplacian_residual = project_residual(
        laplacian_distance, [radius_distance, topology_distance]
    )
    laplacian_score, laplacian_source_norm, laplacian_receiver_norm = signed_correlation(
        source_for_laplacian, laplacian_residual
    )
    return {
        "topology_unique": topology_score,
        "laplacian_unique": laplacian_score,
        "joint_min": min(topology_score, laplacian_score),
        "topology_source_norm": topology_source_norm,
        "topology_receiver_norm": topology_receiver_norm,
        "laplacian_source_norm": laplacian_source_norm,
        "laplacian_receiver_norm": laplacian_receiver_norm,
    }


def coherence_vectors(probabilities: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    root = np.sqrt(probabilities)
    outputs = []
    for kernel in kernels:
        rho = root[:, :, None] * np.asarray(kernel)[None, :, :] * root[:, None, :]
        diagonal = np.zeros_like(rho)
        indices = np.arange(4)
        diagonal[:, indices, indices] = probabilities
        outputs.append(np.linalg.norm(rho - diagonal, axis=(1, 2)))
    return np.column_stack(outputs)

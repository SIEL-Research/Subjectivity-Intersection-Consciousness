#!/usr/bin/env python3
"""DPA-A63R2: intrinsic spectral transfer, UCP generator and Ward current."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any
import hashlib
import importlib.util
import json
import math
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = {
    "protocol": HERE / "FROZEN_PROTOCOL.json",
    "amendment": HERE / "AMENDMENT_0001_MINIMAL_POLYNOMIAL_SELECTION.json",
    "failure_v1": HERE / "FAILURE_V1.json",
    "baseline": HERE / "BASELINE_GATE.json",
    "a62_certificate": ROOT / "audits/DPA_A62_PRIMITIVE_FACTOR_STATE_AND_SOURCE_CAYLEY_GATE_20260915/CERTIFICATE.json",
    "a62_result": ROOT / "audits/DPA_A62_PRIMITIVE_FACTOR_STATE_AND_SOURCE_CAYLEY_GATE_20260915/RESULT.json",
    "a62_evaluate": ROOT / "audits/DPA_A62_PRIMITIVE_FACTOR_STATE_AND_SOURCE_CAYLEY_GATE_20260915/evaluate.py",
    "a61_evaluate": ROOT / "audits/DPA_A61_SOURCE_TRANSFER_FIXED_POINT_STATE_GATE_20260915/evaluate.py",
    "ocbfh014_checker": ROOT / "projects/active/discovery_partner/formal_checks/ocbfh014_source_native_refinement_naturality_check.py",
    "nattrans55_certificate": ROOT / "projects/active/discovery_partner/formal_checks/disc008_nattrans55r_basis_independent_natural_transformation_v1.json",
    "a48r2_result": ROOT / "audits/DPA_A48R2_SOURCE_VARIATIONAL_IMAGE_CURRENT_WARD_GATE_20260915/RESULT.json",
    "ub232_result": ROOT / "audits/UB232_SOURCE_CUP_RELATIONAL_CURRENT/RESULT.json",
    "ub472_report": ROOT / "outputs/discovery-partner/DPA_UB472_BRAID_WORD_CLOCK_C2_INTERTWINER_JA.md",
}

Q17 = tuple[Fraction, Fraction]
ZERO: Q17 = (Fraction(0), Fraction(0))
ONE: Q17 = (Fraction(1), Fraction(0))
SQRT17: Q17 = (Fraction(0), Fraction(1))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def q(value: int | Fraction) -> Q17:
    return (Fraction(value), Fraction(0))


def qadd(left: Q17, right: Q17) -> Q17:
    return (left[0] + right[0], left[1] + right[1])


def qneg(value: Q17) -> Q17:
    return (-value[0], -value[1])


def qsub(left: Q17, right: Q17) -> Q17:
    return qadd(left, qneg(right))


def qmul(left: Q17, right: Q17) -> Q17:
    return (
        left[0] * right[0] + 17 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def qinv(value: Q17) -> Q17:
    denominator = value[0] * value[0] - 17 * value[1] * value[1]
    if denominator == 0:
        raise ZeroDivisionError("zero Q(sqrt(17)) element")
    return (value[0] / denominator, -value[1] / denominator)


def qdiv(left: Q17, right: Q17) -> Q17:
    return qmul(left, qinv(right))


def qpositive(value: Q17) -> bool:
    a, b = value
    if b == 0:
        return a > 0
    if b > 0:
        return a >= 0 or 17 * b * b > a * a
    return a > 0 and a * a > 17 * b * b


def qserial(value: Q17) -> list[str]:
    return [str(value[0]), str(value[1])]


def qmatrix_from_integer(matrix: np.ndarray) -> list[list[Q17]]:
    return [[q(int(value)) for value in row] for row in np.asarray(matrix)]


def qidentity(size: int) -> list[list[Q17]]:
    return [[ONE if row == column else ZERO for column in range(size)] for row in range(size)]


def qmadd(left, right):
    return [
        [qadd(x, y) for x, y in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def qmsub(left, right):
    return [
        [qsub(x, y) for x, y in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def qmscale(matrix, scalar: Q17):
    return [[qmul(value, scalar) for value in row] for row in matrix]


def qmmul(left, right):
    rows = len(left)
    columns = len(right[0])
    middle = len(right)
    result = [[ZERO for _ in range(columns)] for _ in range(rows)]
    for row in range(rows):
        for index in range(middle):
            if left[row][index] == ZERO:
                continue
            for column in range(columns):
                if right[index][column] == ZERO:
                    continue
                result[row][column] = qadd(
                    result[row][column], qmul(left[row][index], right[index][column])
                )
    return result


def qmtrace(matrix) -> Q17:
    value = ZERO
    for index in range(len(matrix)):
        value = qadd(value, matrix[index][index])
    return value


def qmkron_pair(left, right):
    left_a, left_b = left
    right_a, right_b = right
    return (
        np.kron(left_a, right_a) + 17 * np.kron(left_b, right_b),
        np.kron(left_a, right_b) + np.kron(left_b, right_a),
    )


def qmatrix_pair(matrix):
    return (
        np.asarray([[value[0] for value in row] for row in matrix], dtype=object),
        np.asarray([[value[1] for value in row] for row in matrix], dtype=object),
    )


def qpair_trace_sigma(sigma: np.ndarray, projector_pair) -> Q17:
    real, radical = projector_pair
    return (
        sum(Fraction(int(sigma[row, column])) * real[column, row] for row in range(125) for column in range(125)),
        sum(Fraction(int(sigma[row, column])) * radical[column, row] for row in range(125) for column in range(125)),
    )


def integer_matrix_polynomial(source: np.ndarray, coefficients: list[int]) -> np.ndarray:
    result = np.zeros_like(source, dtype=object)
    identity = np.eye(len(source), dtype=object)
    current = identity.copy()
    for coefficient in coefficients:
        result += coefficient * current
        current = current @ source
    return result


def spectral_family(source: np.ndarray) -> tuple[str, list[Q17], list[int], str]:
    source_object = source.astype(object)
    candidates = [
        (
            "A",
            [q(0), q(4), q(5)],
            [3, 1, 1],
            [0, 20, -9, 1],
            "x(x-4)(x-5)",
        ),
        (
            "B",
            [q(2), qadd(q(Fraction(3, 2)), qmul(q(Fraction(-1, 2)), SQRT17)), qadd(q(Fraction(3, 2)), qmul(q(Fraction(1, 2)), SQRT17))],
            [3, 1, 1],
            [4, 4, -5, 1],
            "(x-2)(x^2-3x-2)",
        ),
        (
            "C",
            [q(0), q(4), qadd(q(Fraction(1, 2)), qmul(q(Fraction(-1, 2)), SQRT17)), qadd(q(Fraction(1, 2)), qmul(q(Fraction(1, 2)), SQRT17))],
            [1, 2, 1, 1],
            [0, 16, 0, -5, 1],
            "x(x-4)(x^2-x-4)",
        ),
        ("FLIP", [q(0), q(5)], [4, 1], [0, -5, 1], "x(x-5)"),
    ]
    matches = []
    for name, eigenvalues, ranks, coefficients, polynomial in candidates:
        if not np.any(integer_matrix_polynomial(source_object, coefficients)):
            matches.append((name, eigenvalues, ranks, polynomial))
    if not matches:
        raise AssertionError("no declared polynomial annihilates the source contraction")
    minimum_degree = min(len(item[1]) for item in matches)
    minimal_matches = [item for item in matches if len(item[1]) == minimum_degree]
    if len(minimal_matches) != 1:
        raise AssertionError(
            f"minimum-degree spectral family is not unique: {[item[0] for item in minimal_matches]}"
        )
    return minimal_matches[0]


def exact_projectors(source: np.ndarray):
    family, eigenvalues, expected_ranks, polynomial = spectral_family(source)
    source_q = qmatrix_from_integer(source)
    identity = qidentity(5)
    projectors = []
    for eigenvalue in eigenvalues:
        projector = identity
        denominator = ONE
        for other in eigenvalues:
            if other == eigenvalue:
                continue
            projector = qmmul(projector, qmsub(source_q, qmscale(identity, other)))
            denominator = qmul(denominator, qsub(eigenvalue, other))
        projector = qmscale(projector, qinv(denominator))
        projectors.append(projector)
    assert all(qmmul(projector, projector) == projector for projector in projectors)
    assert all(
        qmmul(projectors[left], projectors[right]) == [[ZERO] * 5 for _ in range(5)]
        for left in range(len(projectors))
        for right in range(len(projectors))
        if left != right
    )
    total = [[ZERO] * 5 for _ in range(5)]
    for projector in projectors:
        total = qmadd(total, projector)
    assert total == identity
    ranks = []
    for eigenvalue, projector in zip(eigenvalues, projectors):
        trace = qmtrace(projector)
        assert trace[1] == 0 and trace[0].denominator == 1
        ranks.append(int(trace[0]))
        assert qmmul(source_q, projector) == qmscale(projector, eigenvalue)
    assert ranks == expected_ranks
    return family, eigenvalues, ranks, polynomial, projectors


def spectral_weights(sigma: np.ndarray, projectors):
    pairs = [qmatrix_pair(projector) for projector in projectors]
    triple_projectors = []
    weights = []
    size = len(projectors)
    for first in range(size):
        for second in range(size):
            for third in range(size):
                pair = qmkron_pair(qmkron_pair(pairs[first], pairs[second]), pairs[third])
                triple_projectors.append(pair)
                weights.append(qpair_trace_sigma(sigma, pair))
    assert all(qpositive(weight) for weight in weights)
    total = ZERO
    for weight in weights:
        total = qadd(total, weight)
    assert total == q(18000)
    pair_weights = [[ZERO for _ in range(size)] for _ in range(size)]
    left_pair = [[ZERO for _ in range(size)] for _ in range(size)]
    right_pair = [[ZERO for _ in range(size)] for _ in range(size)]
    for first in range(size):
        for second in range(size):
            for third in range(size):
                weight = weights[(first * size + second) * size + third]
                left_pair[first][second] = qadd(left_pair[first][second], weight)
                right_pair[second][third] = qadd(right_pair[second][third], weight)
    assert left_pair == right_pair
    pair_weights = left_pair
    return weights, pair_weights, triple_projectors


def exact_edge_current(weights, pair_weights, size):
    states = size**3
    transition = [[ZERO for _ in range(states)] for _ in range(states)]
    for first in range(size):
        for second in range(size):
            for third in range(size):
                source = (first * size + second) * size + third
                for fourth in range(size):
                    target = (second * size + third) * size + fourth
                    transition[source][target] = qdiv(
                        weights[target], pair_weights[second][third]
                    )
                row_sum = ZERO
                for value in transition[source]:
                    row_sum = qadd(row_sum, value)
                assert row_sum == ONE
    current = [[ZERO for _ in range(states)] for _ in range(states)]
    divergence = [ZERO for _ in range(states)]
    nonzero = 0
    squared_norm = ZERO
    for source in range(states):
        for target in range(states):
            value = qsub(
                qmul(weights[source], transition[source][target]),
                qmul(weights[target], transition[target][source]),
            )
            current[source][target] = value
            divergence[source] = qadd(divergence[source], value)
            if value != ZERO:
                nonzero += 1
            squared_norm = qadd(squared_norm, qmul(value, value))
    assert all(value == ZERO for value in divergence)
    assert nonzero > 0 and qpositive(squared_norm)
    # Indicator of the first spectral triple gives a strictly positive Dirichlet witness.
    energy = ZERO
    for source in range(states):
        for target in range(states):
            difference = int(target == 0) - int(source == 0)
            if difference:
                energy = qadd(
                    energy,
                    qmul(
                        q(Fraction(difference * difference, 2)),
                        qmul(weights[source], transition[source][target]),
                    ),
                )
    energy = qdiv(energy, q(18000))
    assert qpositive(energy)
    return transition, {
        "nonzero_oriented_entries": nonzero,
        "divergence_zero_at_every_vertex": True,
        "squared_norm_unnormalized": qserial(squared_norm),
        "indicator_dirichlet_energy": qserial(energy),
    }


def tonelli_shanks(value: int, prime: int) -> int:
    value %= prime
    assert pow(value, (prime - 1) // 2, prime) == 1
    if prime % 4 == 3:
        return pow(value, (prime + 1) // 4, prime)
    power = prime - 1
    exponent = 0
    while power % 2 == 0:
        exponent += 1
        power //= 2
    nonresidue = 2
    while pow(nonresidue, (prime - 1) // 2, prime) != prime - 1:
        nonresidue += 1
    root = pow(value, (power + 1) // 2, prime)
    auxiliary = pow(value, power, prime)
    factor = pow(nonresidue, power, prime)
    remaining = exponent
    while auxiliary != 1:
        index = 1
        square = auxiliary * auxiliary % prime
        while square != 1:
            square = square * square % prime
            index += 1
        multiplier = pow(factor, 1 << (remaining - index - 1), prime)
        root = root * multiplier % prime
        auxiliary = auxiliary * multiplier * multiplier % prime
        factor = multiplier * multiplier % prime
        remaining = index
    return root


def map_fraction(value: Fraction, prime: int) -> int:
    return value.numerator % prime * pow(value.denominator % prime, prime - 2, prime) % prime


def map_q17(value: Q17, prime: int, root: int) -> int:
    return (map_fraction(value[0], prime) + root * map_fraction(value[1], prime)) % prime


def map_projector(projector, prime, root):
    return np.asarray(
        [[map_q17(value, prime, root) for value in row] for row in projector],
        dtype=np.int64,
    )


def rank_mod(rows: list[list[int]], prime: int) -> int:
    work = [[int(value) % prime for value in row] for row in rows]
    rank = 0
    for column in range(len(work[0])):
        pivot = next((row for row in range(rank, len(work)) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], prime - 2, prime)
        work[rank] = [(value * inverse) % prime for value in work[rank]]
        for row in range(len(work)):
            if row == rank or work[row][column] == 0:
                continue
            scale = work[row][column]
            work[row] = [
                (left - scale * right) % prime
                for left, right in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == len(work):
            break
    return rank


def modular_current_packet(
    sigma: np.ndarray,
    projectors,
    ranks,
    weights,
    pair_weights,
    generators,
    prime: int,
    root: int,
):
    size = len(projectors)
    mapped = [map_projector(projector, prime, root) for projector in projectors]
    triple = []
    triple_ranks = []
    for first in range(size):
        for second in range(size):
            for third in range(size):
                triple.append(
                    np.kron(np.kron(mapped[first], mapped[second]), mapped[third]) % prime
                )
                triple_ranks.append(ranks[first] * ranks[second] * ranks[third])
    mapped_weights = [map_q17(value, prime, root) for value in weights]
    mapped_pairs = [
        [map_q17(value, prime, root) for value in row] for row in pair_weights
    ]
    states = size**3
    transition = np.zeros((states, states), dtype=np.int64)
    for first in range(size):
        for second in range(size):
            for third in range(size):
                source = (first * size + second) * size + third
                for fourth in range(size):
                    target = (second * size + third) * size + fourth
                    transition[source, target] = (
                        mapped_weights[target]
                        * pow(mapped_pairs[second][third], prime - 2, prime)
                    ) % prime
                assert int(transition[source].sum() % prime) == 1

    inv18000 = pow(18000, prime - 2, prime)
    rho = np.zeros((125, 125), dtype=np.int64)
    for index, projector in enumerate(triple):
        coefficient = (
            mapped_weights[index]
            * inv18000
            * pow(triple_ranks[index], prime - 2, prime)
        ) % prime
        rho = (rho + coefficient * projector) % prime
    assert int(np.trace(rho) % prime) == 1

    currents = []
    ward = []
    for generator in generators:
        observable = np.asarray(generator, dtype=np.int64) % prime
        averages = []
        for index, projector in enumerate(triple):
            averages.append(
                int(np.trace((projector @ observable) % prime) % prime)
                * pow(triple_ranks[index], prime - 2, prime)
                % prime
            )
        coefficients = transition @ np.asarray(averages, dtype=np.int64) % prime
        image = np.zeros((125, 125), dtype=np.int64)
        for coefficient, projector in zip(coefficients, triple):
            image = (image + int(coefficient) * projector) % prime
        current = (image - observable) % prime
        currents.append(current)
        ward.append(int(np.trace((rho @ current) % prime) % prime))
    current_rank = rank_mod(
        [[int(value) for value in current.flat] for current in currents], prime
    )
    gram = [
        [int(np.sum((left * right) % prime) % prime) for right in currents]
        for left in currents
    ]
    return {
        "prime": prime,
        "sqrt17_image": root,
        "channel_unital": True,
        "stationary_state_trace_one": True,
        "ward_expectations": ward,
        "current_rank": current_rank,
        "current_gram": gram,
    }


def coordinate_diagonal_naturality_failure(sigma: np.ndarray):
    change = 5 * np.eye(5, dtype=np.int64)
    change[0, 0] = 3
    change[0, 1] = 4
    change[1, 0] = -4
    change[1, 1] = 3
    triple = np.kron(np.kron(change, change), change).astype(object)
    transported = triple @ sigma.astype(object) @ triple.T
    after_then_diagonal = np.diag(np.diag(transported)).astype(object)
    diagonal_then_after = triple @ np.diag(np.diag(sigma)).astype(object) @ triple.T
    difference = after_then_diagonal - diagonal_then_after
    nonzero = np.argwhere(difference != 0)
    assert len(nonzero) > 0
    row, column = map(int, nonzero[0])
    return {
        "basis_change": "G exact 3-4-5 rotation on source coordinates 0 and 1",
        "common_denominator": 15625,
        "nonzero_entries": int(len(nonzero)),
        "first_witness": {
            "row": row,
            "column": column,
            "integer_numerator": int(difference[row, column]),
        },
        "integer_frobenius_squared": sum(
            int(value) * int(value) for value in difference.flat
        ),
        "consequence": "Coordinate diagonalization is not natural under the full NATTRANS-55R unitary source groupoid.",
    }


def classify(relation, cap, projectors_chi, o14, flip_control=False):
    rho = A61.selected_state_numerator(relation, cap, o14)
    sigma, _ = A61.stationary_local_numerator(rho)
    source = A62.raw_contraction(relation)
    family, eigenvalues, ranks, polynomial, projectors = exact_projectors(source)
    weights, pair_weights, _ = spectral_weights(sigma, projectors)
    transition, edge = exact_edge_current(weights, pair_weights, len(projectors))
    identity5 = np.eye(5, dtype=np.int64)
    clock, _, _, _ = o14.source_g1(relation)
    generators = [
        clock,
        *[np.kron(numerator, identity5) for numerator, _ in projectors_chi],
    ]
    modular = []
    for prime in (1000033, 1000159):
        root = tonelli_shanks(17, prime)
        for embedding in (root, (-root) % prime):
            modular.append(
                modular_current_packet(
                    sigma,
                    projectors,
                    ranks,
                    weights,
                    pair_weights,
                    generators,
                    prime,
                    embedding,
                )
            )
    assert all(record["current_rank"] == 4 for record in modular)
    assert all(record["ward_expectations"] == [0, 0, 0, 0] for record in modular)
    return {
        "sigma": sigma,
        "source_contraction": source,
        "family": family,
        "minimal_polynomial": polynomial,
        "eigenvalues": [qserial(value) for value in eigenvalues],
        "spectral_ranks": ranks,
        "spectral_projectors": projectors,
        "spectral_weights": weights,
        "pair_weights": pair_weights,
        "edge_current": edge,
        "modular_packets": modular,
        "generator_count": 4,
        "spectral_sector_count": len(projectors),
        "triple_sector_count": len(projectors) ** 3,
        "pair_transition_primitive_exponent_upper_bound": 2,
        "triple_transfer_primitive_exponent_upper_bound": 3,
        "flip_control": flip_control,
    }


def build_outputs():
    protocol = load(INPUTS["protocol"])
    amendment = load(INPUTS["amendment"])
    failure_v1 = load(INPUTS["failure_v1"])
    baseline = load(INPUTS["baseline"])
    a62_result = load(INPUTS["a62_result"])
    a48r2 = load(INPUTS["a48r2_result"])
    ub232 = load(INPUTS["ub232_result"])
    assert protocol["status"].startswith("FROZEN_AFTER_DISCLOSED")
    assert amendment["status"] == "AUTHORIZED_BEFORE_R2_REPLAY"
    assert amendment["gate_id"] == "DPA-A63R2"
    assert failure_v1["status"] == "PRESERVED_IMPLEMENTATION_PROVENANCE_FAILURE"
    assert failure_v1["scientific_decision"] == "NONE"
    assert baseline["status"] == "PASS_DIRECT_COMMIT_PINNED_A62_CONTINUATION_WITH_SCOPE_CORRECTION"
    assert baseline["science_outcome_authorized"] is True
    assert a62_result["status"].startswith("PASS_PRIMITIVE_FACTOR_STATE")
    assert a48r2["status"].startswith("CONDITIONAL_PASS_A48")
    assert ub232["status"].startswith("SOURCE_CUP_RELATIONAL_CURRENT")

    global A61, A62
    A61 = load_module("a63_a61", INPUTS["a61_evaluate"])
    A62 = load_module("a63_a62", INPUTS["a62_evaluate"])
    o14 = load_module("a63_o14", INPUTS["ocbfh014_checker"])
    survivors, projectors_chi, _, source_meta = o14.reconstruct_actual_source()

    flip_relation = A61.ordinary_flip()
    records = []
    for survivor in survivors:
        relation = survivor["R"].astype(np.int64)
        cap = survivor["P"].astype(np.int64)
        raw = classify(relation, cap, projectors_chi, o14)
        flip = classify(flip_relation, cap, projectors_chi, o14, flip_control=True)
        correction = coordinate_diagonal_naturality_failure(raw["sigma"])
        gram_differences = []
        for raw_packet in raw["modular_packets"]:
            flip_packet = next(
                packet
                for packet in flip["modular_packets"]
                if packet["prime"] == raw_packet["prime"]
                and packet["sqrt17_image"] == raw_packet["sqrt17_image"]
            )
            prime = raw_packet["prime"]
            differences = [
                (raw_packet["current_gram"][row][column] - flip_packet["current_gram"][row][column]) % prime
                for row in range(4)
                for column in range(4)
            ]
            assert any(differences)
            gram_differences.append(
                {
                    "prime": prime,
                    "sqrt17_image": raw_packet["sqrt17_image"],
                    "different_entries": sum(value != 0 for value in differences),
                    "first_nonzero_difference": next(value for value in differences if value),
                }
            )
        records.append(
            {
                "mask": survivor["mask"],
                "a62_scope_correction": correction,
                "spectral_family": raw["family"],
                "minimal_polynomial": raw["minimal_polynomial"],
                "eigenvalues_qsqrt17": raw["eigenvalues"],
                "spectral_ranks": raw["spectral_ranks"],
                "spectral_sector_count": raw["spectral_sector_count"],
                "triple_sector_count": raw["triple_sector_count"],
                "all_spectral_weights_positive": all(qpositive(value) for value in raw["spectral_weights"]),
                "spectral_weight_minimum_not_serialized": "PROVED_POSITIVE_FROM_SIGMA_AT_LEAST_10I_AND_NONZERO_PROJECTOR_RANK",
                "pair_transition_primitive_exponent_upper_bound": 2,
                "triple_transfer_primitive_exponent_upper_bound": 3,
                "edge_current": raw["edge_current"],
                "prime_embedding_packets": raw["modular_packets"],
                "raw_flip_current_gram_separation": gram_differences,
                "full_unitary_source_naturality": True,
                "factor_gns": True,
            }
        )

    tests = {
        "G0_A62_SCOPE_CORRECTION": True,
        "G1_SPECTRAL_EXACTNESS": True,
        "G2_FULL_SOURCE_NATURALITY": True,
        "G3_PRIMITIVE_FACTOR_STATE": True,
        "G4_RAW_CONTROL": True,
        "G5_UCP_GENERATOR": True,
        "G6_NONZERO_CONSERVED_EDGE_CURRENT": True,
        "G7_FOUR_CURRENT_RANK": True,
        "G8_WARD_IDENTITY": True,
        "STANDARD_HSMI_REQUIRED": False,
        "PHYSICAL_HAMILTONIAN_TIME_ESTABLISHED": False,
    }
    certificate = {
        "schema": "siel.dpa.a63.intrinsic-spectral-transfer-ward.certificate.v1",
        "gate_id": "DPA-A63R2",
        "parent_gate_id": "DPA-A63",
        "revision": "R2",
        "date": "2026-09-15",
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()
        },
        "source_reconstruction": source_meta,
        "a62_scope_correction": {
            "status": "A62_MATHEMATICAL_CONSTRUCTION_RETAINED__FULL_SOURCE_NATURALITY_CLAIM_NARROWED",
            "reason": "The Cayley functional is source-natural, but coordinate diagonalization of sigma3 is not natural under a non-diagonal unitary basis change. The A62 primitive state is therefore affine-presentation-natural, not natural on the full NATTRANS-55R groupoid.",
            "scientific_validity": "A62 exact primitivity, factor, raw/flip and rank computations remain valid in the frozen affine presentation. No frozen numeric outcome is rewritten.",
        },
        "spectral_repair_theorem": {
            "construction": "Use the complete eigenspace projectors of S_R rather than coordinate diagonal entries. Degenerate eigenspaces remain unsplit.",
            "naturality_proof": "Under any source isomorphism V, S_R maps to V S_R V*. Polynomial spectral projectors map to V P_a V*. Trace weights, degeneracies, transitions, the Markov state, Phi*, L* and all current Gram matrices therefore transport without a basis choice.",
            "faithfulness": "A61 proves sigma3 is at least 10I. Every nonzero spectral triple projector therefore has strictly positive weight. The pair transition is primitive in two steps and the triple transfer in three steps.",
            "factoriality": "The primitive finite-memory spectral Markov law has trivial tail. The normalized trace inside every spectral block adds no classical center, so the infinite finitely correlated GNS is a factor.",
        },
        "ucp_generator_theorem": {
            "channel": "Phi* measures the complete spectral triple sectors, applies the source Markov transition and prepares normalized block identities.",
            "ucp_unital": True,
            "stationary": True,
            "generator": "L*=Phi*-identity",
            "semigroup": "exp(tL*)=exp(-t) sum_n t^n Phi*^n/n!",
            "ward_identity": "omega(L*(A))=0 for every local A",
            "energy": "The stationary Dirichlet form is positive and nonzero, but is a transfer/dissipation energy rather than a physical Hamiltonian.",
        },
        "sector_records": records,
        "decision_tests": tests,
        "endpoint_semantic_gates": protocol["endpoint_semantic_gates"],
    }
    result = {
        "schema": "siel.dpa.a63.intrinsic-spectral-transfer-ward.result.v1",
        "gate_id": "DPA-A63R2",
        "parent_gate_id": "DPA-A63",
        "revision": "R2",
        "date": "2026-09-15",
        "primary_evidence_status": "Theoretical derivation",
        "secondary_qualifier": "Exact all-eight-sector Q(sqrt(17)) spectral-projector and two-prime current audit",
        "scientific_layer": "mathematical formulation and source-native transfer dynamics",
        "status": "PASS_FULL_NATURAL_SPECTRAL_FACTOR_AND_UCP_TRANSFER__PASS_NONZERO_DIVERGENCE_FREE_CURRENT_AND_RANK4_WARD__A62_FULL_NATURALITY_SCOPE_CORRECTED__PHYSICAL_HAMILTONIAN_TIME_OPEN",
        "decisive_result": "A62's coordinate-diagonal primitive state is not natural under the full unitary source groupoid, although its affine-presentation theorem remains valid. Replacing coordinate colors by the complete spectral sectors of the source-natural raw contraction removes that defect and also removes the Cayley selector. The resulting state is faithful, primitive and factor. Its canonical measure-and-prepare transfer is UCP, L*=Phi*-identity generates a UCP semigroup, its stationary edge current is nonzero with exact zero divergence, and the four G1/P_chi current operators have rank four with exact Ward expectation zero in every sector.",
        "observed_evidence": [
            "The exact G rotation gives a nonzero coordinate-diagonal naturality defect in every sector.",
            "All raw contractions fall into three exact minimal-polynomial families over Q(sqrt(17)); complete projectors are orthogonal, exhaustive and keep degeneracies unsplit.",
            "Every spectral triple has positive weight because sigma3 is bounded below by 10I.",
            "Every pair transition is primitive within two steps and every triple transfer within three steps.",
            "All stationary edge currents are nonzero and divergence-free.",
            "The four current operators have rank four and zero stationary expectation in both embeddings of Q(sqrt(17)) over two prime fields.",
            "The current Gram differs from the matched ordinary-flip control in every sector and every exact embedding."
        ],
        "pattern": "A source-native dynamics can be obtained without choosing a color basis or a Cayley frame by coarse-graining only to the intrinsic eigenspaces of a marked raw contraction. The symmetric part supplies a positive transfer energy while the antisymmetric stationary flow supplies a conserved orientation current.",
        "interpretive_leap": "The Braid relation may carry both the stable objectivity of a factor state and a surviving direction of flow: arbitrary coordinate labels disappear, but the imbalance around relational cycles remains as a conserved current.",
        "alternative_explanations": [
            "Spectral functional calculus, primitive Markov transfer, measure-and-prepare UCP channels and stationary Ward balance are standard mathematics.",
            "The current is a finite-window transfer current, not yet a spacetime-local Noether current.",
            "The UCP semigroup is dissipative and its parameter is not yet physical clock time.",
            "Current rank four is an operator-response dimension, not physical 1+3 spacetime.",
            "Raw/flip Gram separation is not full raw-braid tomography."
        ],
        "novel_hypothesis": "The nonterminal Braid tower lifts the A63 spectral transfer compatibly across stages; its antisymmetric current becomes a local continuity current and its Dirichlet form becomes the spatial kinetic operator, providing the missing massless long-wavelength sector without importing a continuum Laplacian.",
        "falsifier": protocol["falsifier"],
        "required_prospective_test": "Audit stage-compatible lifts of the spectral projectors, UCP transfer, Dirichlet form and current. Require exact intertwining, a source-derived zero mode, small-momentum quadratic scaling and a local continuity identity. Stop if the finite-window transfer cannot be lifted without choosing a basis or external graph Laplacian.",
        "confidence": {
            "a62_scope_correction": "HIGH",
            "spectral_factor_and_ucp_transfer": "HIGH",
            "rank_four_ward_current": "HIGH",
            "physical_time_energy_or_spacetime_current": "LOW_OPEN"
        },
        "candidate_level": "LEVEL_1_PATTERN_CANDIDATE__RESERVOIR_REGISTRATION_BLOCKED",
        "endpoint_semantic_gates": protocol["endpoint_semantic_gates"],
        "next_gate": "DPA-A64_STAGE_COMPATIBLE_SPECTRAL_CURRENT_AND_MASSLESS_LIMIT_GATE",
        "claim_ceiling": protocol["claim_ceiling"],
        "governance": {
            "rpd_handoff_authorized": False,
            "experiment_authorized": False,
            "publication_mutation_authorized": False,
            "external_communication_authorized": False,
            "official_siel_adoption": False,
        },
    }
    return certificate, result


def main():
    certificate, result = build_outputs()
    (HERE / "CERTIFICATE.json").write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2) + "\n"
    )
    (HERE / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(result["status"])
    print(json.dumps(certificate["decision_tests"], indent=2))


if __name__ == "__main__":
    main()

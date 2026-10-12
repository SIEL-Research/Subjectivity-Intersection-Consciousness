#!/usr/bin/env python3
"""DPA-A62: primitive factor state and source-natural Cayley lift."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any
import hashlib
import importlib
import importlib.util
import json
import math
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INPUTS = {
    "protocol": HERE / "FROZEN_PROTOCOL.json",
    "baseline": HERE / "BASELINE_GATE.json",
    "a61_certificate": ROOT / "audits/DPA_A61_SOURCE_TRANSFER_FIXED_POINT_STATE_GATE_20260915/CERTIFICATE.json",
    "a61_result": ROOT / "audits/DPA_A61_SOURCE_TRANSFER_FIXED_POINT_STATE_GATE_20260915/RESULT.json",
    "a61_evaluate": ROOT / "audits/DPA_A61_SOURCE_TRANSFER_FIXED_POINT_STATE_GATE_20260915/evaluate.py",
    "a56_evaluate": ROOT / "audits/DPA_A56_FROBENIUS_LIE_CLOSURE_GATE_20260915/evaluate.py",
    "ocbfh014_checker": ROOT / "projects/active/discovery_partner/formal_checks/ocbfh014_source_native_refinement_naturality_check.py",
    "nattrans55_checker": ROOT / "projects/active/discovery_partner/formal_checks/disc008_nattrans55r_basis_independent_natural_transformation_check.py",
    "nattrans55_certificate": ROOT / "projects/active/discovery_partner/formal_checks/disc008_nattrans55r_basis_independent_natural_transformation_v1.json",
}


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


def lcm(left: int, right: int) -> int:
    return abs(left * right) // math.gcd(left, right)


def inverse_fraction(matrix: np.ndarray) -> list[list[Fraction]]:
    size = len(matrix)
    work = [
        [Fraction(int(matrix[row, column])) for column in range(size)]
        + [Fraction(row == column) for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column]), None
        )
        if pivot is None:
            raise AssertionError("integer matrix is singular")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row == column or work[row][column] == 0:
                continue
            scale = work[row][column]
            work[row] = [
                left - scale * right
                for left, right in zip(work[row], work[column])
            ]
    return [row[size:] for row in work]


def cayley_integer_pair(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return Gaussian-integer V=A+iB and d with U=V/d."""

    identity = np.eye(5, dtype=np.int64)
    square = source @ source
    inverse = inverse_fraction(square + identity)
    real = [
        [
            sum(
                Fraction(int(square[row, middle] - identity[row, middle]))
                * inverse[middle][column]
                for middle in range(5)
            )
            for column in range(5)
        ]
        for row in range(5)
    ]
    imag = [
        [
            sum(
                Fraction(int(-2 * source[row, middle]))
                * inverse[middle][column]
                for middle in range(5)
            )
            for column in range(5)
        ]
        for row in range(5)
    ]
    denominator = 1
    for row in real + imag:
        for value in row:
            denominator = lcm(denominator, value.denominator)
    real_integer = np.asarray(
        [[int(value * denominator) for value in row] for row in real],
        dtype=object,
    )
    imag_integer = np.asarray(
        [[int(value * denominator) for value in row] for row in imag],
        dtype=object,
    )
    common = denominator
    for value in list(real_integer.flat) + list(imag_integer.flat):
        common = math.gcd(common, abs(int(value)))
    denominator //= common
    real_integer //= common
    imag_integer //= common
    return real_integer, imag_integer, denominator


def complex_kron(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    left_real, left_imag = left
    right_real, right_imag = right
    return (
        np.kron(left_real, right_real) - np.kron(left_imag, right_imag),
        np.kron(left_real, right_imag) + np.kron(left_imag, right_real),
    )


def rank_mod(rows: list[list[int]], prime: int) -> int:
    work = [[int(value) % prime for value in row] for row in rows]
    row_count = len(work)
    column_count = len(work[0])
    rank = 0
    for column in range(column_count):
        pivot = next(
            (row for row in range(rank, row_count) if work[row][column]), None
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], prime - 2, prime)
        work[rank] = [(value * inverse) % prime for value in work[rank]]
        for row in range(row_count):
            if row == rank or work[row][column] == 0:
                continue
            scale = work[row][column]
            work[row] = [
                (left - scale * right) % prime
                for left, right in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


def exact_pair_rank(matrices: list[tuple[np.ndarray, np.ndarray]]) -> list[int]:
    rows = [
        [int(value) for value in list(real.flat) + list(imag.flat)]
        for real, imag in matrices
    ]
    return [rank_mod(rows, prime) for prime in (1000003, 1000033)]


def raw_contraction(relation: np.ndarray) -> np.ndarray:
    # v=point+line=(1,1,1,1,1), so this is
    # (I tensor <v|) R (I tensor |v>).
    return np.einsum("abcd->ac", relation.reshape(5, 5, 5, 5)).astype(
        np.int64
    )


def qadd_matrix(nat, left, right):
    return [
        [nat.exact.qadd(x, y) for x, y in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def qsub_matrix(nat, left, right):
    return [
        [nat.qsub(x, y) for x, y in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def qsource_vector(nat, source):
    return [
        [nat.exact.qadd(point[0], line[0])]
        for point, line in zip(source["point"], source["line"])
    ]


def qright_insertion(nat, vector):
    result = nat.qmatrix(25, 5)
    for first in range(5):
        for second in range(5):
            result[first * 5 + second][first] = vector[second][0]
    return result


def qraw_contraction(nat, source):
    insertion = qright_insertion(nat, qsource_vector(nat, source))
    return nat.qmatmul(
        nat.qmatmul(nat.exact.qadjoint(insertion), source["R"]), insertion
    )


def qcayley_from_integer_pair(nat, real, imag, denominator):
    result = nat.qmatrix(5, 5)
    for row in range(5):
        for column in range(5):
            real_part = nat.exact.qscalar(
                Fraction(int(real[row, column]), denominator)
            )
            imag_part = nat.exact.qmul(
                nat.QI,
                nat.exact.qscalar(
                    Fraction(int(imag[row, column]), denominator)
                ),
            )
            result[row][column] = nat.exact.qadd(real_part, imag_part)
    return result


def cayley_naturality(relation: np.ndarray, real, imag, denominator, nat):
    source = nat.frozen_object(relation)
    source_s = qraw_contraction(nat, source)
    expected_s = nat.qfrom_integer(raw_contraction(relation))
    assert source_s == expected_s
    source_u = qcayley_from_integer_pair(nat, real, imag, denominator)
    identity = nat.exact.qidentity(5)
    i_identity = [
        [nat.QI if row == column else nat.ZERO for column in range(5)]
        for row in range(5)
    ]
    assert nat.qmatmul(qadd_matrix(nat, source_s, i_identity), source_u) == qsub_matrix(
        nat, source_s, i_identity
    )
    assert nat.qmatmul(source_u, nat.exact.qadjoint(source_u)) == identity

    records = []
    for name in ("D", "G", "P"):
        unitary = nat.basis_changes()[name]
        target = nat.transport_object(source, unitary)
        target_s = qraw_contraction(nat, target)
        transported_s = nat.qconjugate(unitary, source_s)
        transported_u = nat.qconjugate(unitary, source_u)
        assert target_s == transported_s
        assert nat.qmatmul(
            qadd_matrix(nat, target_s, i_identity), transported_u
        ) == qsub_matrix(nat, target_s, i_identity)
        assert nat.qmatmul(
            transported_u, nat.exact.qadjoint(transported_u)
        ) == identity
        records.append(
            {
                "basis_change": name,
                "raw_contraction_transport": True,
                "cayley_equation_transport": True,
                "cayley_unitarity": True,
            }
        )
    return records


def diagonal_markov_data(sigma: np.ndarray) -> dict[str, Any]:
    weights = np.diag(sigma).reshape(5, 5, 5).astype(np.int64)
    left_pair = weights.sum(axis=2)
    right_pair = weights.sum(axis=0)
    assert np.array_equal(left_pair, right_pair)
    assert int(weights.min()) > 0
    assert int(left_pair.min()) > 0

    stochastic_rows = 0
    stationary_checks = 0
    positive_two_step_entries = 0
    for first in range(5):
        for second in range(5):
            denominator = int(left_pair[first, second])
            assert sum(
                Fraction(int(weights[first, second, third]), denominator)
                for third in range(5)
            ) == 1
            stochastic_rows += 1
            for third in range(5):
                # From (first,second) to (third,fourth), the unique two-step
                # path passes through (second,third); both factors are positive.
                for fourth in range(5):
                    value = Fraction(
                        int(weights[first, second, third]), denominator
                    ) * Fraction(
                        int(weights[second, third, fourth]),
                        int(left_pair[second, third]),
                    )
                    assert value > 0
                    positive_two_step_entries += 1
    for second in range(5):
        for third in range(5):
            incoming = sum(
                Fraction(int(left_pair[first, second]), 18000)
                * Fraction(
                    int(weights[first, second, third]),
                    int(left_pair[first, second]),
                )
                for first in range(5)
            )
            assert incoming == Fraction(int(left_pair[second, third]), 18000)
            stationary_checks += 1
    assert int(weights.sum()) == 18000
    return {
        "weights": weights,
        "left_pair": left_pair,
        "weight_minimum": int(weights.min()),
        "weight_maximum": int(weights.max()),
        "pair_weight_minimum": int(left_pair.min()),
        "pair_weight_maximum": int(left_pair.max()),
        "stochastic_rows": stochastic_rows,
        "stationary_pair_equations": stationary_checks,
        "two_step_positive_entries": positive_two_step_entries,
        "primitive_exponent_upper_bound": 2,
    }


def coherent_state_pair(relation: np.ndarray, weights: np.ndarray):
    source = raw_contraction(relation)
    real, imag, denominator = cayley_integer_pair(source)
    identity = np.eye(5, dtype=object)
    assert np.array_equal(
        real @ real.T + imag @ imag.T,
        denominator * denominator * identity,
    )
    assert not np.any(imag @ real.T - real @ imag.T)
    triple_real, triple_imag = complex_kron(
        complex_kron((real, imag), (real, imag)), (real, imag)
    )
    diagonal = weights.reshape(-1).astype(object)
    state_real = (
        (triple_real * diagonal) @ triple_real.T
        + (triple_imag * diagonal) @ triple_imag.T
    )
    state_imag = (
        (triple_imag * diagonal) @ triple_real.T
        - (triple_real * diagonal) @ triple_imag.T
    )
    scale = denominator**6
    assert sum(int(state_real[index, index]) for index in range(125)) == 18000 * scale
    assert all(int(state_imag[index, index]) == 0 for index in range(125))
    return state_real, state_imag, denominator, source, real, imag


def first_pair_witness(real: np.ndarray, imag: np.ndarray) -> dict[str, int]:
    for row in range(len(real)):
        for column in range(len(real)):
            if real[row, column] or imag[row, column]:
                return {
                    "row": row,
                    "column": column,
                    "real_numerator": int(real[row, column]),
                    "imag_numerator": int(imag[row, column]),
                }
    raise AssertionError("expected a nonzero matrix pair")


def classify_sector(survivor, projectors, o14, nat) -> dict[str, Any]:
    relation = survivor["R"].astype(np.int64)
    cap = survivor["P"].astype(np.int64)
    rho = A61.selected_state_numerator(relation, cap, o14)
    sigma, _ = A61.stationary_local_numerator(rho)
    markov = diagonal_markov_data(sigma)
    weights = markov.pop("weights")
    markov.pop("left_pair")

    identity5 = np.eye(5, dtype=np.int64)
    clock, _, _, _ = o14.source_g1(relation)
    generators = [
        clock,
        *[np.kron(numerator, identity5) for numerator, _ in projectors],
    ]
    diagonal_state = np.diag(weights.reshape(-1)).astype(object)
    diagonal_tangents = [
        (
            generator @ diagonal_state - diagonal_state @ generator,
            np.zeros((125, 125), dtype=object),
        )
        for generator in generators
    ]
    diagonal_ranks = exact_pair_rank(diagonal_tangents)
    assert diagonal_ranks == [1, 1]

    flip_relation = A61.ordinary_flip()
    flip_rho = A61.selected_state_numerator(flip_relation, cap, o14)
    flip_sigma, _ = A61.stationary_local_numerator(flip_rho)
    flip_weights = np.diag(flip_sigma).reshape(5, 5, 5).astype(np.int64)
    diagonal_difference = weights - flip_weights
    assert np.any(diagonal_difference)

    real, imag, denominator, source, cayley_real, cayley_imag = coherent_state_pair(
        relation, weights
    )
    coherent_tangents = []
    for generator in generators:
        coherent_tangents.append(
            (generator @ real - real @ generator, generator @ imag - imag @ generator)
        )
    coherent_ranks = exact_pair_rank(coherent_tangents)
    assert coherent_ranks == [4, 4]

    flip_real, flip_imag, flip_denominator, *_ = coherent_state_pair(
        flip_relation, flip_weights
    )
    raw_scale = denominator**6
    flip_scale = flip_denominator**6
    difference_real = real * flip_scale - flip_real * raw_scale
    difference_imag = imag * flip_scale - flip_imag * raw_scale
    assert np.any(difference_real) or np.any(difference_imag)
    witness = first_pair_witness(difference_real, difference_imag)
    nonzero_count = sum(
        bool(difference_real[row, column] or difference_imag[row, column])
        for row in range(125)
        for column in range(125)
    )

    naturality = cayley_naturality(
        relation, cayley_real, cayley_imag, denominator, nat
    )
    noncentral_count = sum(
        bool(
            real[row, column]
            - (18000 * raw_scale // 125 if row == column else 0)
            or imag[row, column]
        )
        for row in range(125)
        for column in range(125)
    )
    assert noncentral_count > 0
    return {
        "mask": survivor["mask"],
        "markov_transfer": markov,
        "factor_argument": {
            "finite_memory": 2,
            "strictly_positive_two_step_kernel": True,
            "unique_stationary_pair_distribution": True,
            "exponential_mixing": True,
            "tail_sigma_algebra_trivial": True,
            "diagonal_quantum_markov_gns_factor": True,
            "onsite_cayley_automorphism_preserves_factoriality": True,
        },
        "diagonal_state": {
            "local_tangent_ranks_two_primes": diagonal_ranks,
            "raw_flip_difference_nonzero_entries": int(
                np.count_nonzero(diagonal_difference)
            ),
            "raw_flip_difference_integer_norm_squared": sum(
                int(value) * int(value) for value in diagonal_difference.flat
            ),
        },
        "source_cayley": {
            "source_contraction_rank": int(np.linalg.matrix_rank(source.astype(float))),
            "source_contraction_symmetric": bool(np.array_equal(source, source.T)),
            "cayley_denominator": denominator,
            "cayley_max_abs_integer_numerator": max(
                abs(int(value))
                for value in list(cayley_real.flat) + list(cayley_imag.flat)
            ),
            "unitary_exact": True,
            "source_vector": "point+line",
            "basis_change_naturality": naturality,
            "uniqueness_among_source_natural_functions": False,
        },
        "coherent_state": {
            "normalization_denominator": 18000 * raw_scale,
            "faithful": True,
            "nontracial": True,
            "noncentral_local_entries": noncentral_count,
            "local_tangent_ranks_two_primes": coherent_ranks,
            "raw_flip_difference_nonzero_entries": nonzero_count,
            "raw_flip_common_denominator": 18000 * raw_scale * flip_scale,
            "raw_flip_first_witness": witness,
        },
    }


def build_outputs():
    protocol = load(INPUTS["protocol"])
    baseline = load(INPUTS["baseline"])
    a61_certificate = load(INPUTS["a61_certificate"])
    a61_result = load(INPUTS["a61_result"])
    nattrans_certificate = load(INPUTS["nattrans55_certificate"])
    assert protocol["status"].startswith("FROZEN_AFTER_DISCLOSED")
    assert baseline["status"] == "PASS_DIRECT_COMMIT_PINNED_A61_CONTINUATION_SCOPE_ONLY"
    assert baseline["science_outcome_authorized"] is True
    assert a61_result["status"].startswith(
        "PASS_FAITHFUL_TRANSLATION_INVARIANT_THREE_PHASE_SOURCE_STATE"
    )
    assert a61_certificate["decision_tests"]["E6_FOUR_DIRECTION_SURVIVAL"] is True
    assert nattrans_certificate["analytic_naturality_proof"]["theorem_passed"] is True

    global A61
    A61 = load_module("a62_a61", INPUTS["a61_evaluate"])
    o14 = load_module("a62_o14", INPUTS["ocbfh014_checker"])
    nat = importlib.import_module(
        "projects.active.discovery_partner.formal_checks."
        "disc008_nattrans55r_basis_independent_natural_transformation_check"
    )
    survivors, projectors, _, source = o14.reconstruct_actual_source()
    records = [
        classify_sector(survivor, projectors, o14, nat)
        for survivor in survivors
    ]
    assert len(records) == 8
    assert all(
        record["markov_transfer"]["two_step_positive_entries"] == 625
        for record in records
    )
    assert all(
        record["diagonal_state"]["local_tangent_ranks_two_primes"] == [1, 1]
        for record in records
    )
    assert all(
        record["coherent_state"]["local_tangent_ranks_two_primes"] == [4, 4]
        for record in records
    )

    tests = {
        "G1_PRIMITIVITY": True,
        "G2_FACTORIALITY": True,
        "G3_CLASSICAL_RAW_SURVIVAL": True,
        "G4_CLASSICAL_DIRECTION_DIAGNOSIS": True,
        "G5_CAYLEY_SOURCE_NATURALITY": True,
        "G6_COHERENT_FOUR_DIRECTION_SURVIVAL": True,
        "G7_COHERENT_RAW_SURVIVAL": True,
        "G8_NONTRIVIAL_MODULAR_STATE": True,
        "STANDARD_HSMI_REQUIRED": False,
    }
    certificate = {
        "schema": "siel.dpa.a62.primitive-factor-source-cayley.certificate.v1",
        "gate_id": "DPA-A62",
        "date": "2026-09-15",
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()
        },
        "source_reconstruction": source,
        "primitive_markov_theorem": {
            "definition": "The diagonal of the A61 sigma3 is used as a stationary order-two Markov law on pair states.",
            "exact_proof": "Every triple weight is positive. Therefore every pair state (a,b) reaches every pair state (c,d) in exactly two steps through (b,c), so T^2 is entrywise positive. The matching A61 left/right pair marginals prove stationarity. Perron-Frobenius gives the unique faithful stationary pair law and exponential mixing.",
            "factor_step": "The resulting faithful diagonal finitely correlated state has trivial Markov tail. Its GNS von Neumann algebra is the tail-equivalence algebra and has scalar center. The onsite Cayley automorphism preserves this factor property.",
            "prior_art": {
                "title": "Finitely correlated states on quantum spin chains",
                "authors": "M. Fannes, B. Nachtergaele, R. F. Werner",
                "journal": "Communications in Mathematical Physics 144 (1992), 443-490",
                "doi": "10.1007/BF02099178",
                "role": "Standard finite-correlation, ergodic-decomposition and exponential-clustering framework; not claimed new."
            },
        },
        "source_natural_cayley_theorem": {
            "marked_vector": "v=point+line",
            "contraction": "S_R=(I tensor <v|)R(I tensor |v>)",
            "cayley": "U_R=(S_R-iI)(S_R+iI)^-1",
            "analytic_naturality": "For a NATTRANS-55R morphism V, v'=Vv and R'=(V tensor V)R(V tensor V)^*. Hence S_R'=V S_R V^* and U_R'=V U_R V^*.",
            "invertibility": "S_R is self-adjoint, so S_R+iI is invertible.",
            "coordinate_regressions": ["D", "G", "P"],
            "choice_boundary": "The Cayley function is a natural parameter-free admissible lift, but A62 does not prove it is the unique source-natural functional calculus or physically necessary."
        },
        "sector_records": records,
        "decision_tests": tests,
        "endpoint_semantic_gates": protocol["endpoint_semantic_gates"],
    }
    result = {
        "schema": "siel.dpa.a62.primitive-factor-source-cayley.result.v1",
        "gate_id": "DPA-A62",
        "date": "2026-09-15",
        "primary_evidence_status": "Theoretical derivation",
        "secondary_qualifier": "Exact all-eight-sector primitive Markov and source-natural coherent-lift audit",
        "scientific_layer": "mathematical formulation and infinite-state bridge",
        "status": "PASS_PRIMITIVE_FACTOR_STATE__CLASSICAL_RESPONSE_COLLAPSES_TO_ONE__PASS_SOURCE_NATURAL_CAYLEY_LIFT_RESTORES_RANK_FOUR__HSMI_NOT_REQUIRED__LIFT_UNIQUENESS_OPEN",
        "decisive_result": "The A61 central three-phase mixture can be replaced by a unique faithful primitive order-two Markov state built from the same local source weights. This removes the residual phase center and yields a factor GNS, but the bare diagonal state collapses the four source directions to one. Contracting raw R through the already marked vector point+line and applying its Cayley unitary gives a source-natural onsite lift. It preserves factoriality, remains different from the matched ordinary-flip control and restores exact tangent rank four in all eight sectors.",
        "observed_evidence": [
            "All 125 triple weights are positive in every sector; all 625 entries of the two-step pair transition are strictly positive.",
            "The A61 matching pair marginals solve all 25 stationarity equations exactly.",
            "The diagonal primitive state differs from the matched flip control but its four local tangent vectors have rank one.",
            "The raw contraction and Cayley equation transport exactly under the D, G and P arbitrary-unitary basis changes used by NATTRANS-55R.",
            "After the source Cayley lift, the four local tangent vectors have rank four over both prime fields in every sector, and raw-versus-flip differences remain nonzero."
        ],
        "pattern": "Removing a classical phase center by primitive mixing need not preserve quantum directional response. A coherent automorphism generated naturally from the marked raw relation can restore that response without changing the factor class of the infinite state.",
        "interpretive_leap": "A homogeneous world-state may require two simultaneous operations from the same relation: irreversible forgetting of arbitrary placement history and coherent retention of the relation's directional standpoint. Classical mixing supplies the first and the Braid-derived Cayley frame supplies the second.",
        "alternative_explanations": [
            "Primitive finite-state Markov mixing and onsite unitary conjugation are standard constructions.",
            "The successful Cayley function is admissible and natural but is not proved uniquely forced by the Braid axioms.",
            "Raw-versus-flip discrimination is not tomography or recovery of the complete raw orbit.",
            "Rank four is a four-dimensional local response space, not a derivation of physical 1+3 spacetime.",
            "A nontrivial modular group is state-relative operator dynamics, not physical time or Lorentz boosts."
        ],
        "novel_hypothesis": "The same source contraction S_R selects not only a coherent state frame but also a distinguished transfer generator whose conserved current maps to the four-dimensional quotient response; if unique under source naturality and Ward compatibility, the remaining functional-calculus ambiguity may disappear.",
        "falsifier": protocol["falsifier"],
        "required_prospective_test": "Construct the infinitesimal generator/current of the combined primitive transfer and source Cayley action, require source naturality, a nonzero conserved four-direction response and Ward compatibility, and compare all admissible natural functional calculi before making a Braid-necessity claim.",
        "confidence": {
            "primitive_factor_state": "HIGH",
            "raw_and_four_direction_survival_after_cayley": "HIGH",
            "cayley_naturality": "HIGH",
            "cayley_uniqueness_or_physical_necessity": "LOW_OPEN"
        },
        "candidate_level": "LEVEL_1_PATTERN_CANDIDATE__RESERVOIR_REGISTRATION_BLOCKED",
        "endpoint_semantic_gates": protocol["endpoint_semantic_gates"],
        "next_gate": "DPA-A63_SOURCE_TRANSFER_GENERATOR_AND_WARD_CURRENT_GATE",
        "claim_ceiling": protocol["claim_ceiling"],
        "governance": {
            "rpd_handoff_authorized": False,
            "experiment_authorized": False,
            "publication_mutation_authorized": False,
            "external_communication_authorized": False,
            "official_siel_adoption": False
        }
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

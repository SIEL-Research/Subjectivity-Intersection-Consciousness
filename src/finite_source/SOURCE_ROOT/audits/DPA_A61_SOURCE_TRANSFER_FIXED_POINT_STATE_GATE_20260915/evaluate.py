#!/usr/bin/env python3
"""DPA-A61: source three-phase transfer fixed-point state."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any
import hashlib
import importlib.util
import json
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = {
    "protocol": HERE / "FROZEN_PROTOCOL.json",
    "baseline": HERE / "BASELINE_GATE.json",
    "a55_certificate": ROOT / "audits/DPA_A55_MAXIMAL_INTERSECTION_OBSERVABILITY_SELECTOR_GATE_20260915/CERTIFICATE.json",
    "a56_evaluate": ROOT / "audits/DPA_A56_FROBENIUS_LIE_CLOSURE_GATE_20260915/evaluate.py",
    "a57_certificate": ROOT / "audits/DPA_A57_CANONICAL_FOUR_ABELIANIZATION_GATE_20260915/CERTIFICATE.json",
    "a60_certificate": ROOT / "audits/DPA_A60_ACTIVE_TRANSLATED_FACTOR_NET_AND_GNS_SCALE_GATE_20260915/CERTIFICATE.json",
    "a60_result": ROOT / "audits/DPA_A60_ACTIVE_TRANSLATED_FACTOR_NET_AND_GNS_SCALE_GATE_20260915/RESULT.json",
    "ocbfh014_checker": ROOT / "projects/active/discovery_partner/formal_checks/ocbfh014_source_native_refinement_naturality_check.py",
    "kraustower127r": ROOT / "projects/active/discovery_partner/formal_checks/disc008_kraustower127r_active_mark_cp_semigroup_v1.json",
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


def selected_state_numerator(relation: np.ndarray, cap: np.ndarray, o14) -> np.ndarray:
    identity5 = np.eye(5, dtype=np.int64)
    identity125 = np.eye(125, dtype=np.int64)
    _, _, r1, r2 = o14.source_g1(relation)
    pointed = np.kron(np.eye(25, dtype=np.int64) + cap, identity5)
    braid_group = [identity125, r1, r2, r1 @ r2, r2 @ r1, r1 @ r2 @ r1]
    neutral = sum(
        (element @ pointed @ element.T for element in braid_group),
        start=np.zeros_like(pointed),
    )
    result = 2 * pointed + neutral
    assert int(np.trace(result)) == 1200
    return result


def marginals(rho: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    tensor = rho.reshape(5, 5, 5, 5, 5, 5)
    one = []
    for keep in range(3):
        removed = [index for index in range(3) if index != keep]
        current = tensor
        active = 3
        for index in sorted(removed, reverse=True):
            current = np.trace(current, axis1=index, axis2=index + active)
            active -= 1
        one.append(current.reshape(5, 5))
    left = np.trace(tensor, axis1=2, axis2=5).reshape(25, 25)
    right = np.trace(tensor, axis1=0, axis2=3).reshape(25, 25)
    return one, left, right


def stationary_local_numerator(rho: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    one, left, right = marginals(rho)
    identity5 = np.eye(5, dtype=np.int64)
    assert all(np.array_equal(item, 240 * identity5) for item in one)
    # The three phase densities share denominator 6000:
    # rho/1200, (rho23/1200) tensor I/5, I/5 tensor (rho12/1200).
    phase_numerators = [5 * rho, np.kron(right, identity5), np.kron(identity5, left)]
    sigma = sum(phase_numerators, start=np.zeros_like(rho))
    assert int(np.trace(sigma)) == 18000
    return sigma, phase_numerators


def ordinary_flip() -> np.ndarray:
    flip = np.zeros((25, 25), dtype=np.int64)
    for left in range(5):
        for right in range(5):
            flip[right * 5 + left, left * 5 + right] = 1
    return flip


def gram_determinant_three(matrices: list[np.ndarray]) -> int:
    gram = [[sum(int(x) * int(y) for x, y in zip(left.reshape(-1), right.reshape(-1))) for right in matrices] for left in matrices]
    a, b, c = gram[0]
    d, e, f = gram[1]
    g, h, i = gram[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def classify_sector(survivor: dict[str, Any], projectors, o14, a56) -> dict[str, Any]:
    relation = survivor["R"].astype(np.int64)
    cap = survivor["P"].astype(np.int64)
    rho = selected_state_numerator(relation, cap, o14)
    _, original_left, original_right = marginals(rho)
    original_difference = original_left - original_right
    assert sum(int(value) * int(value) for value in original_difference.reshape(-1)) == 8352

    sigma, phases = stationary_local_numerator(rho)
    _, sigma_left, sigma_right = marginals(sigma)
    assert np.array_equal(sigma_left, sigma_right)
    phase_gram_determinant = gram_determinant_three(phases)
    assert phase_gram_determinant == 1212501594112000

    point = cap
    assert np.array_equal(point, point.T)
    assert np.array_equal(point @ point, 5 * point)
    # Hence point is positive semidefinite and pointed=(I+point) tensor I >= I.
    # rho >= 2 I and sigma >= 5 rho >= 10 I, so all are exactly faithful.

    identity5 = np.eye(5, dtype=np.int64)
    clock, _, _, _ = o14.source_g1(relation)
    local_generators = [clock, *[np.kron(numerator, identity5) for numerator, _ in projectors]]
    tangents = [generator @ sigma - sigma @ generator for generator in local_generators]
    tangent_rank = a56.exact_rank(tangents)
    assert tangent_rank == 4

    flip_rho = selected_state_numerator(ordinary_flip(), cap, o14)
    flip_sigma, _ = stationary_local_numerator(flip_rho)
    raw_flip_difference = sigma - flip_sigma
    raw_flip_norm_squared = sum(int(value) * int(value) for value in raw_flip_difference.reshape(-1))
    nonzero = np.argwhere(raw_flip_difference != 0)
    assert raw_flip_norm_squared > 0
    assert tuple(int(value) for value in nonzero[0]) == (0, 12)
    assert int(raw_flip_difference[0, 12]) == -2

    scalar = int(np.trace(sigma)) // 125
    nonscalar_difference = sigma - scalar * np.eye(125, dtype=np.int64)
    assert np.any(nonscalar_difference)
    return {
        "mask": survivor["mask"],
        "original_overlap_difference_frobenius_squared": 8352,
        "stationary_density_numerator_trace": 18000,
        "stationary_left_right_two_site_equal": True,
        "stationary_left_right_difference_frobenius_squared": 0,
        "three_phase_gram_determinant": phase_gram_determinant,
        "three_phase_orbit_exactly_three_distinct_local_states": True,
        "faithfulness_lower_bound_for_sigma_numerator": 10,
        "local_observer_tangent_rank": tangent_rank,
        "raw_flip_difference_nonzero_entries": int(len(nonzero)),
        "raw_flip_difference_frobenius_squared": raw_flip_norm_squared,
        "raw_flip_first_witness": {"row": 0, "column": 12, "stationary_numerator_value": -2},
        "stationary_density_noncentral": True,
        "stationary_density_noncentral_entries": int(np.count_nonzero(nonscalar_difference)),
    }


def build_outputs():
    protocol = load(INPUTS["protocol"])
    baseline = load(INPUTS["baseline"])
    a55 = load(INPUTS["a55_certificate"])
    a57 = load(INPUTS["a57_certificate"])
    a60 = load(INPUTS["a60_certificate"])
    a60_result = load(INPUTS["a60_result"])
    kraus127 = load(INPUTS["kraustower127r"])
    assert protocol["status"].startswith("FROZEN_AFTER_EXPLORATORY_ALL_SECTOR")
    assert baseline["status"] == "PASS_DIRECT_COMMIT_PINNED_A60_CONTINUATION_SCOPE_ONLY"
    assert baseline["science_outcome_authorized"] is True
    assert a55["selector_theorem"]["unique_global_maximizer"] == "3/4"
    assert a57["canonical_exact_sequence"]["a_dimension"] == 4
    assert a60_result["status"].startswith("PASS_ACTIVE_OVERLAPPING_LOCAL_FACTOR_NET")
    assert a60["state_extension_no_go"]["left_right_two_site_equal"] is False

    o14 = load_module("a61_ocbfh014", INPUTS["ocbfh014_checker"])
    a56 = load_module("a61_a56", INPUTS["a56_evaluate"])
    survivors, projectors, _, source = o14.reconstruct_actual_source()
    records = [classify_sector(survivor, projectors, o14, a56) for survivor in survivors]
    stable_keys = [
        "original_overlap_difference_frobenius_squared",
        "stationary_density_numerator_trace",
        "stationary_left_right_two_site_equal",
        "stationary_left_right_difference_frobenius_squared",
        "three_phase_gram_determinant",
        "three_phase_orbit_exactly_three_distinct_local_states",
        "faithfulness_lower_bound_for_sigma_numerator",
        "local_observer_tangent_rank",
        "stationary_density_noncentral",
        "stationary_density_noncentral_entries",
    ]
    assert all({key: row[key] for key in stable_keys} == {key: records[0][key] for key in stable_keys} for row in records)

    tests = {
        "E1_A60_REPLAY": True,
        "E2_GLOBAL_STATE_EXISTENCE": True,
        "E3_TRANSLATION_STATIONARITY": True,
        "E4_FAITHFULNESS": True,
        "E5_RAW_MARK_SURVIVAL": True,
        "E6_FOUR_DIRECTION_SURVIVAL": True,
        "E7_NONTRIVIAL_MODULAR_DYNAMICS": True,
        "E8_STANDARD_HSMI": False,
    }
    certificate = {
        "schema": "siel.dpa.a61.source-transfer-fixed-point-state.certificate.v1",
        "gate_id": "DPA-A61",
        "date": "2026-09-15",
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()},
        "source_reconstruction": source,
        "stationary_state_theorem": {
            "local_block_arity": 3,
            "phase_group": "Z3 generated by the one-site shift",
            "phase_states": "omega_s is the product of rho3 on disjoint three-site blocks with phase s",
            "stationary_state": "omega_star=(omega_0+omega_1+omega_2)/3",
            "exact_stationarity_proof": "The one-site shift cyclically permutes omega_0, omega_1, omega_2, hence fixes their uniform average.",
            "unique_on_phase_simplex": "A probability vector fixed by the three-cycle obeys p0=p1=p2, so the normalized fixed vector is unique.",
            "not_unique_among_all_chain_states": True,
            "finite_local_consistency": "Each omega_s is an explicit block-product state, so all finite restrictions are projectively consistent. Their average is therefore an explicit infinite quasi-local state.",
            "faithfulness": "P is symmetric and P^2=5P, so pointed=(I+P) tensor I is at least I. Thus rho numerator is at least 2I and every block marginal is positive definite. Tensor products and their convex average remain faithful on every finite local algebra.",
            "three_site_density": "sigma3=(rho3 + rho23 tensor I5/5 + I5/5 tensor rho12)/3",
            "three_site_density_denominator": 18000,
            "three_site_overlap_consistency": "PASS_EXACT",
        },
        "modular_typing": {
            "each_phase_component_nontracial": True,
            "component_modular_group": "blockwise Ad(rho3^(it)) in the product-state representation",
            "component_modular_group_nontrivial": True,
            "reason": "rho3 is positive definite and noncentral, so its modular automorphism is not the identity.",
            "uniform_phase_average_factoriality": "NOT_ESTABLISHED_AND_EXPECTED_TO_HAVE_A_PHASE_DECOMPOSITION",
            "positive_HSMI_translation": False,
            "physical_time_scale": False,
        },
        "prior_channel_boundary": {
            "127R_channel_status": kraus127["status"],
            "relation": "127R supplies a different active-mark random-unitary CP semigroup. A61 uses the tensor shift orbit of the A55 block state and does not identify the two channels.",
        },
        "sector_records": records,
        "decision_tests": tests,
    }
    result = {
        "schema": "siel.dpa.a61.source-transfer-fixed-point-state.result.v1",
        "gate_id": "DPA-A61",
        "date": "2026-09-15",
        "primary_evidence_status": "Theoretical derivation",
        "secondary_qualifier": "Exact all-eight-sector three-phase stationary-state construction",
        "scientific_layer": "mathematical formulation and infinite-state bridge",
        "status": "PASS_FAITHFUL_TRANSLATION_INVARIANT_THREE_PHASE_SOURCE_STATE__PASS_RAW_FLIP_AND_FOUR_DIRECTION_SURVIVAL__NONTRIVIAL_COMPONENT_MODULAR_FLOW__STANDARD_HSMI_OPEN",
        "decisive_result": "The A60 local-state obstruction is bypassed without altering rho3: form product states on source-sized three-site blocks and uniformly average their three translated phases. The resulting infinite quasi-local state is exactly one-site translation invariant and faithful. Its three-site density has matching left/right two-site marginals, differs exactly from the ordinary-flip control and retains rank-four G1/P_chi response in every sector. Each phase component has nontrivial modular dynamics, but the uniform phase mixture is not yet a canonical factor HSMI.",
        "observed_evidence": [
            "The new stationary three-site density has equal left and right two-site marginals exactly in all eight sectors, while the original rho3 difference remains 8352.",
            "Its numerator is bounded below by 10I, so no support reduction is needed at finite stages.",
            "The three phase-local densities are linearly independent; their exact Gram determinant is 1212501594112000.",
            "The stationary local density differs from the matched ordinary-flip control in every sector; the first exact witness is entry (0,12)=-2 before normalization.",
            "The four local state tangents generated by G1 and the three P_chi have exact rank four in every sector."
        ],
        "pattern": "A locally selected oriented state need not tile by overlapping copies, but its full translation orbit can define a faithful homogeneous global state while retaining the source response.",
        "interpretive_leap": "A physical state may be selected by averaging all observer-origin placements of a source-local relation, so homogeneity appears only after the standpoints are combined rather than inside each local standpoint.",
        "alternative_explanations": [
            "This is the standard translation average of a periodic block-product state.",
            "The three-phase mixture may retain a classical central phase label and therefore fail to be a factor state.",
            "The block size three is source-local here, but another source-derived blocking or a primitive transfer channel may yield a different state.",
            "Raw-versus-flip sensitivity does not prove injective recovery of the raw braid."
        ],
        "novel_hypothesis": "Replace the residual three-phase mixture by a source-derived primitive CP transfer operator whose unique faithful fixed state has the same local raw mark and four-direction quotient response; its factorial GNS may then support a nontrivial half-sided modular inclusion.",
        "falsifier": "Reject this route if every primitive CP perturbation constructed from the frozen raw/cap/local-generator library either loses the raw-versus-flip discriminator, collapses the rank-four response, or has only a tracial fixed state.",
        "required_prospective_test": "Audit a frozen finite library of primitive source transfer channels that couple the three phase sectors; require unique faithful stationarity, exact raw sensitivity, rank-four response, factoriality or a proved center-free GNS, and only then test the HSMI scaling relation.",
        "confidence": {"stationary_state_construction": "HIGH", "raw_and_four_direction_survival": "HIGH", "factor_HSMI_route": "LOW_OPEN"},
        "candidate_level": "LEVEL_1_PATTERN_CANDIDATE__RESERVOIR_REGISTRATION_BLOCKED",
        "next_gate": "DPA-A62_PRIMITIVE_SOURCE_TRANSFER_AND_FACTOR_GNS_GATE",
        "claim_ceiling": "A61 constructs a faithful one-site translation-invariant state on the A60 quasi-local tensor-chain algebra by uniformly averaging the three translations of the A55 block-product state. It proves local overlap consistency, matched raw-versus-flip sensitivity and rank-four local response. It does not prove uniqueness among all states, a primitive source channel, factoriality of the averaged GNS, a positive HSMI translation, physical time, Lorentz dynamics, AQFT, QFT, spacetime, gravity, GR, natural evidence, ontology, subjectivity, consciousness or unification."
    }
    return certificate, result


def main():
    certificate, result = build_outputs()
    (HERE / "CERTIFICATE.json").write_text(json.dumps(certificate, ensure_ascii=False, indent=2) + "\n")
    (HERE / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"])
    print(json.dumps(certificate["decision_tests"], indent=2))


if __name__ == "__main__":
    main()

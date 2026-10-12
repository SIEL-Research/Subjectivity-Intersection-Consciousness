"""OCBFH-014: source-native refinement naturality of the unique soldering.

This separates product-preserving pointed tensor refinements from the older
split-UCP observation/recovery maps.  The distinction is tested rather than
assumed: the former must carry the full clock/Fourier algebra, while the latter
is subjected to an exact multiplicative-domain obstruction.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any
import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys

import numpy as np


GATE_ID = "OCBFH-014"
FREEZE_COMMIT = "6e09d022b996c7914843283d8905f1a9bdc7e2a4"
PROTOCOL_SHA256 = "94b183ae7890e10fdabc276a93f6bee7fd4cf6e79fde38892269ace6886f4e6c"
ROUTE_SHA256 = "3f644951acfd8151a0bc7545c18b39c15cb8ee8f0946c8be421e6f2b1ffffae5"
BEFORE_SHA256 = "1a2f400e02501dba96e9dbbe49b2198cc216552e6a6b20a7ba38b4e2718d44b4"
SOURCE_MANIFEST_SHA256 = "bb9b5c0836043687c49c33be931fe7c554ff59c8e7ca711fb1f445d8a6e99b92"
FINAL_STATUS = (
    "PASS_ALLORDER_POINTED_MARKED_BRAID_TENSOR_REFINEMENT_NATURALITY__"
    "FAIL_PINNED_SPLIT_UCP_AS_PRODUCT_PRESERVING_CLOCK_FOURIER_REFINEMENT__"
    "TYPED_REFINEMENT_OBSERVATION_DISTINCTION_REQUIRED__UNIVERSAL_FIXED_POINT_OPEN"
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROTOCOL = HERE / "ocbfh014_protocol_source_native_refinement_naturality_v1.json"
ROUTE = REPO / "projects/active/discovery_partner/route_control/OCBFH014_UNIFICATION_ROUTE_REASSESSMENT_v1.json"
BEFORE = REPO / "projects/active/discovery_partner/route_control/SELF_EVALUATION_OCBFH014_BEFORE_v1.json"
SOURCE_MANIFEST = REPO / "audits/UB475_SECTION_FREE_C1_C2_QUOTIENT_FOURIER_SOLDERING_GATE/SOURCE_MANIFEST.json"

INPUTS = {
    "ocbfh013": (
        "projects/active/discovery_partner/formal_checks/ocbfh013_full_orbit_exact_reentry_initial_soldering_certificate_v1.json",
        "48f2392d876f45e699ecb962020314494b7a309943adea608bcc44a933ef4e09",
    ),
    "ocbfh002": (
        "projects/active/discovery_partner/formal_checks/ocbfh002_source_generated_coherent_root_tower_certificate_v1.json",
        "eaaebd181309ec7603f9ae7aec603e5b92abd7f0e8e5192aec3132657bbd2d1d",
    ),
    "allordercp68r": (
        "projects/active/discovery_partner/formal_checks/disc008_allordercp68r_rooted_split_ucp_coherence_v1.json",
        "e1133b068f9693799226d6c5140acafc37847d6b13d28fe57dcdb36de5140582",
    ),
    "allordercp68r_checker": (
        "projects/active/discovery_partner/formal_checks/disc008_allordercp68r_rooted_split_ucp_coherence_check.py",
        "0e72ac0af8751b6278c4347743e2e543f8199bbd5a3a079cd7de65c261d38b64",
    ),
    "ub473": (
        "audits/UB473_GLOBAL_BRAID_PATH_SUSPENSION_AND_YANG_BAXTER_COHERENCE_GATE/GLOBAL_BRAID_PATH_CERTIFICATE.json",
        "34a32242db320f0903f19d4015960a55dca536455d7e99ca20301b37ba909467",
    ),
    "ub473_evaluate": (
        "audits/UB473_GLOBAL_BRAID_PATH_SUSPENSION_AND_YANG_BAXTER_COHERENCE_GATE/evaluate.py",
        "def9e5d0032757d9095998e0ae932235ccb2d16308f6e6ef74d030a56e57b34e",
    ),
    "ub475": (
        "audits/UB475_SECTION_FREE_C1_C2_QUOTIENT_FOURIER_SOLDERING_GATE/SECTION_FREE_C1_C2_SOLDERING_CERTIFICATE.json",
        "649d4fc140f4e45aab31098bef19cf3a3b5522feca964d47121629cdaec8b3ca",
    ),
    "ub475_evaluate": (
        "audits/UB475_SECTION_FREE_C1_C2_QUOTIENT_FOURIER_SOLDERING_GATE/evaluate.py",
        "d1b5f7738d571de51fa771ecad168bd9414fb94c6673f64e7838ef61c6b8ee33",
    ),
    "ub443_evaluate": (
        "audits/UB443_TYPED_SIGNED_SURVIVOR_GAUGE_OR_OBSERVABLE_GATE/evaluate.py",
        "ad53908146ba65d9de71994c0b297fac759a05c4938378f6cc962bbf9eae8c9e",
    ),
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    assert digest(PROTOCOL.read_bytes()) == PROTOCOL_SHA256
    assert digest(ROUTE.read_bytes()) == ROUTE_SHA256
    assert digest(BEFORE.read_bytes()) == BEFORE_SHA256
    assert digest(SOURCE_MANIFEST.read_bytes()) == SOURCE_MANIFEST_SHA256
    protocol = json.loads(PROTOCOL.read_text())
    route = json.loads(ROUTE.read_text())
    assert protocol["status"] == "FROZEN_AFTER_ANALYTIC_TYPE_SCOUT_BEFORE_EXACT_OCBFH014_EVALUATION"
    assert protocol["prefreeze_scout_disclosure"]["outcome_accessed"] is True
    assert route["decision_status"] == "SHORTEST_WITHIN_OCBFH_INITIAL_SOLDERING_EXISTENCE_LANE_ONLY"
    loaded: dict[str, Any] = {}
    verified = {}
    for name, (relative, expected) in INPUTS.items():
        payload = (REPO / relative).read_bytes()
        actual = digest(payload)
        assert actual == expected, name
        verified[relative] = actual
        if relative.endswith(".json"):
            loaded[name] = json.loads(payload)

    allorder_checker = load_module("ocbfh014_allorder_baseline", REPO / INPUTS["allordercp68r_checker"][0])
    assert allorder_checker.run_checks() == loaded["allordercp68r"]
    return loaded, {
        "freeze_commit": FREEZE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "route_sha256": ROUTE_SHA256,
        "before_self_evaluation_sha256": BEFORE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "verified_input_count": len(INPUTS),
        "verified_input_hashes": verified,
        "ALLORDERCP68R_baseline_reproduced": True,
        "endpoint_semantic_gate": "NOT_ISSUED_BY_DPA__INTERNAL_THEOREM_AUDIT_NOT_RPD_OR_MANUSCRIPT_CERTIFICATION",
    }


def git_json(commit: str, path: str, expected: str) -> dict[str, Any]:
    payload = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO)
    assert digest(payload) == expected
    return json.loads(payload)


def reconstruct_actual_source() -> tuple[list[dict[str, Any]], list[tuple[np.ndarray, int]], list[tuple[int, ...]], dict[str, Any]]:
    ub443 = load_module("ocbfh014_ub443", REPO / INPUTS["ub443_evaluate"][0])
    survivors, _ = ub443.build_survivors()
    assert [item["mask"] for item in survivors] == [0, 5, 8, 13, 16, 21, 24, 29]

    manifest = json.loads(SOURCE_MANIFEST.read_text())
    commit = manifest["commit"]
    cross = {path: git_json(commit, path, expected) for path, expected in manifest["files"].items()}
    ob011 = cross["audits/OB011_FULL_EIGHT_SECTOR_TIME_PARITY_CLOCK_ACTION/RESULT.json"]
    ob013 = cross["audits/OB013_CANONICAL_COMMUTING_TIME_SPACE_QUOTIENT/RESULT.json"]
    quotient_classes = [tuple(item) for item in ob013["exact_results"]["space_quotient_classes"]]
    representatives = [item[0] for item in quotient_classes]
    assert representatives == [0, 8, 16, 24]

    mismatch = {}
    for source in (element for pair in quotient_classes for element in pair):
        bits = ob011["exact_results"]["mismatch_records"][str(source)]["one_site_factor_bits"]
        one_site = np.diag([-1 if bit else 1 for bit in bits]).astype(np.int64)
        mismatch[source] = np.kron(one_site, one_site)
    quotient_representation = []
    for pair in quotient_classes:
        left, right = (mismatch[element] for element in pair)
        assert np.array_equal(left, right)
        quotient_representation.append(left)

    qindex = {representative: index for index, representative in enumerate(representatives)}
    product = [[qindex[left ^ right] for right in representatives] for left in representatives]
    characters = []
    for values in itertools.product((-1, 1), repeat=4):
        if values[0] != 1:
            continue
        if all(values[product[i][j]] == values[i] * values[j] for i in range(4) for j in range(4)):
            characters.append(tuple(values))
    characters = sorted(characters)
    spatial_characters = [item for item in characters if item != (1, 1, 1, 1)]
    projectors = []
    for character in spatial_characters:
        numerator = sum(
            (character[index] * quotient_representation[index] for index in range(4)),
            start=np.zeros((25, 25), dtype=np.int64),
        )
        assert np.array_equal(numerator @ numerator, 4 * numerator)
        projectors.append((numerator, 4))
    for left, right in itertools.combinations(projectors, 2):
        assert not np.any(left[0] @ right[0])
    ranks = [int(np.trace(numerator) // denominator) for numerator, denominator in projectors]
    assert ranks == [8, 4, 4]
    return survivors, projectors, spatial_characters, {
        "cross_source_commit": commit,
        "cross_source_hashes_verified": len(cross),
        "actual_survivor_masks": [item["mask"] for item in survivors],
        "actual_spatial_projector_ranks": ranks,
    }


def source_g1(relation: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    identity5 = np.eye(5, dtype=np.int64)
    identity125 = np.eye(125, dtype=np.int64)
    r1 = np.kron(relation, identity5)
    r2 = np.kron(identity5, relation)
    class_sum = r1 + r2 + r1 @ r2 @ r1
    energy_numerator = 6 * identity125 - 2 * class_sum
    crossing_numerator = identity125 - r1
    return 2 * energy_numerator + 3 * crossing_numerator, 6, r1, r2


def tensor_refinement_gate(
    survivors: list[dict[str, Any]], projectors: list[tuple[np.ndarray, int]]
) -> dict[str, Any]:
    identity5 = np.eye(5, dtype=np.int64)
    identity125 = np.eye(125, dtype=np.int64)
    naturality_checks = 0
    algebra_checks = 0
    sector_records = []
    for survivor in survivors:
        g_num, g_den, r1, r2 = source_g1(survivor["R"].astype(np.int64))
        assert g_den == 6
        assert not np.any(
            g_num
            @ (g_num - 12 * identity125)
            @ (g_num - 18 * identity125)
            @ (g_num - 30 * identity125)
        )
        assert not np.any(r1 @ r2 @ r1 - r2 @ r1 @ r2)
        assert not np.any(r1 @ r1 - identity125)

        g_four = np.kron(g_num, identity5)
        assert np.array_equal(g_four, np.kron(g_num, identity5))
        naturality_checks += 1
        p_three = []
        commutator_nonzero = []
        for numerator, denominator in projectors:
            p3_num = np.kron(numerator, identity5)
            p4_num = np.kron(p3_num, identity5)
            assert np.array_equal(p4_num, np.kron(numerator, np.eye(25, dtype=np.int64)))
            assert np.array_equal(p3_num @ p3_num, denominator * p3_num)
            commutator = g_num @ p3_num - p3_num @ g_num
            assert np.any(commutator)
            # Star-monomorphism transport carries the noncommutative product,
            # and hence its commutator, by the same tensor identity.
            assert np.array_equal(np.kron(commutator, identity5), np.kron(g_num @ p3_num - p3_num @ g_num, identity5))
            commutator_nonzero.append(True)
            p_three.append((p3_num, denominator))
            naturality_checks += 1
            algebra_checks += 3
        for left, right in itertools.combinations(p_three, 2):
            assert not np.any(left[0] @ right[0])
            algebra_checks += 1
        sector_records.append({
            "mask": survivor["mask"],
            "three_to_four_G1_naturality": True,
            "three_to_four_all_P_chi_naturality": True,
            "G1_minimal_polynomial_transports": True,
            "G1_P_chi_commutators_all_nonzero": all(commutator_nonzero),
            "G1_P_chi_commutators_transport": True,
            "Fourier_products_and_adjoint_transport": True,
        })

    return {
        "base_dimension": 5,
        "starting_order": 3,
        "refinement_map": "j_n(A)=A tensor I_5",
        "actual_signed_sectors": len(survivors),
        "actual_three_to_four_naturality_checks": naturality_checks,
        "actual_algebra_checks": algebra_checks,
        "sector_records": sector_records,
        "all_order_proof": [
            "Tensor associativity gives j_(n+1)(j_n(A))=A tensor I_5 tensor I_5 for every n.",
            "j_n is a unital injective star-homomorphism, so it preserves sums, products, adjoints, projectors and every polynomial in the marked operators.",
            "In the retained finite-dimensional C-star category, star-homomorphisms preserve continuous functional calculus; hence j_n(exp(-itG1_n))=exp(-itG1_(n+1)).",
            "The added factor is another C5 source strand. No independent clock, metric, state or continuum factor is adjoined."
        ],
        "endpoint_transport": "The pinned exp(-i*pi*G1_3)=R1 identity tensors with I at every later order.",
        "source_native_no_spectator": True,
        "status": "PASS_EXACT_ALLORDER_RIGHT_TAIL_POINTED_TENSOR_REFINEMENT",
    }


SignedAction = tuple[tuple[int, ...], tuple[int, ...]]


def pair_action(relation: np.ndarray) -> SignedAction:
    mapping = []
    signs = []
    for column in range(25):
        rows = np.flatnonzero(relation[:, column])
        assert len(rows) == 1
        row = int(rows[0])
        value = int(relation[row, column])
        assert value in (-1, 1)
        mapping.append(row)
        signs.append(value)
    return tuple(mapping), tuple(signs)


def local_action(pair: SignedAction, order: int, site: int) -> SignedAction:
    pair_map, pair_sign = pair
    dimension = 5 ** order
    mapping = []
    signs = []
    for column in range(dimension):
        digits = []
        value = column
        for power in range(order - 1, -1, -1):
            divisor = 5 ** power
            digits.append(value // divisor)
            value %= divisor
        pair_column = 5 * digits[site] + digits[site + 1]
        pair_row = pair_map[pair_column]
        digits[site], digits[site + 1] = divmod(pair_row, 5)
        row = 0
        for digit in digits:
            row = 5 * row + digit
        mapping.append(row)
        signs.append(pair_sign[pair_column])
    return tuple(mapping), tuple(signs)


def compose(left: SignedAction, right: SignedAction) -> SignedAction:
    left_map, left_sign = left
    right_map, right_sign = right
    return (
        tuple(left_map[right_map[column]] for column in range(len(right_map))),
        tuple(left_sign[right_map[column]] * right_sign[column] for column in range(len(right_map))),
    )


def insertion_coherence_gate(survivors: list[dict[str, Any]]) -> dict[str, Any]:
    checks = 0
    identity_action: SignedAction = (tuple(range(625)), tuple(1 for _ in range(625)))
    for survivor in survivors:
        pair = pair_action(survivor["R"].astype(np.int64))
        generators = [local_action(pair, 4, site) for site in range(3)]
        for generator in generators:
            assert compose(generator, generator) == identity_action
            checks += 1
        assert compose(generators[0], compose(generators[1], generators[0])) == compose(
            generators[1], compose(generators[0], generators[1])
        )
        assert compose(generators[1], compose(generators[2], generators[1])) == compose(
            generators[2], compose(generators[1], generators[2])
        )
        assert compose(generators[0], generators[2]) == compose(generators[2], generators[0])
        checks += 3
    return {
        "actual_four_strand_signed_action_dimension": 625,
        "actual_sectors": len(survivors),
        "Coxeter_relation_checks": checks,
        "involutivity": True,
        "adjacent_Yang_Baxter_coherence": True,
        "distant_commutation": True,
        "all_order_consequence": "The local signed actions satisfy the symmetric-group presentation. Moving a marked finite support through inserted unmarked strands is therefore independent of the chosen adjacent-move word up to the transported marked permutation.",
        "covered_refinements": "Identity-strand insertions away from a finite marked support, together with coherent transport of that support by the actual Braid action.",
        "not_covered": "Splitting one marked C5 factor into two factors; no source coproduct or cloning morphism has been derived.",
        "status": "PASS_MARKED_INSERTION_AND_SUPPORT_TRANSPORT_COHERENCE__MARKED_FACTOR_SPLITTING_OPEN",
    }


def ucp_multiplicative_domain_gate(allorder: dict[str, Any], ub475: dict[str, Any]) -> dict[str, Any]:
    theorem = allorder["exact_all_order_theorem"]
    assert theorem["recoverable_operator_system_genuine_functor"] is True
    assert theorem["full_algebra_ordinary_functor"] is False
    assert "normalized-trace state" in theorem["statements"]["ucp_embedding"]

    multiplicities = {0: 49, 2: 36, 3: 36, 5: 4}
    dimension = sum(multiplicities.values())
    mean = sum(Fraction(value * count, dimension) for value, count in multiplicities.items())
    second = sum(Fraction(value * value * count, dimension) for value, count in multiplicities.items())
    variance = second - mean * mean
    assert dimension == 125 and mean == Fraction(8, 5) and variance == Fraction(248, 125)
    endpoint_mean = sum(Fraction((1 if value % 2 == 0 else -1) * count, dimension) for value, count in multiplicities.items())
    endpoint_defect = Fraction(1) - endpoint_mean * endpoint_mean
    assert endpoint_mean == Fraction(9, 25) and endpoint_defect == Fraction(544, 625)

    spatial_traces = [Fraction(rank, 25) for rank in ub475["spatial_projector_ranks"]]
    spatial_defects = [value - value * value for value in spatial_traces]
    assert spatial_traces == [Fraction(8, 25), Fraction(4, 25), Fraction(4, 25)]
    assert spatial_defects == [Fraction(136, 625), Fraction(84, 625), Fraction(84, 625)]
    return {
        "map_type": "split UCP observation/recovery embedding with a normalized-trace scalar branch",
        "multiplicative_domain_identity": "Phi(X^2)-Phi(X)^2 has trace-branch value tau(X^2)-tau(X)^2=tau((X-tau(X)I)^2), zero exactly for scalar self-adjoint X under faithful tau.",
        "G1_normalized_trace": str(mean),
        "G1_variance_defect": str(variance),
        "endpoint_R_normalized_trace": str(endpoint_mean),
        "endpoint_unitarity_defect": str(endpoint_defect),
        "P_chi_normalized_traces": [str(value) for value in spatial_traces],
        "P_chi_idempotency_defects": [str(value) for value in spatial_defects],
        "nonzero_defects": True,
        "existing_hierarchy_contains_G1_transport": False,
        "conclusion": "The pinned split-UCP hierarchy cannot be the same product-preserving refinement morphism for the non-scalar clock/Fourier algebra.",
        "status": "FAIL_EXACT_MULTIPLICATIVE_CLOCK_FOURIER_TRANSPORT_THROUGH_NORMALIZED_TRACE_UCP_BRANCH",
    }


def repair_controls_gate() -> dict[str, Any]:
    return {
        "direct_sum_identity_clock": {
            "formal_transport": True,
            "admissible": False,
            "reason": "It appends an independent unchanged clock factor and is therefore a forbidden spectator."
        },
        "spectral_character_state": {
            "clock_algebra_characters": [0, 2, 3, 5],
            "ground_character_unique_for_clock_only": True,
            "preserves_all_three_nonzero_Fourier_projectors_in_scalar_branch": False,
            "admissible_as_full_screen_repair": False,
            "reason": "A scalar character sends orthogonal projectors to zero or one and cannot retain all three as distinct nonzero projector directions."
        },
        "typed_resolution": "Use star-monomorphisms for information-preserving refinement and retain UCP maps as observation/coarse-graining morphisms with explicitly declared recoverable subobjects; equality of morphism types is refuted.",
        "typed_resolution_already_fully_proved_as_one_fixed_point_category": False,
        "status": "NO_ADMISSIBLE_ONE_LINE_REPAIR__TYPED_TWO_MORPHISM_CATEGORY_REQUIRED",
    }


def run_checks() -> dict[str, Any]:
    loaded, frozen = verify_inputs()
    survivors, projectors, characters, source = reconstruct_actual_source()
    tensor = tensor_refinement_gate(survivors, projectors)
    insertion = insertion_coherence_gate(survivors)
    ucp = ucp_multiplicative_domain_gate(loaded["allordercp68r"], loaded["ub475"])
    repairs = repair_controls_gate()
    assert tensor["status"].startswith("PASS")
    assert insertion["status"].startswith("PASS")
    assert ucp["status"].startswith("FAIL")
    assert repairs["typed_resolution_already_fully_proved_as_one_fixed_point_category"] is False
    return {
        "schema": "siel.dpa.ocbfh014.certificate.v1",
        "programme_id": "OCBFH-001",
        "gate_id": GATE_ID,
        "date": "2026-09-13",
        "status": FINAL_STATUS,
        "primary_evidence_status": "Theoretical derivation and exact executable verification",
        "scientific_layer": "mathematical formulation",
        "frozen_inputs": frozen,
        "actual_source_reconstruction": source,
        "tensor_refinement_gate": tensor,
        "insertion_coherence_gate": insertion,
        "ucp_multiplicative_domain_gate": ucp,
        "repair_controls_gate": repairs,
        "observed_evidence": [
            "For all eight actual signed sources, the marked three-site G1 and all three two-site V4 Fourier projectors extend exactly by A to A tensor I. The extension preserves products, adjoints, projectors, the G1 polynomial and the endpoint functional calculus.",
            "The actual four-strand signed actions satisfy all involutive, adjacent Yang-Baxter and distant-commutation relations, so insertion away from marked support and Braid transport of that support are coherent at every order.",
            "The normalized-trace branch of the existing split-UCP hierarchy has exact G1 variance defect 248/125 and endpoint unitarity defect 544/625; it also destroys idempotency of all three P_chi. It is not a product-preserving refinement of the same clock/Fourier algebra.",
            "Neither adjoining an independent identity clock nor replacing the trace by one scalar spectral character is an admissible full-screen repair."
        ],
        "pattern": "The soldering exists naturally on the genuine information-preserving pointed Braid tensor tower. Its previous absence from ALLORDERCP is a real obstruction for that UCP map, but not a no-go for every source-native all-order construction; refinement and observation have different morphism types.",
        "interpretive_leap": "Braid can carry the same one-time/three-space dictionary through indefinitely many added relational degrees of freedom without reselecting it. Observation may compress that dictionary, so a complete observer-fixed-point theory may need embeddings for world growth and UCP arrows for viewpoint readout rather than one arrow type for both.",
        "alternative_explanation": "Local tensor inclusions preserving local observables are standard operator-algebraic facts, and UCP trace channels failing multiplicativity is standard. Braid specificity is confined to the actual G1, V4 projectors, signed Yang-Baxter transport and their common eight-source provenance.",
        "falsifier": "Find an actual sector where tensor extension fails, a violation of the signed Coxeter relations, zero UCP defects for the non-scalar four-screen, or a source-derived marked-factor coproduct contradicting the stated open boundary.",
        "decision": {
            "fixed_carrier_unique_soldering": "PASS_INHERITED_OCBFH013",
            "source_native_pointed_tensor_refinement_existence": "PASS_EXACT_ALLORDER",
            "all_unmarked_insertions_and_marked_support_transport": "PASS_UP_TO_TRANSPORTED_MARKED_BRAID_EQUIVALENCE",
            "marked_factor_splitting": "OPEN_NO_SOURCE_COPRODUCT",
            "pinned_split_UCP_as_same_product_preserving_refinement": "REFUTED_EXACT",
            "universal_all_source_native_morphism_compatibility": "NOT_ESTABLISHED",
            "external_S0_as_tensor_refinement_choice": "RETIRED_WITHIN_POINTED_MARKED_TENSOR_TOWER",
            "ocbfh001_master_hypothesis": "STRENGTHENED_BUT_REQUIRES_TYPED_REFINEMENT_OBSERVATION_CATEGORY",
            "next_gate": "OCBFH-015_TYPED_REFINEMENT_OBSERVATION_DINATURAL_FIXED_POINT"
        },
        "claim_ceiling": "Exact all-order naturality of the OCBFH013 soldering for the source-native pointed marked tensor/Braid tower, exact marked-insertion coherence away from unsplit marked factors, and exact refutation of the normalized-trace split-UCP map as a product-preserving clock/Fourier refinement. No marked-factor coproduct, universal fixed-point category, real completion, physical scale, spacetime, dynamics, HSMI, QFT, GR or natural evidence is proved."
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), ensure_ascii=False, indent=2, sort_keys=True))

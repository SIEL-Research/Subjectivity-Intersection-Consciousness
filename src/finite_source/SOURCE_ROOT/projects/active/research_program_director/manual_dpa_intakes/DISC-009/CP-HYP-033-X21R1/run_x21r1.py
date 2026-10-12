#!/usr/bin/env python3
"""Direct characteristic-zero reconstruction for CP-HYP-033-X21R1."""

from __future__ import annotations

from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

from charzero_core import cross, curvature, hchi, lorentz_completion, ray_mod


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
A63_PATH = ROOT / "audits/DPA_A63_INTRINSIC_SPECTRAL_TRANSFER_GENERATOR_WARD_GATE_20260915/evaluate.py"
O14_PATH = ROOT / "projects/active/discovery_partner/formal_checks/ocbfh014_source_native_refinement_naturality_check.py"
X13_PATH = ROOT / "audits/SRA_DPA_A57S_X13_CURVATURE_ODD_DUAL_COMPLETION_SCOUT_20260917/RAW_OUTPUT.json"
PRIMES = (1000033, 1000159)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def commutator(left, right):
    return left @ right - right @ left


def vector_dict(values):
    return {index: Q(value) for index, value in enumerate(values) if Q(value)}


class ExactSpan:
    """Small-rank exact RREF over Q, with large sparse ambient vectors."""

    def __init__(self):
        self.rows = {}

    def add(self, values):
        work = vector_dict(values)
        for pivot in sorted(self.rows):
            factor = work.get(pivot, Q(0))
            if not factor:
                continue
            for index, value in self.rows[pivot].items():
                updated = work.get(index, Q(0)) - factor * value
                if updated:
                    work[index] = updated
                else:
                    work.pop(index, None)
        if not work:
            return False
        pivot = min(work)
        factor = work[pivot]
        work = {index: value / factor for index, value in work.items()}
        for old_pivot, row in list(self.rows.items()):
            factor = row.get(pivot, Q(0))
            if not factor:
                continue
            updated = dict(row)
            for index, value in work.items():
                entry = updated.get(index, Q(0)) - factor * value
                if entry:
                    updated[index] = entry
                else:
                    updated.pop(index, None)
            self.rows[old_pivot] = updated
        self.rows[pivot] = work
        return True

    @property
    def rank(self):
        return len(self.rows)


def qeye(size):
    return [[Q(int(row == column)) for column in range(size)] for row in range(size)]


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matmul(left, right):
    right_t = transpose(right)
    return [[sum((x * y for x, y in zip(row, column)), Q(0)) for column in right_t] for row in left]


def inverse(matrix):
    size = len(matrix)
    work = [[Q(value) for value in row] + qeye(size)[index] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column]), None)
        if pivot is None:
            raise ValueError("singular exact matrix")
        work[column], work[pivot] = work[pivot], work[column]
        factor = work[column][column]
        work[column] = [value / factor for value in work[column]]
        for row in range(size):
            if row == column or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [x - factor * y for x, y in zip(work[row], work[column])]
    return [row[size:] for row in work]


def exact_rank_columns(columns):
    span = ExactSpan()
    indices = []
    for index, column in enumerate(columns):
        if span.add(column):
            indices.append(index)
    return indices


def pivot_rows(column_matrix):
    rows = column_matrix
    selected = exact_rank_columns(rows)
    return selected


def module_closure(screen, omegas):
    basis = []
    span = ExactSpan()
    for item in screen:
        if not span.add(item.reshape(-1)):
            raise AssertionError("source screen is not rank four over Q")
        basis.append(np.asarray(item, dtype=np.int64))
    growth = [len(basis)]
    frontier = 0
    while True:
        snapshot = list(basis)
        for item in snapshot[frontier:]:
            for omega in omegas:
                candidate = commutator(omega, item).astype(np.int64)
                if span.add(candidate.reshape(-1)):
                    basis.append(candidate)
        if len(basis) == len(snapshot):
            break
        frontier = len(snapshot)
        growth.append(len(basis))
        if len(basis) > 64:
            raise AssertionError("unexpected rational module growth")
    columns = [item.reshape(-1).tolist() for item in basis]
    rows = [list(row) for row in zip(*columns)]
    pivots = pivot_rows(rows)
    if len(pivots) != len(basis):
        raise AssertionError("failed to find rational pivot rows")
    square = [[Q(rows[row][column]) for column in range(len(basis))] for row in pivots]
    square_inverse = inverse(square)

    def coordinates(target):
        values = [[Q(int(target.reshape(-1)[row]))] for row in pivots]
        answer = [row[0] for row in matmul(square_inverse, values)]
        for ambient_row, original_row in enumerate(rows):
            if sum((Q(value) * coefficient for value, coefficient in zip(original_row, answer)), Q(0)) != Q(int(target.reshape(-1)[ambient_row])):
                raise AssertionError("rational coordinate solve failed")
        return answer

    actions = []
    for omega in omegas:
        action_columns = [coordinates(commutator(omega, item)) for item in basis]
        actions.append([list(row) for row in zip(*action_columns)])
    return basis, rows, actions, growth


def build_carrier(screen, omegas):
    basis, ambient_rows, actions, growth = module_closure(screen, omegas)
    size = len(basis)
    if size != 20:
        raise AssertionError(f"rational module dimension {size}, expected 20")
    image_columns = [column for action in actions for column in transpose(action)]
    image_indices = exact_rank_columns(image_columns)
    kernel_columns = [image_columns[index] for index in image_indices]
    if len(kernel_columns) != 16:
        raise AssertionError(f"rational image dimension {len(kernel_columns)}, expected 16")
    direct_columns = [qeye(size)[column] for column in range(4)] + kernel_columns
    direct = transpose(direct_columns)
    direct_inverse = inverse(direct)
    blocks = [matmul(matmul(direct_inverse, action), direct) for action in actions]
    coefficient = []
    target = []
    for block in blocks:
        coefficient.extend([row[4:] for row in block[4:]])
        target.extend([[-value for value in row[:4]] for row in block[4:]])
    independent_rows = pivot_rows(coefficient)
    if len(independent_rows) != 16:
        raise AssertionError(f"splitting coefficient rank {len(independent_rows)}, expected 16")
    square = [coefficient[index] for index in independent_rows]
    rhs = [target[index] for index in independent_rows]
    x_map = matmul(inverse(square), rhs)
    if matmul(coefficient, x_map) != target:
        raise AssertionError("rational extension does not split")
    graph = qeye(4) + x_map
    carrier_coordinates = matmul(direct, graph)
    carrier = []
    for ambient_row in ambient_rows:
        carrier.append([
            sum((Q(value) * carrier_coordinates[index][column] for index, value in enumerate(ambient_row)), Q(0))
            for column in range(4)
        ])
    return carrier, {
        "module_dimension": size,
        "module_growth": growth,
        "image_dimension": len(kernel_columns),
        "splitting_coefficient_rank": len(independent_rows),
        "splitting_unique": len(coefficient[0]) == len(independent_rows),
    }


def qdot(a63, left, right):
    result = a63.ZERO
    for x, y in zip(left, right):
        result = a63.qadd(result, a63.qmul(x, y))
    return result


def ordered_edge_gram_q17(a63, raw, generators):
    projector_pairs = [a63.qmatrix_pair(projector) for projector in raw["spectral_projectors"]]
    triple_pairs = []
    triple_ranks = []
    size = len(projector_pairs)
    for first in range(size):
        for second in range(size):
            for third in range(size):
                triple_pairs.append(a63.qmkron_pair(a63.qmkron_pair(projector_pairs[first], projector_pairs[second]), projector_pairs[third]))
                triple_ranks.append(raw["spectral_ranks"][first] * raw["spectral_ranks"][second] * raw["spectral_ranks"][third])

    def averages(observable):
        return [
            a63.qdiv(a63.qpair_trace_sigma(observable, pair), a63.q(rank))
            for pair, rank in zip(triple_pairs, triple_ranks)
        ]

    transition, _ = a63.exact_edge_current(raw["spectral_weights"], raw["pair_weights"], size)
    stationary = [a63.qdiv(weight, a63.q(18000)) for weight in raw["spectral_weights"]]
    av = [averages(generator) for generator in generators]
    products = [[averages(generators[left].T @ generators[right]) for right in range(4)] for left in range(4)]
    conditional = []
    for values in av:
        conditional.append([qdot(a63, row, values) for row in transition])
    gram = []
    for left in range(4):
        row = []
        for right in range(4):
            integrand = []
            for state in range(len(stationary)):
                twice_product = a63.qmul(a63.q(2), products[left][right][state])
                cross_lr = a63.qmul(conditional[left][state], av[right][state])
                cross_rl = a63.qmul(av[left][state], conditional[right][state])
                integrand.append(a63.qsub(a63.qsub(twice_product, cross_lr), cross_rl))
            row.append(qdot(a63, stationary, integrand))
        gram.append(row)
    return gram


def serial_fraction(value):
    return str(value)


def main():
    protocol = load(HERE / "FROZEN_PROTOCOL.json")
    baseline = load(HERE / "BASELINE_GATE.json")
    endpoint = load(HERE / "E0_RECORD.json")
    assert protocol["status"] == "FROZEN_AFTER_SOURCE_DERIVED_E0_BEFORE_FOCAL_OUTCOME_ACCESS"
    assert baseline["science_outcome_authorized"] is True
    assert endpoint["overall_status"] == "PASS"
    for relative, expected in protocol["fixed_source_hashes"].items():
        assert sha(ROOT / relative) == expected

    a63 = module("x21r1_a63", A63_PATH)
    a63.A61 = a63.load_module("x21r1_a61", a63.INPUTS["a61_evaluate"])
    a63.A62 = a63.load_module("x21r1_a62", a63.INPUTS["a62_evaluate"])
    o14 = module("x21r1_o14", O14_PATH)
    survivors, projectors_chi, _, source_metadata = o14.reconstruct_actual_source()
    identity5 = np.eye(5, dtype=np.int64)
    spatial = [np.kron(numerator, identity5) for numerator, _ in projectors_chi]

    carriers = {}
    carrier_records = []
    gram_by_mask = {}
    for survivor in survivors:
        mask = int(survivor["mask"])
        relation = survivor["R"].astype(np.int64)
        cap = survivor["P"].astype(np.int64)
        clock, _, _, _ = o14.source_g1(relation)
        screen = [clock, *spatial]
        omegas = [commutator(screen[0], screen[index]) for index in range(1, 4)]
        carrier, diagnostics = build_carrier(screen, omegas)
        carriers[mask] = carrier
        raw = a63.classify(relation, cap, projectors_chi, o14)
        q17_gram = ordered_edge_gram_q17(a63, raw, screen)
        radical_zero = all(value[1] == 0 for row in q17_gram for value in row)
        if not radical_zero:
            raise AssertionError(f"ordered-edge Gram retains sqrt17 at mask {mask}")
        rational_gram = [[value[0] for value in row] for row in q17_gram]
        gram_by_mask[mask] = rational_gram
        carrier_records.append({
            "mask": mask,
            **diagnostics,
            "ordered_edge_gram_radical_zero": radical_zero,
            "ordered_edge_gram": [[serial_fraction(value) for value in row] for row in rational_gram],
        })

    rational_hchi = {}
    source_records = []
    for mask in sorted(carriers):
        face = curvature(carriers, mask, 8, 16)
        eta = lorentz_completion(gram_by_mask[mask])
        value = hchi(eta, face)
        rational_hchi[mask] = value
        denominator_lcm = math.lcm(*(entry.denominator for entry in value))
        primitive = [int(entry * denominator_lcm) for entry in value]
        divisor = math.gcd(*(abs(entry) for entry in primitive if entry))
        primitive = [entry // divisor for entry in primitive]
        first = next(entry for entry in primitive if entry)
        if first < 0:
            primitive = [-entry for entry in primitive]
        source_records.append({
            "mask": mask,
            "h_chi": [serial_fraction(entry) for entry in value],
            "primitive_integer_ray": primitive,
            "orientation_reversal_negates": hchi(eta, [[-entry for entry in row] for row in face]) == [-entry for entry in value],
            "zero_curvature_control": hchi(eta, [[Q(0) for _ in range(4)] for _ in range(4)]) == [Q(0)] * 4,
        })

    # The focal target is loaded only after the direct Q construction is complete.
    x13 = load(X13_PATH)
    comparisons = []
    for record in x13["records"]:
        mask = int(record["mask"])
        prime = int(record["prime"])
        produced = ray_mod(rational_hchi[mask], prime)
        target = ray_mod([Q(int(value)) for value in record["h_chi"]], prime)
        comparisons.append({
            "mask": mask,
            "prime": prime,
            "sqrt17_image": int(record["sqrt17_image"]),
            "prefix": record["prefix"],
            "produced_ray": produced,
            "target_ray": target,
            "match": produced == target,
        })

    tests = {
        "B0_SOURCE_DERIVED_E0_AND_BASELINE": True,
        "A1_RATIONAL_MODULE_DIMENSION_20": all(record["module_dimension"] == 20 for record in carrier_records),
        "A2_RATIONAL_IMAGE_DIMENSION_16": all(record["image_dimension"] == 16 for record in carrier_records),
        "A3_UNIQUE_RATIONAL_SPLITTING": all(record["splitting_unique"] for record in carrier_records),
        "A4_ORDERED_EDGE_GRAM_DESCENDS_TO_Q": all(record["ordered_edge_gram_radical_zero"] for record in carrier_records),
        "A5_ALL_RATIONAL_HCHI_NONZERO": all(any(entry for entry in value) for value in rational_hchi.values()),
        "A6_ALL_64_MODULAR_RAYS_REPRODUCED": len(comparisons) == 64 and all(record["match"] for record in comparisons),
        "B1_ORIENTATION_REVERSAL": all(record["orientation_reversal_negates"] for record in source_records),
        "B2_ZERO_CURVATURE_CONTROL": all(record["zero_curvature_control"] for record in source_records),
        "X20_ORDER_RECOVERY": False,
        "PHYSICAL_GRAVITY": False,
    }
    mandatory = [key for key in tests if key.startswith("A") or key.startswith("B")]
    status = (
        "EXPLORATORY_PASS_DIRECT_CHARACTERISTIC_ZERO_HCHI_LIFT_REPRODUCES_ALL_64_X13_RAYS__X20_ORDER_RECOVERY_AND_PHYSICAL_GRAVITY_OPEN"
        if all(tests[key] for key in mandatory)
        else "EXPLORATORY_NO_GO_DIRECT_CHARACTERISTIC_ZERO_HCHI_LIFT"
    )
    raw_output = {
        "schema": "siel.rpd.cp-hyp-033-x21r1.raw-output.v1",
        "experiment_id": "CP-HYP-033-X21R1",
        "source_reconstruction": source_metadata,
        "input_hashes": {
            str(A63_PATH.relative_to(ROOT)): sha(A63_PATH),
            str(O14_PATH.relative_to(ROOT)): sha(O14_PATH),
            str(X13_PATH.relative_to(ROOT)): sha(X13_PATH),
        },
        "carrier_records": carrier_records,
        "source_records": source_records,
        "comparisons": comparisons,
        "summary": {
            "source_count": len(source_records),
            "comparison_count": len(comparisons),
            "matched_comparisons": sum(record["match"] for record in comparisons),
            "module_dimensions": sorted({record["module_dimension"] for record in carrier_records}),
            "image_dimensions": sorted({record["image_dimension"] for record in carrier_records}),
        },
        "decision_tests": tests,
    }
    result = {
        "schema": "siel.rpd.cp-hyp-033-x21r1.result.v1",
        "experiment_id": "CP-HYP-033-X21R1",
        "date": "2026-09-17",
        "primary_evidence_status": "Exploratory result",
        "qualifiers": ["exact finite-source", "direct rational reconstruction", "not independently replicated", "not natural evidence"],
        "scientific_layer": "mathematical formulation and finite-source field descent",
        "status": status,
        "summary": raw_output["summary"],
        "decision_tests": tests,
        "claim_ceiling": protocol["claim_ceiling"],
    }
    (HERE / "RAW_PACKET_OUTPUTS.json").write_text(json.dumps(raw_output, indent=2) + "\n")
    (HERE / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(status)
    print(json.dumps(raw_output["summary"], indent=2))
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()

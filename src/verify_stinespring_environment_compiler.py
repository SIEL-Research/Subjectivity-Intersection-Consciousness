#!/usr/bin/env python3
"""Exact source-side checks for the Q-014 Stinespring compiler theorem."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations, product
import hashlib
import json
import math
import subprocess


COMMIT = "269a832e497c901a2b24bb2167ca97b9e6ce2bd1"
RAW_PATH = (
    "projects/active/research_program_director/manual_dpa_intakes/"
    "DISC-009/CP-HYP-033-X21R1/RAW_PACKET_OUTPUTS.json"
)
EXPECTED_RAW_SHA256 = (
    "7ba1c987a9389f7426becdde136986be7fbc7b17b92dac938cc2a43cc39b373e"
)


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"])


def poly_add(a: tuple[Fraction, ...], b: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    size = max(len(a), len(b))
    return tuple(
        (a[index] if index < len(a) else Fraction(0))
        + (b[index] if index < len(b) else Fraction(0))
        for index in range(size)
    )


def poly_scale(a: tuple[Fraction, ...], scale: Fraction) -> tuple[Fraction, ...]:
    return tuple(scale * value for value in a)


def poly_mul(a: tuple[Fraction, ...], b: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            out[i + j] += left * right
    return tuple(out)


def poly_eval(a: tuple[Fraction, ...], value: float) -> float:
    return sum(float(coefficient) * value**power for power, coefficient in enumerate(a))


def interval_add(left, right):
    return left[0] + right[0], left[1] + right[1]


def interval_multiply(left, right):
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def poly_interval(a: tuple[Fraction, ...], domain):
    """Exact Horner enclosure of a rational polynomial on an interval."""
    value = (Fraction(0), Fraction(0))
    for coefficient in reversed(a):
        value = interval_add(
            interval_multiply(value, domain),
            (coefficient, coefficient),
        )
    return value


def interval_excludes_zero(value) -> bool:
    return value[1] < 0 or value[0] > 0


def matrix_multiply(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def transpose(a):
    return [list(row) for row in zip(*a)]


def identity():
    return [[Fraction(int(i == j)) for j in range(4)] for i in range(4)]


def signed_permutation(perm, signs):
    matrix = [[Fraction(0) for _ in range(4)] for _ in range(4)]
    for column, row in enumerate(perm):
        matrix[row][column] = Fraction(signs[column])
    return matrix


def common_signed_stabilizer(grams):
    elements = []
    for perm in permutations(range(4)):
        for signs in product((-1, 1), repeat=4):
            candidate = signed_permutation(perm, signs)
            if all(
                matrix_multiply(transpose(candidate), matrix_multiply(gram, candidate))
                == gram
                for gram in grams
            ):
                elements.append(candidate)
    return elements


def joint_binary_pairs(elements) -> int:
    ident = identity()
    count = 0
    for left, right in combinations(elements, 2):
        if matrix_multiply(left, left) != ident or matrix_multiply(right, right) != ident:
            continue
        if left != transpose(left) or right != transpose(right):
            continue
        if matrix_multiply(left, right) != matrix_multiply(right, left):
            continue
        # With only the common +/- identity stabilizer, no pair can have four
        # nonzero one-dimensional joint eigenspaces. Retain the explicit count
        # instead of importing a numerical-rank dependency.
        if left not in (ident, poly_matrix_scale(ident, Fraction(-1))) and right not in (
            ident,
            poly_matrix_scale(ident, Fraction(-1)),
        ):
            count += 1
    return count


def poly_matrix_scale(matrix, scale):
    return [[scale * value for value in row] for row in matrix]


def main() -> None:
    payload = git_bytes(RAW_PATH)
    raw = json.loads(payload)
    grams = [
        [[Fraction(value) for value in row] for row in record["ordered_edge_gram"]]
        for record in raw["carrier_records"]
    ]

    # p=(1/4,1/4,(1+s)/4,(1-s)/4), s=sin(0.05).
    probability_polynomials = (
        (Fraction(1, 4),),
        (Fraction(1, 4),),
        (Fraction(1, 4), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(-1, 4)),
    )
    diagonal_purity = (Fraction(1, 4), Fraction(0), Fraction(1, 8))
    s_value = math.sin(0.05)
    x = Fraction(1, 20)
    # Alternating Taylor bounds: x-x^3/3! < sin(x) < x-x^3/3!+x^5/5!.
    sin_interval = (
        x - x**3 / math.factorial(3),
        x - x**3 / math.factorial(3) + x**5 / math.factorial(5),
    )

    records = []
    signatures = set()
    for source_record, gram in zip(raw["carrier_records"], grams):
        purity = (Fraction(0),)
        for i in range(4):
            for j in range(4):
                squared_correlation = (
                    gram[i][j] * gram[i][j] / (gram[i][i] * gram[j][j])
                )
                term = poly_scale(
                    poly_mul(probability_polynomials[i], probability_polynomials[j]),
                    squared_correlation,
                )
                purity = poly_add(purity, term)
        purity = purity + (Fraction(0),) * (3 - len(purity))
        gain = tuple(purity[index] - diagonal_purity[index] for index in range(3))
        signatures.add(purity)
        records.append(
            {
                "mask": int(source_record["mask"]),
                "purity_polynomial_coefficients": [str(value) for value in purity],
                "gain_polynomial_coefficients": [str(value) for value in gain],
                "ideal_action_purity": poly_eval(purity, s_value),
                "ideal_action_gain": poly_eval(gain, s_value),
            }
        )

    actual_values = [record["ideal_action_purity"] for record in records]
    purity_polynomials = [
        tuple(Fraction(value) for value in record["purity_polynomial_coefficients"])
        for record in records
    ]
    gain_polynomials = [
        tuple(Fraction(value) for value in record["gain_polynomial_coefficients"])
        for record in records
    ]
    pairwise_difference_intervals = [
        poly_interval(
            poly_add(left, poly_scale(right, Fraction(-1))),
            sin_interval,
        )
        for index, left in enumerate(purity_polynomials)
        for right in purity_polynomials[index + 1 :]
    ]
    gain_intervals = [poly_interval(gain, sin_interval) for gain in gain_polynomials]
    stabilizer = common_signed_stabilizer(grams)
    report = {
        "schema": "q014-stinespring-environment-compiler-certificate-v1",
        "source_commit": COMMIT,
        "source_path": RAW_PATH,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_masks": [record["mask"] for record in records],
        "checks": {
            "source_hash_matches": hashlib.sha256(payload).hexdigest() == EXPECTED_RAW_SHA256,
            "eight_source_records": len(records) == 8,
            "eight_distinct_ideal_action_purities_exactly_certified": all(
                interval_excludes_zero(value)
                for value in pairwise_difference_intervals
            ),
            "eight_distinct_symbolic_purity_polynomials": len(signatures) == 8,
            "all_ideal_action_purity_gains_positive_exactly_certified": all(
                value[0] > 0 for value in gain_intervals
            ),
            "common_signed_stabilizer_is_plus_minus_identity": len(stabilizer) == 2,
            "common_joint_binary_pair_count_zero": joint_binary_pairs(stabilizer) == 0,
        },
        "common_signed_stabilizer_size": len(stabilizer),
        "common_joint_binary_pair_count": joint_binary_pairs(stabilizer),
        "sin_0_05_rational_interval": [str(value) for value in sin_interval],
        "minimum_ideal_action_purity_separation": min(
            abs(left - right)
            for index, left in enumerate(actual_values)
            for right in actual_values[index + 1 :]
        ),
        "records": records,
        "claim_ceiling": (
            "Exact source-side compiler certificate only; no external Ankaa-3 "
            "realization, mask selection in nature, EEG readout, gravity, "
            "subjectivity, or consciousness claim."
        ),
    }
    report["decision"] = (
        "PHASE_ROBUST_STINESPRING_ENVIRONMENT_COMPILER_EXISTS"
        if all(report["checks"].values())
        else "COMPILER_CERTIFICATE_INCOMPLETE"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

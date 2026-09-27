import math

import numpy as np

from relational_compression.experiments.transductive_relational_distortion.run_scripts.analyze_relative_distortions import (
    COLLISION,
    COLLISION_ENTROPY,
    EDGE,
    FOURIER,
    Match,
    average_ranks,
    collision_jaccard,
    complete_rate_matched_case,
    select_rate_match,
)


def test_collision_jaccard_matches_contingency_formula() -> None:
    first = np.asarray([0, 0, 0, 1, 1, 2])
    second = np.asarray([3, 3, 4, 4, 4, 5])

    # Intersections: choose2({0,1} in first cluster 0 and second cluster 3) = 1,
    # plus choose2({3,4} in first cluster 1 and second cluster 4) = 1.
    # Pair counts are 4 for the first partition and 4 for the second, so union = 6.
    assert math.isclose(collision_jaccard(first, second), 2.0 / 6.0)


def test_collision_jaccard_is_label_permutation_invariant() -> None:
    assignments = np.asarray([0, 0, 2, 2, 2, 7, 7])
    permuted = np.asarray([10, 10, 11, 11, 11, 4, 4])

    assert math.isclose(collision_jaccard(assignments, assignments), 1.0)
    assert math.isclose(collision_jaccard(assignments, permuted), 1.0)


def test_collision_jaccard_returns_nan_for_no_co_clustered_pairs() -> None:
    first = np.arange(5)
    second = np.arange(5) + 10

    assert math.isnan(collision_jaccard(first, second))


def test_select_rate_match_uses_tolerance_and_lower_lambda_tie_break() -> None:
    rows = [
        {"hard_h2": "0.95", "lambda_org": "0.20"},
        {"hard_h2": "0.95", "lambda_org": "0.10"},
        {"hard_h2": "1.50", "lambda_org": "0.03"},
    ]

    match = select_rate_match(rows, r_target=1.0, tolerance=0.10)

    assert match is not None
    assert match.row["lambda_org"] == "0.10"
    assert math.isclose(match.distance, 0.05)
    assert select_rate_match(rows, r_target=1.0, tolerance=0.04) is None


def test_average_ranks_uses_average_ties_and_lower_is_better() -> None:
    ranks = average_ranks({"edge": 0.3, "fourier": 0.1, "collision": 0.1, "collision_entropy": 0.7})

    assert ranks["fourier"] == 1.5
    assert ranks["collision"] == 1.5
    assert ranks["edge"] == 3.0
    assert ranks["collision_entropy"] == 4.0


def _synthetic_matches(
    values: dict[str, float], *, target: float = 1.0
) -> dict[tuple[tuple[str, str, str, str, str, str], str, float], Match]:
    key = ("collection", "source", "split", "0", "0", "0")
    return {
        (key, criterion, target): Match(
            row={"criterion": criterion, "hard_h2": str(value), "lambda_org": "0.1"},
            distance=abs(value - target),
        )
        for criterion, value in values.items()
    }


def test_complete_rate_matched_case_rejects_individually_valid_but_wide_span() -> None:
    key = ("collection", "source", "split", "0", "0", "0")
    matches = _synthetic_matches({EDGE: 0.91, FOURIER: 1.00, COLLISION: 1.09, COLLISION_ENTROPY: 1.00})

    assert complete_rate_matched_case(matches, key, 1.0, (EDGE, FOURIER, COLLISION), 0.10) is None
    assert complete_rate_matched_case(matches, key, 1.0, (EDGE, FOURIER, COLLISION, COLLISION_ENTROPY), 0.10) is None


def test_complete_rate_matched_case_accepts_tightly_matched_three_and_four_way_cases() -> None:
    key = ("collection", "source", "split", "0", "0", "0")
    matches = _synthetic_matches({EDGE: 0.96, FOURIER: 1.00, COLLISION: 1.05, COLLISION_ENTROPY: 1.02})

    three_way = complete_rate_matched_case(matches, key, 1.0, (EDGE, FOURIER, COLLISION), 0.10)
    four_way = complete_rate_matched_case(matches, key, 1.0, (EDGE, FOURIER, COLLISION, COLLISION_ENTROPY), 0.10)

    assert three_way is not None
    assert tuple(three_way) == (EDGE, FOURIER, COLLISION)
    assert four_way is not None
    assert tuple(four_way) == (EDGE, FOURIER, COLLISION, COLLISION_ENTROPY)

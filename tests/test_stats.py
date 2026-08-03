"""Ranking maths, checked against inputs whose answer is known in closed form.

Bradley-Terry on a single pair has an exact solution: the ratio of strengths
equals the ratio of wins. That makes the two-model cases real verification
rather than a regression lock on whatever the code happened to produce.
"""

import pytest

from konkord.stats import (
    PairOutcome,
    bootstrap_win_rate_intervals,
    bradley_terry,
    models_in,
    tie_groups,
    win_rates,
)


def battles(
    model_a: str, model_b: str, a_wins: int, b_wins: int, ties: int = 0
) -> list[PairOutcome]:
    return (
        [PairOutcome(model_a, model_b, model_a)] * a_wins
        + [PairOutcome(model_a, model_b, model_b)] * b_wins
        + [PairOutcome(model_a, model_b, None)] * ties
    )


class TestWinRates:
    def test_counts_are_arithmetic(self) -> None:
        rates = win_rates(battles("alpha", "beta", a_wins=3, b_wins=1))
        assert rates == {"alpha": 0.75, "beta": 0.25}

    def test_tie_is_half_a_win_each(self) -> None:
        rates = win_rates(battles("alpha", "beta", a_wins=0, b_wins=0, ties=4))
        assert rates == {"alpha": 0.5, "beta": 0.5}

    def test_models_are_listed_once_and_sorted(self) -> None:
        outcomes = battles("beta", "alpha", 1, 1)
        assert models_in(outcomes) == ["alpha", "beta"]


class TestBradleyTerry:
    def test_two_model_fit_matches_the_closed_form(self) -> None:
        """With one pair, the MLE strength ratio equals the win ratio exactly."""
        ratings = bradley_terry(battles("alpha", "beta", a_wins=3, b_wins=1), prior=0.0)
        assert ratings["alpha"] / ratings["beta"] == pytest.approx(3.0, rel=1e-6)

    def test_another_closed_form_ratio(self) -> None:
        ratings = bradley_terry(battles("alpha", "beta", a_wins=9, b_wins=3), prior=0.0)
        assert ratings["alpha"] / ratings["beta"] == pytest.approx(3.0, rel=1e-6)

    def test_even_record_gives_equal_strengths(self) -> None:
        ratings = bradley_terry(battles("alpha", "beta", a_wins=5, b_wins=5))
        assert ratings["alpha"] == pytest.approx(ratings["beta"])

    def test_normalised_to_geometric_mean_one(self) -> None:
        ratings = bradley_terry(battles("alpha", "beta", a_wins=3, b_wins=1))
        product = 1.0
        for value in ratings.values():
            product *= value
        assert product ** (1 / len(ratings)) == pytest.approx(1.0)

    def test_schedule_strength_is_accounted_for(self) -> None:
        """Raw win counts would tie these two; Bradley-Terry should not.

        `alpha` and `gamma` each win 2 and lose 0, but alpha beat the strong
        model and gamma beat the weak one.
        """
        outcomes = [
            *battles("alpha", "beta", 2, 0),
            *battles("beta", "delta", 2, 0),
            *battles("gamma", "delta", 2, 0),
        ]
        ratings = bradley_terry(outcomes)
        assert ratings["alpha"] > ratings["gamma"]

    def test_undefeated_model_still_converges(self) -> None:
        """Without the prior this fit runs away; the default keeps it finite."""
        ratings = bradley_terry(battles("alpha", "beta", a_wins=6, b_wins=0))
        assert ratings["alpha"] > ratings["beta"]
        assert all(0.0 < value < 1e6 for value in ratings.values())

    def test_empty_input(self) -> None:
        assert bradley_terry([]) == {}


class TestBootstrap:
    def test_is_reproducible_from_the_seed(self) -> None:
        """A published interval has to be reproducible exactly."""
        outcomes = battles("alpha", "beta", 7, 3)
        assert bootstrap_win_rate_intervals(
            outcomes, resamples=200, seed=1
        ) == bootstrap_win_rate_intervals(outcomes, resamples=200, seed=1)

    def test_different_seeds_give_different_intervals(self) -> None:
        outcomes = battles("alpha", "beta", 7, 3)
        assert bootstrap_win_rate_intervals(
            outcomes, resamples=200, seed=1
        ) != bootstrap_win_rate_intervals(outcomes, resamples=200, seed=2)

    def test_interval_brackets_the_point_estimate(self) -> None:
        outcomes = battles("alpha", "beta", 7, 3)
        rates = win_rates(outcomes)
        for name, (low, high) in bootstrap_win_rate_intervals(
            outcomes, resamples=500, seed=3
        ).items():
            assert low <= rates[name] <= high

    def test_bounds_stay_within_zero_and_one(self) -> None:
        for low, high in bootstrap_win_rate_intervals(
            battles("alpha", "beta", 5, 5), resamples=200, seed=4
        ).values():
            assert 0.0 <= low <= high <= 1.0

    def test_unanimous_record_gives_a_degenerate_interval(self) -> None:
        """Every resample of an all-wins record is still all wins."""
        intervals = bootstrap_win_rate_intervals(
            battles("alpha", "beta", 8, 0), resamples=200, seed=5
        )
        assert intervals["alpha"] == (1.0, 1.0)
        assert intervals["beta"] == (0.0, 0.0)

    def test_empty_input(self) -> None:
        assert bootstrap_win_rate_intervals([]) == {}


class TestTieGroups:
    def test_overlapping_intervals_are_one_group(self) -> None:
        """Ordering within an overlap would assert a difference the data lacks."""
        groups = tie_groups(
            {"alpha": 1.2, "beta": 1.0},
            {"alpha": (0.4, 0.8), "beta": (0.5, 0.9)},
        )
        assert groups["alpha"] == groups["beta"]

    def test_disjoint_intervals_are_separate_groups(self) -> None:
        groups = tie_groups(
            {"alpha": 2.0, "beta": 0.5},
            {"alpha": (0.7, 0.9), "beta": (0.1, 0.3)},
        )
        assert groups["alpha"] < groups["beta"]

    def test_touching_intervals_still_count_as_overlapping(self) -> None:
        groups = tie_groups(
            {"alpha": 1.5, "beta": 1.0},
            {"alpha": (0.5, 0.7), "beta": (0.3, 0.5)},
        )
        assert groups["alpha"] == groups["beta"]

    def test_groups_follow_rating_order(self) -> None:
        groups = tie_groups(
            {"alpha": 3.0, "beta": 2.0, "gamma": 1.0},
            {"alpha": (0.8, 0.9), "beta": (0.4, 0.6), "gamma": (0.1, 0.2)},
        )
        assert [groups[n] for n in ("alpha", "beta", "gamma")] == [0, 1, 2]

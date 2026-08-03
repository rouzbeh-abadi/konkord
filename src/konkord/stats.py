"""Ranking maths: Bradley-Terry ratings, bootstrap intervals, and tie groups.

Raw win counts are not a ranking. A model that only ever faced weak opponents
can outscore a better model that faced strong ones, and with an incomplete
schedule the two are not comparable at all. Bradley-Terry fits a strength per
model such that the observed results are most likely, which accounts for who
played whom.

Every function here is deterministic. The bootstrap takes an explicit seed so a
published interval can be reproduced exactly.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: A tie counts as half a win to each side. Bradley-Terry has no native notion
#: of a draw, and this is the conventional treatment.
_TIE_CREDIT = 0.5


@dataclass(frozen=True, slots=True)
class PairOutcome:
    """One resolved comparison: who faced whom, and who won."""

    model_a: str
    model_b: str
    winner: str | None  # a model name, or None for a tie


def models_in(outcomes: Sequence[PairOutcome]) -> list[str]:
    """Every model appearing in the outcomes, sorted for a stable result."""
    return sorted({m for outcome in outcomes for m in (outcome.model_a, outcome.model_b)})


def win_rates(outcomes: Sequence[PairOutcome]) -> dict[str, float]:
    """Share of comparisons each model won, counting a tie as half."""
    wins: dict[str, float] = {}
    played: dict[str, float] = {}
    for outcome in outcomes:
        for model in (outcome.model_a, outcome.model_b):
            played[model] = played.get(model, 0.0) + 1.0
            wins.setdefault(model, 0.0)
        if outcome.winner is None:
            wins[outcome.model_a] += _TIE_CREDIT
            wins[outcome.model_b] += _TIE_CREDIT
        else:
            wins[outcome.winner] += 1.0
    return {model: wins[model] / played[model] for model in sorted(wins)}


def bradley_terry(
    outcomes: Sequence[PairOutcome],
    *,
    prior: float = 0.5,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths, normalised to a geometric mean of 1.

    Uses the minorisation-maximisation update, which is monotone and needs no
    optimiser. `prior` adds a fictitious split result to every pair that was
    actually compared: without it, a model that won every one of its matchups
    has an unbounded strength and the fit does not converge. Pass `prior=0.0`
    to recover the plain maximum-likelihood estimate on well-mixed data.
    """
    names = models_in(outcomes)
    if not names:
        return {}
    index = {name: i for i, name in enumerate(names)}
    size = len(names)

    wins = np.zeros(size, dtype=float)
    played = np.zeros((size, size), dtype=float)
    for outcome in outcomes:
        i, j = index[outcome.model_a], index[outcome.model_b]
        played[i, j] += 1.0
        played[j, i] += 1.0
        if outcome.winner is None:
            wins[i] += _TIE_CREDIT
            wins[j] += _TIE_CREDIT
        else:
            wins[index[outcome.winner]] += 1.0

    if prior > 0:
        faced = played > 0
        played[faced] += 2 * prior
        wins += prior * faced.sum(axis=1)

    strengths = np.ones(size, dtype=float)
    for _ in range(max_iterations):
        denominators = np.zeros(size, dtype=float)
        for i in range(size):
            others = np.arange(size) != i
            sums = strengths[i] + strengths
            denominators[i] = np.sum(played[i, others] / sums[others])
        updated = np.where(denominators > 0, wins / np.maximum(denominators, 1e-300), strengths)
        updated = np.maximum(updated, 1e-300)
        updated /= np.exp(np.mean(np.log(updated)))  # geometric mean of 1
        if np.max(np.abs(updated - strengths)) < tolerance:
            strengths = updated
            break
        strengths = updated

    return {name: float(strengths[index[name]]) for name in names}


def bootstrap_win_rate_intervals(
    outcomes: Sequence[PairOutcome],
    *,
    resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap intervals for each model's win rate.

    Resamples comparisons with replacement. A model absent from a resample
    contributes nothing to that draw rather than a zero, since a model that was
    not compared has no win rate, and scoring it zero would drag the interval
    down for a reason that has nothing to do with its quality.
    """
    names = models_in(outcomes)
    if not names or not outcomes:
        return dict.fromkeys(names, (0.0, 0.0))

    rng = np.random.default_rng(seed)
    draws: defaultdict[str, list[float]] = defaultdict(list)
    count = len(outcomes)
    for _ in range(resamples):
        picks = rng.integers(0, count, size=count)
        sample = [outcomes[int(p)] for p in picks]
        for name, rate in win_rates(sample).items():
            draws[name].append(rate)

    tail = (1.0 - confidence) / 2.0
    intervals: dict[str, tuple[float, float]] = {}
    for name in names:
        values = draws[name]
        if not values:
            intervals[name] = (0.0, 0.0)
            continue
        low, high = np.quantile(values, [tail, 1.0 - tail])
        intervals[name] = (float(low), float(high))
    return intervals


def tie_groups(
    ratings: dict[str, float],
    intervals: dict[str, tuple[float, float]],
) -> dict[str, int]:
    """Assign a rank group per model, joining models whose intervals overlap.

    Models sharing a group must be rendered as tied. Presenting an order within
    a group would assert a difference the data does not support, which is the
    single most common way a leaderboard misleads.
    """
    ordered = sorted(ratings, key=lambda name: (-ratings[name], name))
    groups: dict[str, int] = {}
    group = 0
    leader: str | None = None
    for name in ordered:
        if leader is None:
            leader = name
        else:
            low, high = intervals.get(name, (0.0, 0.0))
            leader_low, leader_high = intervals.get(leader, (0.0, 0.0))
            if low > leader_high or high < leader_low:  # disjoint
                group += 1
                leader = name
        groups[name] = group
    return groups

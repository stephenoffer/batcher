"""The join-strategy bandit's reward must be stationary, and its radius commensurate.

`ucb1_best_arm` minimizes a latency by lower-confidence bound. Two properties have to hold
for that to be regret-minimizing rather than merely greedy:

1. **The reward is stationary per arm.** UCB1 assumes each arm draws from a fixed
   distribution. Raw wall time is not: the same plan signature runs over 1M rows today and
   50M tomorrow, so an arm that happened to be sampled on the large input carries a
   permanently inflated mean. `record_join_strategy` divides by the join's input size.
2. **The confidence radius is in reward units.** Textbook UCB1 assumes rewards in `[0, 1]`.
   Against an unbounded millisecond mean, a bare `sqrt(2*ln N / n)` radius is a fraction of
   a percent — the bandit is greedy in all but name. Scaling by the pooled spread makes the
   exploration constant dimensionless.

The tests below pin both, and pin the boundary between them: the scaled radius deliberately
does *not* resurrect an arm whose mean is genuinely worse.
"""

from __future__ import annotations

import math
import random

import pytest

from batcher.kyber.learned_tuning import _reward_scale, ucb1_best_arm

pytestmark = pytest.mark.unit

_ARMS = ("hash", "sort_merge")


def _arm(n: int, mean: float) -> dict:
    """`n` identical observations of `mean` — a Welford state with no spread."""
    return {"n": float(n), "mean": mean, "m2": 0.0}


def _pull(stats: dict, arm: str, reward: float) -> None:
    """Fold one reward into an arm, exactly as `record_arm` does (Welford, no discount)."""
    a = stats[arm]
    a["n"] += 1.0
    delta = reward - a["mean"]
    a["mean"] += delta / a["n"]
    a["m2"] += delta * (reward - a["mean"])


def _bare_radius_pick(stats: dict, c: float = 1.0) -> str:
    """The pre-fix rule: identical to `ucb1_best_arm` but with no reward scale on the radius."""
    tried = {a: s for a, s in stats.items() if a in _ARMS and s["n"] > 0}
    total = sum(s["n"] for s in tried.values())
    untried = sorted(a for a in _ARMS if a not in tried)
    if untried and total >= len(tried):
        return untried[0]
    return min(
        sorted(tried),
        key=lambda a: (
            tried[a]["mean"] - c * math.sqrt(2.0 * math.log(max(2, total)) / tried[a]["n"])
        ),
    )


def _simulate(pick, *, unit: float, rounds: int = 80, seed: int = 11, noise: float = 0.15):
    """`hash` costs 2 ms/Mrow, `sort_merge` 4. The first run sees a 50x larger input.

    Returns cumulative regret in ms and the per-arm pull counts.
    """
    rng = random.Random(seed)
    stats = {a: {"n": 0.0, "mean": 0.0, "m2": 0.0} for a in _ARMS}
    truth = {"hash": 2.0, "sort_merge": 4.0}
    sizes = [50.0] + [rng.choice([1.0, 2.0, 3.0]) for _ in range(rounds)]
    regret = 0.0
    for mrows in sizes:
        arm = pick(stats) if any(s["n"] for s in stats.values()) else "hash"
        wall = truth[arm] * mrows * (1 + rng.gauss(0, noise))
        regret += (truth[arm] - truth["hash"]) * mrows
        _pull(stats, arm, (wall / mrows) * unit)  # size-normalized reward
    return regret, {a: stats[a]["n"] for a in _ARMS}


def test_untried_arm_is_explored_before_any_ranking():
    stats = {"hash": _arm(2, 10.0)}
    assert ucb1_best_arm(stats, _ARMS) == "sort_merge"


def test_no_tried_arm_defers_to_the_cost_model():
    assert ucb1_best_arm({}, _ARMS) is None
    assert ucb1_best_arm({"hash": _arm(0, 0.0)}, _ARMS) is None


def test_reward_scale_is_the_pooled_standard_deviation():
    stats = {"hash": _arm(1, 600.0), "sort_merge": _arm(1, 10.0)}
    # Pooled over both observations: mean 305, variance 295^2.
    assert _reward_scale(stats) == pytest.approx(295.0)


def test_reward_scale_floors_on_a_zero_spread_history():
    """Every run identical => sd 0 => a bare radius would freeze the bandit forever."""
    stats = {"hash": _arm(5, 10.0), "sort_merge": _arm(1, 10.0)}
    assert _reward_scale(stats) == pytest.approx(2.5)  # 0.25 * mean
    assert _reward_scale({}) == 0.0


def test_arm_ranking_is_invariant_to_the_reward_unit():
    """The property a bare radius lacks: ms, ms-per-row and microseconds must rank alike."""
    results = [_simulate(lambda s: ucb1_best_arm(s, _ARMS), unit=u) for u in (1.0, 1e-6, 1e3)]
    regrets = {r for r, _ in results}
    pulls = {tuple(sorted(p.items())) for _, p in results}
    assert len(regrets) == 1, f"regret changed with the reward's unit: {regrets}"
    assert len(pulls) == 1, f"arm pulls changed with the reward's unit: {pulls}"


def test_a_bare_radius_is_unit_sensitive_and_the_scaled_one_is_not():
    """Pins *why* the scale exists, by exhibiting the failure it removes."""
    bare = [_simulate(_bare_radius_pick, unit=u)[0] for u in (1.0, 1e-6)]
    scaled = [_simulate(lambda s: ucb1_best_arm(s, _ARMS), unit=u)[0] for u in (1.0, 1e-6)]
    assert bare[0] != bare[1], "expected the un-scaled radius to depend on the unit"
    assert scaled[0] == scaled[1]


def test_size_normalized_reward_finds_the_faster_arm_despite_a_large_first_input():
    """The whole point: a 50x input on round 0 must not condemn the arm that drew it."""
    regret, pulls = _simulate(lambda s: ucb1_best_arm(s, _ARMS), unit=1.0)
    assert pulls["hash"] > pulls["sort_merge"] * 10, f"bandit did not converge to hash: {pulls}"
    assert regret < 20.0, f"regret {regret} — the bandit is not converging"


def test_the_scaled_radius_does_not_resurrect_a_genuinely_worse_arm():
    """The boundary of the fix, stated honestly.

    A scale-aware radius is not a cure for non-stationarity. Once an arm's *recorded* mean is
    far worse, the pooled spread decays as 1/sqrt(N) — faster than the gap closes — so UCB
    stops paying to re-check it. That is correct under UCB1's stationarity assumption; making
    the reward actually stationary is `record_join_strategy`'s job, not the radius's.
    """
    stats = {"hash": _arm(1, 600.0), "sort_merge": _arm(1, 10.0)}
    for _ in range(400):
        arm = ucb1_best_arm(stats, _ARMS)
        _pull(stats, arm, 5.0 if arm == "hash" else 10.0)
    assert stats["hash"]["n"] == 1.0, "expected the poisoned arm to stay retired"


def test_record_join_strategy_normalizes_by_input_rows():
    # Patched on the defining module (`learned_tuning.bandit`), which is where
    # `record_join_strategy` resolves `record_arm` from.
    from batcher.kyber.learned_tuning import bandit

    recorded: list[tuple[str, float]] = []
    original = bandit.record_arm
    try:
        bandit.record_arm = lambda hub, ns, key, arm, reward: recorded.append((arm, reward))
        # 40 ms over 2M input rows => 20 ms per million rows.
        bandit.record_join_strategy(object(), "sig", "hash", 40.0, 2_000_000.0)
        # An unknown input size must fall back to the raw wall time, not divide by zero.
        bandit.record_join_strategy(object(), "sig", "hash", 40.0, 0.0)
    finally:
        bandit.record_arm = original
    assert recorded == [("hash", 20.0), ("hash", 40.0)]


def test_discounting_lets_a_changed_arm_be_re_examined():
    """The non-stationary case UCB1 cannot handle, and discounting exists for.

    An arm measured badly long ago keeps its mean under undiscounted statistics, and its
    confidence radius shrinks to nothing, so it can never be re-checked even after the world
    changed. Discounting decays the *evidence* (the effective count and the accumulated
    spread) while keeping the estimate, so the radius reopens over time.
    """
    from batcher.kyber.learned_tuning.bandit import _ARM_DISCOUNT, _decayed

    state = {"n": 200.0, "mean": 600.0, "m2": 400.0}
    for _ in range(200):
        state = _decayed(state)
    assert state["mean"] == 600.0, "the estimate itself must not drift toward zero"
    assert state["n"] < 200.0 * _ARM_DISCOUNT**100, "evidence must decay geometrically"
    assert state["n"] > 0.0


def test_a_consistent_arm_is_not_explored_as_hard_as_an_erratic_one():
    """UCB-V: the radius uses the arm's *own* spread, not the pooled one.

    Two arms with the same mean, one measured identically every time and one wildly noisy.
    The consistent one has nothing left to learn, so its lower-confidence bound must sit
    above the noisy one's — the noisy arm is the one worth another look.
    """
    consistent = {"n": 20.0, "mean": 100.0, "m2": 0.0}
    erratic = {"n": 20.0, "mean": 100.0, "m2": 20.0 * 400.0}  # sd 20
    assert ucb1_best_arm({"hash": consistent, "sort_merge": erratic}, _ARMS) == "sort_merge"


def test_variance_is_read_without_catastrophic_cancellation():
    """A tight spread on a large mean must survive.

    `sumsq/n - mean^2` at mean 1e6 with sd 1 subtracts two numbers that agree to 12 digits,
    and the answer is noise. The Welford `m2` never forms that difference.
    """
    stats = {a: {"n": 0.0, "mean": 0.0, "m2": 0.0} for a in _ARMS}
    for i in range(200):
        _pull(stats, "hash", 1e6 + (i % 2))
    sd = math.sqrt(stats["hash"]["m2"] / stats["hash"]["n"])
    assert 0.4 < sd < 0.6, f"recovered sd {sd}, expected ~0.5"

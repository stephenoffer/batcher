#!/usr/bin/env python3
"""Draw `bandit_tuning.svg` -- the learned-tuning bandit: arms, reward, and the bound
on exploration.

Source, all of it `python/batcher/kyber/learned_tuning/bandit.py` unless said otherwise.
Every number below is a module constant; if one moves, move the drawing.

* Arms. `JOIN_ARMS = ("hash", "broadcast", "sort_merge")` and the two execution routes
  `("one_shot", "staged")`. The tuple order is load-bearing: `ucb1_best_arm` takes the
  first untried arm, so it decides which arm a cold signature probes first.
* Reward. A measured latency in milliseconds, folded in per arm as a discounted Welford
  state `(n, mean, m2)` keyed by plan signature -- and, for anything in machine units,
  scoped by hardware fingerprint (`metadata/hardware_scope.py`). Lower is better; the
  bandit minimizes.
* The bound on exploration, which is the half a bandit diagram usually omits:
  - `_MIN_ARM_TOTAL = 3` -- below three observations `learned_arm` returns `None` and the
    caller keeps the cost model's choice. (The two-arm route bandit uses
    `_MIN_ROUTE_TOTAL = 1`, because each of its samples is a whole query.)
  - an untried arm is given a turn only once `total >= len(tried)`.
  - the pick is `mean - c * scale * sqrt(2 ln N / n)` with `_UCB_C = 1.0`, `scale` the
    arm's own spread (UCB-V) floored at `_UCB_SCALE_FLOOR = 0.25` of the pooled spread.
  - `_ARM_DISCOUNT = 0.975` per observation, an effective horizon of about
    `1/(1-gamma)` = 40 runs, which is what lets a genuinely changed arm be re-examined
    instead of frozen out by a confidence radius that has shrunk to nothing.
* Determinism. No RNG anywhere; ties break by arm name, so a plan is reproducible.
* Result-invariance (`learned_tuning/__init__.py`): every arm emits the same relation, so
  the worst a wrong learned value can do is cost throughput. That is what makes
  exploration safe, and it is why it is drawn.
"""

from __future__ import annotations

from _authoring import arrow, band, card, curve, label, note, svg, write

W, H = 980, 620

body = [
    band(20, 20, 940, 210, "THE ARMS, AND WHAT IS MEASURED", "blue"),
    card(44, 84, 280, 90, "Join strategy", "hash / broadcast / sort_merge"),
    card(350, 84, 280, 90, "Execution route", "one_shot / staged"),
    card(656, 84, 280, 90, "The reward", "measured latency, minimized"),
    note(490, 206, "Every arm emits the same relation. A wrong pick costs throughput, never correctness -- which is what makes exploring safe at all.", anchor="middle"),

    card(44, 290, 260, 96, "Under 3 observations?", "yes: the cost model decides"),
    card(360, 290, 260, 96, "An arm never tried?", "yes: give it exactly one turn"),
    card(676, 290, 260, 96, "Otherwise", "the lowest bound wins"),
    arrow(304, 338, 360, 338, "blue"),
    label(332, 328, "no", anchor="middle"),
    arrow(620, 338, 676, 338, "blue"),
    label(648, 328, "no", anchor="middle"),
    note(490, 412, "mean - 1.0 x spread x sqrt(2 ln N / n)", anchor="middle"),
    note(490, 430, "ties break by name. No RNG anywhere, so a plan is reproducible.", anchor="middle"),

    card(676, 452, 260, 84, "Evidence decays", "0.975 per observation"),
    arrow(806, 386, 806, 452, "amber"),
    label(818, 424, "record the ms", anchor="start"),
    note(806, 560, "an effective horizon of about 40 runs", anchor="middle"),
    curve(676, 516, 420, 580, 176, 394, "amber"),
    label(420, 548, "so an arm that got faster is asked again", anchor="middle"),

    note(490, 600, "Statistics are per plan signature, and anything in machine units is additionally scoped by hardware fingerprint, so unlike machines never blend.", anchor="middle"),
]

write("bandit_tuning", svg(W, H, "".join(body)))
print("wrote bandit_tuning.svg")

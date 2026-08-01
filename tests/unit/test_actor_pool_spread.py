"""An inference actor pool must spread its partitions before it stacks them.

The pool drivers assigned work to "the first actor with a free in-flight slot", which
fills actor 0 to its submit depth before actor 1 receives anything. An inference stage's
partitions are few and wide, so the partition count is routinely at or below
``len(actors) x depth`` — and then the tail of the pool never runs.

The lever that made it worst is the one meant to help: `recommend_inflight_depth` raises
the depth precisely when a GPU looks starved. Measured on four T4s with four partitions,
depth 2 left two GPUs at 0%; at depth 4 a single actor ran the entire stage while three
GPUs idled, at a throughput high enough to look healthy.

These tests pin the assignment order itself, so they stay meaningful however the pool is
driven.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.map import _emptiest_actor

pytestmark = pytest.mark.unit


def _assign(n_actors: int, n_partitions: int, depth: int) -> list[int]:
    """Partitions landing on each actor, replaying the drivers' assignment loop.

    Models the worst case for spreading: nothing completes, so no slot is ever returned —
    which is exactly the window in which every actor should have been given work.
    """
    actors = [f"a{i}" for i in range(n_actors)]
    slots = dict.fromkeys(actors, depth)
    landed = dict.fromkeys(actors, 0)
    for _ in range(n_partitions):
        actor = _emptiest_actor(actors, slots)
        if actor is None:
            break
        slots[actor] -= 1
        landed[actor] += 1
    return [landed[a] for a in actors]


def test_every_actor_gets_work_before_any_gets_seconds() -> None:
    assert _assign(n_actors=4, n_partitions=4, depth=2) == [1, 1, 1, 1]


def test_a_deep_submit_window_no_longer_starves_the_fleet() -> None:
    """The regression: at depth 4 this put all four partitions on one actor."""
    assert _assign(n_actors=4, n_partitions=4, depth=4) == [1, 1, 1, 1]


def test_depth_still_stacks_once_every_actor_is_busy() -> None:
    """Spreading must not cost the submit-ahead depth its purpose."""
    assert _assign(n_actors=4, n_partitions=8, depth=2) == [2, 2, 2, 2]


def test_a_partial_round_spreads_across_distinct_actors() -> None:
    landed = _assign(n_actors=4, n_partitions=2, depth=4)
    assert sorted(landed) == [0, 0, 1, 1], f"two partitions shared an actor: {landed}"


def test_the_pool_stops_when_every_slot_is_taken() -> None:
    assert sum(_assign(n_actors=2, n_partitions=99, depth=3)) == 6


def test_an_empty_pool_has_nothing_to_assign() -> None:
    assert _emptiest_actor([], {}) is None

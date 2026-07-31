"""A task whose host is leaving gets backed up immediately, not once it looks slow.

The straggler heuristic answers "is duplicating this worth the resource?" by watching for a
task that runs far past the median. That question is already settled for a task on a node
being reclaimed: the copy is going to be needed. Making it wait to also look slow spends the
notice period, which is the one window in which a backup can still finish somewhere else,
and then the loss costs a full recovery round instead of a duplicate already in flight.

`max_backups` still caps the total, because a correlated preemption dooms many slots at
once and that is exactly when unbounded duplicate load is least affordable.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience import SpeculationPolicy, stragglers_to_backup

pytestmark = pytest.mark.unit


def _policy(**kw) -> SpeculationPolicy:
    base = {"straggler_factor": 1.5, "min_finished_frac": 0.75, "max_backups": 1}
    return SpeculationPolicy(**{**base, **kw})


class TestDoomedBypassesTheHeuristic:
    def test_doomed_is_backed_up_with_nothing_finished(self):
        """The relative test needs a median and so cannot fire early. A doomed task never
        needed one, and early is the only time a backup helps it."""
        picked = stragglers_to_backup(
            n=4, finished={}, elapsed={0: 0.1, 1: 0.1}, policy=_policy(), doomed=frozenset({1})
        )
        assert picked == [1]

    def test_doomed_is_backed_up_below_the_min_elapsed_floor(self):
        """The absolute floor exists to stop duplicating a 8 ms task to save 3 ms. A doomed
        task is not being duplicated to save time, but to survive."""
        picked = stragglers_to_backup(
            n=4,
            finished={2: 5.0, 3: 5.0, 0: 5.0},
            elapsed={1: 0.001},
            policy=_policy(min_elapsed_s=1.0),
            doomed=frozenset({1}),
        )
        assert picked == [1]

    def test_doomed_outranks_a_slower_straggler(self):
        """A scarce backup slot goes to the task that will certainly need it."""
        picked = stragglers_to_backup(
            n=4,
            finished={2: 1.0, 3: 1.0},
            elapsed={0: 100.0, 1: 2.0},
            policy=_policy(max_backups=1),
            doomed=frozenset({1}),
        )
        assert picked == [1], "the doomed slot must win the only backup slot"

    def test_both_are_taken_when_there_is_room(self):
        # `min_finished_frac=0.5` so the straggler branch has actually engaged (2 of 4
        # finished); at the 0.75 default it would legitimately not have, and the doomed
        # slot alone would be the correct answer.
        picked = stragglers_to_backup(
            n=4,
            finished={2: 1.0, 3: 1.0},
            elapsed={0: 100.0, 1: 2.0},
            policy=_policy(max_backups=2, min_finished_frac=0.5),
            doomed=frozenset({1}),
        )
        assert picked == [1, 0], "doomed first, then the straggler"

    def test_the_cap_still_bounds_a_preemption_wave(self):
        """Many doomed slots at once is exactly when duplicate load is least affordable."""
        picked = stragglers_to_backup(
            n=8,
            finished={},
            elapsed=dict.fromkeys(range(6), 1.0),
            policy=_policy(max_backups=2),
            doomed=frozenset(range(6)),
        )
        assert len(picked) == 2

    def test_doomed_slots_are_ordered_by_elapsed(self):
        """Among equally-doomed slots, the one with most work invested goes first."""
        picked = stragglers_to_backup(
            n=8,
            finished={},
            elapsed={0: 1.0, 1: 9.0, 2: 5.0},
            policy=_policy(max_backups=3),
            doomed=frozenset({0, 1, 2}),
        )
        assert picked == [1, 2, 0]

    def test_a_doomed_slot_is_never_listed_twice(self):
        """It qualifies on both counts when it is also slow; it must still be one entry."""
        picked = stragglers_to_backup(
            n=4,
            finished={2: 1.0, 3: 1.0},
            elapsed={1: 100.0},
            policy=_policy(max_backups=3),
            doomed=frozenset({1}),
        )
        assert picked == [1]


class TestUnchangedWithoutDoomedSlots:
    """Every pre-existing behavior must survive, since this path runs on every barrier."""

    def test_speculation_off_backs_up_nothing(self):
        assert (
            stragglers_to_backup(
                n=4,
                finished={0: 1.0},
                elapsed={1: 100.0},
                policy=_policy(max_backups=0),
                doomed=frozenset({1}),
            )
            == []
        )

    def test_nothing_finished_backs_up_nothing(self):
        assert stragglers_to_backup(n=4, finished={}, elapsed={0: 99.0}, policy=_policy()) == []

    def test_below_the_finished_fraction_backs_up_nothing(self):
        assert (
            stragglers_to_backup(n=4, finished={0: 1.0}, elapsed={1: 99.0}, policy=_policy()) == []
        )

    def test_a_plain_straggler_is_still_picked(self):
        picked = stragglers_to_backup(
            n=4, finished={0: 1.0, 1: 1.0, 2: 1.0}, elapsed={3: 99.0}, policy=_policy()
        )
        assert picked == [3]

    def test_a_uniform_stage_backs_up_nothing(self):
        picked = stragglers_to_backup(
            n=4, finished={0: 5.0, 1: 5.0, 2: 5.0}, elapsed={3: 5.1}, policy=_policy()
        )
        assert picked == []

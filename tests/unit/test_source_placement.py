"""`SourcePlacement` — a relocated source's host is not its source id.

On a clean run source `s` is published by worker `s`, so a recovery loop can use one
number for both, and the flat reduce path did. The two diverge permanently after the
first recompute, and code that still conflates them marks the *original* (already-dead)
worker on the next failure while the genuinely dead host keeps looking alive — so
`_pick_live` hands it out again and the recovery budget drains on a host that cannot
answer, failing the stage with `ResourceError` while survivors sit idle.

These pin the identity-until-relocated behavior (so a clean run is unchanged) and the
divergence after it (the bug). `test_second_failure_of_a_relocated_source_marks_its_real
_host` is the one that fails against the old `dead.add(src)`.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience import SourcePlacement

pytestmark = pytest.mark.unit


def test_a_clean_run_is_the_identity() -> None:
    """Nothing relocated ⇒ every source is on its own worker, exactly as before."""
    placement = SourcePlacement(4)

    assert [placement.host_of(s) for s in range(4)] == [0, 1, 2, 3]
    assert placement.sources_on(2) == {2}


def test_relocation_moves_the_host_and_only_that_source() -> None:
    placement = SourcePlacement(4)

    placement.relocate(1, 3)

    assert placement.host_of(1) == 3
    assert [placement.host_of(s) for s in (0, 2, 3)] == [0, 2, 3]


def test_a_relocated_source_leaves_its_old_host_holding_nothing() -> None:
    """The vacated worker must stop being blamed for a source it no longer holds."""
    placement = SourcePlacement(4)

    placement.relocate(1, 3)

    assert placement.sources_on(1) == set()
    assert placement.sources_on(3) == {1, 3}  # its own, plus the one that moved in


def test_second_failure_of_a_relocated_source_marks_its_real_host() -> None:
    """The regression: the host that dies is where the source *is*, not its id.

    Worker 1 dies, so source 1 is recomputed onto worker 3. Worker 3 then dies too. The
    recovery loop must mark worker 3 — the host actually holding source 1. The old code
    computed `dead.add(src)` = worker 1, which was already dead, so worker 3 stayed
    "live", `_pick_live` kept returning it, and every remaining attempt was spent
    republishing onto a corpse.
    """
    placement = SourcePlacement(4)
    dead = {1}
    placement.relocate(1, 3)

    dead.add(placement.host_of(1))

    assert dead == {1, 3}
    assert 3 in dead, "the host actually holding source 1 must be marked dead"


def test_a_dead_host_loses_every_source_it_accumulated() -> None:
    """One host can hold several sources after a preemption wave; its death loses all.

    This is why the reduce stage translates a dead *host* into sources through the
    placement rather than assuming the host id is the source id: after two relocations
    onto the same survivor, one death costs three recomputes, not one.
    """
    placement = SourcePlacement(4)
    placement.relocate(0, 2)
    placement.relocate(1, 2)

    assert placement.sources_on(2) == {0, 1, 2}


def test_a_source_can_be_relocated_repeatedly() -> None:
    """A churning spot cluster moves the same source more than once; only the last counts."""
    placement = SourcePlacement(4)

    placement.relocate(0, 1)
    placement.relocate(0, 2)

    assert placement.host_of(0) == 2
    assert placement.sources_on(1) == {1}
    assert placement.sources_on(2) == {0, 2}

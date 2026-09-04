"""Two materialized combiner trees in one query must not publish at the same ticket stage.

A published bucket is addressed by `(plan, stage, src, dst, epoch)`. The plan id fences one
query from another, so within a query the *stage* is the only thing separating two
intermediates -- which is why `fleet.plan_id.next_result_stage` exists and why the flat
reduce mints one per published result.

The combiner tree did not. `_tree_reduce(materialize=True)` called
`combine_finalize_publish` without a `result_stage`, taking the parameter's default, so
**every** tree-reduced intermediate in the process published at the literal stage 100. Two of
them in one query were byte-identical tickets on the same worker and the second overwrote the
first.

It reproduced on `left.join(right)` over two `group_by` aggregates at 12 workers and not at 8
-- `shuffle_fan_in` is what selects the tree -- and surfaced three frames from the cause, as
`KeyError: Field "a" does not exist in schema` when the join projected the survivor's columns
off the wrong bucket. This test needs no cluster: it reads the stage off the call.
"""

from __future__ import annotations

import pytest

from batcher.dist import flight_aggregate

pytestmark = pytest.mark.unit


class _Recorder:
    """A stub fleet actor. Every `.remote(...)` records its arguments and returns a marker."""

    def __init__(self, calls: list[tuple[str, tuple]]) -> None:
        self._calls = calls

    def __getattr__(self, name: str):
        calls = self._calls

        class _Method:
            @staticmethod
            def remote(*args, **kwargs):
                calls.append((name, args))
                return ("ok", None)

        return _Method


def _published_stages(monkeypatch, workers: int = 12, reducers: int = 3) -> list[int]:
    """Run one materialized `_tree_reduce` and return the `result_stage` of each root call."""
    calls: list[tuple[str, tuple]] = []
    actors = [_Recorder(calls) for _ in range(workers)]

    # The tree's own arithmetic is exercised for real; only the fan-out around it is stubbed.
    monkeypatch.setattr(flight_aggregate, "placement_probe", lambda *a, **k: None)
    monkeypatch.setattr(flight_aggregate, "replicate_interior_outputs", lambda *a, **k: None)
    monkeypatch.setattr(
        flight_aggregate, "gather_in_windows", lambda launch, items, _w: [launch(i) for i in items]
    )

    flight_aggregate._tree_reduce(
        actors,
        [f"addr-{i}" for i in range(workers)],
        reducers,
        None,
        None,
        fan_in=8,
        workers=workers,
        stage_base=flight_aggregate.next_stage_base(17),
        materialize=True,
    )
    # Read positionally, falling back to the parameter's own default when the call passed
    # nothing -- so the pre-fix state reads as "stage 100" here rather than an `IndexError`,
    # and each assertion below fails for the reason it names.
    from batcher.dist.flight_worker import _RESULT_STAGE

    return [
        args[5] if len(args) > 5 else _RESULT_STAGE
        for name, args in calls
        if name == "combine_finalize_publish"
    ]


def test_every_bucket_of_one_result_shares_one_stage(monkeypatch):
    """The other half of the contract: a *single* result's buckets must agree, or the reader
    that walks its handles addresses a stage nothing published."""
    stages = _published_stages(monkeypatch)
    assert stages, "the tree published no bucket, so this test measured nothing"
    assert len(set(stages)) == 1, f"one result split across stages {sorted(set(stages))}"


def test_two_results_never_share_a_stage(monkeypatch):
    first = _published_stages(monkeypatch)
    second = _published_stages(monkeypatch)
    assert not set(first) & set(second), (
        f"two published intermediates collided on stage {sorted(set(first) & set(second))}"
    )


def test_the_stage_is_minted_not_the_worker_default(monkeypatch):
    """Pins the actual defect rather than only its consequence. `next_result_stage` never
    returns the worker-side default, so a call that lands on it is one that passed nothing."""
    from batcher.dist.flight_worker import _RESULT_STAGE

    assert _RESULT_STAGE not in set(_published_stages(monkeypatch))

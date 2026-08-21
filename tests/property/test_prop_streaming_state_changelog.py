"""Property: a changelog checkpoint reconstructs exactly the state a whole snapshot holds.

This needs no oracle. It is a law of the mergeable algebra: `combine` is associative and
commutative (invariant #7), so combining a base snapshot with every partial recorded after
it is the same state as folding those partials in directly. Incremental checkpointing is
that law applied to durability, and if the law holds for a schedule the hand-written tests
did not think of, the feature holds for it too.

Hypothesis searches the two axes those tests fix by hand: the **shape of the stream** (how
many micro-batches, how many rows each, how many distinct group keys) and the **schedule**
(which epochs snapshot whole and which record a delta). The engine picks the schedule from a
size rule and an interval; here it is chosen adversarially, so a chain of length one and a
chain of length twenty are both reachable, as is a snapshot on the very last epoch and a
delta on it.

A counterexample is unambiguously a bug in the chain, not a disagreement with an oracle —
and it is the kind of bug that produces a short total rather than an error, which is why it
is worth searching for rather than waiting to observe.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher.core.mergeable import RunningAggregate
from batcher.io.formats.streaming.checkpoint.store import CheckpointStore

pytestmark = pytest.mark.property


def _fold() -> RunningAggregate:
    """A running `sum`/`count` grouped by `k`, the shape a changelog is written for."""
    plan = (
        bt.from_pydict({"k": [0], "v": [0]})
        .group_by("k")
        .agg(total=bt.col("v").sum(), n=bt.col("v").count())
    )
    return RunningAggregate(plan._plan)


def _finalized(state: pa.RecordBatch | None) -> list[tuple[int, int, int]]:
    """A running state's finalized rows as sorted ``(k, total, n)`` triples."""
    if state is None:
        return []
    fold = _fold()
    fold.restore(state)
    result = fold.finalize()
    if result is None:
        return []
    out = result.to_pydict()
    return sorted(zip(out["k"], out["total"], out["n"], strict=True))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    batches=st.lists(
        st.lists(
            st.tuples(st.integers(min_value=0, max_value=4), st.integers(-50, 50)),
            min_size=0,
            max_size=6,
        ),
        min_size=1,
        max_size=12,
    ),
    schedule=st.lists(st.booleans(), min_size=1, max_size=12),
)
def test_a_chain_reconstructs_the_state_a_whole_snapshot_would(tmp_path_factory, batches, schedule):
    """For any stream shape and any snapshot/delta schedule, the two agree.

    `schedule[i]` is True to snapshot epoch `i` whole and False to record its delta. The
    first eligible epoch always snapshots regardless, because a chain with no base under it
    is not a state the store can be asked to rebuild.
    """
    store = CheckpointStore(str(tmp_path_factory.mktemp("ckpt")))
    live = _fold()
    wrote_base = False

    for index, rows in enumerate(batches):
        batch = pa.record_batch(
            {
                "k": pa.array([k for k, _ in rows], type=pa.int64()),
                "v": pa.array([v for _, v in rows], type=pa.int64()),
            }
        )
        delta = live.push([batch]) if batch.num_rows else None
        whole = schedule[index % len(schedule)] or not wrote_base or delta is None
        if whole:
            state = live.state()
            if state is not None:
                store.state.snapshot(index, state)
                wrote_base = True
        else:
            store.state.snapshot_delta(index, delta)
        store.state.prune(index)

    chain = store.state.restore_chain(len(batches) - 1)
    replayed = _fold()
    replayed.combine_all(chain)

    assert _finalized(replayed.state()) == _finalized(live.state())


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    partials=st.lists(
        st.lists(st.tuples(st.integers(0, 3), st.integers(-20, 20)), min_size=1, max_size=5),
        min_size=1,
        max_size=8,
    )
)
def test_combining_partials_in_any_order_gives_one_state(partials):
    """The commutativity the chain rests on, stated directly.

    The store replays a chain oldest-first because that is legible to a reader, not because
    order matters. If it ever did, a recovery that listed files differently — an object
    store's ordering is not a filesystem's — would silently produce a different aggregate.
    """
    built = []
    for rows in partials:
        fold = _fold()
        fold.push(
            [
                pa.record_batch(
                    {
                        "k": pa.array([k for k, _ in rows], type=pa.int64()),
                        "v": pa.array([v for _, v in rows], type=pa.int64()),
                    }
                )
            ]
        )
        state = fold.state()
        if state is not None:
            built.append(state)

    forward, backward = _fold(), _fold()
    forward.combine_all(built)
    backward.combine_all(list(reversed(built)))

    assert _finalized(forward.state()) == _finalized(backward.state())

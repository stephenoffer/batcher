"""Skew salting must never engage when the shuffle reducer finalizes an aggregate.

`_distributed_join_aggregate` fuses a post-join aggregate into the join's reducer and
finalizes each bucket locally. That is correct only because co-partitioning by the join
key puts every group in exactly one bucket. Salting deliberately spreads a hot key across
several buckets, which breaks that precondition — each salted reducer would finalize a
*partial* group and the union would carry several half-summed rows for the hot key
instead of one correct row. No error; just a wrong answer.

The guard matters on the **default** path, not only under an opt-in: `skew_join_salt`
defaults to 0, but a measured hot key engages salting anyway
(`if hot and salt <= 0: salt = DEFAULT_LEARNED_SALT`).

These are unit tests over the pure predicate rather than an end-to-end distributed run,
and that is deliberate. Reaching the hazard end-to-end needs the disk transport AND a
fusable join-aggregate AND a detected hot key to line up; `resolve_transport` picks Flight
on any multi-node cluster, and `docs/internals/databricks_parity.md` records that Flight
join behaviour "cannot be validated in a single-node dev environment." An integration test
written here passes whether or not the guard exists, which is worse than no test — so the
invariant is pinned where it can actually be checked.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist.executors.join import salting_is_safe
from batcher.plan.logical import Join, Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([("k", pa.int64()), ("a", pa.int64()), ("b", pa.int64())])


def _join(join_type: str = "inner", left_keys=("k",), right_keys=("k",)) -> Join:
    return Join(
        left=Scan(source_id=0, schema=SchemaRef(_SCHEMA)),
        right=Scan(source_id=1, schema=SchemaRef(_SCHEMA)),
        left_keys=list(left_keys),
        right_keys=list(right_keys),
        join_type=join_type,
        output=[],
    )


def test_plain_join_reducer_may_salt():
    # The reducer concatenates its bucket, so moving a hot key between buckets cannot
    # change the relation. This is the case salting exists for.
    assert salting_is_safe(_join(), reducer_ir=None) is True


def test_finalizing_reducer_may_not_salt():
    # The whole point: a fused join+aggregate finalizes per bucket, so a split group
    # would be silently half-summed.
    assert salting_is_safe(_join(), reducer_ir='{"op": "aggregate"}') is False


@pytest.mark.parametrize("join_type", ["right", "full"])
def test_non_left_driven_join_may_not_salt(join_type):
    # Salting replicates build rows to every salt bucket; a join type that must emit
    # unmatched *right* rows exactly once would duplicate them.
    assert salting_is_safe(_join(join_type=join_type), reducer_ir=None) is False


def test_composite_key_may_not_salt():
    # The salting kernel is single-key; a composite key must keep the plain co-partition.
    assert salting_is_safe(_join(left_keys=("a", "b"), right_keys=("a", "b")), None) is False


def test_finalizing_reducer_wins_over_an_otherwise_eligible_join():
    # Guard against a future refactor reordering the predicate so an eligible join type
    # short-circuits past the reducer check — the exact shape that reintroduces the bug.
    j = _join(join_type="inner")
    assert salting_is_safe(j, reducer_ir=None) is True
    assert salting_is_safe(j, reducer_ir="{}") is False

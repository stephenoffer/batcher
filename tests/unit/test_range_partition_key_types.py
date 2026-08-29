"""Which key types range-partition — measured against the primitives, not asserted about them.

`range_partitionable` gates three separate paths: the distributed sort
(`dist.executor._range_partitionable_sort_key`), the out-of-core sort
(`dist.spill_breakers.sort.supports_spilling_sort`) and the ordered-bucket global window
(`dist.global_window.offsets.supports_ordered_bucket_offsets`). All three are asking one
question about one pair of primitives — `sample_key_grid` sketches the key and `bucketize`
routes it — and the predicate exists so they cannot answer it differently.

They did anyway, twice, in opposite directions:

* the global-window predicate had no type test at all, so `rank()` over a Boolean column
  collected fine and raised a bare Rust ``RuntimeError`` the moment the same plan was
  streamed;
* then the distributed sort found the predicate too *narrow* for a temporal key and widened
  it at its own call site instead of here — so ``ORDER BY <timestamp>`` distributed
  perfectly while the out-of-core sort of the same key declined to spill and a global window
  ordered by it had no distributed path at all. The canonical time-series shape, refused by
  exactly the two paths that exist because the relation does not fit in memory.

Neither failed a test, because each predicate was only ever asked its own question. So this
file asks the primitives instead: for every key type, whether `bucketize` actually routes it
is the ground truth, and the predicate has to match. A widening that outruns the partitioner
fails here as a ``RuntimeError``, and one that lags behind it fails as a disagreement.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

from batcher.dist.executors.partition_io import (
    bucketize,
    merge_boundaries,
    range_partitionable,
    sample_key_grid,
)

pytestmark = pytest.mark.unit

_N = 240

#: One representative array per key type family anyone sorts or windows by. Values are spread
#: so a three-way cut is a genuine three-way cut rather than one bucket and two empties.
_KEYS: dict[str, pa.Array] = {
    "int64": pa.array([(i * 7) % 97 for i in range(_N)], pa.int64()),
    "uint32": pa.array([(i * 7) % 97 for i in range(_N)], pa.uint32()),
    "float64": pa.array([float((i * 7) % 97) for i in range(_N)], pa.float64()),
    "string": pa.array([f"k{(i * 7) % 97:03d}" for i in range(_N)]),
    "binary": pa.array([bytes([(i * 7) % 97]) for i in range(_N)], pa.binary()),
    "date32": pa.array(
        [datetime.date(2020, 1, 1) + datetime.timedelta(days=(i * 7) % 97) for i in range(_N)],
        pa.date32(),
    ),
    "timestamp": pa.array(
        [
            datetime.datetime(2020, 1, 1) + datetime.timedelta(seconds=(i * 97) % 8000)
            for i in range(_N)
        ],
        pa.timestamp("us"),
    ),
    "time64": pa.array(
        [datetime.time((i * 7) % 24, (i * 11) % 60, i % 60) for i in range(_N)], pa.time64("us")
    ),
    "duration": pa.array(
        [datetime.timedelta(seconds=(i * 7) % 97) for i in range(_N)], pa.duration("us")
    ),
    "decimal128": pa.array([float((i * 7) % 97) for i in range(_N)], pa.float64()).cast(
        pa.decimal128(12, 2)
    ),
    "bool": pa.array([i % 3 == 0 for i in range(_N)], pa.bool_()),
    "list": pa.array([[i] for i in range(_N)], pa.list_(pa.int64())),
}


def _routes(array: pa.Array) -> bool:
    """Whether the sample-and-scatter primitives actually carry this key. The ground truth."""
    batch = pa.RecordBatch.from_arrays([array], ["k"])
    try:
        grid = sample_key_grid([batch], "k", [0.25, 0.5, 0.75])
        bucketize([batch], "k", merge_boundaries([(grid, len(array))], 3), 3, False, False)
    except Exception:
        return False
    return True


@pytest.mark.parametrize("name", sorted(_KEYS))
def test_the_predicate_matches_what_the_partitioner_does(name):
    """`range_partitionable` says yes exactly when `sample_key_grid` + `bucketize` succeed."""
    array = _KEYS[name]
    assert range_partitionable(array.type) is _routes(array), (
        f"{name}: the predicate and the primitives disagree — a `True` here is a Rust "
        "`RuntimeError` inside a Ray task, and a `False` is a whole relation on one node"
    )


def test_the_table_covers_both_answers():
    """Guard against a vacuous sweep: the parametrization must contain a yes and a no.

    Every assertion above is "predicate == primitives". If the key table drifted to types the
    partitioner all accepts (or all refuses) the file would still pass while testing that one
    constant equals itself.
    """
    answers = {range_partitionable(a.type) for a in _KEYS.values()}
    assert answers == {True, False}


def test_the_three_callers_read_one_predicate():
    """The sort's own key test is `range_partitionable` and nothing more.

    It used to prepend ``is_decimal(dtype) or is_temporal(dtype) or``, which is how the other
    two callers came to refuse a timestamp the partitioner routes. Asserting the *identity* of
    the answer rather than re-listing the types is the point: a future widening lands in one
    place or fails here.
    """
    from batcher.dist.executor import _range_partitionable_sort_key
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Scan, Sort
    from batcher.plan.logical.aggregate import SortKeySpec
    from batcher.plan.schema import SchemaRef

    for name, array in _KEYS.items():
        schema = SchemaRef(pa.schema([("k", array.type), ("v", pa.int64())]))
        sort = Sort(input=Scan(source_id=0, schema=schema), keys=(SortKeySpec(Col("k")),))
        assert _range_partitionable_sort_key(sort) is range_partitionable(array.type), name

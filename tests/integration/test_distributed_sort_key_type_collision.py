"""Two sorts on a same-named column of different types must not share a learned grid.

`sort_shape_key` hashed the mapped plan IR and the key's column *name*. A mapped prefix that
is a bare scan serializes to `{"op": "scan", "source_id": 0}` -- no schema, no column types --
so `sort("k")` over a `float64` column and `sort("k")` over a `string` column produced
byte-identical IR and therefore the same digest. The second sort loaded the first's grid, and
a float boundary list handed to the string range partitioner raises

    TypeError: argument 'boundaries': 'float' object cannot be converted to 'PyString'

from inside a Ray task, after two retries. The reverse direction fails in NumPy instead.

This is the one way a learned grid can do more than cost balance, which is what the rest of
`dist/sort_boundaries.py`'s safety argument rests on -- so it is worth an end-to-end test and
not only the unit ones. Ordering is asserted positionally on the key, because that is the
sort's whole contract.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

_N = 400
_FLOAT = pa.schema([("k", pa.float64()), ("v", pa.int64())])
_STRING = pa.schema([("k", pa.string()), ("v", pa.int64())])


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(3)
    yield
    shutdown_test_ray(started)


def _floats():
    return bt.from_pydict(
        {"k": [float(i % 37) for i in range(_N)], "v": list(range(_N))}, schema=_FLOAT
    )


def _strings():
    # Nulls included: they are what the sampler drops, and what left the numeric grid
    # looking plausible enough to be reused.
    return bt.from_pydict(
        {
            "k": [None if i % 9 == 0 else f"g{i % 11:02d}" for i in range(_N)],
            "v": list(range(_N)),
        },
        schema=_STRING,
    )


def _keys(table):
    return table.to_pydict()["k"]


@pytest.mark.parametrize("descending", [False, True])
def test_a_string_sort_survives_a_float_sort_of_the_same_shape(descending):
    """The float sort runs first and persists its grid; the string sort must not load it."""
    _floats().sort("k", descending=descending).collect(distributed=True, num_workers=3)
    got = _strings().sort("k", descending=descending).collect(distributed=True, num_workers=3)
    assert _keys(got) == _keys(_strings().sort("k", descending=descending).collect())


@pytest.mark.parametrize("descending", [False, True])
def test_a_float_sort_survives_a_string_sort_of_the_same_shape(descending):
    """The other direction, which failed inside NumPy rather than at the FFI boundary."""
    _strings().sort("k", descending=descending).collect(distributed=True, num_workers=3)
    got = _floats().sort("k", descending=descending).collect(distributed=True, num_workers=3)
    assert _keys(got) == _keys(_floats().sort("k", descending=descending).collect())


def test_a_disjoint_relation_does_not_inherit_the_other_s_boundaries(tmp_path):
    """The silent half of the same defect: correct rows, and no distribution left.

    Two tables with the same schema and the same key column but disjoint ranges shared a
    grid, so every key of the second fell past the last boundary of the first and the whole
    relation landed in one bucket. Nothing raised. Asserted on the *bucket loads* rather
    than on wall time, because the damage is a scheduling property and timing on a shared
    box is not evidence.
    """
    from batcher.dist.executors.partition_io import (
        bucketize,
        merge_boundaries,
        sample_key_grid,
    )

    schema = pa.schema([("k", pa.int64()), ("v", pa.int64())])
    n, buckets = 4_000, 8
    low = [pa.record_batch({"k": list(range(n)), "v": [0] * n}, schema=schema)]
    high = [pa.record_batch({"k": [10**9 + i for i in range(n)], "v": [0] * n}, schema=schema)]
    probs = [i / 32 for i in range(1, 32)]

    own = merge_boundaries([(sample_key_grid(high, "k", probs), n)], buckets)
    foreign = merge_boundaries([(sample_key_grid(low, "k", probs), n)], buckets)

    def loads(bounds):
        return [
            sum(b.num_rows for b in bucket)
            for bucket in bucketize(high, "k", bounds, buckets, True, False)
        ]

    assert min(loads(own)) > 0, "its own grid must keep every reducer fed"
    assert loads(foreign).count(0) == buckets - 1, "the fixture must show the damage"

    # And the two shapes must not share a store entry, which is what stops it.
    from batcher.dist.sort_boundaries import sort_key_identity

    a = bt.from_pydict({"k": list(range(n)), "v": [0] * n}, schema=schema)
    b = bt.from_pydict({"k": [10**9 + i for i in range(n)], "v": [0] * n}, schema=schema)
    assert sort_key_identity(a._sources[0], "k") != sort_key_identity(b._sources[0], "k")


def test_each_type_still_reuses_its_own_grid():
    """Separating the shapes must not cost the optimization it exists to keep."""
    from batcher.dist.sort_boundaries import (
        load_learned_grids,
        sort_key_identity,
        sort_key_is_string,
        sort_shape_key,
    )

    # ONE dataset, sorted and then inspected. Two `_strings()` calls would be two in-memory
    # relations, and an in-memory source is keyed by a per-instance serial precisely so that
    # two unrelated relations of the same shape cannot share a grid -- so the second would
    # correctly miss, and the miss would be a property of the test.
    query = _strings()
    query.sort("k").collect(distributed=True, num_workers=3)
    source = query._sources[0]
    assert sort_key_is_string(source, "k") is True
    key = sort_shape_key('{"op": "scan", "source_id": 0}', "k", sort_key_identity(source, "k"))
    assert load_learned_grids(key, True), "the string sort's own grid must be reusable"

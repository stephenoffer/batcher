"""`LIMIT` over an unordered relation: which rows may differ, and what still may not.

`.claude/rules/python-control-plane.md` states that a distributed result is identical to the
single-node one with three exceptions, all of them places where the *query* does not determine
an answer. This is the third: a `group_by(k).agg(...)` emits its groups in whatever order the
hash table walks, which is not a property of the query, so `LIMIT n` over it keeps *some* `n`
groups rather than a defined `n`.

A rule that only says "these may differ" is the kind that turns into a licence. This file is
the other half of it — the part that still binds, and the part someone chasing a suspicious
`LIMIT` should check first:

* the row **count** is exact;
* every row returned is a row of the *unlimited* result, so the limit selects rather than
  invents;
* the column types match.

And the control: `sort(...).limit(n)` is a top-N, has a defined answer, and both paths compute
it. Without that arm the file would be consistent with a distributed `LIMIT` that was simply
broken, since "the rows may differ" would explain any result at all.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")

_N = 600
_KEYS = 17
_LIMIT = 3
_WORKERS = 2


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def splittable(tmp_path_factory) -> str:
    """Four Parquet files, so the read genuinely fans out and the groups genuinely shuffle."""
    table = pa.table(
        {
            "k": pa.array([i % _KEYS for i in range(_N)], pa.int64()),
            "t": pa.array([(i * 7) % 101 for i in range(_N)], pa.int64()),
        }
    )
    directory = tmp_path_factory.mktemp("unordered_limit")
    for part in range(4):
        pq.write_table(table, directory / f"p{part}.parquet")
    return str(directory)


_UNORDERED = {
    "aggregate": lambda ds: ds.group_by("k").agg(n=bt.col("t").sum()),
    # A whole-row `DISTINCT` is the group-by over every column, and diverges the same way.
    "distinct": lambda ds: ds.select("k").distinct(),
}


def _rows(table: pa.Table) -> set[tuple]:
    names = sorted(table.column_names)
    data = table.to_pydict()
    return {tuple(data[n][i] for n in names) for i in range(table.num_rows)}


@pytest.mark.parametrize("shape", sorted(_UNORDERED))
def test_an_unordered_limit_selects_rather_than_invents(splittable, shape):
    build = _UNORDERED[shape]
    whole = build(bt.read.parquet(splittable)).collect()
    single = build(bt.read.parquet(splittable)).limit(_LIMIT).collect()
    limited = build(bt.read.parquet(splittable)).limit(_LIMIT)
    distributed = limited.collect(distributed=True, num_workers=_WORKERS)

    assert single.num_rows == _LIMIT
    assert distributed.num_rows == _LIMIT, "the row count is exact even where the rows are not"
    assert distributed.schema == single.schema
    everything = _rows(whole)
    assert _rows(single) <= everything
    assert _rows(distributed) <= everything, (
        "a distributed LIMIT returned a row the unlimited answer does not contain — that is a "
        "defect, not the unordered-limit exception"
    )


@pytest.mark.parametrize("shape", sorted(_UNORDERED))
def test_an_ordered_limit_is_exact_on_both_paths(splittable, shape):
    """The control. A `sort` gives the limit a defined answer, and both paths must find it."""
    build = _UNORDERED[shape]
    single = build(bt.read.parquet(splittable)).sort("k").limit(_LIMIT).collect()
    distributed = (
        build(bt.read.parquet(splittable))
        .sort("k")
        .limit(_LIMIT)
        .collect(distributed=True, num_workers=_WORKERS)
    )
    assert distributed.to_pydict() == single.to_pydict()

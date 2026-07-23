"""Property: the answer is invariant across *execution paths*, not just chunkings.

``CLAUDE.md`` names the cross-product that has repeatedly hidden bugs — ``{collect,
spill, iter_batches, distributed}`` by ``{nulls, empty, dups, -0.0/NaN}``. ``sort(descending)``
once returned unsorted data *only under spill*; a distributed ``GROUP BY`` on a float key
once split one group into two. Those are path bugs: the same plan, executed a different
way, gives a different answer. This module makes that a property — for a random table and a
random unordered pipeline it asserts

    collect() == collect(spill=True) == concat(iter_batches(random size))
              == collect(distributed=True, num_workers=2)

so an execution path that diverges is a counterexample. ``num_workers`` is pinned at 2
(schedulable in this sandbox; 3+ hangs). Float data *may* carry ``-0.0``/``NaN`` here — the
comparison is Batcher-vs-Batcher, so the total-order question (ledger B26) does not enter;
we only require the paths agree with *each other*, and a NaN sentinel makes ``nan == nan``.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = [pytest.mark.property, pytest.mark.integration]


def _coerce(v: object) -> object:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v != v:
            return "<nan>"
        if not math.isfinite(v):
            return float(v)
        r = round(v, 9)
        return int(r) if r == int(r) else r
    return v


def _rowset(table: pa.Table) -> list[tuple]:
    cols = table.column_names
    rows = [tuple(_coerce(r[c]) for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((v is None, str(type(v)), str(v)) for v in t))


_SCHEMA = pa.schema([("k", pa.int64()), ("i", pa.int64()), ("f", pa.float64()), ("s", pa.string())])
_floats = st.one_of(st.none(), st.sampled_from([0.0, -0.0, 1.5, -1.5, 2.5, float("nan")]))
_ints = st.one_of(st.none(), st.sampled_from([-3, -1, 0, 1, 2, 3, 7]))
_strs = st.one_of(st.none(), st.sampled_from(["a", "b", "", "c"]))


@st.composite
def _table(draw: st.DrawFn) -> pa.Table:
    n = draw(st.integers(min_value=0, max_value=40))
    return pa.table(
        {
            "k": pa.array(draw(st.lists(st.integers(0, 3), min_size=n, max_size=n)), pa.int64()),
            "i": pa.array(draw(st.lists(_ints, min_size=n, max_size=n)), pa.int64()),
            "f": pa.array(draw(st.lists(_floats, min_size=n, max_size=n)), pa.float64()),
            "s": pa.array(draw(st.lists(_strs, min_size=n, max_size=n)), pa.string()),
        },
        schema=_SCHEMA,
    )


# Unordered pipelines only — path equivalence is a *set* property; row order across paths
# (especially distributed) is not part of the contract unless a sort fixes it.
_PIPES = {
    "aggregate": lambda ds: ds.group_by("k").agg(
        s=col("i").sum(), n=bt.count(), md=col("i").median(), sd=col("f").std(), mx=col("i").max()
    ),
    "distinct": lambda ds: ds.select("k", "i", "f").distinct(),
    "filter": lambda ds: ds.filter(col("i") > bt.lit(0)),
    "self_join": lambda ds: ds.join(ds.select("k", "s"), on="k", how="inner"),
    "value_counts": lambda ds: ds.value_counts("k", sort=False),
    "fill_null": lambda ds: ds.fill_null({"i": 0, "s": "x"}),
}

_HC = [HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
# The distributed path is the slow one, so this test keeps a small budget; the sort test
# below (spill only) can afford more.
_PROP = settings(max_examples=30, deadline=None, suppress_health_check=_HC)
_PROP_FAST = settings(max_examples=60, deadline=None, suppress_health_check=_HC)


@_PROP
@given(_table(), st.sampled_from(sorted(_PIPES)), st.sampled_from([1, 2, 7, None]))
def test_paths_agree(table: pa.Table, pipe_name: str, batch_size: int | None) -> None:
    """collect == spill == iter_batches(bs) == distributed(2) for a random pipeline."""
    pipe = _PIPES[pipe_name]
    base = _rowset(pipe(bt.from_arrow(table)).collect())

    spilled = _rowset(pipe(bt.from_arrow(table)).collect(spill=True))
    assert spilled == base, f"spill diverged ({pipe_name}):\n base={base}\n spill={spilled}"

    batches = list(pipe(bt.from_arrow(table)).iter_batches(batch_size=batch_size))
    streamed = _rowset(pa.Table.from_batches(batches, schema=batches[0].schema)) if batches else []
    assert streamed == base, (
        f"iter_batches(bs={batch_size}) diverged ({pipe_name}):\n base={base}\n iter={streamed}"
    )

    dist = _rowset(pipe(bt.from_arrow(table)).collect(distributed=True, num_workers=2))
    assert dist == base, f"distributed(2) diverged ({pipe_name}):\n base={base}\n dist={dist}"


@_PROP_FAST
@given(_table(), st.sampled_from(["i", "k", "f", "s"]), st.booleans(), st.booleans())
def test_sort_path_agrees_ordered(
    table: pa.Table, key: str, descending: bool, nulls_first: bool
) -> None:
    """A sort's *row order* survives spill (the ``sort(descending)``-under-spill bug class).

    Ordered comparison on the sort key sequence: tie rows may reorder, but the key column
    itself must come out in the same order regardless of path.
    """
    plan = bt.from_arrow(table).sort(key, descending=descending, nulls_first=nulls_first)
    base = [_coerce(r[key]) for r in plan.collect().to_pylist()]
    spilled = [_coerce(r[key]) for r in plan.collect(spill=True).to_pylist()]
    assert spilled == base, (
        f"sort order changed under spill "
        f"(key={key}, desc={descending}, nf={nulls_first}):\n base={base}\n spill={spilled}"
    )

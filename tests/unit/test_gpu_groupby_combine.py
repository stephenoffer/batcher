"""The GPU group-by fan-out's `combine` step: mergeable, and inside its layer.

`dist` schedules operators; it must not reach back up through the public API to run one. This
combine used to build a `Dataset`, which forged a `dist -> api` edge and broke the backend
contract. It now emits IR, so the property worth pinning is that the IR still computes what
the fan-out promises: combining per-shard partials equals aggregating the whole input once.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.dist.gpu.groupby import combine_ops, partial_aggs
from batcher.plan.distribution import nest_ops

pytestmark = pytest.mark.unit

ROWS = {
    "k": ["a", "b", "a", "c", "b", "a", None, "c"],
    "v": [1.0, 2.0, 3.0, 4.0, 5.0, None, 7.0, 8.0],
    "n": [1, 2, 3, 4, 5, 6, 7, 8],
}


def _run(ops: list[dict], table: pa.Table) -> pa.Table:
    """Run an operator chain on the native engine, the way the fan-out's driver does."""
    from batcher._internal.native import engine
    from batcher.dist.executors.ray_runtime import engine_config_json

    out = engine().execute_plan(
        json.dumps(nest_ops(ops)), [table.to_batches()], engine_config_json()
    )
    return pa.Table.from_batches(out, schema=out[0].schema)


def _shard_partials(table: pa.Table, key: str, aggs: dict, shards: int) -> pa.Table:
    """What each shard's device would return, computed here with the engine instead.

    The fan-out's contract is over the *partials*, not over how they were produced, so
    computing them on the CPU engine tests the combine without needing a GPU.
    """
    reductions = partial_aggs(aggs)
    ops = [
        {
            "op": "aggregate",
            "group_keys": [{"expr": {"e": "col", "name": key}, "alias": key}],
            "aggregates": [
                {"func": func, "alias": alias, "input": {"e": "col", "name": src}}
                for alias, (src, func) in reductions.items()
            ],
        }
    ]
    rows = table.num_rows
    step = -(-rows // shards)
    return pa.concat_tables([_run(ops, table.slice(i, step)) for i in range(0, rows, step)])


@pytest.mark.parametrize("shards", [1, 2, 3, 8])
@pytest.mark.parametrize(
    "aggs",
    [
        {"total": ("v", "sum")},
        {"rows": ("v", "count")},
        {"lo": ("v", "min"), "hi": ("v", "max")},
        {"avg": ("v", "mean")},
        {"avg": ("v", "mean"), "total": ("v", "sum"), "rows": ("n", "count")},
        {"avg_int": ("n", "mean")},
    ],
)
def test_combining_shard_partials_equals_aggregating_once(shards, aggs):
    """`combine(partition(partial(p)))` == the single-node answer, over any shard count."""
    table = pa.table(ROWS)
    combined = _run(combine_ops("k", aggs), _shard_partials(table, "k", aggs, shards))

    funcs = {"sum": "sum", "count": "count", "min": "min", "max": "max", "mean": "mean"}
    expected = (
        bt.from_arrow(table)
        .group_by("k")
        .agg(**{a: getattr(col(c), funcs[f])() for a, (c, f) in aggs.items()})
        .collect()
    )

    got = {r["k"]: r for r in combined.to_pylist()}
    want = {r["k"]: r for r in expected.to_pylist()}
    assert got.keys() == want.keys()
    for key, row in want.items():
        for alias in aggs:
            assert got[key][alias] == pytest.approx(row[alias]), (key, alias)


def test_a_count_of_counts_is_summed_not_recounted():
    """The bug this shape invites: counting the partials returns the number of shards."""
    ops = combine_ops("k", {"rows": ("v", "count")})
    assert ops[0]["aggregates"] == [
        {"func": "sum", "alias": "rows", "input": {"e": "col", "name": "rows__count"}}
    ]


def test_a_mean_is_divided_after_both_halves_are_folded():
    ops = combine_ops("k", {"avg": ("v", "mean")})
    assert [a["func"] for a in ops[0]["aggregates"]] == ["sum", "sum"]
    assert ops[1]["op"] == "project"
    assert ops[1]["exprs"][-1]["expr"]["op"] == "div"


def test_no_projection_when_nothing_needs_dividing():
    assert len(combine_ops("k", {"total": ("v", "sum")})) == 1


def test_the_projection_preserves_the_users_column_order():
    aggs = {"a": ("v", "sum"), "b": ("v", "mean"), "c": ("v", "max")}
    exprs = combine_ops("k", aggs)[1]["exprs"]
    assert [e["alias"] for e in exprs] == ["k", "a", "b", "c"]

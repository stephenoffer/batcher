"""A tree of combines returns exactly what a line of combines returned.

`executors/aggregate._tree_combine_buckets` rebrackets the disk shuffle's reduce so no
reducer reads more than `shuffle_fan_in` mapper partials. That is only sound because
`combine` is associative and commutative, and "sound" here has to mean the *engine's*
combine, not a stand-in: the orchestration is checked with a stub in
`test_reduction_tree.py`, and this file checks the algebra it assumes, over real Arrow
partial state, for every aggregate whose merge is non-trivial.

No Ray: the map, combine and reduce tasks are plain functions of their inputs, so calling
them directly exercises the identical code the cluster runs while staying a unit test.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher._internal.native import engine
from batcher.dist.executors import aggregate as agg_mod
from batcher.dist.reduction import chunks
from batcher.plan.ir_specs import agg_spec_json

pytestmark = pytest.mark.unit


@pytest.fixture
def wd(tmp_path):
    return str(tmp_path)


def _specs(keys, **aggs):
    """The `(group_keys_json, aggregates_json)` pair the shuffle tasks take.

    Built through the public API rather than by hand so the spec is the one a real query
    produces — an `Aggregate` assembled directly would let this test agree with itself
    about a shape the planner never emits.
    """
    import batcher as bt

    ds = bt.from_pydict({"k": [1], "v": [1.0], "i": [1], "s": ["a"]})
    node = (ds.group_by(*keys).agg(**aggs) if keys else ds.agg(**aggs))._plan
    return agg_spec_json(node), node


def _mapper_partials(nat, gk, aj, tables):
    """One partial per simulated mapper."""
    return [nat.partial_aggregate(gk, aj, t.to_batches()) for t in tables]


def _write(nat, wd, partials):
    from batcher.dist.shuffle_io import write_ipc

    paths = []
    for m, p in enumerate(partials):
        path = f"{wd}/m{m}_r0.arrow"
        write_ipc([p], path)
        paths.append(path)
    return paths


def _by_key(path, key="k"):
    """The reduce output keyed by group, read back from its IPC file."""
    from batcher.dist.shuffle_io import read_ipc

    return {r[key]: r for r in pa.Table.from_batches(read_ipc(path)).to_pylist()}


def _assert_same_groups(got: dict, want: dict) -> None:
    """Identical groups and identical values, floats up to reassociation.

    The tolerance is not slack, it is the contract: `combine` is associative in exact
    arithmetic and IEEE addition is not, so rebracketing a float sum moves its last bits and
    nothing can prevent that while the bracketing is free. Everything else — the group set,
    the counts, the integer extremes — is exact, which is where a real rebracketing bug
    would show.
    """
    assert got.keys() == want.keys()
    for k, row in want.items():
        for col, expected in row.items():
            actual = got[k][col]
            if isinstance(expected, float):
                assert actual == pytest.approx(expected, rel=1e-12), (k, col)
            else:
                assert actual == expected, (k, col)


def _tables(n_mappers, rows_each, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_mappers):
        out.append(
            pa.table(
                {
                    "k": rng.integers(0, 12, rows_each),
                    "v": rng.random(rows_each) * 1000,
                    "i": rng.integers(-50, 50, rows_each),
                    "s": [f"s{x}" for x in rng.integers(0, 5, rows_each)],
                }
            )
        )
    return out


@pytest.mark.parametrize("fan_in", [2, 3, 8])
@pytest.mark.parametrize("n_mappers", [5, 9, 33])
def test_a_tree_of_combines_equals_the_flat_fold(wd, fan_in, n_mappers):
    """The claim the rebracketing rests on, over the engine's own partial state."""
    import batcher as bt

    nat = engine()
    (gk, aj), _node = _specs(
        ["k"],
        total=bt.col("v").sum(),
        n=bt.col("v").count(),
        lo=bt.col("i").min(),
        hi=bt.col("i").max(),
        avg=bt.col("v").mean(),
    )
    partials = _mapper_partials(nat, gk, aj, _tables(n_mappers, 400))
    paths = _write(nat, wd, partials)

    flat = agg_mod._reduce_task(gk, aj, paths, wd, 900)

    # One interior level by hand, then the same reduce over its output.
    level: list[str] = []
    for i, chunk in enumerate(chunks(paths, fan_in)):
        if len(chunk) == 1:
            level.append(chunk[0])
            continue
        out = agg_mod._combine_task(gk, aj, list(chunk), wd, f"c0_r0_{i}.arrow")
        assert len(out) == 1
        level.extend(out)
    treed = agg_mod._reduce_task(gk, aj, level, wd, 901)

    assert flat[1] == treed[1]
    _assert_same_groups(_by_key(treed[0]), _by_key(flat[0]))


def test_the_tree_holds_for_a_keyless_global_aggregate(wd):
    """The shape the tree helps most: one bucket, so the flat fold was a `W`-long line on a
    single node with the rest of the cluster idle."""
    import batcher as bt

    nat = engine()
    (gk, aj), _node = _specs([], total=bt.col("v").sum(), n=bt.col("v").count())
    partials = _mapper_partials(nat, gk, aj, _tables(40, 100))
    paths = _write(nat, wd, partials)

    flat = agg_mod._reduce_task(gk, aj, paths, wd, 800)
    level = [
        p
        for i, c in enumerate(chunks(paths, 4))
        for p in agg_mod._combine_task(gk, aj, list(c), wd, f"c0_r0_{i}.arrow")
    ]
    treed = agg_mod._reduce_task(gk, aj, level, wd, 801)

    from batcher.dist.shuffle_io import read_ipc

    a = pa.Table.from_batches(read_ipc(flat[0])).to_pylist()[0]
    b = pa.Table.from_batches(read_ipc(treed[0])).to_pylist()[0]
    assert a["n"] == b["n"]
    assert a["total"] == pytest.approx(b["total"], rel=1e-12)


def test_an_interior_combine_does_not_finalize(wd):
    """An interior node's output is another combine's input, so it must stay partial state.
    Finalizing early is the classic average-of-averages, and it would be invisible on `sum`
    while being wrong on `mean` — which is why this asserts on `mean`."""
    import batcher as bt

    nat = engine()
    (gk, aj), _node = _specs(["k"], avg=bt.col("v").mean())
    tables = _tables(8, 300, seed=11)
    partials = _mapper_partials(nat, gk, aj, tables)
    paths = _write(nat, wd, partials)

    flat = agg_mod._reduce_task(gk, aj, paths, wd, 700)
    mid = [
        p
        for i, c in enumerate(chunks(paths, 3))
        for p in agg_mod._combine_task(gk, aj, list(c), wd, f"c0_r0_{i}.arrow")
    ]
    treed = agg_mod._reduce_task(gk, aj, mid, wd, 701)

    from batcher.dist.shuffle_io import read_ipc

    got = {r["k"]: r["avg"] for r in pa.Table.from_batches(read_ipc(treed[0])).to_pylist()}
    want = {r["k"]: r["avg"] for r in pa.Table.from_batches(read_ipc(flat[0])).to_pylist()}
    assert got.keys() == want.keys()
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=1e-12)

    # And the true mean, so the test cannot pass by both sides being wrong the same way.
    whole = pa.concat_tables(tables)
    expected: dict = {}
    for row in whole.to_pylist():
        expected.setdefault(row["k"], []).append(row["v"])
    for k, vs in expected.items():
        assert got[k] == pytest.approx(sum(vs) / len(vs), rel=1e-9)


def test_a_chunk_of_empty_partials_invents_no_rows(wd):
    """A mapper whose partition matched nothing still publishes a schema-bearing partial, so
    an interior combine sees zero-row inputs and must produce a zero-row partial. Inventing
    a group here would add a row to the answer that no input contained."""
    import batcher as bt

    nat = engine()
    (gk, aj), _node = _specs(["k"], n=bt.col("v").count())
    schema = pa.schema(
        [("k", pa.int64()), ("v", pa.float64()), ("i", pa.int64()), ("s", pa.string())]
    )
    blank = pa.RecordBatch.from_pylist([], schema=schema)
    partials = [nat.partial_aggregate(gk, aj, [blank]) for _ in range(5)]
    paths = _write(nat, wd, partials)

    merged = agg_mod._combine_task(gk, aj, paths, wd, "c0_r0_0.arrow")
    assert len(merged) == 1  # schema-bearing, as the mappers' own partials are
    assert agg_mod._reduce_task(gk, aj, merged, wd, 601) == (None, 0)


def test_a_chunk_of_unreadable_partials_is_dropped_not_written(wd):
    """`write_ipc` refuses a batch list with nothing in it, so a chunk that read back as
    nothing must be dropped rather than written — the difference between one fewer reducer
    input and an IOError in the middle of a shuffle."""
    import batcher as bt

    (gk, aj), _node = _specs(["k"], n=bt.col("v").count())
    assert agg_mod._combine_task(gk, aj, [], wd, "c0_r0_0.arrow") == []


def test_a_chunk_over_the_memory_envelope_declines_instead_of_merging(wd):
    """An interior combine has no out-of-core fold — spilling is only defined for the step
    that finalizes — so it must hand its inputs back rather than build a state it cannot
    hold. The reducer then sees the bucket at full width, which is what
    `combine_finalize_spilling` is written for. Without this the tree would turn an
    aggregate that used to spill and finish into one that dies in a level nobody can see."""
    import json

    import batcher as bt

    nat = engine()
    (gk, aj), _node = _specs(["k"], n=bt.col("v").count(), s=bt.col("v").sum())
    paths = _write(nat, wd, _mapper_partials(nat, gk, aj, _tables(6, 500, seed=5)))

    tiny = json.dumps({"memory_budget_bytes": 1})
    assert agg_mod._combine_task(gk, aj, paths, wd, "c0_r0_0.arrow", tiny) == paths

    roomy = json.dumps({"memory_budget_bytes": 1 << 40})
    assert len(agg_mod._combine_task(gk, aj, paths, wd, "c0_r0_1.arrow", roomy)) == 1

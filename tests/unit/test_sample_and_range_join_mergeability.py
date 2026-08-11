"""The algebra behind the distributed `Sample(n=)` and `RangeJoin` paths.

`CLAUDE.md` #7: a stateful operator distributes only if it has a mergeable form, and the
invariant `combine_finalize(partition(partial(p_k))) == single-node` is the test that must
stay green. These are that test for the two operators that until now had no distributed
path at all — and they are deliberately *unit* tests over the real engine rather than Ray
integration tests, because it is the algebra that makes a distributed answer right or
wrong. The Ray plumbing is covered by
`tests/integration/test_distributed_sample_and_range_join.py`; a wrong answer would show
up here first, and in milliseconds.

- **`Sample(n=)`** keeps the `n` smallest-hash rows of the whole relation. It is mergeable
  top-N: a row among the globally `n` smallest is also among its own partition's `n`
  smallest (its partition holds a subset, so its rank there is no worse than its global
  rank), so the union of the per-partition results *contains* the global answer and
  re-applying the operator to that union selects exactly it.

- **`RangeJoin`** has no equality to co-partition on, so it is distributed by broadcasting
  the build side: each probe partition is joined against the *whole* right. Every left row
  is in exactly one partition and sees every right row, so the union of the per-partition
  joins is the full relation, with nothing duplicated and nothing missed.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

ROWS = 2_000


def _table(seed: int = 11) -> pa.Table:
    rng = np.random.default_rng(seed)
    return pa.table(
        {
            "x": np.arange(ROWS, dtype="int64"),
            "g": rng.integers(0, 5, ROWS).astype("int64"),
        }
    )


def _rows(table: pa.Table) -> list[tuple]:
    return sorted(tuple(r.values()) for r in table.to_pylist())


def _partitions(table: pa.Table, k: int) -> list[pa.Table]:
    """`k` contiguous slices — the shape `partition_descriptors` hands the workers."""
    per = table.num_rows // k
    return [table.slice(i * per, per if i < k - 1 else table.num_rows - i * per) for i in range(k)]


# --- Sample(n=): mergeable top-N -----------------------------------------------


def _sample_n(table: pa.Table, n: int, seed: int) -> pa.Table:
    return bt.from_arrow(table).sample(n=n, seed=seed).collect()


@pytest.mark.unit
@pytest.mark.parametrize("n", [0, 1, 2, 7, 100, 999, 1_500, ROWS, 5_000])
@pytest.mark.parametrize("parts", [1, 2, 4, 7])
def test_fixed_count_sample_is_mergeable(n, parts):
    """partial per partition, then combine, equals the single-node answer — at every size
    and every partition count, including sizes that straddle the relation."""
    t = _table()
    whole = _sample_n(t, n, seed=42)

    partials = [_sample_n(p, n, seed=42) for p in _partitions(t, parts)]
    combined = _sample_n(pa.concat_tables(partials), n, seed=42)

    assert combined.num_rows == min(n, ROWS)
    assert _rows(combined) == _rows(whole)


@pytest.mark.unit
def test_fixed_count_sample_partial_is_a_superset_of_the_global_answer():
    """The property the merge rests on: every globally-selected row survives its own
    partition's `partial`. If this fails, no combine can recover it."""
    t = _table()
    n = 50
    global_rows = set(_rows(_sample_n(t, n, seed=42)))
    survivors: set = set()
    for p in _partitions(t, 4):
        survivors |= set(_rows(_sample_n(p, n, seed=42)))
    assert global_rows <= survivors


@pytest.mark.unit
def test_fixed_count_sample_is_partition_count_independent():
    """The same rows whatever the fan-out — a distributed result must not depend on how
    many workers happened to run it."""
    t = _table()
    answers = {
        parts: tuple(
            _rows(
                _sample_n(
                    pa.concat_tables([_sample_n(p, 25, 3) for p in _partitions(t, parts)]), 25, 3
                )
            )
        )
        for parts in (1, 2, 3, 5, 8)
    }
    assert len(set(answers.values())) == 1


@pytest.mark.unit
def test_fraction_sample_needs_no_merge_step():
    """The contrast that justifies the separate branch: a *fraction* sample is a per-row
    predicate, so concatenating the partials is already the answer."""
    t = _table()
    whole = bt.from_arrow(t).sample(fraction=0.25, seed=3).collect()
    partials = pa.concat_tables(
        [bt.from_arrow(p).sample(fraction=0.25, seed=3).collect() for p in _partitions(t, 4)]
    )
    assert _rows(partials) == _rows(whole)


@pytest.mark.unit
def test_sample_hash_reads_every_column_so_pruning_below_it_is_unsound():
    """Why the distributed path must not let projection pruning reach below the sample:
    the row hash is over the whole row, so a pruned worker samples different rows."""
    t = _table()
    over_both = _rows(_sample_n(t, 5, seed=42))
    over_one = _rows(_sample_n(t.select(["x"]), 5, seed=42))
    assert [r[0] for r in over_both] != [r[0] for r in over_one]


@pytest.mark.unit
def test_distributed_sample_pushdown_keeps_every_column():
    """...and the pushdown the distributed path actually computes does keep them."""
    from batcher.dist.executors.map import _scan_pushdown
    from batcher.dist.executors.plan_analysis import _relabel_single_source

    ds = bt.from_arrow(_table()).sample(n=5, seed=42)
    plan0, _sid = _relabel_single_source(ds._plan)
    projection, _predicate = _scan_pushdown(plan0)
    assert projection is None or set(projection) == {"x", "g"}


# --- RangeJoin: broadcast-partitionable ----------------------------------------


def _range_join(left: pa.Table, right: pa.Table, op: str) -> pa.Table:
    pred = {
        "<": bt.col("x") < bt.col("lo"),
        "<=": bt.col("x") <= bt.col("lo"),
        ">": bt.col("x") > bt.col("lo"),
        ">=": bt.col("x") >= bt.col("lo"),
    }[op]
    return bt.from_arrow(left).join(bt.from_arrow(right), how="cross").filter(pred).collect()


@pytest.fixture
def bands() -> pa.Table:
    return pa.table({"lo": [0, 400, 900, 1_500], "tier": ["a", "b", "c", "d"]})


@pytest.mark.unit
@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
@pytest.mark.parametrize("parts", [1, 2, 4, 7])
def test_range_join_partitions_over_the_probe_side(bands, op, parts):
    """Splitting the probe (left) side and joining each part against the WHOLE build side
    reproduces the single-node relation exactly — the broadcast argument."""
    t = _table()
    whole = _range_join(t, bands, op)
    per_partition = pa.concat_tables([_range_join(p, bands, op) for p in _partitions(t, parts)])
    assert per_partition.num_rows == whole.num_rows
    assert _rows(per_partition) == _rows(whole)


@pytest.mark.unit
@pytest.mark.parametrize("parts", [2, 4])
def test_band_join_two_conditions_partitions_over_the_probe_side(parts):
    """Two inequalities (interval containment) — the IEJoin shape."""
    t = _table()
    iv = pa.table({"lo": [0, 400, 900], "hi": [300, 800, 1_400], "tier": ["a", "b", "c"]})

    def q(left):
        return (
            bt.from_arrow(left)
            .join(bt.from_arrow(iv), how="cross")
            .filter((bt.col("x") >= bt.col("lo")) & (bt.col("x") <= bt.col("hi")))
            .collect()
        )

    assert _rows(pa.concat_tables([q(p) for p in _partitions(t, parts)])) == _rows(q(t))


def _find_ir(node: object, op: str) -> dict | None:
    """The first `{"op": op, ...}` dict in an IR tree."""
    if isinstance(node, dict):
        if node.get("op") == op:
            return node
        for value in node.values():
            found = _find_ir(value, op)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_ir(item, op)
            if found is not None:
                return found
    return None


@pytest.mark.unit
def test_range_join_reducer_ir_matches_the_wire_contract(bands):
    """The per-task IR must be the planner's own `range_join` node with only the inputs
    swapped for the per-task scans. A drift in any other field is a silent wire-contract
    bug that no differential test would see, because only the distributed path sends it."""
    import batcher.kyber as kyber
    from batcher.dist.executor import _range_join_reducer_ir

    ds = (
        bt.from_arrow(_table())
        .join(bt.from_arrow(bands), how="cross")
        .filter(bt.col("x") < bt.col("lo"))
    )
    physical = kyber.optimize(ds._plan, sources=ds._sources)
    planned = _find_ir(physical.ir, "range_join")
    assert planned is not None, "the cartesian+filter rewrite did not produce a range_join"

    node = _real_range_join(planned, bt.from_arrow(_table()), bt.from_arrow(bands))
    reducer = _range_join_reducer_ir(node)
    assert reducer["op"] == "range_join"
    assert reducer["left"] == {"op": "scan", "source_id": 0}
    assert reducer["right"] == {"op": "scan", "source_id": 1}
    for key in ("conditions", "join_type", "output"):
        assert reducer[key] == planned[key], key
    # The drift guard: every key the single-node lowering emits, other than the two inputs,
    # must appear in the reducer with the same value. A field added to `RangeJoin` and
    # forgotten in the distributed path fails here — which is the whole point, since only
    # the distributed path sends this IR and no differential test would see it.
    single = node.to_ir()
    assert {k: v for k, v in single.items() if k not in ("left", "right")} == {
        k: v for k, v in reducer.items() if k not in ("left", "right")
    }


def _real_range_join(ir: dict, left, right):
    """A **real** `RangeJoin` carrying the conditions/join_type/output the planner emitted.

    A look-alike object with the same attribute names would pass this test forever. The
    reducer and the single-node lowering now share `RangeJoin.shape_ir`, so only a real node
    exercises that sharing — and only a real node grows a new field when the node does,
    which is what makes the drift guard below able to fail.
    """
    from batcher.plan.logical import JoinOutputCol, RangeCondition, RangeJoin

    return RangeJoin(
        left._plan,
        right._plan,
        tuple(
            RangeCondition(left_key=c["left_key"], right_key=c["right_key"], op=c["op"])
            for c in ir["conditions"]
        ),
        ir["join_type"],
        tuple(
            JoinOutputCol(side=o["side"], name=o["name"], alias=o["alias"]) for o in ir["output"]
        ),
    )


# --- the dispatcher routes these shapes instead of refusing them ----------------
#
# Both used to fall through every branch of `_dispatch` to `_unsupported`, which raises on
# splittable (real distributed) data rather than silently running the whole query on one
# node. These pin the routing itself: the delegate is stubbed, so nothing reaches Ray and
# the test is about the decision, not the plumbing.


class _FakeSplittableSource:
    """A source `_is_splittable_source` accepts, with no IO behind it."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def splits(self, *_a, **_k):
        return [object(), object()]


def _routes(monkeypatch, ds, delegate_module: str, delegate_name: str):
    """Run `_dispatch` over `ds` with a splittable source, capturing the delegate call."""
    import importlib

    from batcher.dist import executor as dist_executor

    seen: list = []
    module = importlib.import_module(delegate_module)
    monkeypatch.setattr(
        module, delegate_name, lambda *a, **k: seen.append((a, k)) or _SENTINEL, raising=True
    )
    monkeypatch.setattr(dist_executor, "_is_splittable_source", lambda _s: True)
    monkeypatch.setattr(dist_executor, "_ensure_ray", lambda *_a, **_k: None)
    sources = [_FakeSplittableSource(_table()) for _ in ds._sources]
    return seen, dist_executor, sources


_SENTINEL = pa.table({"sentinel": [1]})


@pytest.mark.unit
def test_dispatcher_routes_fixed_count_sample_to_the_map_partial(monkeypatch):
    ds = bt.from_arrow(_table()).sample(n=5, seed=42)
    seen, dist_executor, sources = _routes(
        monkeypatch, ds, "batcher.dist.executors.map", "_distributed_map"
    )
    monkeypatch.setattr(dist_executor, "_apply_above", lambda _above, table: table, raising=True)
    out = dist_executor._dispatch(ds._plan, sources, 4, "disk")
    assert seen, "a fixed-count sample did not reach the distributed map partial"
    assert out is _SENTINEL


@pytest.mark.unit
def test_dispatcher_routes_range_join_to_the_broadcast_probe(monkeypatch, bands):
    import batcher.kyber as kyber
    from batcher.plan.logical import RangeJoin
    from batcher.plan.visitor import walk

    ds = (
        bt.from_arrow(_table())
        .join(bt.from_arrow(bands), how="cross")
        .filter(bt.col("x") < bt.col("lo"))
    )
    # The rewrite lives in Kyber, so dispatch against the plan the optimizer produces.
    kyber.optimize(ds._plan, sources=ds._sources)
    logical = _optimized_logical(ds)
    assert any(isinstance(n, RangeJoin) for n in walk(logical))

    seen, dist_executor, sources = _routes(
        monkeypatch, ds, "batcher.dist.executors.join", "broadcast_probe_join"
    )
    out = dist_executor._dispatch(logical, sources, 4, "disk")
    assert seen, "a range join did not reach the broadcast probe"
    assert out is _SENTINEL


def _optimized_logical(ds):
    """The optimized *logical* plan — what `api.orchestration.stages` actually hands the
    distributed dispatcher ("the *optimized* logical plan is what gets distributed, not the
    raw one"). The `RangeJoin` only exists after Kyber's rewrite, so dispatching the raw
    plan would test a shape the executor never receives."""
    import batcher.kyber as kyber

    _physical, logical_opt, _decisions = kyber.optimize_full(ds._plan, sources=ds._sources)
    return logical_opt

"""A distributed top-N returns the top-N, with no exchange.

`ORDER BY ... LIMIT k` needs its rows compared, not co-located: a row among the global `k`
smallest is among its own partition's `k` smallest, so the union of the per-partition
top-`k`s contains the answer and re-applying the operator to that union selects it. The
Flight transport has always taken that route; the disk transport range-partitioned every row
across the cluster and then sliced the first `k` off the front, so the same query paid a full
all-to-all exchange or none depending on which transport the topology resolved to.

`_topn_task` is a plain function of its partition file — `_ensure_ray` only rebinds it to a
remote wrapper at query time — so these run the real engine over real Arrow with no cluster,
which is the only way to check it on a box whose Ray is contended.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pyarrow as pa
import pytest

from batcher._internal.native import engine
from batcher.dist.executors import sort as sort_mod
from batcher.dist.shuffle_io import read_ipc, write_ipc

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([("v", pa.int64()), ("w", pa.int64())])


def _plans(descending: bool, k: int):
    """The `(local_ir, merge_ir)` pair `_topn_task` takes, built the way the executor builds
    them — through the plan node's own `shape_ir`, so the test cannot agree with itself
    about an encoding the engine would reject."""
    import batcher as bt

    node = bt.from_pydict({"v": [1], "w": [1]}).sort("v", descending=descending)._plan
    node = dataclasses.replace(node, limit=k)
    ir = json.dumps({**node.shape_ir(), "input": {"op": "scan", "source_id": 0}})
    return ir, ir


def _table(n: int, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    return pa.table(
        {"v": rng.integers(0, 10_000, n).astype("int64"), "w": np.arange(n, dtype="int64")}
    )


def _partition(tmp_path, name: str, table: pa.Table) -> str:
    batches = table.to_batches() or [pa.RecordBatch.from_pylist([], schema=_SCHEMA)]
    return write_ipc(batches, str(tmp_path / name))


def _run(tmp_path, tables, k, descending, tag="p"):
    """Drive the whole shape: per-partition task, then the driver's running merge."""
    nat = engine()
    local_ir, merge_ir = _plans(descending, k)
    merged: list = []
    for i, t in enumerate(tables):
        part = _partition(tmp_path, f"{tag}{i}.arrow", t)
        out = sort_mod._topn_task(local_ir, merge_ir, part, str(tmp_path), f"{tag}{i}", "")
        if out is None:
            continue
        arrived = [b for b in read_ipc(out) if b.num_rows > 0]
        if arrived:
            merged = list(nat.execute_plan(merge_ir, [merged + arrived], ""))
    return pa.Table.from_batches(merged).to_pylist() if merged else []


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("k", [1, 5, 50])
@pytest.mark.parametrize("n_parts", [1, 3, 8])
def test_the_merged_top_n_is_the_single_node_top_n(tmp_path, descending, k, n_parts):
    """The whole claim, against the engine's own answer over the concatenated relation."""
    nat = engine()
    tables = [_table(500, seed=i) for i in range(n_parts)]
    got = _run(tmp_path, tables, k, descending)

    whole = pa.concat_tables(tables)
    _local, merge_ir = _plans(descending, k)
    want = pa.Table.from_batches(
        list(nat.execute_plan(merge_ir, [whole.to_batches()], ""))
    ).to_pylist()

    # Sort keys must match exactly and in order; which of several rows tied at the k-th place
    # is returned is genuinely free, so compare the key column rather than whole rows.
    assert [r["v"] for r in got] == [r["v"] for r in want]


@pytest.mark.parametrize("descending", [False, True])
def test_k_larger_than_the_relation_returns_everything_sorted(tmp_path, descending):
    tables = [_table(20, seed=i) for i in range(3)]
    got = [r["v"] for r in _run(tmp_path, tables, 1000, descending)]
    assert len(got) == 60
    assert got == sorted(got, reverse=descending)


def test_a_partition_with_no_rows_publishes_nothing(tmp_path):
    """An empty partition must return `None`, not a file with no batch in it — `write_ipc`
    refuses the latter, so the difference is a dropped input against a raised IOError in the
    middle of the stage."""
    local_ir, merge_ir = _plans(False, 5)
    part = _partition(tmp_path, "empty.arrow", _SCHEMA.empty_table())
    assert sort_mod._topn_task(local_ir, merge_ir, part, str(tmp_path), "e", "") is None


def test_an_empty_partition_beside_full_ones_does_not_change_the_answer(tmp_path):
    """The mergeability argument has to survive a partition contributing nothing, which is
    the ordinary case once a pushed-down predicate empties one split."""
    full = [_table(200, seed=1), _table(200, seed=2)]
    with_empty = [full[0], _SCHEMA.empty_table(), full[1]]
    assert [r["v"] for r in _run(tmp_path, with_empty, 10, False, tag="a")] == [
        r["v"] for r in _run(tmp_path, full, 10, False, tag="b")
    ]


def test_the_fold_is_bounded_by_k_not_by_the_partition_count(tmp_path):
    """The driver's peak is what makes this scale: it holds one running top-`k` plus the
    partition it is merging, never `partitions x k`. Twenty partitions, and the running
    result never exceeds `k`."""
    nat = engine()
    local_ir, merge_ir = _plans(False, 10)
    merged: list = []
    widths = []
    for i in range(20):
        part = _partition(tmp_path, f"q{i}.arrow", _table(300, seed=i))
        out = sort_mod._topn_task(local_ir, merge_ir, part, str(tmp_path), f"q{i}", "")
        arrived = [b for b in read_ipc(out) if b.num_rows > 0]
        merged = list(nat.execute_plan(merge_ir, [merged + arrived], ""))
        widths.append(sum(b.num_rows for b in merged))
    assert max(widths) == 10


def test_a_worker_never_materializes_its_partition_to_pick_k_rows(tmp_path):
    """`streaming_topn` folds the partition a chunk at a time, so a worker holding a node's
    share of a large input keeps `k` rows plus one chunk — not the share. Checked by giving
    the fold a chunk budget far below the partition and demanding the same answer."""
    from batcher.dist.executors.partition_io import streaming_topn

    nat = engine()
    local_ir, merge_ir = _plans(False, 5)
    table = _table(4_000, seed=42)

    whole = list(nat.execute_plan(local_ir, [table.to_batches()], ""))
    chunked = streaming_topn(
        nat, local_ir, merge_ir, iter(table.to_batches()), "", chunk_bytes=1024
    )
    assert [r["v"] for r in pa.Table.from_batches(whole).to_pylist()] == [
        r["v"] for r in pa.Table.from_batches(list(chunked)).to_pylist()
    ]


# --- the keyed row shuffle reports its exact row count -------------------------------


def test_the_keyed_reducer_reports_the_rows_it_wrote(tmp_path):
    """A caller keeping the dedup's result partitioned has to size the intermediate without
    reading it back — reading it back being the Θ(relation) driver term the partitioned path
    exists to remove. So the reducer returns `(path, rows, metrics)`, and `rows` has to be
    the count actually in the file."""
    import batcher as bt
    from batcher.dist.executors import keyed_shuffle
    from batcher.dist.executors.keyed_shuffle import scan_rooted_ir

    node = bt.from_pydict({"v": [1], "w": [1]}).distinct(subset=["v"])._plan
    reduce_ir = scan_rooted_ir(node)

    rows = pa.table({"v": [1, 1, 2, 3, 3, 3], "w": [1, 2, 3, 4, 5, 6]})
    part = write_ipc(rows.to_batches(), str(tmp_path / "bucket.arrow"))
    path, n, _metrics = keyed_shuffle._reduce_task(reduce_ir, [part], str(tmp_path), 0, "", "t")
    assert path is not None
    assert n == sum(b.num_rows for b in read_ipc(path)) == 3


def test_the_keyed_reducer_reports_zero_for_an_empty_bucket(tmp_path):
    """An empty bucket must be `(None, 0, ...)`, not a path with nothing behind it — the
    materialized source drops it by the `None` and sizes itself by the count."""
    import batcher as bt
    from batcher.dist.executors import keyed_shuffle
    from batcher.dist.executors.keyed_shuffle import scan_rooted_ir

    node = bt.from_pydict({"v": [1], "w": [1]}).distinct(subset=["v"])._plan
    part = write_ipc(
        [pa.RecordBatch.from_pylist([], schema=_SCHEMA)], str(tmp_path / "empty_bucket.arrow")
    )
    path, rows, _metrics = keyed_shuffle._reduce_task(
        scan_rooted_ir(node), [part], str(tmp_path), 0, "", "t"
    )
    assert (path, rows) == (None, 0)

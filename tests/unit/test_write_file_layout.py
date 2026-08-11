"""`FileLayout` arithmetic, and that a distributed write shard actually applies it.

The layout (`max_rows_per_file` / `num_files` / `target_size_mb`) used to be resolved on
the driver and dropped on the floor by every distributed path, so a clustered
``repartition(target_size_mb=...).write(...)`` produced one unbounded file per shard. These
tests pin both halves: the arithmetic, and the fact that the shard entry points hand the
resolved cap (and `resume`) to the sink.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist.executors.write import _shard_rows_per_file, _write_shard, _write_shards
from batcher.io.base._layout import FileLayout
from batcher.io.manifest import WrittenFile

pytestmark = pytest.mark.unit


class _RecordingSink:
    """A sink that records the keywords each write entry point was called with.

    `_write_plan_shard` reaches a sink three ways, and which one it picks is itself part
    of the contract: a layout naming a *total* (`num_files`) has to materialize the shard
    to resolve it, while the default layout streams. Recording all three in one list lets
    a test assert the route as well as the keywords.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.suffix = ".rec"

    def write_partitioned(self, table, path, **kwargs):
        self.calls.append({"via": "partitioned", "rows": table.num_rows, "path": path, **kwargs})
        return [WrittenFile(path=path, rows=table.num_rows, bytes=0)]

    def write_stream_shard(self, batches, directory, **kwargs):
        rows = sum(b.num_rows for b in batches)
        self.calls.append({"via": "stream_shard", "rows": rows, "path": directory, **kwargs})
        return WrittenFile(path=directory, rows=rows, bytes=0)

    def write_stream_parts(self, batches, directory, **kwargs):
        rows = sum(b.num_rows for b in batches)
        self.calls.append({"via": "stream_parts", "rows": rows, "path": directory, **kwargs})
        return [WrittenFile(path=directory, rows=rows, bytes=0)]


def test_default_layout_imposes_no_cap():
    assert FileLayout().is_default
    assert FileLayout().rows_per_file(1000, 4000) is None


def test_explicit_row_cap_wins_over_the_derived_ones():
    layout = FileLayout(max_rows_per_file=7, num_files=2, target_bytes_per_file=1)
    assert layout.rows_per_file(1000, 4000) == 7


def test_num_files_divides_the_rows_and_never_undershoots():
    # 10 rows into 3 files is 4+4+2 -- ceiling, so three files rather than four.
    assert FileLayout(num_files=3).rows_per_file(10, 40) == 4
    assert FileLayout(num_files=1).rows_per_file(10, 40) == 10
    # More files than rows still caps at one row per file rather than zero.
    assert FileLayout(num_files=100).rows_per_file(3, 12) == 1


def test_target_bytes_scales_by_measured_bytes_per_row():
    # 1000 rows occupying 8000 bytes is 8 bytes/row, so a 400-byte target is 50 rows.
    assert FileLayout(target_bytes_per_file=400).rows_per_file(1000, 8000) == 50
    # A target smaller than one row still writes one row per file, never zero.
    assert FileLayout(target_bytes_per_file=1).rows_per_file(1000, 8000) == 1


def test_empty_or_sizeless_input_resolves_to_no_cap():
    assert FileLayout(num_files=4).rows_per_file(0, 0) is None
    # nbytes==0 with rows>0 cannot yield a bytes-per-row, so the byte target declines.
    assert FileLayout(target_bytes_per_file=1024).rows_per_file(10, 0) is None


def test_num_files_is_a_global_budget_split_across_shards():
    layout = FileLayout(num_files=7)
    shares = [layout.for_shard(i, 3).num_files for i in range(3)]
    assert shares == [3, 2, 2]
    assert sum(shares) == 7


def test_a_row_cap_and_a_byte_target_are_already_per_shard():
    layout = FileLayout(max_rows_per_file=100, target_bytes_per_file=999)
    assert layout.for_shard(2, 5) == layout


def test_every_shard_gets_at_least_one_file_when_shards_outnumber_files():
    layout = FileLayout(num_files=2)
    assert [layout.for_shard(i, 4).num_files for i in range(4)] == [1, 1, 1, 1]


def test_shard_rows_per_file_tolerates_an_absent_layout():
    assert _shard_rows_per_file(pa.table({"x": [1, 2]}), None) is None


def test_write_shard_passes_the_resolved_cap_and_resume_to_the_sink():
    sink = _RecordingSink()
    table = pa.table({"x": list(range(100))})
    _write_shard(sink, table, "out", None, 3, FileLayout(num_files=4), True)
    (call,) = sink.calls
    assert call["file_index"] == 3
    assert call["resume"] is True
    assert call["max_rows_per_file"] == 25


def test_write_shard_without_a_layout_leaves_the_file_uncapped():
    sink = _RecordingSink()
    _write_shard(sink, pa.table({"x": [1, 2, 3]}), "out", None, 0)
    (call,) = sink.calls
    assert call["max_rows_per_file"] is None
    assert call["resume"] is False


def _keys(shard: pa.Table, column: str = "p") -> set:
    return set(shard.column(column).to_pylist())


def test_unpartitioned_shards_are_contiguous_row_ranges():
    table = pa.table({"x": list(range(10))})
    shards = _write_shards(table, None, 4)
    assert [s.num_rows for s in shards] == [3, 3, 3, 1]
    assert [v for s in shards for v in s.column("x").to_pylist()] == list(range(10))


def test_an_empty_result_still_produces_one_shard():
    for partition_by in (None, ["p"]):
        shards = _write_shards(pa.table({"p": pa.array([], pa.string())}), partition_by, 4)
        assert len(shards) == 1
        assert shards[0].num_rows == 0


def test_a_partition_key_lands_wholly_inside_one_shard():
    # Interleaved keys: a row-range split would put every key in every shard.
    table = pa.table({"p": [f"k{i % 5}" for i in range(100)], "x": list(range(100))})
    shards = _write_shards(table, ["p"], 4)
    seen: set = set()
    for shard in shards:
        assert not (_keys(shard) & seen), "a key was split across shards"
        seen |= _keys(shard)
    assert seen == {f"k{i}" for i in range(5)}
    assert sum(s.num_rows for s in shards) == 100


def test_partitioned_sharding_preserves_every_row():
    table = pa.table({"p": [i % 7 for i in range(200)], "x": list(range(200))})
    shards = _write_shards(table, ["p"], 3)
    got = sorted(v for s in shards for v in s.column("x").to_pylist())
    assert got == list(range(200))


def test_skewed_keys_are_packed_largest_first_so_no_shard_holds_everything():
    # One key holds half the rows; the rest are small. Greedy LPT should isolate the
    # hot key and spread the remainder rather than trailing them all behind it.
    rows = ["hot"] * 100 + [f"c{i}" for i in range(100)]
    table = pa.table({"p": rows, "x": list(range(200))})
    shards = _write_shards(table, ["p"], 4)
    loads = sorted(s.num_rows for s in shards)
    assert loads[-1] == 100  # the hot key alone
    assert sum(loads) == 200
    assert all(load > 0 for load in loads)


def test_sharding_is_deterministic_so_resume_can_rely_on_it():
    table = pa.table({"p": [i % 11 for i in range(150)], "x": list(range(150))})
    first = _write_shards(table, ["p"], 5)
    second = _write_shards(table, ["p"], 5)
    assert [s.to_pylist() for s in first] == [s.to_pylist() for s in second]


def test_null_and_nan_keys_do_not_shatter_into_one_shard_per_row():
    table = pa.table(
        {
            "p": pa.array([None, None, float("nan"), float("nan"), 1.0], pa.float64()),
            "x": list(range(5)),
        }
    )
    shards = _write_shards(table, ["p"], 4)
    # Three distinct keys (null, NaN, 1.0), so at most three shards -- not five.
    assert len(shards) <= 3
    assert sum(s.num_rows for s in shards) == 5


def test_more_workers_than_keys_produces_one_shard_per_key():
    table = pa.table({"p": ["a", "b", "a", "b"], "x": [1, 2, 3, 4]})
    shards = _write_shards(table, ["p"], 16)
    assert len(shards) == 2


def test_distributed_write_hands_every_shard_its_layout_and_resume(monkeypatch):
    """The end-to-end threading: driver options reach the per-shard `write_partitioned`."""
    import batcher.dist.executors.write as dw

    sink = _RecordingSink()
    original = dw._write_shard

    class _Remote:
        """Stand in for a `ray.remote` wrapper: `.remote(...)` yields a callable thunk."""

        @staticmethod
        def remote(*args):
            return lambda: original(*args)

    monkeypatch.setattr("batcher.dist.executors.ray_runtime._ensure_ray", lambda workers: None)
    monkeypatch.setattr(dw, "_write_shard", _Remote)
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.gather_map_results",
        lambda submit, n, *a, **k: [submit(i)() for i in range(n)],
    )
    table = pa.table({"p": ["a"] * 40 + ["b"] * 40, "x": list(range(80))})
    manifest = dw._distributed_write(
        sink, table, "out", ["p"], 2, layout=FileLayout(num_files=4), resume=True
    )
    assert len(sink.calls) == 2, "one call per partition-aligned shard"
    for call in sink.calls:
        assert call["resume"] is True
        assert call["partition_by"] == ["p"]
        # num_files=4 over 2 shards is 2 each; 40 rows / 2 files = 20 rows per file.
        assert call["max_rows_per_file"] == 20
    assert manifest.total_rows == 80


@pytest.fixture(scope="module")
def splittable() -> pa.Table:
    """Two keys, each just over the floor below which a key is never split.

    Built once: the point is only to clear `skew_min_bucket_rows`, and materializing a
    six-figure key column per test costs more than everything else here put together.
    """
    from batcher.config import active_config

    per_key = active_config().execution.skew_min_bucket_rows + 1
    keys = pa.chunked_array([pa.array(["a"] * per_key), pa.array(["b"] * per_key)])
    return pa.table({"p": keys, "x": pa.array(range(2 * per_key), pa.int64())})


def test_a_key_larger_than_a_worker_share_is_split_so_workers_are_not_idle(splittable):
    """Grouping by key alone starves a cluster when there are few keys.

    Two partitions across five workers must not become two shards. The table clears
    `skew_min_bucket_rows`, which is what turns the splitting on; below that floor a
    small result is deliberately left whole.
    """
    shards = _write_shards(splittable, ["p"], 5)
    assert len(shards) > 2, "more workers than keys must still all get work"
    assert sum(s.num_rows for s in shards) == splittable.num_rows
    assert max(s.num_rows for s in shards) <= splittable.num_rows // 2


def test_a_small_result_is_never_split_below_the_skew_floor():
    # 4 rows over 16 workers: the even share is one row, but splitting there would make
    # one file per row -- the small-files problem manufactured from nothing.
    table = pa.table({"p": ["a", "b", "a", "b"], "x": [1, 2, 3, 4]})
    shards = _write_shards(table, ["p"], 16)
    assert len(shards) == 2


@pytest.mark.parametrize("workers", [2, 3, 5, 8])
def test_splitting_a_key_never_duplicates_or_drops_a_row(splittable, workers):
    """The clamp on the last piece of a split run.

    `Table.slice(offset, length)` clamps to the end of the table, not to the end of the
    run being cut, so an unclamped final piece runs on into the next key -- and those rows
    are then written twice, silently, and only on inputs large enough to be split at all.
    """
    shards = _write_shards(splittable, ["p"], workers)
    total = sum(s.num_rows for s in shards)
    assert total == splittable.num_rows
    seen = set()
    for shard in shards:
        seen.update(shard.column("x").to_pylist())
    assert len(seen) == splittable.num_rows, "a row was written twice"


def test_the_streaming_write_shard_resolves_the_layout_on_the_worker(monkeypatch):
    """The path that never materializes on the driver, so only the worker can size a file.

    `_write_plan_shard` is what a Ray task runs: it executes the plan over its own
    partition and writes the result. The layout has to arrive with it and be resolved
    against the rows *it* produced, since the driver never sees them.
    """
    import json

    import batcher as bt
    import batcher.io.sink as sink_module
    from batcher.dist.executors.ray_runtime import engine_config_json
    from batcher.dist.executors.write import _write_plan_shard

    sink = _RecordingSink()
    real_get = sink_module.SINKS.get
    monkeypatch.setattr(
        sink_module.SINKS,
        "get",
        lambda name: (lambda **_: sink) if name == "rec" else real_get(name),
    )

    plan_ir = json.dumps(bt.from_pydict({"v": list(range(100))})._plan.to_ir())
    batch = pa.record_batch({"v": pa.array(range(100))})
    written = _write_plan_shard(
        plan_ir,
        {"batches": [batch]},
        "rec",
        {},
        "out",
        None,
        2,
        engine_config_json(),
        FileLayout(num_files=4),
        True,
    )
    (call,) = sink.calls
    assert call["file_index"] == 2
    assert call["resume"] is True
    assert call["max_rows_per_file"] == 25  # 100 rows over the shard's 4-file budget
    assert [w.rows for w in written] == [100]


def test_the_streaming_write_shard_without_a_layout_streams_one_uncapped_file(monkeypatch):
    """No layout is one uncapped file per shard — and the shard *streams* it.

    The route matters as much as the cap. Materializing the shard to hand the sink a table
    made a worker's peak memory its whole share of the result, so doubling the input on a
    fixed cluster doubled every worker's peak. With no total to resolve there is nothing to
    count first, so this shape goes to `write_stream_shard` and the worker holds one batch.
    """
    import json

    import batcher as bt
    import batcher.io.sink as sink_module
    from batcher.dist.executors.ray_runtime import engine_config_json
    from batcher.dist.executors.write import _write_plan_shard

    sink = _RecordingSink()
    real_get = sink_module.SINKS.get
    monkeypatch.setattr(
        sink_module.SINKS,
        "get",
        lambda name: (lambda **_: sink) if name == "rec" else real_get(name),
    )

    plan_ir = json.dumps(bt.from_pydict({"v": [1, 2, 3]})._plan.to_ir())
    batch = pa.record_batch({"v": pa.array([1, 2, 3])})
    written = _write_plan_shard(
        plan_ir, {"batches": [batch]}, "rec", {}, "out", None, 0, engine_config_json()
    )
    (call,) = sink.calls
    assert call["via"] == "stream_shard"  # not materialized
    assert "max_rows_per_file" not in call  # nothing caps the file
    assert call["file_index"] == 0
    assert call["resume"] is False
    assert [w.rows for w in written] == [3]

"""A merge's output files must not collide with the files it deliberately did not read.

This is the one distributed failure mode that is *specific to merge*, and it needs no Ray
to provoke — only the sink and a file index. A copy-on-write merge writes new data files
**into a directory that still holds live data**: the files pruning proved could not match
are untouched, and are the table. `FileSink` names shards ``part-{file_index}.parquet``,
which is deterministic on purpose (a retried shard overwrites itself, so a resumed write is
idempotent) — and which, here, means shard 0 writes straight over ``part-00000.parquet``, a
file this merge never read. That is silent data loss.

The guard is a per-write token in the file name (`ParquetSink.file_token`). It has to ride in
`sink_kwargs`, not on a sink *object*, because a distributed write reconstructs its sink on
every worker from those kwargs — a token carried on the object would be lost in transit and
every worker would collide.

`tests/integration/test_distributed_merge.py` proves the same property end to end on a real
Ray cluster; these prove it at the seam, where the failure actually lives.
"""

from __future__ import annotations

import glob
import pathlib

import pyarrow as pa

import batcher as bt
from batcher.api.merge import plan_merge, simple_clauses
from batcher.io.formats.base import SINKS
from batcher.io.formats.structured.parquet.sink import ParquetSink

_CLAUSES = simple_clauses("update", "insert")


def test_a_tokenized_sink_never_reuses_a_part_name() -> None:
    plain = ParquetSink()
    tokenized = ParquetSink(file_token="abc123")
    assert plain.suffix == ".parquet"
    assert tokenized.suffix == "-abc123.parquet"
    assert tokenized.suffix != plain.suffix


def test_the_token_survives_sink_reconstruction_from_sink_kwargs() -> None:
    """A worker rebuilds its sink from `sink_kwargs`; the token must come back with it."""
    sink_kwargs = {"file_token": "deadbeef"}
    worker_sink = SINKS.get("parquet")(**sink_kwargs)
    assert worker_sink.suffix == "-deadbeef.parquet"


def test_shards_of_one_merge_do_not_collide_with_each_other(tmp_path) -> None:
    """Every worker writes with the same token but its own `file_index`."""
    path = str(tmp_path / "t")
    sink = ParquetSink(file_token="tok")
    written = []
    for index in range(4):  # four workers, four shards
        shard = pa.table({"id": [index], "v": [index * 10]})
        written += sink.write_partitioned(shard, path, file_index=index)

    names = [f.path for f in written]
    assert len(set(names)) == 4, f"two shards wrote the same file: {names}"
    assert bt.read.parquet(path).collect().num_rows == 4


def test_merge_output_never_overwrites_a_file_pruning_preserved(tmp_path) -> None:
    """The real thing: the survivors' bytes must be exactly what they were.

    A merge writes its output beside files it never read. If a shard reused one of their
    names, the file's *content* would change — so comparing the surviving files byte for
    byte is what actually proves nothing was clobbered.
    """
    path = str(tmp_path / "t")
    table = pa.table({"id": list(range(20)), "v": [i * 10 for i in range(20)]})
    bt.from_arrow(table).write.parquet(path, max_rows_per_file=2)  # 10 files

    changes = pa.table({"id": [19], "v": [-1]})  # one key ⇒ one file can match
    plan = plan_merge(bt.from_arrow(changes), path, ["id"], _CLAUSES, format="parquet")
    assert plan.pruned and len(plan.rewritten) == 1, "expected pruning to skip 9 of 10 files"

    survivors = {p: pathlib.Path(p).read_bytes() for p in plan.skipped}

    (
        bt.from_arrow(changes)
        .write.merge_into(path, on="id", format="parquet")
        .when_matched()
        .update_all()
        .when_not_matched()
        .insert_all()
        .execute()
    )

    for skipped_path, original in survivors.items():
        assert glob.glob(skipped_path), f"{skipped_path} was deleted; it was never read"
        assert pathlib.Path(skipped_path).read_bytes() == original, (
            f"{skipped_path} was rewritten — a merge output file took its name"
        )

    out = bt.read.parquet(path).collect().to_pydict()
    rows = dict(zip(out["id"], out["v"], strict=True))
    assert len(rows) == 20 and rows[19] == -1 and rows[0] == 0


def test_the_replaced_files_are_deleted_so_no_row_is_duplicated(tmp_path) -> None:
    """The other half of the swap: the file whose rows were rewritten must go away."""
    path = str(tmp_path / "t")
    bt.from_arrow(
        pa.table({"id": list(range(20)), "v": [i * 10 for i in range(20)]})
    ).write.parquet(path, max_rows_per_file=2)
    changes = pa.table({"id": [19], "v": [-1]})
    plan = plan_merge(bt.from_arrow(changes), path, ["id"], _CLAUSES, format="parquet")
    replaced = plan.rewritten

    (
        bt.from_arrow(changes)
        .write.merge_into(path, on="id", format="parquet")
        .when_matched()
        .update_all()
        .when_not_matched()
        .insert_all()
        .execute()
    )

    for gone in replaced:
        assert not glob.glob(gone), f"{gone} survived; its rows are now duplicated"
    out = bt.read.parquet(path).collect().to_pydict()
    assert len(out["id"]) == len(set(out["id"])) == 20


def test_a_source_with_duplicate_keys_is_rejected(tmp_path) -> None:
    """SQL calls this a cardinality violation, and it must be an error, not a coin flip.

    Two source rows claiming the same target row have no defined winner. Silently picking
    one (or, worse, emitting the target row twice) is the failure mode a merge must not
    have — so the source's row count and its distinct-key count are compared, and a
    mismatch raises before anything is written.
    """
    import pytest

    from batcher._internal.errors import PlanError

    path = str(tmp_path / "t.parquet")
    bt.from_arrow(pa.table({"id": [1, 2], "v": [10, 20]})).write.parquet(path)

    dupes = pa.table({"id": [2, 2], "v": [98, 99]})  # two rows for key 2
    with pytest.raises(PlanError, match="cardinality violation"):
        (
            bt.from_arrow(dupes)
            .write.merge_into(path, on="id")
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
            .execute()
        )

    # And it raised *before* writing: the target is untouched.
    out = bt.read.parquet(path).collect().to_pydict()
    assert dict(zip(out["id"], out["v"], strict=True)) == {1: 10, 2: 20}


def test_deduplicating_the_source_makes_it_valid(tmp_path) -> None:
    """The error message names the fix; this is that fix working."""
    path = str(tmp_path / "t.parquet")
    bt.from_arrow(pa.table({"id": [1, 2], "v": [10, 20]})).write.parquet(path)

    feed = pa.table({"id": [2, 2], "v": [98, 99], "seq": [1, 2]})
    latest = (
        bt.from_arrow(feed).distinct(subset=["id"], keep="last", order_by="seq").select("id", "v")
    )
    latest.write.merge_into(
        path, on="id"
    ).when_matched().update_all().when_not_matched().insert_all().execute()

    out = bt.read.parquet(path).collect().to_pydict()
    assert dict(zip(out["id"], out["v"], strict=True)) == {1: 10, 2: 99}  # the latest won


def test_the_merge_forwards_distributed_and_the_token_to_the_writer(tmp_path, monkeypatch) -> None:
    """The distributed knobs must actually reach `_write` — silently dropping them would make
    every "distributed" merge a single-node one, and nothing would fail.

    Checked here rather than only on a cluster because this is a *plumbing* bug, not a
    concurrency one: it reproduces with no workers at all.
    """
    import batcher.api.terminal.core as core

    real = core._write
    seen: dict = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "_write", spy)

    path = str(tmp_path / "t")
    table = pa.table({"id": list(range(20)), "v": [i * 10 for i in range(20)]})
    bt.from_arrow(table).write.parquet(path, max_rows_per_file=2)

    changes = pa.table({"id": [19], "v": [-1]})
    (
        bt.from_arrow(changes)
        .write.merge_into(path, on="id", format="parquet", distributed=False, num_workers=3)
        .when_matched()
        .update_all()
        .when_not_matched()
        .insert_all()
        .execute()
    )

    assert seen.get("directory") is True, "a directory target must not be written as a single file"
    assert seen.get("distributed") is False, "distributed= was dropped on the way to the writer"
    assert seen.get("num_workers") == 3, "num_workers= was dropped on the way to the writer"
    token = (seen.get("sink_kwargs") or {}).get("file_token")
    assert token, "no per-merge file token — a shard could overwrite a file pruning preserved"

"""A streaming write honors `max_rows_per_file`, or says why it cannot.

Without a cap a micro-batch is one file whatever its size, which on a long-running stream
is the small-files problem in its purest form: the file size is whatever the trigger
interval happened to produce, and nothing in the query says otherwise. The option was
accepted and silently ignored, which is worse than either honoring or refusing it.
"""

from __future__ import annotations

import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration


def _drain(path: str, rows: int, **opts):
    query = bt.from_pydict({"v": list(range(rows))}).write(
        path, format="parquet", trigger=bt.Trigger.available_now(), **opts
    )
    query.await_termination()
    return sorted(glob.glob(f"{path}/*.parquet"))


def test_a_capped_micro_batch_is_split_across_files(tmp_path):
    files = _drain(str(tmp_path / "s"), 20, max_rows_per_file=3)
    assert len(files) == 7  # 3,3,3,3,3,3,2
    assert max(pq.read_table(f).num_rows for f in files) == 3


def test_every_row_survives_the_split(tmp_path):
    files = _drain(str(tmp_path / "s"), 20, max_rows_per_file=3)
    got = sorted(v for f in files for v in pq.read_table(f).column("v").to_pylist())
    assert got == list(range(20))


def test_the_chunk_index_joins_the_batch_id_so_names_stay_positional(tmp_path):
    files = _drain(str(tmp_path / "s"), 5, max_rows_per_file=2)
    names = sorted(os.path.basename(f) for f in files)
    assert names == [
        "part-batch00000-00000.parquet",
        "part-batch00000-00001.parquet",
        "part-batch00000-00002.parquet",
    ]


def test_an_uncapped_stream_still_writes_one_file_per_batch(tmp_path):
    files = _drain(str(tmp_path / "s"), 5)
    assert [os.path.basename(f) for f in files] == ["part-batch00000.parquet"]


def test_a_cap_larger_than_the_batch_leaves_the_single_file_name(tmp_path):
    files = _drain(str(tmp_path / "s"), 5, max_rows_per_file=100)
    assert [os.path.basename(f) for f in files] == ["part-batch00000.parquet"]


def test_the_output_reads_back_as_one_relation(tmp_path):
    out = str(tmp_path / "s")
    _drain(out, 20, max_rows_per_file=3)
    assert sorted(bt.read.parquet(out).to_pydict()["v"]) == list(range(20))


def test_a_transactional_target_says_the_cap_has_no_meaning(tmp_path):
    with pytest.raises(PlanError, match="no meaning"):
        bt.from_pydict({"v": [1]}).write(
            str(tmp_path / "d"),
            format="delta",
            trigger=bt.Trigger.available_now(),
            max_rows_per_file=2,
        )


def test_the_distributed_drain_refuses_rather_than_ignoring_the_cap(tmp_path):
    with pytest.raises(PlanError, match="max_rows_per_file"):
        bt.from_pydict({"v": [1]}).write(
            str(tmp_path / "s"),
            format="parquet",
            trigger=bt.Trigger.available_now(),
            distributed=True,
            max_rows_per_file=2,
        )


def _capped_stream(out: str, ckpt: str):
    """Drain a three-batch stream into `out` under a checkpoint, capped at two rows."""
    schema = pa.schema([("v", pa.int64())])

    def feed():
        for i in range(3):
            yield pa.record_batch({"v": pa.array(list(range(i * 10, i * 10 + 5)), pa.int64())})

    query = bt.from_batches(feed, schema, bounded=False).write(
        out,
        format="parquet",
        trigger=bt.Trigger.available_now(),
        checkpoint=ckpt,
        query_name="capped",
        max_rows_per_file=2,
    )
    query.await_termination()
    files = sorted(glob.glob(f"{out}/*.parquet"))
    rows = sorted(v for f in files for v in pq.read_table(f).column("v").to_pylist())
    return files, rows


def test_the_cap_keeps_a_restart_exactly_once(tmp_path):
    """Chunking must not cost the property the batch-position naming exists to give.

    Each chunk is still named by (batch, chunk), so a replayed epoch finds its own files
    and writes nothing — the same idempotence the uncapped single file had.
    """
    out, ckpt = str(tmp_path / "s"), str(tmp_path / "ck")
    first_files, first_rows = _capped_stream(out, ckpt)
    assert len(first_files) == 9  # three batches of five rows, capped at two
    second_files, second_rows = _capped_stream(out, ckpt)
    assert second_rows == first_rows, "a restart duplicated rows"
    assert len(second_files) == len(first_files)


def test_a_chunk_lost_after_the_epoch_is_rewritten_on_restart(tmp_path):
    """Resume is per *chunk*, so losing one does not lose the epoch or duplicate the rest."""
    out, ckpt = str(tmp_path / "s"), str(tmp_path / "ck")
    _, expected = _capped_stream(out, ckpt)
    os.remove(sorted(glob.glob(f"{out}/*.parquet"))[-1])
    _, healed = _capped_stream(out, ckpt)
    assert healed == expected

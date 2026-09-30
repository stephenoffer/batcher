"""A streaming checkpoint written before the 128-bit integer-`SUM` state still resumes.

A running aggregate's checkpoint is its raw partial state, persisted as Arrow IPC under
``<checkpoint>/state/`` and combined with the next micro-batch's partial on restart
(`core.streaming.folds._AggFold`, `core.mergeable.RunningAggregate`). An engine before
`bc-runtime/src/agg/int_sum` wrote an `Int64` `SUM`'s partial as a bare `int64` column; the
engine now writes a marked `struct<__bc_int64_sum_i128: decimal128(38, 0)>` (finding F212). A
restored legacy state therefore meets new partials in the same `combine`, and the engine
widens the legacy column there -- losslessly, since it was already a valid `Int64` total.

The legacy file is produced the only way one can be here: by running the current engine,
then rewriting the state files on disk to the old schema, column for column. A control
asserts the rewrite really produced a bare `int64` column, and that the original was the
marked struct -- otherwise the restart below would be resuming a new-engine checkpoint and
prove nothing.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pytest

import batcher as bt
from batcher.config import option_context

pytestmark = pytest.mark.integration

_MARKER = "__bc_int64_sum_i128"


def _run(ckpt: str, rows: int) -> dict[int, int]:
    """Fold `rows` of the rate source into `SUM(value)` per `value % 3`, against `ckpt`."""
    latest: dict[int, int] = {}

    def collect(table: pa.Table, _batch_id: int) -> None:
        out = table.to_pydict()
        latest.update(zip(out["bucket"], out["total"], strict=True))

    query = (
        bt.read.rate(4, num_rows=rows, pace=False)
        .with_columns(bucket=bt.col("value") % 3)
        .group_by("bucket")
        .agg(total=bt.col("value").sum())
        .write.for_each_batch(
            collect,
            trigger=bt.Trigger.available_now(),
            checkpoint=ckpt,
            output_mode="update",
        )
    )
    query.await_termination()
    return latest


def _to_legacy(table: pa.Table) -> tuple[pa.Table, int]:
    """Rewrite every marked integer-`SUM` column to the bare `int64` the old engine wrote."""
    cols, fields, rewritten = [], [], 0
    for field, column in zip(table.schema, table.columns, strict=True):
        if pa.types.is_struct(field.type) and field.type.names == [_MARKER]:
            column = pc.cast(pc.struct_field(column, [0]), pa.int64())
            field = pa.field(field.name, pa.int64(), field.nullable)
            rewritten += 1
        cols.append(column)
        fields.append(field)
    return pa.Table.from_arrays(cols, schema=pa.schema(fields, table.schema.metadata)), rewritten


def _rewrite_state_as_legacy(ckpt: str) -> int:
    state_dir = os.path.join(ckpt, "state")
    rewritten = 0
    for name in sorted(os.listdir(state_dir)):
        if not name.endswith(".arrow"):
            continue
        path = os.path.join(state_dir, name)
        with open(path, "rb") as fh:
            table = ipc.open_file(fh).read_all()
        legacy, n = _to_legacy(table)
        rewritten += n
        with ipc.new_file(path, legacy.schema) as w:
            w.write_table(legacy)
        with open(path, "rb") as fh:
            back = ipc.open_file(fh).read_all()
        assert all(not pa.types.is_struct(t) for t in back.schema.types), back.schema
    return rewritten


@pytest.mark.parametrize("delta_interval", [0, 1000], ids=["whole-snapshot", "delta-chain"])
def test_a_legacy_int64_sum_checkpoint_resumes_to_the_uninterrupted_answer(
    tmp_path, delta_interval
):
    ckpt = str(tmp_path / "ckpt")
    with option_context("streaming.checkpoint_delta_interval", delta_interval):
        _run(ckpt, rows=12)
        # The control: the current engine's state carried the marked struct, and after the
        # rewrite it carries none -- so what resumes below is genuinely a legacy state.
        assert _rewrite_state_as_legacy(ckpt) > 0, "no marked SUM state was checkpointed"
        resumed = _run(ckpt, rows=24)
    straight = _run(str(tmp_path / "straight"), rows=24)
    assert straight == {b: sum(v for v in range(24) if v % 3 == b) for b in range(3)}
    assert resumed == straight

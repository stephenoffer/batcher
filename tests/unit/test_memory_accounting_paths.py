"""Every byte figure the engine acts on must be readable and must not under-count.

Two failure modes share one cause — a bare `pyarrow` `nbytes` read — and both are silent
until the data is unusual:

* **It raises.** `nbytes` walks the buffer layout and raises `ArrowTypeError` on the *view*
  layouts (`string_view`, `binary_view`, `list_view`). Those arrive from a Parquet reader
  with view types on, from DuckDB's Arrow export, from Polars, and from any Velox-backed
  producer. So a query that only *measured* its input died in the measurement, before a byte
  of work was done, naming a physical layout the user never chose.
* **It under-counts.** Every zero-copy derivation in Arrow — `slice`, `head`, a partition cut
  by offset — addresses a window of its parent's buffers and pins the whole parent. A bound
  charged on `nbytes` therefore admits far more than it means to, which is the OOM the bound
  exists to prevent, arrived at through the bound itself.

`plan.types` answers both: `logical_bytes` where a figure is *reported*, `retained_bytes` /
`total_retained_bytes` where it *decides* something. These tests pin the properties rather
than the call sites, so they keep holding as the sites move.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import Config, MemoryConfig, config_context
from batcher.plan.types import (
    logical_bytes,
    retained_bytes,
    total_logical_bytes,
    total_retained_bytes,
)

pytestmark = pytest.mark.unit

#: The Arrow layouts whose `nbytes` raises. Parameterized as (name, array factory) so a
#: failure names the layout rather than an index.
_VIEW_LAYOUTS = [
    ("string_view", lambda: pa.array(["aa", None, "", "aa"], pa.string_view())),
    ("binary_view", lambda: pa.array([b"aa", None, b"", b"aa"], pa.binary_view())),
    ("list_view", lambda: pa.array([[1, 2], None, [], [1, 2]], pa.list_view(pa.int64()))),
]


@pytest.mark.parametrize(("name", "make"), _VIEW_LAYOUTS, ids=[n for n, _ in _VIEW_LAYOUTS])
def test_a_bare_nbytes_read_still_raises_on_this_layout(name, make):
    """The premise. If pyarrow ever makes `nbytes` total, these helpers become redundant —
    and this test is what says so, rather than the helpers quietly outliving their reason."""
    with pytest.raises(pa.ArrowTypeError):
        _ = pa.record_batch({"x": make()}).nbytes


@pytest.mark.parametrize(("name", "make"), _VIEW_LAYOUTS, ids=[n for n, _ in _VIEW_LAYOUTS])
def test_the_footprint_helpers_answer_on_every_layout(name, make):
    batch = pa.record_batch({"x": make()})
    assert logical_bytes(batch) > 0
    assert retained_bytes(batch) >= logical_bytes(batch)
    assert total_logical_bytes([batch, batch]) == 2 * logical_bytes(batch)
    assert total_retained_bytes([batch, batch]) == 2 * retained_bytes(batch)


def test_retained_bytes_charges_a_slice_for_the_parent_it_pins():
    """The under-count half. A ten-row window of a large column reports a handful of bytes
    and keeps the whole parent alive; a bound reading the first figure does not bind."""
    parent = pa.record_batch({"v": pa.array(range(200_000), pa.int64())})
    window = parent.slice(0, 10)
    assert logical_bytes(window) == 80
    assert retained_bytes(window) > 1_000_000
    assert total_retained_bytes([window]) == retained_bytes(window)


def _view_table(n: int = 4096) -> pa.Table:
    """A table whose key column is a `string_view` — enough rows to cross a morsel."""
    keys = [f"k{i % 97}" if i % 13 else None for i in range(n)]
    return pa.table(
        {
            "s": pa.array(keys, pa.string_view()),
            "b": pa.array([None if v is None else v.encode() for v in keys], pa.binary_view()),
            "v": pa.array(range(n), pa.int64()),
        }
    )


def _plain(table: pa.Table) -> pa.Table:
    """The same rows in the plain spellings — the oracle for every case below."""
    return table.set_column(
        table.schema.get_field_index("s"), "s", table.column("s").cast(pa.string())
    ).set_column(table.schema.get_field_index("b"), "b", table.column("b").cast(pa.binary()))


@pytest.mark.parametrize("spill", [False, True], ids=["in-memory", "spilled"])
def test_a_view_keyed_aggregate_runs_on_both_paths(spill):
    """The spill path reads its input through one tap that sized morsels with a bare
    `nbytes`, so an out-of-core query over a view column failed in the sizing arithmetic."""
    view, plain = _view_table(), None
    plain = _plain(view)
    got = bt.from_arrow(view).group_by("s").agg(n=bt.col("v").count()).collect(spill=spill)
    want = bt.from_arrow(plain).group_by("s").agg(n=bt.col("v").count()).collect()
    assert got.sort_by("s").to_pydict() == want.sort_by("s").to_pydict()


def test_a_view_keyed_aggregate_survives_a_tight_envelope():
    """A budget far below the state forces the spilling breaker, which is the path whose
    input tap does the sizing — so this is the shape the accounting read actually killed."""
    view = _view_table(20_000)
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1 << 21))
    with config_context(cfg):
        got = bt.from_arrow(view).group_by("s").agg(n=bt.col("v").count()).collect()
    want = bt.from_arrow(_plain(view)).group_by("s").agg(n=bt.col("v").count()).collect()
    assert got.sort_by("s").to_pydict() == want.sort_by("s").to_pydict()


def test_a_view_column_streams_through_a_batch_udf():
    """`map_batches` sizes its next batch from the last one's bytes per row — an accounting
    read on the user-callback path, where a raise would look like the user's own bug."""
    view = _view_table(2048)
    got = (
        bt.from_arrow(view)
        .map_batches(lambda b: b, batch_format="pyarrow")
        .agg(n=bt.col("v").count())
        .collect()
    )
    assert got.to_pydict() == {"n": [2048]}


def test_a_view_column_writes_and_reads_back(tmp_path):
    """The writer sizes files and its in-flight concurrency from the table's bytes."""
    view = _view_table(1024)
    out = str(tmp_path / "view.parquet")
    bt.from_arrow(view).write.parquet(out)
    back = bt.read.parquet(out).agg(n=bt.col("v").count()).collect()
    assert back.to_pydict() == {"n": [1024]}

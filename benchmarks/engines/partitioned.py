"""One definition of how a comparator's in-memory handle gets partitioned.

Both Ray Data and Daft build a **single partition** from a whole Arrow table
(``ray.data.from_arrow`` / ``daft.from_arrow``), and for both a partition is the unit of
parallelism — so a whole-table handle runs every downstream operator far narrower than the
box allows. Ray Data's version was found first and fixed in ``engines/ray.py``, with a
paragraph of module docstring explaining it. Daft's went unnoticed for as long as nobody
compared the two adapters, while Daft sat in the *default* single-node lineup: measured on
this box, Daft 0.7.24, a two-key group-by with two aggregates over 4M rows ran 319 ms from
``from_arrow`` against 51 ms from Parquet, a **6.2x** handicap charged silently to the
comparator.

That is the argument for this module. A fix written as prose in one adapter does not
propagate to the adapter beside it; a fix written as a shared function does. Any future
comparator whose ingest has the same shape calls :func:`parquet_handle` and cannot
reintroduce the defect by omission.

The mechanism is the representative one for both engines: write the normalized Arrow table
to Parquet once as **untimed setup**, sized so the engine's own scan splits it across the
available cores, and let the engine read it back through its native reader. That is how a
user of either engine ingests data, and it is the read path each engine's own parallelism
heuristics are written against.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from functools import cache

import pyarrow as pa
import pyarrow.parquet as pq

from batcher._internal.hardware import available_cpu_count

__all__ = ["parquet_handle", "row_group_rows", "scratch_dir"]

#: Floor on row-group rows. TPC-H's tiny dimension tables (nation=25, region=5) must not be
#: split into single-row groups, which costs more in task overhead than the scan saves.
MIN_ROW_GROUP_ROWS = 8192

#: Splits per core the scan should see. Two is Ray Data's own documented read-parallelism
#: default and a reasonable target for Daft; it is the engines' number, not one tuned here.
SPLITS_PER_CORE = 2


@cache
def scratch_dir(tag: str) -> str:
    """A per-engine temp directory, removed at exit. Cached so one engine reuses one dir."""
    path = tempfile.mkdtemp(prefix=f"batcher-bench-{tag}-")
    atexit.register(shutil.rmtree, path, True)
    return path


def row_group_rows(table: pa.Table, max_group_bytes: int | None = None) -> int:
    """Rows per row group: enough splits to fill the box, without exceeding a byte ceiling.

    Args:
        table: The table about to be written.
        max_group_bytes: An engine's own ceiling on block size, when it has one (Ray Data's
            ``DataContext.target_max_block_size``). The smaller of the two bounds wins, so
            the engine gets the parallelism its defaults ask for without ever exceeding its
            own block-size limit.

    Returns:
        A row-group size, never below :data:`MIN_ROW_GROUP_ROWS`.
    """
    if not table.num_rows:
        return MIN_ROW_GROUP_ROWS
    cores = available_cpu_count() or 1
    target = -(-table.num_rows // (SPLITS_PER_CORE * cores))  # ceil
    if max_group_bytes is not None:
        bytes_per_row = max(1, table.nbytes // table.num_rows)
        target = min(target, max(1, max_group_bytes // bytes_per_row))
    return max(MIN_ROW_GROUP_ROWS, target)


def parquet_handle(table: pa.Table, tag: str, max_group_bytes: int | None = None) -> str:
    """Write ``table`` to Parquet once (untimed) and return the path for the engine to read.

    Keyed on the table's identity, so a suite that asks for the same handle across many
    cases pays the write once.

    Args:
        table: The normalized Arrow table every engine in the run receives.
        tag: Short engine name, used for the scratch directory and the file name.
        max_group_bytes: Passed to :func:`row_group_rows`.

    Returns:
        The Parquet path.
    """
    path = os.path.join(scratch_dir(tag), f"{tag}-{id(table):x}.parquet")
    if not os.path.exists(path):
        pq.write_table(table, path, row_group_size=row_group_rows(table, max_group_bytes))
    return path

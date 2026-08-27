"""DuckDB-on-Arrow adapter — the same-input execution comparison.

The default :class:`DuckDBEngine` ingests every table into DuckDB's *native* columnar
storage (an untimed ``CREATE TABLE``) before the timed query. That measures DuckDB's
storage engine — compression, dictionary encoding, min/max zone maps — *plus* its
execution engine, against Batcher's execution engine over raw Arrow. It is the right
bar for "DuckDB at its best," but it conflates two layers: on the same in-memory Arrow
that Batcher runs on, DuckDB is markedly slower (measured 1.3-2.6x on TPC-H sf1), so the
native-vs-Arrow gap is DuckDB's storage advantage, not an execution deficit in Batcher.

This adapter is the honest execution-parity comparator: it binds each table as a
*zero-copy Arrow view* (``con.register``) — exactly the input Batcher receives — so the
timed query exercises DuckDB's execution engine over identical bytes. The original
adapter's rationale for avoiding this (that an Arrow scan is "~100x slower on joins") was
true of older DuckDB; on 1.5.x a registered-Arrow join is ~1.5-3x native, not 100x, so
the fair comparison is now viable and is what this measures.

**DuckDB parallelizes an Arrow scan across the table's chunks, so the chunk count decides
how many threads it can use.** Every table the harness builds arrives as a *single* chunk
(``pa.table`` over whole arrays, and ``datagen`` builds them that way), which pinned this
bar to one scan thread on a 92-core box and made it measure a thread count rather than an
execution engine. Measured on the real H2O groupby table, all ten queries, with the bytes,
the values, the connection and the query held identical and *only* the chunking varied:

===================  ==========
registered as        total (ms)
===================  ==========
1 chunk (was)           6146.9
16 chunks               1186.4
32 chunks                765.4
64 chunks                702.9
128 chunks               736.5
512 chunks               971.2
===================  ==========

**8.7x, entirely from a presentation detail of the input.** So this adapter re-slices to
``_ROWS_PER_CHUNK`` before registering. The re-slice is zero-copy — ``Table.to_batches``
with a ``max_chunksize`` slices the existing buffers — so "identical bytes" stays literally
true; only the batch boundaries move.

The re-slice is untimed, which is what makes it symmetric: the native bar's ``CREATE TABLE``
ingest is untimed too, and it is what produces *its* well-formed row groups. Neither bar is
charged for arranging its input.

Batcher was controlled for and is chunk-insensitive: the same ten queries read 595.7 ms on
the 1-chunk table and 637.2 ms on the 64-chunk one — 7%, and *slower* on the chunked input,
because it morselizes internally regardless of how the input is presented. So the single
chunk was not a shared handicap. It penalized DuckDB alone.

**Which suites this actually moved is narrower than the 8.7x suggests, and the difference is
where the tables come from.** A suite reading Parquet arrives pre-chunked at its row-group
size and was never affected; only a suite whose tables are *generated in-process* lands as
one chunk. Measured:

=====================  ==============  =============  =====================================
suite                  source          chunks         effect on ``b/duckdb_arrow``
=====================  ==============  =============  =====================================
h2o-groupby, h2o-join  ``datagen``     1              severe — 0.10 -> **0.72** on groupby
json                   ``datagen``     1              severe (same shape, not re-run)
clickbench             parquet         2 (500k rows)  mild — 2 scan threads of 92
tpch, tpcds            parquet         ~123k/chunk    **none** — already at the target
=====================  ==============  =============  =====================================

TPC-H sf10 is the control that establishes the "none": ``lineitem`` loads as 481 chunks of
124,711 rows, within 2% of ``_ROWS_PER_CHUNK``, and the three-bar re-run after this fix reads
**0.27** against a 0.26-0.297 recorded before it. So the sf10 and JOB claims stand and must
not be withdrawn; the H2O and JSON ones do not. An earlier revision of this docstring said
every ``duckdb_arrow`` figure was inflated by 8.7x — that generalized one suite's controlled
result to suites whose inputs do not share the property, and it is false. See
``BENCHMARK_RESULTS.md`` (2026-08-26).

Batcher's ``Arrow is the only columnar contract`` invariant means it has no native
compressed store to switch to — so ``duckdb_arrow`` is the like-for-like bar, and
``duckdb`` (native) is the storage-advantaged one. Report both to keep the claim honest.
"""

from __future__ import annotations

import pyarrow as pa

from .base import SqlRunner
from .duckdb import DuckDBEngine

#: Rows per Arrow chunk when registering. DuckDB's own default Parquet row-group size, so
#: it is its native reader's arrangement rather than a constant tuned against this suite.
#: It lands mid-plateau on the sweep above (10M rows -> 81 chunks, between the 64 and 128
#: rows), where the curve is flat to within 5% — the value is not balanced on a peak.
_ROWS_PER_CHUNK = 122_880


class DuckDBArrowEngine(DuckDBEngine):
    name = "duckdb_arrow"

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        import duckdb

        con = duckdb.connect()
        # Zero-copy Arrow views — the same in-memory bytes Batcher executes over, with no
        # untimed ingest/compression step. This is the execution-engine comparison.
        for name, tbl in tables.items():
            con.register(name, _rechunk(tbl))
        return lambda query: con.sql(query).to_arrow_table()


def _rechunk(tbl: pa.Table) -> pa.Table:
    """Re-slice to chunks DuckDB's scan can spread over threads, without copying buffers.

    A table already at or below the target is returned untouched, so a source that arrives
    well-chunked (a Parquet read, say) is never re-sliced finer than it came.
    """
    if tbl.num_rows <= _ROWS_PER_CHUNK:
        return tbl
    batches = tbl.to_batches(max_chunksize=_ROWS_PER_CHUNK)
    return pa.Table.from_batches(batches, schema=tbl.schema) if batches else tbl

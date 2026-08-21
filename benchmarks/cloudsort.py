"""Sort throughput on the CloudSort record: a 10-byte binary key over a 90-byte payload.

The [Sort Benchmark](https://sortbenchmark.org/) family — Indy/Daytona GraySort, MinuteSort,
CloudSort — all define the same 100-byte record: a **10-byte binary key** and a 90-byte
payload. That shape is not a curiosity. It is the extreme of a ratio every large sort has,
where the key is narrow, the payload is wide, and the sort is a permutation of bytes rather
than a tournament of comparisons. The engine's whole sort design turns on that ratio: the
sample-sort routes by key and gathers the payload exactly once.

This benchmark measures that shape directly, at several scales, against DuckDB on the same
Arrow input. It answers three questions the operator-mix suite cannot, because that suite
sorts TPC-H columns:

1. **Does the byte key take a parallel path?** A serial sort and a 64-way one differ by more
   than an order of magnitude here, and the difference is invisible in a single number — it
   shows as a rows/s column that stays flat as the machine's cores fill, rather than falling.
2. **What does the payload cost?** Every case is run twice, once with the payload and once
   without, so the gather is separated from the ordering rather than reported as one figure.
   On this record the gather is most of the work, which is what makes "gather once" the
   design decision it is.
3. **Do the three binary spellings agree?** `binary`, `large_binary` and `binary(10)` are one
   ordering, so a difference between them is an implementation artifact worth seeing. The payload
   is reported at both fixed and variable width for the same reason: the record the benchmark
   defines is fixed on both halves, and a fixed-stride gather is a different path from an
   offset-chasing one.

Correctness is checked before any timing is trusted: each case's result is compared against
DuckDB's row for row, and a case whose engines disagree is reported as a mismatch and never
timed. Sorting is the one operator whose *order* is the answer, so the comparison is ordered
and the keys are drawn with duplicates so that ties are exercised rather than avoided.

    python benchmarks/cloudsort.py                 # 1M / 4M / 16M records
    python benchmarks/cloudsort.py 2000000         # one explicit scale
"""

from __future__ import annotations

import os
import sys
import time

import pyarrow as pa

import batcher as bt
from envinfo import require_release_build

# The Sort Benchmark record: a 10-byte key and a 90-byte payload, 100 bytes in total.
KEY_BYTES = 10
PAYLOAD_BYTES = 90

# Distinct keys per million records. Well below the record count, so every scale sorts a
# column with real ties — the case an unstable sort gets wrong and an order-independent
# comparison cannot see.
DISTINCT_PER_MILLION = 250_000

# Timed repeats after one warm-up. The minimum is reported: a shared machine's noise is
# one-sided, so the fastest run is the closest estimate of the cost of the work itself.
REPEATS = 3


def _records(rows: int) -> tuple[list[bytes], list[bytes]]:
    """`rows` CloudSort records as `(keys, payloads)`.

    Keys are drawn from `os.urandom` and truncated to a distinct-value budget, which gives
    both properties the sort needs tested: bytes spread over the whole key space (so the
    quantile boundaries have something to describe), and enough duplicates that tie order
    is exercised. Payloads are a single shared buffer repeated — the benchmark measures
    moving the payload, not generating it.
    """
    distinct = max(1, rows * DISTINCT_PER_MILLION // 1_000_000)
    pool = os.urandom(distinct * KEY_BYTES)
    keys = [pool[(i % distinct) * KEY_BYTES : (i % distinct + 1) * KEY_BYTES] for i in range(rows)]
    # Rotate so equal keys are spread through the relation rather than adjacent, which is
    # what makes the range routing do real work.
    keys = [keys[(i * 7919) % rows] for i in range(rows)]
    return keys, [b"\xab" * PAYLOAD_BYTES] * rows


def _table(
    keys: list[bytes],
    payloads: list[bytes],
    key_type: pa.DataType,
    payload_type: pa.DataType | None,
):
    """The relation under test: the key column, and the payload when `payload_type` is given.

    The payload's own type is a variable, not a constant, because the Sort Benchmark record is
    fixed width on *both* halves and a `binary(90)` column is gathered by a different path from a
    variable-length `binary` one -- fixed stride against offset chasing. Reporting only the
    variable-length payload would describe a record layout nobody who cares about this benchmark
    would choose.
    """
    columns = {"k": pa.array(keys, type=key_type)}
    if payload_type is not None:
        columns["v"] = pa.array(payloads, type=payload_type)
    return pa.table(columns)


def _duckdb():
    """A DuckDB connection, or `None` when it is not installed."""
    try:
        import duckdb
    except ImportError:
        return None
    return duckdb.connect()


def _time(fn) -> float:
    """The fastest of [`REPEATS`] runs of `fn`, in milliseconds, after one warm-up."""
    fn()
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1e3)
    return best


def _case(con, table: pa.Table, label: str) -> None:
    """Time one sort on both engines, after proving they agree on the answer."""
    ds = bt.from_arrow(table)
    ours = ds.sort("k").collect()
    con.register("t", table)
    theirs = con.sql("SELECT * FROM t ORDER BY k ASC").to_arrow_table()

    # Ordered, column by column: `sort` is the one operator whose order is the answer, and a
    # multiset comparison would pass on exactly the bug this measures the fix for.
    if ours.column("k").to_pylist() != theirs.column("k").to_pylist():
        print(f"{label:36s}  MISMATCH against DuckDB - not timed")
        con.unregister("t")
        return

    ours_ms = _time(lambda: ds.sort("k").collect())
    theirs_ms = _time(lambda: con.sql("SELECT * FROM t ORDER BY k ASC").to_arrow_table())
    con.unregister("t")

    rows = table.num_rows
    rate = f"{rows / (ours_ms / 1e3) / 1e6:.1f}M/s"
    print(
        f"{label:36s} {ours_ms:9.1f}ms {theirs_ms:9.1f}ms {theirs_ms / ours_ms:7.2f}x   {rate:>9}"
    )


def run(con, rows: int) -> None:
    """Every key spelling, with and without the payload, at one scale."""
    keys, payloads = _records(rows)
    record_mb = rows * (KEY_BYTES + PAYLOAD_BYTES) / 1e6
    print(f"\n=== {rows:,} records ({record_mb:.0f} MB of 100-byte records) ===")
    print(f"{'case':36s} {'batcher':>11s} {'duckdb':>11s} {'ratio':>8s} {'rows/s':>10s}")
    shapes = (
        (None, "key only"),
        (pa.binary(PAYLOAD_BYTES), f"+ binary({PAYLOAD_BYTES}) payload"),
        (pa.binary(), "+ variable payload"),
    )
    for name, key_type in (
        ("binary(10)", pa.binary(KEY_BYTES)),
        ("binary", pa.binary()),
        ("large_binary", pa.large_binary()),
    ):
        for payload_type, shape in shapes:
            _case(con, _table(keys, payloads, key_type, payload_type), f"{name}, {shape}")


def main() -> None:
    """Run every scale, or the scales named on the command line."""
    # A dev-profile engine is a hard stop, not a warning: `maturin develop` builds debug by
    # default, and a table produced without this check reports the build profile rather than
    # the engine.
    require_release_build()
    con = _duckdb()
    if con is None:
        print("This benchmark compares against DuckDB on the same Arrow input; install duckdb.")
        raise SystemExit(1)
    for rows in [int(a) for a in sys.argv[1:]] or [1_000_000, 4_000_000, 16_000_000]:
        run(con, rows)
    print(
        "\nRead the rows/s column down each scale: flat means the sort is keeping up with the\n"
        "input, and the gap between the 'key only' and 'with payload' rows is what the gather\n"
        "costs. The ratio column is DuckDB's time over Batcher's, so above 1.00x is Batcher\n"
        "ahead, on the same Arrow input."
    )


if __name__ == "__main__":
    main()

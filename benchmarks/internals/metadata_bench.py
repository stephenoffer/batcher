"""Metadata benchmark — what the *ordinary* API costs when the answer is already written down.

The claim is not "a bit faster". It is that a whole class of questions has the wrong cost
model. `count()`, `min(x)`, `null_count()`, "does this filter match anything", "does this join
match anything", "does this data satisfy its contract" are all treated as queries — and they
are all *already written down*, in a Parquet footer, a manifest, a catalog. Reading them is
O(metadata); computing them is O(rows). So the gap does not shrink as the data grows: it grows.

Every case here is a call a user already writes. **None of them mentions `ds.meta`.** That is
the point of the exercise: the metadata layer is not a surface to opt into, it is the cost of
the surface you already use.

Each query is timed twice over the same Parquet file: once normally, and once with the metadata
layer genuinely switched off (`map_batches` is opaque to the IR, so Kyber declines to reason
about the plan at all; the identity callback changes no row, so it is the same relation the
long way round).

Correctness first, as everywhere here: each pair is asserted **equal** before either is timed.
A shortcut that returned a different answer would be a bug, not a benchmark result.

Run:
    python benchmarks/internals/metadata_bench.py [rows]
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import batcher as bt

#: Repeats per measurement — the metadata path is sub-millisecond, so one sample is noise.
REPEATS = 5


def build(rows: int, path: str) -> None:
    """A Parquet file whose footer records everything these queries ask for."""
    table = pa.table(
        {
            "id": pa.array(range(rows), pa.int64()),
            "amount": pa.array([(i % 1000) + 1 for i in range(rows)], pa.int64()),  # 1..1000
            "day": pa.array([i % 365 for i in range(rows)], pa.int64()),
            # A string column with nulls — the shape most real tables are made of, and the one
            # whose exact footer null count used to be discarded with its truncatable bounds.
            "name": pa.array([None if i % 5 == 0 else f"n{i}" for i in range(rows)]),
        }
    )
    pq.write_table(table, path, row_group_size=max(1, rows // 16))


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    """Best-of-`REPEATS` wall time in milliseconds, plus the answer (to compare)."""
    best = math.inf
    answer = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        answer = fn()
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best, answer


def forced(ds: bt.Dataset) -> bt.Dataset:
    """The same relation with the metadata layer switched off — `map_batches` is IR-opaque."""
    return ds.map_batches(lambda batch: batch)


def cases() -> list[tuple[str, Callable[[bt.Dataset], Any]]]:
    """The ordinary calls, each written exactly as a user would write it."""
    # A dimension whose key range is disjoint from `id` — nothing can match.
    absent = bt.from_pydict({"id": [-3, -2, -1]})

    return [
        ("ds.count()", lambda d: d.count()),
        ("ds.min('amount')", lambda d: d.min("amount")),
        ("ds.max('amount')", lambda d: d.max("amount")),
        ("ds.n_null('amount')", lambda d: d.n_null("amount")),
        ("ds.n_null('name') [string]", lambda d: d.n_null("name")),
        ("ds.null_count()", lambda d: d.null_count().to_pydict()),
        (
            "ds.filter(amount > 1e9).collect()",
            lambda d: d.filter(bt.col("amount") > 10**9).collect().num_rows,
        ),
        (
            "ds.filter(amount > 0).count()",
            lambda d: d.filter(bt.col("amount") > 0).count(),
        ),
        ("ds.drop_nulls(['id']).count()", lambda d: d.drop_nulls(["id"]).count()),
        ("ds.limit(2*rows).count()", lambda d: d.limit(10**12).count()),
        # The two that change what a query *costs*, not merely what it shaves.
        (
            "ds.join(disjoint).collect()",
            lambda d: d.join(absent, on="id", how="inner").collect().num_rows,
        ),
        (
            "ds.dq.not_null.in_range.fail()",
            lambda d: d.dq.not_null("id").in_range("amount", 0, 10_000).fail().columns,
        ),
        (
            "ds.dq.in_range(...).validate()",
            lambda d: d.dq.in_range("amount", 0, 10_000).validate().ok,
        ),
    ]


def main() -> None:
    """Time every ordinary call against the same call with the metadata layer switched off."""
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "bench.parquet")
        print(f"building {rows:,} rows → {path}")
        build(rows, path)
        ds = bt.read.parquet(path)

        print(f"\n{'query (no ds.meta anywhere)':<36} {'metadata':>10} {'executed':>11} {'x':>9}")
        print("-" * 70)
        for name, call in cases():
            fast_ms, fast_answer = timed(lambda call=call: call(ds))
            slow_ms, slow_answer = timed(lambda call=call: call(forced(ds)))
            # Correctness gate: never report a timing for an answer that disagrees.
            assert fast_answer == slow_answer, (
                f"{name}: metadata said {fast_answer!r}, executing said {slow_answer!r}"
            )
            speedup = slow_ms / fast_ms if fast_ms > 0 else math.inf
            print(f"{name:<36} {fast_ms:>8.2f}ms {slow_ms:>9.2f}ms {speedup:>8.0f}x")


if __name__ == "__main__":
    main()

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

That control is not free, and the run says so before the table rather than leaving the reader
to assume it. An identity `map_batches` buys a Python callback and an Arrow round trip per
morsel, and the `executed` column pays for both — so a raw speedup credits the metadata layer
with work the UDF did. There is no config switch that turns the shortcuts off, so this is the
only available control; `udf_floor` therefore measures the callback on its own, over a
predicate no footer can answer, and prints it as the floor under every ratio in the table.
Read the `x` column as an upper bound, not a measurement.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envinfo import machine_fingerprint, require_quiet_box, require_release_build

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
    """The same relation with the metadata layer switched off — `map_batches` is IR-opaque.

    This is the only mechanism available: there is no config switch that disables the
    metadata shortcuts, so the control has to be a plan Kyber cannot reason about, and an
    identity `map_batches` is the cheapest such plan.

    It is not free, and the `x` column below is inflated by exactly that. See
    :func:`udf_floor`, which measures it, and read every speedup against the floor it
    prints rather than against zero.
    """
    return ds.map_batches(lambda batch: batch)


def udf_floor(ds: bt.Dataset) -> tuple[float, float]:
    """What the control arm costs before the metadata layer is involved at all.

    `forced()` disables the metadata shortcut by wrapping the relation in an identity
    `map_batches`. That also buys a Python callback and an Arrow round trip per morsel,
    and those are charged to the `executed` column even though they have nothing to do
    with the shortcut being measured. Reporting a speedup without saying so overstates
    the metadata layer by whatever the UDF costs.

    So this measures the UDF alone: the same predicate, with and without the identity
    callback, over a filter **no footer can answer** (`amount % 7`), where the metadata
    layer has nothing to contribute and the entire difference is the callback.

    Returns:
        `(plain_ms, with_udf_ms)` — their ratio is the floor under every `x` in the table.
    """
    predicate = bt.col("amount") % 7 == 3
    plain, _ = timed(lambda: ds.filter(predicate).collect().num_rows)
    wrapped, _ = timed(lambda: forced(ds).filter(predicate).collect().num_rows)
    return plain, wrapped


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
    # Refuse to time a dev-profile engine: it is 8-60x slower, so a number taken from one
    # compares an unoptimized Batcher against release competitors. `BENCH_ALLOW_DEBUG_BUILD=1`
    # overrides deliberately.
    require_release_build()
    # Print the machine before any number: a timing is only reproducible beside the
    # box that produced it, and this file's own history has ratios quoted across four
    # different machines as if they were comparable.
    print(machine_fingerprint())
    # ...and refuse a contended one: a neighbour's load is not a fact about any
    # engine. `BENCH_ALLOW_BUSY_BOX=1` overrides.
    require_quiet_box()
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "bench.parquet")
        print(f"building {rows:,} rows → {path}")
        build(rows, path)
        ds = bt.read.parquet(path)

        plain_ms, udf_ms = udf_floor(ds)
        print(
            f"\ncontrol-arm calibration: an identity map_batches costs "
            f"{udf_ms - plain_ms:.2f}ms on this data "
            f"({udf_ms / plain_ms:.2f}x a filter no footer can answer: "
            f"{plain_ms:.2f}ms -> {udf_ms:.2f}ms)."
        )
        print(
            "  Every 'executed' below carries that, because switching the metadata layer "
            "off requires it.\n  Read the x column as an upper bound."
        )

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

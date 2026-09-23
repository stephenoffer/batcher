"""What a partitioned dataset holds on disk, what the sketches say about it, and membership checks.

Three parts of ``ds.meta`` that read metadata rather than rows:

* ``ds.meta.storage`` lists the files, row groups, bytes, and partition keys a scan would
  read. ``num_files`` is the small-files check: many files holding little data each means a
  query spends its time opening files rather than reading them.
* ``ds.meta.approx`` reads the sketches a previous run recorded (top values, a quantile
  grid, a learned selectivity). It never executes, so it returns ``None`` until something
  has run, and an approximation rather than an answer afterwards.
* ``ds.meta.col(...).check`` answers ``contains`` / ``any_in`` / ``none_in`` from the column
  bounds when they refute the value, and runs the filter when they cannot.

    python examples/dataset/meta_storage_and_approx.py
"""

from __future__ import annotations

import glob
import os
import random
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import batcher as bt

DAYS = ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"]
BATCHES = 3  # three writes per day, as a job appending hourly would leave behind
ROWS_PER_BATCH = 250


def write_events(root: str) -> pa.Table:
    """Write a day-partitioned event table in several small appends; return all the rows."""
    rng = random.Random(42)
    written = []
    for batch in range(BATCHES):
        n = ROWS_PER_BATCH * len(DAYS)
        table = pa.table(
            {
                "event_id": list(range(batch * n, (batch + 1) * n)),
                "day": [DAYS[i % len(DAYS)] for i in range(n)],
                "country": [rng.choice(["US"] * 6 + ["DE"] * 3 + ["FR"]) for _ in range(n)],
                "amount": [round(rng.uniform(1.0, 500.0), 2) for _ in range(n)],
            }
        )
        pq.write_to_dataset(table, root, partition_cols=["day"])
        written.append(table)
    return pa.concat_tables(written)


def storage_report(ds: bt.Dataset, root: str) -> None:
    """The layout, without reading a row: files, row groups, bytes, partition keys."""
    storage = ds.meta.storage
    on_disk = glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
    print("files:", storage.num_files(), "row groups:", storage.row_group_count())
    print("partitioned by:", storage.partition_keys(), "rows:", storage.row_count())

    assert storage.num_files() == len(on_disk) == len(DAYS) * BATCHES
    assert sorted(map(os.path.realpath, storage.files())) == sorted(map(os.path.realpath, on_disk))
    assert storage.is_partitioned() and storage.partition_keys() == ("day",)
    assert storage.row_count() == len(DAYS) * BATCHES * ROWS_PER_BATCH
    assert storage.has_exact_row_count()

    # The small-files check: how much data does each file carry on average?
    per_file = storage.total_bytes() / storage.num_files()
    print(f"average recorded bytes per file: {per_file:,.0f}")
    too_small = per_file < 64 * 1024 * 1024
    assert too_small, "a dozen files of a few KiB each is the textbook small-files shape"
    print("small files: compact this directory before scanning it repeatedly")

    # An in-memory relation has no files: the question does not apply, so the list is empty.
    assert bt.from_pydict({"x": [1]}).meta.storage.files() == []


def compact(root: str, target: str) -> str:
    """Rewrite the small files as one, which is what the small-files check asks for."""
    manifest = bt.read.parquet(root).write.parquet(target)
    assert bt.read.parquet(target).meta.storage.num_files() == 1
    assert bt.read.parquet(target).meta.storage.row_count() == manifest.total_rows
    return target


def approx_report(path: str, rows: pa.Table) -> None:
    """Sketches recorded by one run, read back for free and held to the exact answers."""
    total = rows.num_rows
    before = bt.read.parquet(path).meta.approx
    assert before.top_k("country") is None, "nothing has read `country` yet"

    # Ordinary queries record sketches for the columns they group and filter on.
    ds = bt.read.parquet(path)
    ds.group_by("country").agg(n=bt.count()).collect()
    big = bt.col("amount") > 400
    ds.filter(big).collect()

    approx = bt.read.parquet(path).meta.approx
    top = approx.top_k("country", 2)
    print("top countries:", top)
    counts = rows.group_by("country").aggregate([("country", "count")]).to_pydict()
    share = {c: n / total for c, n in zip(counts["country"], counts["country_count"], strict=True)}
    assert top is not None and top[0][0] == "US"
    for value, estimated in top:
        assert abs(estimated - share[value]) < 0.02, (value, estimated, share[value])

    buckets = approx.histogram("amount", 4)
    print("amount quartile buckets:", buckets)
    assert buckets is not None and len(buckets) == 4
    amounts = rows.column("amount").to_pylist()
    for low, high in buckets:
        inside = sum(low <= a <= high for a in amounts) / total
        assert abs(inside - 0.25) < 0.05, (low, high, inside)

    exact = sum(a > 400 for a in amounts) / total
    estimated = approx.selectivity(big)
    print(f"selectivity of amount > 400: estimated {estimated:.3f}, exact {exact:.3f}")
    assert estimated is not None and abs(estimated - exact) < 0.02

    # Behind a stage the planner cannot see into, the estimate is unknown -- None, not 0.0.
    opaque = bt.read.parquet(path).map_batches(lambda batch: batch).meta.approx
    assert opaque.rows() is None and opaque.selectivity(big) is None


def membership_checks(ds: bt.Dataset) -> None:
    """`contains` / `any_in` / `none_in`: refuted from the bounds, else one filter."""
    event = ds.meta.col("event_id").check
    last = len(DAYS) * BATCHES * ROWS_PER_BATCH - 1

    assert event.contains(17)
    assert not event.contains(10**9)  # above the recorded maximum: no scan
    assert event.never_equals(-1)
    assert event.any_in([-5, 17, 10**9])
    assert event.none_in([-5, last + 1])
    assert not event.may_contain(last + 1)  # False is a proof of absence

    country = ds.meta.col("country").check
    assert country.any_in(["FR", "JP"])
    assert country.none_in(["JP", "BR"])

    # A value the column's type cannot hold is a mistake, and it says so the same way
    # whether metadata or the engine would have answered.
    try:
        event.contains("17")
    except bt.PlanError as err:
        print("refused:", err)
    else:
        raise AssertionError("comparing an integer column with a string must raise")


def main() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        root = os.path.join(scratch, "events")
        rows = write_events(root)
        ds = bt.read.parquet(root)
        storage_report(ds, root)
        membership_checks(ds)
        compacted = compact(root, os.path.join(scratch, "events_compacted.parquet"))
        approx_report(compacted, rows)


if __name__ == "__main__":
    main()

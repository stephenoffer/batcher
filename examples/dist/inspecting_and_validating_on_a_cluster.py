"""Profile, question and validate a table on a cluster, and get the answers one node gives.

The inspection tools come in two shapes. `null_count()` and the `ds.dq` row splits return
lazy Datasets, so they take `collect(distributed=..., num_workers=...)` like any query. The
rest execute on their own with `distributed="auto"`: `describe()` and `profile()` run their
aggregate as soon as you call them, and so do the `ds.meta` questions, `ds.dq.validate()`,
`count()` and `approx_median()`. The session option `distributed.mode` is how you pin those
to the cluster (`"always"`) or to this process (`"never"`).

Runs single-node by default; pass ``--distributed`` (or set
``BATCHER_EXAMPLES_DISTRIBUTED=1``) to run the cluster half on Ray.

    python examples/dist/inspecting_and_validating_on_a_cluster.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_distributed
from batcher.config import option_context

WORKERS = 4


def write_orders(root: Path, files: int = 4, rows: int = 50_000) -> str:
    """A small partitioned table: a key, a measure with a few NaN, and a nullable count."""
    rng = np.random.default_rng(3)
    for i in range(files):
        amount = rng.normal(50.0, 20.0, rows)
        amount[rng.random(rows) < 0.01] = np.nan
        pq.write_table(
            pa.table(
                {
                    "order_id": np.arange(i * rows, (i + 1) * rows, dtype="int64"),
                    "amount": amount,
                    "qty": pa.array(rng.integers(0, 40, rows), mask=rng.random(rows) < 0.05),
                }
            ),
            root / f"part-{i}.parquet",
        )
    return str(root)


def same(a: object, b: object) -> bool:
    """Equal, or equal up to float reassociation across partitions."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9)
    return a == b


def by_key(table: pa.Table, key: str) -> list[dict]:
    return sorted(table.to_pylist(), key=lambda row: str(row[key]))


def profile_under(ds: bt.Dataset, mode: str) -> pa.Table:
    with option_context("distributed.mode", mode):
        return ds.describe().collect(distributed=False)


def main() -> None:
    distributed = resolve_distributed()
    mode = "always" if distributed else "never"

    with tempfile.TemporaryDirectory() as tmp:
        orders = bt.read.parquet(write_orders(Path(tmp)))

        # 1. `describe()` computes when called, so the mode decides where its aggregate runs.
        local = by_key(profile_under(orders, "never"), "statistic")
        there = by_key(profile_under(orders, mode), "statistic")
        for left, right in zip(local, there, strict=True):
            assert all(same(left[c], right[c]) for c in left), (left, right)
        # `null_count()` is lazy, so it takes the cluster through `collect` directly.
        nulls_frame = orders.null_count()
        on_one = nulls_frame.collect(distributed=False).to_pylist()
        on_many = nulls_frame.collect(distributed=distributed, num_workers=WORKERS).to_pylist()
        assert on_one == on_many
        print(f"describe(): {len(local)} statistics, identical on one node and under {mode!r}")

        # 2. Metadata questions. A footer answers these with no query at all...
        assert orders.meta.shape() == (200_000, 3)
        assert orders.meta.col("order_id").is_key()
        # ...and a filter turns them into real queries, which `mode` routes.
        recent = orders.filter(bt.col("order_id") >= 100_000)
        with option_context("distributed.mode", mode):
            distinct_qty = recent.meta.col("qty").n_unique()
            nulls = recent.meta.nulls.counts()
            median = recent.approx_median("amount")
        assert distinct_qty == 40
        assert nulls["qty"] == recent.filter(bt.col("qty").is_null()).count()
        assert 45.0 < median < 55.0
        print(f"meta (mode={mode!r}): {distinct_qty} distinct qty, {nulls['qty']} null qty")

        # 3. A data contract. `validate()` runs where `mode` says; the row split is lazy.
        gate = orders.dq.not_null("qty").in_range("amount", 0.0, 100.0).unique("order_id")
        with option_context("distributed.mode", mode):
            report = gate.validate()
        clean, rejected = gate.quarantine()
        clean_rows = clean.collect(distributed=distributed, num_workers=WORKERS).num_rows
        rejected_rows = rejected.collect(distributed=distributed, num_workers=WORKERS).num_rows
        assert clean_rows + rejected_rows == 200_000  # a total partition of the input
        assert report.violations["unique(order_id)"] == 0
        assert report.violations["not_null(qty)"] > 0
        print(f"dq: {report.violations}; {clean_rows} clean, {rejected_rows} quarantined")


if __name__ == "__main__":
    main()

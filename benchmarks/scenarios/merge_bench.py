"""MERGE benchmark — an upsert should cost the change set, not the table.

The claim under test decides whether Batcher's ``MERGE`` is usable on a warehouse table at
all. A naive copy-on-write merge rewrites every data file, so merging 1,000 rows into a
20M-row table reads and rewrites 20M rows, and its cost is *flat* no matter how little you
changed. A merge that skips the files whose key statistics prove they cannot match is
sublinear in the table's size, and its cost tracks the selectivity of the change set. The
gap between those two is the whole benchmark.

Each row reports:

* ``pruned``  — the merge as it ships (rewrite only what can match)
* ``full``    — the same merge with ``prune=False`` (rewrite everything), the old cost
* ``files``   — how many of the target's data files were actually rewritten

## Two things this harness does on purpose

**Every measurement runs in its own process.** Batcher learns from execution: Core records
measured cardinalities into the MetadataHub and Kyber plans the next query with them. That
is the point of the engine — and it makes an in-process A/B *invalid*, because the first
configuration teaches the optimizer things the second then benefits (or suffers) from.
Measured in one process, this benchmark reported the pruned path as 5x **slower** on a
full-table restatement; measured in separate processes, the two land within a couple of
percent — as they must, since with nothing to prune they run the identical plan. The bias
was entirely an artifact of shared learned state. So: fork per point.

**Correctness is gated before any timing is reported.** Every configuration is checked
against DuckDB's own ``MERGE INTO`` over the same two tables. A fast wrong answer is not a
result — `benchmarks/harness.py` holds the same line for the query benchmarks.

    python benchmarks/scenarios/merge_bench.py              # 20M-row target
    python benchmarks/scenarios/merge_bench.py 5000000      # a different scale
    python benchmarks/scenarios/merge_bench.py --scaling    # speedup vs table size
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa

ROWS_PER_FILE = 250_000


# ======================================================================================
# The child: one measurement, one process, no inherited learned state.
# ======================================================================================


def _target_table(rows: int) -> pa.Table:
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "id": np.arange(rows, dtype=np.int64),
            "amount": rng.integers(0, 10_000, rows).astype(np.int64),
            "region": pa.array([f"r{i % 20}" for i in range(rows)]),
        }
    )


def _key_set(shape: str, rows: int) -> np.ndarray:
    """The change set's keys.

    Their *shape* is what decides how many files they touch, not merely how many there are:
    a thousand keys clustered in recent history touch one file; a thousand scattered at
    random touch a thousand.
    """
    rng = np.random.default_rng(1)
    if shape == "cdc_tail":
        return np.arange(rows - 1_000, rows, dtype=np.int64)
    if shape == "scattered_1k":
        return rng.choice(rows, 1_000, replace=False).astype(np.int64)
    if shape == "recent_100k":
        return np.arange(rows - 100_000, rows, dtype=np.int64)
    if shape == "one_percent":
        return rng.choice(rows, max(1, rows // 100), replace=False).astype(np.int64)
    if shape == "ten_percent":
        return rng.choice(rows, max(1, rows // 10), replace=False).astype(np.int64)
    if shape == "restatement":
        return np.arange(rows, dtype=np.int64)
    raise SystemExit(f"unknown shape {shape!r}")


def _changes(keys: np.ndarray) -> pa.Table:
    return pa.table(
        {
            "id": keys,
            "amount": np.full(len(keys), -1, dtype=np.int64),
            "region": pa.array(["updated"] * len(keys)),
        }
    )


def _duckdb_expected(target: pa.Table, changes: pa.Table) -> set:
    """The oracle: DuckDB's own MERGE INTO over the same two tables."""
    import duckdb

    con = duckdb.connect()
    con.register("_t", target)
    con.register("_s", changes)
    con.execute("CREATE TABLE t AS SELECT * FROM _t")
    con.execute("CREATE TABLE s AS SELECT * FROM _s")
    con.execute(
        """MERGE INTO t USING s ON t.id = s.id
           WHEN MATCHED THEN UPDATE SET amount = s.amount, region = s.region
           WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.amount, s.region)"""
    )
    return set(con.execute("SELECT id, amount, region FROM t").fetchall())


def _child(rows: int, shape: str, prune: bool) -> dict:
    """Run exactly one measurement; the parent reads its JSON off stdout."""
    import batcher as bt
    from batcher.api.merge import plan_merge, simple_clauses

    table = _target_table(rows)
    changes = _changes(_key_set(shape, rows))
    root = tempfile.mkdtemp(prefix="bc-merge-bench-")
    path = f"{root}/t"
    try:
        bt.from_arrow(table).write.parquet(path, max_rows_per_file=ROWS_PER_FILE)

        plan = plan_merge(
            bt.from_arrow(changes),
            path,
            ["id"],
            simple_clauses("update", "insert"),
            prune=prune,
            format="parquet",
        )

        start = time.perf_counter()
        (
            bt.from_arrow(changes)
            .write.merge_into(path, on="id", format="parquet", prune=prune)
            .when_matched()
            .update_all()
            .when_not_matched()
            .insert_all()
            .execute()
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Correctness gate: refuse to report a timing for a wrong answer.
        out = bt.read.parquet(path).collect().to_pydict()
        got = set(zip(out["id"], out["amount"], out["region"], strict=True))
        expected = _duckdb_expected(table, changes)
        if got != expected:
            missing, extra = len(expected - got), len(got - expected)
            return {"error": f"{missing} rows missing, {extra} unexpected"}
        return {
            "ms": elapsed_ms,
            "changed": len(changes),
            "rewritten": len(plan.rewritten),
            "files": plan.total,
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ======================================================================================
# The parent: fork a child per point, tabulate.
# ======================================================================================


def _measure(rows: int, shape: str, prune: bool) -> dict:
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", str(rows), shape, str(prune)],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = proc.stdout.strip().splitlines()
    line = lines[-1] if lines else ""
    if not line.startswith("{"):
        raise SystemExit(f"child failed ({shape}, prune={prune}):\n{proc.stderr[-2000:]}")
    result = json.loads(line)
    if "error" in result:
        raise SystemExit(f"CORRECTNESS FAILURE ({shape}, prune={prune}): {result['error']}")
    return result


_SHAPES = [
    ("cdc_tail", "1k recent (CDC tail)"),
    ("scattered_1k", "1k scattered"),
    ("recent_100k", "100k contiguous"),
    ("one_percent", "1% of table"),
    ("ten_percent", "10% of table"),
    ("restatement", "100% (restatement)"),
]


def _row(label: str, pruned: dict, full: dict) -> None:
    speedup = full["ms"] / pruned["ms"] if pruned["ms"] else float("nan")
    files = f"{pruned['rewritten']}/{pruned['files']}"
    print(
        f"{label:<22} {pruned['changed']:>10,} {files:>9} "
        f"{pruned['ms']:>8.0f}m {full['ms']:>8.0f}m {speedup:>8.1f}x"
    )


def _sweep(rows: int) -> None:
    print(f"\nMERGE into a {rows:,}-row Parquet table ({ROWS_PER_FILE:,} rows/file)")
    print("each point in its own process; correctness gated against DuckDB MERGE INTO\n")
    header = (
        f"{'change set':<22} {'rows':>10} {'files':>9} {'pruned':>9} {'full':>9} {'speedup':>9}"
    )
    print(header)
    print("-" * len(header))
    for shape, label in _SHAPES:
        _row(label, _measure(rows, shape, True), _measure(rows, shape, False))
    print(
        "\npruned = rewrite only the files whose key bounds can match"
        "\nfull   = prune=False, rewrite the whole table (the cost before this change)"
    )


def _scaling() -> None:
    """Pruning's win *grows* with the table: the pruned cost stays ~one file while the full
    cost is O(table). This is the shape of the claim, and the reason it is not a fixed 10x."""
    print("\nA 1,000-row CDC batch merged into a table of growing size")
    print("each point in its own process; correctness gated against DuckDB MERGE INTO\n")
    header = f"{'table rows':>12} {'files':>9} {'pruned':>9} {'full':>9} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for rows in (1_000_000, 5_000_000, 20_000_000, 50_000_000):
        pruned = _measure(rows, "cdc_tail", True)
        full = _measure(rows, "cdc_tail", False)
        speedup = full["ms"] / pruned["ms"] if pruned["ms"] else float("nan")
        files = f"{pruned['rewritten']}/{pruned['files']}"
        print(f"{rows:>12,} {files:>9} {pruned['ms']:>8.0f}m {full['ms']:>8.0f}m {speedup:>8.1f}x")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--child":
        print(json.dumps(_child(int(args[1]), args[2], args[3] == "True")))
        return
    if args and args[0] == "--scaling":
        _scaling()
        return
    _sweep(int(args[0]) if args else 20_000_000)


if __name__ == "__main__":
    main()

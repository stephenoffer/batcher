"""Batcher vs Ray Data — distributed fraud feature aggregation (tabular preprocessing).

The dominant cost of the fraud-detection batch path is **feature engineering**: per-account
aggregations over transaction history (count/velocity, sum, mean, max) that become the model
features. The optimization guides call this out as the big speedup lever ("feature
preprocessing 10x").

This is Batcher's *structural* home turf, not a physics race: the aggregation is
**relational**, so Batcher runs it in the Rust engine as a mergeable group-by (partial per
partition → Flight shuffle by key → combine) with JIT-compiled derived-feature expressions,
while Ray Data runs its own `groupby().aggregate(...)`. Both read the same Parquet
transaction shards distributed; the per-account mean must match before any timing.

Note: the *enrich* shape (join fact rows back to their per-account aggregate) is a known
distributed gap — the Flight co-partition join can't take an aggregate build side (each
mapper would compute a wrong per-partition aggregate); it needs aggregate-then-broadcast.
This benchmark measures the aggregation itself, which is the supported, dominant-cost step.

Run:
    python benchmarks/cluster/fraud_scoring.py                 # 20M txns, 200k accounts
    BENCH_FRAUD_N=50000000 python benchmarks/cluster/fraud_scoring.py
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np
import pyarrow as pa

print = functools.partial(print, flush=True)

_SEED = 1234
# Derived-risk-feature weights (native vectorized expression over the aggregates).
_W_RATIO, _W_VELOCITY, _BIAS = 0.8, 0.05, -1.2


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_FRAUD_N", "20000000")),
        "accounts": int(os.environ.get("BENCH_FRAUD_ACCOUNTS", "200000")),
        "shards": int(os.environ.get("BENCH_FRAUD_SHARDS", "64")),
        "dir": os.environ.get("BENCH_FRAUD_PARQUET", "/mnt/cluster_storage/fraud_scoring"),
        "runs": int(os.environ.get("BENCH_RUNS", "3")),
    }


def write_shards(directory: str, n: int, accounts: int, shards: int) -> str:
    import pyarrow.parquet as pq

    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(_SEED)
    per = -(-n // shards)
    idx = 0
    for s in range(shards):
        rows = min(per, n - idx)
        if rows <= 0:
            break
        acct = rng.integers(0, accounts, size=rows, dtype=np.int64)
        amt = np.round(np.abs(rng.normal(60.0, 40.0, size=rows)) + 1.0, 2)
        hour = rng.integers(0, 24, size=rows, dtype=np.int64)
        tbl = pa.table({"acct": acct, "amt": amt, "hour": hour})
        pq.write_table(tbl, os.path.join(directory, f"part-{s:05d}.parquet"))
        idx += rows
    return directory


def _init() -> None:
    import importlib.util

    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        v = os.environ.get(var)
        if v:
            head = v.lstrip("[{\"' ").split(".")[0].split("[")[0]
            if head and importlib.util.find_spec(head) is None:
                os.environ.pop(var, None)
    import ray

    if not ray.is_initialized():
        ray.init(
            address="auto", runtime_env={"pip": None}, logging_level="ERROR", log_to_driver=False
        )


def batcher_features(directory: str):
    """Native distributed per-account risk features (a group-by aggregate in the Rust
    engine, shuffled over Flight) plus a derived risk expression."""
    import batcher as bt
    from batcher import col

    stats = (
        bt.read.parquet(directory)
        .group_by("acct")
        .agg(
            n=col("amt").count(),
            total=col("amt").sum(),
            avg=col("amt").mean(),
            mx=col("amt").max(),
        )
    )
    # A derived risk feature over the aggregates (still native, vectorized).
    return stats.with_columns(
        risk=(_W_VELOCITY * col("n") + _W_RATIO * (col("mx") / col("avg")) + _BIAS)
    )


def ray_features(directory: str):
    """Ray Data's native multi-aggregate over the same group-by (its optimized path)."""
    import ray.data as rd
    from ray.data.aggregate import Count, Max, Mean, Sum

    ds = (
        rd.read_parquet(directory)
        .groupby("acct")
        .aggregate(Count(), Sum("amt"), Mean("amt"), Max("amt"))
    )
    return ds


def _ray_table(directory: str) -> pa.Table:
    return pa.concat_tables(list(ray_features(directory).iter_batches(batch_format="pyarrow")))


def _acct_avg(tbl: pa.Table, avg_col: str) -> dict:
    """A acct -> mean(amt) map for order-independent correctness comparison."""
    d = tbl.to_pydict()
    return dict(zip(d["acct"], d[avg_col], strict=True))


def main() -> int:
    cfg = _cfg()
    _init()
    directory = write_shards(cfg["dir"], cfg["n"], cfg["accounts"], cfg["shards"])
    print(f"txns={cfg['n']}  accounts={cfg['accounts']}  shards={cfg['shards']}\n")

    # Correctness: per-account mean amount must match between engines on a sample.
    sample_dir = write_shards(cfg["dir"] + "_s", 20000, 2000, 4)
    bs = _acct_avg(batcher_features(sample_dir).collect(), "avg")
    r_tbl = _ray_table(sample_dir)
    # Ray names the mean column "mean(amt)".
    mean_col = next(c for c in r_tbl.column_names if "mean" in c.lower())
    rs = _acct_avg(r_tbl, mean_col)
    common = set(bs) & set(rs)
    if not common:
        print("FAIL: no comparable accounts")
        return 1
    max_err = max(abs(bs[k] - rs[k]) for k in common)
    print(f"per-account mean agreement on {len(common)} accounts: max abs err {max_err:.2e}")
    if max_err > 1e-6:
        print("FAIL: aggregates diverge — not timing")
        return 1

    def batcher_run():
        return batcher_features(directory).collect()

    def ray_run():
        return ray_features(directory).materialize()

    res = {}
    for name, fn in (("batcher", batcher_run), ("ray", ray_run)):
        fn()  # warm-up
        best = float("inf")
        for _ in range(cfg["runs"]):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        res[name] = best
        print(f"{name:<8} {cfg['n'] / best / 1e6:6.1f} M rows/s  ({best * 1000:.0f} ms)")

    ratio = res["ray"] / res["batcher"]
    print(f"\nfraud feature aggregation (per-account risk features): batcher {ratio:.2f}x Ray Data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

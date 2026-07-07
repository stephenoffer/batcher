"""Batcher vs Ray Data — distributed fraud batch scoring (tabular feature engineering).

The full fraud-detection batch path: per-account **feature engineering** (aggregations over
transaction history) → join the features back onto every transaction → a risk score → a
decision. Feature preprocessing is the dominant cost and the piece the optimization guides
call out as the big speedup lever ("feature preprocessing 10x").

This is Batcher's *structural* home turf, not a physics race: the feature engineering and
the enrich are **relational** — a mergeable group-by aggregate, a join back, and vectorized
derived-column/score expressions — so Batcher runs the whole pipeline in the Rust engine
(distributed aggregate over Flight → join → JIT-compiled score) while Ray Data has no
relational optimizer and runs it as a per-account **Python** function (``groupby().
map_groups``) over pandas. The score is a deterministic logistic so both engines must produce
identical risk scores before any timing.

The enrich shape (join fact rows to their per-account aggregate) runs fully distributed via
the adaptive aggregate → materialize → join → project staging (fixed 2026-07-02). Both read
the same Parquet transaction shards distributed.

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


def _score_np(amt, avg, n):
    """The shared logistic risk score — identical math in both engines (a stand-in for
    the trained model; deterministic so decisions can be compared exactly)."""
    z = _W_RATIO * (amt / avg) + _W_VELOCITY * n + _BIAS
    return 1.0 / (1.0 + np.exp(-z))


def batcher_pipeline(directory: str):
    """The full fraud enrich pipeline, native in the Rust engine: distributed group-by
    aggregate (per-account features) → join the aggregates back onto every fact row →
    vectorized logistic risk score → decision. Enabled by the enrich-join distributed
    route (aggregate → materialize → join → project); all stages run distributed."""
    import batcher as bt
    from batcher import col

    ds = bt.read.parquet(directory)
    stats = ds.group_by("acct").agg(avg=col("amt").mean(), n=col("amt").count())
    joined = ds.join(stats, on="acct", how="inner")
    z = _W_RATIO * (col("amt") / col("avg")) + _W_VELOCITY * col("n") + _BIAS
    return joined.with_columns(score=(1.0 / (1.0 + (-z).exp()))).select("acct", "amt", "score")


def _score_group(g):
    """Ray Data's idiomatic per-account feature+score: a Python fn over one group (pandas)."""
    amt = g["amt"].to_numpy()
    score = _score_np(amt, amt.mean(), len(g))
    return g.assign(score=score)[["acct", "amt", "score"]]


def ray_dataset(directory: str):
    import ray.data as rd

    return (
        rd.read_parquet(directory).groupby("acct").map_groups(_score_group, batch_format="pandas")
    )


def _ray_table(directory: str) -> pa.Table:
    return pa.concat_tables(list(ray_dataset(directory).iter_batches(batch_format="pyarrow")))


def _key_scores(tbl: pa.Table) -> dict:
    """A (acct, amt-rounded) -> score map for order-independent correctness comparison."""
    d = tbl.to_pydict()
    return {(a, round(m, 2)): s for a, m, s in zip(d["acct"], d["amt"], d["score"], strict=True)}


def main() -> int:
    cfg = _cfg()
    _init()
    directory = write_shards(cfg["dir"], cfg["n"], cfg["accounts"], cfg["shards"])
    print(f"txns={cfg['n']}  accounts={cfg['accounts']}  shards={cfg['shards']}\n")

    # Correctness: per-row risk scores must match between engines on a sample.
    sample_dir = write_shards(cfg["dir"] + "_s", 20000, 2000, 4)
    bs = _key_scores(batcher_pipeline(sample_dir).collect())
    rs = _key_scores(_ray_table(sample_dir))
    common = set(bs) & set(rs)
    if not common:
        print("FAIL: no comparable rows")
        return 1
    max_err = max(abs(bs[k] - rs[k]) for k in common)
    print(f"per-row score agreement on {len(common)} rows: max abs err {max_err:.2e}")
    if max_err > 1e-6:
        print("FAIL: scores diverge — not timing")
        return 1

    def batcher_run():
        return batcher_pipeline(directory).collect()

    def ray_run():
        return ray_dataset(directory).materialize()

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
    print(f"\nfraud enrich pipeline (features + join + score): batcher {ratio:.2f}x Ray Data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

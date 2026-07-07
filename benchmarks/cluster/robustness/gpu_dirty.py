"""Batcher vs Ray Data — dirty-data tolerance (corrupt-record survival + retention).

Real-world AI data is messy: a fraction of images/records fail to decode. A robust engine
must *survive* them, not crash — and ideally drop only the bad rows, not good ones. This
injects ~1% corrupt rows (a UDF that raises on them) and compares:

* **completion** — does the job finish?
* **rows retained** — Batcher's `max_errored_rows` drops the corrupt ROWS (keeps ~99%);
  Ray Data's `max_errored_blocks` drops whole BLOCKS containing any bad row (loses good
  rows with it).
* throughput on the surviving data.

Run:
    python benchmarks/cluster/robustness/gpu_dirty.py            # 200k rows, ~1% corrupt
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np
import pyarrow as pa

print = functools.partial(print, flush=True)

_SEED = 1234
_BAD_EVERY = int(os.environ.get("BENCH_DIRTY_EVERY", "100"))  # ~1% corrupt


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_DIRTY_N", "200000")),
        "batch": int(os.environ.get("BENCH_DIRTY_BATCH", "1024")),
        "runs": int(os.environ.get("BENCH_RUNS", "3")),
    }


def _process(d: dict) -> dict:
    """A per-row transform that RAISES on a corrupt row (id divisible by _BAD_EVERY)."""
    ids = d["id"]
    if (ids % _BAD_EVERY == 0).any():
        raise ValueError("corrupt record")
    return {"id": ids, "y": np.sqrt(ids.astype(np.float64) + 1.0)}


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


def main() -> int:
    cfg = _cfg()
    _init()
    import ray.data as rd

    import batcher as bt

    n = cfg["n"]
    n_bad = len([i for i in range(n) if i % _BAD_EVERY == 0])
    t = pa.table({"id": np.arange(n, dtype=np.int64)})
    print(f"rows={n}  corrupt={n_bad} (~{100 * n_bad / n:.1f}%)  batch={cfg['batch']}\n")

    # Batcher: per-ROW tolerance — drops only the corrupt rows.
    def batcher_run():
        out = (
            bt.from_arrow(t)
            .map_batches(
                _process, output_columns=["id", "y"], batch_format="numpy", max_errored_rows=n
            )
            .collect()
        )
        return out.num_rows

    # Ray Data: per-BLOCK tolerance (max_errored_blocks) — the closest equivalent.
    ctx = rd.DataContext.get_current()
    ctx.max_errored_blocks = -1  # unlimited, so it completes rather than crash

    def ray_run():
        ds = rd.from_arrow(t).map_batches(
            lambda b: _process(b), batch_format="numpy", batch_size=cfg["batch"]
        )
        rows = sum(b["id"].shape[0] for b in ds.iter_batches(batch_format="numpy"))
        return rows

    res = {}
    for name, fn in (("batcher", batcher_run), ("ray", ray_run)):
        try:
            kept = fn()  # warmup
            best = float("inf")
            for _ in range(cfg["runs"]):
                t0 = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - t0)
            res[name] = {"kept": kept, "s": best}
            pct = 100 * kept / n
            print(f"{name:<10} completed  kept={kept}/{n} ({pct:.1f}%)  {n / best:.0f} rows/s")
        except Exception as e:
            res[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{name:<10} CRASHED: {type(e).__name__}: {e}")

    b, r = res.get("batcher", {}), res.get("ray", {})
    if "kept" in b and "kept" in r:
        print(
            f"\nretention: batcher {b['kept']} vs ray {r['kept']} "
            f"→ batcher keeps {b['kept'] - r['kept']} more good rows "
            f"(Ray drops whole blocks; Batcher drops only bad rows)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

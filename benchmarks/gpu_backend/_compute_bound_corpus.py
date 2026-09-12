"""Generate the compute-bound corpus: narrow rows, so the mount is never the constraint.

The image corpus was 74 KB an image against 14 MFLOP of model, which made every arm a
measurement of `/mnt/cluster_storage` (0.9 GiB/s). This is the opposite ratio on purpose:
1 KB a row against hundreds of MFLOP, so a 60-second run reads a few GiB and spends the rest
of its time in the CPU stage and on the devices -- which is what a batch-inference pipeline
actually looks like and the only shape on which two engines can be told apart.
"""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

print = functools.partial(print, flush=True)
OUT = Path(os.environ.get("FEAT_DIR", "/mnt/cluster_storage/feat_infer_bench"))
ROWS = int(os.environ.get("FEAT_ROWS", "4000000"))
DIM = int(os.environ.get("FEAT_DIM", "256"))  # int8 features -> 256 B a row
SHARDS = int(os.environ.get("FEAT_SHARDS", "256"))


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from envinfo import require_release_build

    # Writing a corpus is not a timing entry point, but the gate reads file shape rather than
    # intent and it is right to: a generator that ships in the same directory as the benchmark
    # will be run from the same shell, and a debug engine here would write the corpus slowly
    # and silently. Costs one import.
    require_release_build()
    OUT.mkdir(parents=True, exist_ok=True)
    per = ROWS // SHARDS
    rng = np.random.default_rng(7)
    t0 = time.perf_counter()
    for s in range(SHARDS):
        path = OUT / f"part-{s:04d}.parquet"
        if path.exists():
            continue
        raw = rng.integers(-128, 127, size=(per, DIM), dtype=np.int8)
        tbl = pa.table(
            {
                "id": pa.array(np.arange(s * per, (s + 1) * per, dtype=np.int64)),
                "feat": pa.FixedSizeListArray.from_arrays(pa.array(raw.reshape(-1)), DIM),
            }
        )
        pq.write_table(tbl, path, compression="zstd")
        if s % 32 == 0:
            print(f"  shard {s}/{SHARDS} ({time.perf_counter() - t0:.0f}s)")
    tot = sum(p.stat().st_size for p in OUT.glob("*.parquet"))
    print(f"# {SHARDS} shards, {ROWS:,} rows x {DIM} int8, {tot / 2**30:.2f} GiB on disk")


if __name__ == "__main__":
    main()

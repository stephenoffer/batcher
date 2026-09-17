"""Prove an installed Batcher works, the way a user's first session would exercise it.

Run against an *installed* package from outside the source tree, so the checkout cannot shadow
it: the release workflow runs it on every wheel, in Alpine for the musllinux wheels, and inside
the Docker image. It covers the four things a broken install has actually got wrong here — an
undeclared import (NumPy), the native extension failing to load, a host probe crashing where no
cgroup mount exists, and the parallel path, which only engages past `MIN_ROWS_TO_SHARD`.

Exits non-zero on the first failure.
"""

from __future__ import annotations

import os
import platform
import sys
import sysconfig
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import batcher as bt


def main() -> None:
    """Run the checks and print one line describing the install that passed them."""
    here = os.path.dirname(os.path.abspath(bt.__file__))
    # An editable install or a PYTHONPATH pointing at the checkout would pass every check below
    # while proving nothing about the wheel, so refuse a package that sits in a source tree.
    if os.path.exists(os.path.join(here, os.pardir, os.pardir, "Cargo.toml")):
        sys.exit(f"batcher was imported from a source checkout ({here}), not an installed wheel")

    ds = bt.from_pydict({"a": [1, 2, 3]})
    assert ds.agg(s=bt.col("a").sum()).to_pydict() == {"s": [6]}
    assert bt.sql("select 1 + 1 as x").to_pydict() == {"x": [2]}

    with tempfile.TemporaryDirectory() as tmp:
        table = pa.table({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
        pq.write_table(table, os.path.join(tmp, "p.parquet"))
        out = bt.read(tmp).group_by("k").agg(t=bt.col("v").sum()).sort("k").to_pydict()
        assert out == {"k": [1, 2], "t": [3.0, 3.0]}, out

    # 300,000 rows clears the 65,536-row sharding threshold, so this runs the parallel path.
    n = 300_000
    big = bt.from_pydict({"g": [i % 7 for i in range(n)], "x": list(range(n))})
    counts = big.group_by("g").agg(c=bt.col("x").count()).sort("g").to_pydict()
    assert counts["g"] == list(range(7)) and sum(counts["c"]) == n, counts

    versions = bt.versions()
    print(
        f"ok: batcher {versions['batcher']} ({versions['engine_profile']}) on "
        f"{sysconfig.get_platform()} {platform.libc_ver()[0] or 'musl'} "
        f"python {platform.python_version()} from {here}"
    )


if __name__ == "__main__":
    main()

"""How wrong a sample is, and how that shrinks with size.

The standard error falls with the square root of the sample size, so a sample four times
larger is twice as precise, not four times. Seeing that on real data is the fastest way to
calibrate how big a sample actually needs to be.

    python examples/statistics/sampling_error.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    # Keep the key alongside the measure: it makes the sample auditable, and a
    # projection down to a single column is not what a real pipeline samples.
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity")
    truth = lineitem.agg(m=col("l_quantity").mean()).to_pydict()["m"][0]
    print(f"population mean: {truth:.4f}")

    errors: dict[int, float] = {}
    for size in (100, 1_000, 10_000, 100_000):
        sample = lineitem.sample(n=size, seed=13)
        estimate = sample.agg(m=col("l_quantity").mean()).to_pydict()["m"][0]
        errors[size] = abs(estimate - truth)
        sem = sample.agg(s=col("l_quantity").std()).to_pydict()["s"][0] / size**0.5
        print(f"  n={size:>6} estimate={estimate:8.4f} error={errors[size]:.4f} sem={sem:.4f}")

    # Every estimate is in the neighbourhood of the truth.
    assert all(value < 2.0 for value in errors.values())

    # The largest sample is the most accurate, and by a wide margin over the smallest.
    assert errors[100_000] < errors[100]

    # The reported standard error really is sd/sqrt(n): a hundredfold sample is a
    # tenfold reduction in error.
    small = lineitem.sample(n=100, seed=13)
    large = lineitem.sample(n=10_000, seed=13)
    small_sem = small.agg(s=col("l_quantity").std()).to_pydict()["s"][0] / 100**0.5
    large_sem = large.agg(s=col("l_quantity").std()).to_pydict()["s"][0] / 10_000**0.5
    ratio = small_sem / large_sem
    print(f"sem ratio for a 100x sample: {ratio:.2f} (expected about 10)")
    assert 8.0 < ratio < 12.0


if __name__ == "__main__":
    main()

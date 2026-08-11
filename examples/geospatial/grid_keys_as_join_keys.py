"""Grid keys: turning a point into a joinable string.

A geohash or tile id makes a spatial join into an equality join, which can hash. The
precision picks the cell size, and that is the whole trade: coarser cells mean more
candidates to check exactly, finer cells mean more cells to enumerate.

    python examples/geospatial/grid_keys_as_join_keys.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    places = bt.from_pydict(
        {
            "name": ["ferry", "pier", "bridge", "london"],
            "lon": [-122.3937, -122.3941, -122.4783, -0.1276],
            "lat": [37.7955, 37.7960, 37.8199, 51.5074],
        }
    ).with_columns(point=bt.st_point(col("lon"), col("lat")))

    keyed = places.select(
        "name",
        coarse=bt.st_geohash(col("point"), 4),
        fine=bt.st_geohash(col("point"), 9),
    )
    result = keyed.to_pydict()
    for name, coarse, fine in zip(result["name"], result["coarse"], result["fine"], strict=True):
        print(f"  {name:<8} {coarse}  {fine}")

    # Precision is the length of the key.
    assert all(len(value) == 4 for value in result["coarse"])
    assert all(len(value) == 9 for value in result["fine"])

    # A finer key always extends its coarser prefix.
    assert all(
        fine.startswith(coarse)
        for coarse, fine in zip(result["coarse"], result["fine"], strict=True)
    )

    # Nearby points share a coarse cell; distant ones do not. That is the join key.
    ferry, pier, bridge, london = result["coarse"]
    assert ferry == pier
    assert ferry != london

    # At high precision even the two nearby points separate, which is the trade.
    fine_ferry, fine_pier = result["fine"][0], result["fine"][1]
    assert fine_ferry != fine_pier
    assert bridge is not None


if __name__ == "__main__":
    main()

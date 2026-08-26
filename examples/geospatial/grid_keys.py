"""Turning positions into cell ids you can group, sort and join on.

Latitude and longitude are floats, so no two observations share a value and
`GROUP BY lat, lon` returns the input. A grid function turns a position into a discrete
id, and the engine then hashes, sorts, shuffles and joins that at full speed with no
spatial index at all. This is usually the cheapest useful geospatial thing you can do.

    python examples/geospatial/grid_keys.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def pickups() -> bt.Dataset:
    """Seven positions: two pairs that should co-locate, three singletons."""
    return bt.from_pydict(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "lon": [-122.4194, -122.4190, -122.2712, -74.0060, -74.0059, -0.1278, 151.2093],
            "lat": [37.7749, 37.7751, 37.8044, 40.7128, 40.7130, 51.5074, -33.8688],
        }
    )


def geohash_is_a_prefix_code_over_space() -> None:
    """Sharing a prefix means being close, which turns a region filter into a LIKE."""
    ds = pickups()
    hashed = ds.with_columns(
        cell=bt.geohash_encode(col("lon"), col("lat"), 6),
        coarse=bt.geohash_encode(col("lon"), col("lat"), 3),
    )
    print("--- one cell id per position ---")
    cells_out = hashed.select("id", "cell", "coarse").to_pydict()
    print(cells_out)
    # The two San Francisco pickups are 4 metres apart, so a 6-character cell must not
    # separate them — that co-location is the property the whole prefix code rests on.
    assert cells_out["cell"][0] == cells_out["cell"][1] == "9q8yyk"
    assert cells_out["cell"][6] == "r3gx2f", "Sydney is nowhere near the others"

    print("--- and integers group, so this is an ordinary hash aggregate ---")
    grouped = hashed.group_by("cell").agg(n=bt.count()).sort("cell").to_pydict()
    print(grouped)
    assert sum(grouped["n"]) == 7 and grouped["n"] == [2, 1, 2, 1, 1]

    print("--- hashes nest, so a rollup is substr rather than a recomputation ---")
    nested = hashed.select(
        "id",
        same=col("cell").str.substr(1, 3) == col("coarse"),
    ).to_pydict()
    print(nested)
    # If this were ever false the "rollup is a substring" claim above would be a lie, and
    # a coarser aggregate built by truncation would silently mix unrelated regions.
    assert all(nested["same"]), "a 3-character geohash must be the 6-character one's prefix"

    print("--- decode a cell back to a position, or to the cell itself ---")
    cells = bt.from_pydict({"h": ["9q8yyk", "gcpvj0"]})
    decoded = cells.select(
        lon=bt.geohash_decode_lon(col("h")).round(4),
        lat=bt.geohash_decode_lat(col("h")).round(4),
        box=bt.st_as_text(bt.st_geom_from_geohash(col("h"))),
    ).to_pydict()
    print(decoded)
    # Decoding is lossy by construction — it returns the cell's centre, not the original
    # point — so the test is that the centre lands inside the cell, within its ~600m width.
    assert abs(decoded["lon"][0] - (-122.4194)) < 0.01
    assert abs(decoded["lat"][0] - 37.7749) < 0.01
    assert decoded["box"][0].startswith("POLYGON((")

    print("--- st_geohash takes a geometry, reducing it to its centroid ---")
    areas = bt.from_pydict(
        {"g": ["POLYGON((-122.42 37.77, -122.41 37.77, -122.41 37.78, -122.42 37.77))"]}
    )
    print(areas.select(cell=bt.st_geohash(col("g"), 6)).to_pydict())


def tiles_are_the_grid_maps_are_served_on() -> None:
    ds = pickups()
    tiled = ds.with_columns(
        x=bt.st_tile_x(col("lon"), col("lat"), 12),
        # y increases *southward*: row 0 is the top of the map, near 85 degrees north.
        y=bt.st_tile_y(col("lon"), col("lat"), 12),
        key=bt.st_quadkey(col("lon"), col("lat"), 12),
    )
    print("--- tile addresses at zoom 12 ---")
    addresses = tiled.select("id", "x", "y", "key").to_pydict()
    print(addresses)
    # Zoom 12 is a 4096x4096 grid, so every address is in range and the quadkey has one
    # digit per zoom level.
    assert all(0 <= v < 2**12 for v in addresses["x"] + addresses["y"])
    assert all(len(k) == 12 for k in addresses["key"])

    print("--- northern positions have smaller y ---")
    by_lat = tiled.select("id", "lat", "y").sort("lat", descending=True).to_pydict()
    print(by_lat)
    # The comment above the `st_tile_y` call claims y increases southward. Sorting by
    # latitude descending must therefore produce a non-decreasing y, or the claim is wrong.
    assert by_lat["y"] == sorted(by_lat["y"]), "y must increase as latitude decreases"

    print("--- a quadkey nests by digit, like a geohash by character ---")
    zooms = ds.filter(col("id") == 1)
    keys = zooms.select(
        z4=bt.st_quadkey(col("lon"), col("lat"), 4),
        z8=bt.st_quadkey(col("lon"), col("lat"), 8),
        z12=bt.st_quadkey(col("lon"), col("lat"), 12),
    ).to_pydict()
    print(keys)
    # One digit per zoom level, and each zoom's key is the next one's prefix.
    assert [len(keys[z][0]) for z in ("z4", "z8", "z12")] == [4, 8, 12]
    assert keys["z12"][0].startswith(keys["z8"][0])
    assert keys["z8"][0].startswith(keys["z4"][0])
    assert set(keys["z12"][0]) <= set("0123"), "a quadkey is base 4"


def s2_cells_sort_spatially_and_are_near_equal_area() -> None:
    """The best general-purpose spatial key: fair across latitudes, and range-scannable."""
    ds = pickups()
    celled = ds.with_columns(
        fine=bt.st_s2_cell(col("lon"), col("lat"), 15),
        coarse=bt.st_s2_cell(col("lon"), col("lat"), 8),
    )
    print("--- cell ids at two levels ---")
    ids = celled.select("id", "fine", "coarse").to_pydict()
    print(ids)
    assert ids["fine"][0] == ids["fine"][1], "the two San Francisco rows share a level-15 cell"
    assert ids["fine"][0] != ids["fine"][6]

    print("--- rolling up is a bit mask, not a recomputation ---")
    rolled = celled.select(
        "id",
        same=bt.st_s2_cell_parent(col("fine"), 8) == col("coarse"),
    ).to_pydict()
    print(rolled)
    # The claim in this function's docstring — that a coarser key is derivable from a finer
    # one — is only true if this holds for every row.
    assert all(rolled["same"]), "st_s2_cell_parent must agree with encoding at level 8"

    print("--- neighbouring positions land in the same coarse cell ---")
    coarse_groups = celled.group_by("coarse").agg(n=bt.count()).sort("coarse").to_pydict()
    print(coarse_groups)
    assert sum(coarse_groups["n"]) == 7 and max(coarse_groups["n"]) == 2

    # Because the id is a Hilbert index, sorting by it clusters neighbours onto the same
    # pages: the two San Francisco rows end up adjacent, and Sydney is far away.
    print("--- sorting by cell id is sorting by locality ---")
    by_locality = celled.select("id", "fine").sort("fine").to_pydict()["id"]
    print(by_locality)
    # The comment above claims neighbours end up adjacent. Check it rather than assert it in
    # prose: the two San Francisco rows (1, 2) and the two New York rows (4, 5) must each be
    # side by side in Hilbert order.
    assert abs(by_locality.index(1) - by_locality.index(2)) == 1
    assert abs(by_locality.index(4) - by_locality.index(5)) == 1


def hexagons_remove_the_grid_bias_of_squares() -> None:
    """All six neighbours are equidistant, which a square grid cannot offer."""
    # Project first: this bins whatever coordinates it is given, and it is not H3.
    ds = pickups()
    projected = ds.with_columns(
        m=bt.st_transform(bt.st_point(col("lon"), col("lat")), 4326, 3857)
    ).with_columns(x=bt.st_x(col("m")), y=bt.st_y(col("m")))

    binned = projected.with_columns(cell=bt.st_hex_bin(col("x"), col("y"), 500.0))
    print("--- 500 metre hexagons in Web Mercator ---")
    counts = binned.group_by("cell").agg(n=bt.count()).sort("n", descending=True).to_pydict()["n"]
    print(counts)
    # The two pairs 4m and 20m apart each fall in one hexagon; the other three are alone.
    assert counts == [2, 2, 1, 1, 1] and sum(counts) == 7

    print("--- recover a plottable centre from the group key ---")
    centres = binned.select(
        "id",
        cx=bt.st_hex_center_x(col("cell"), 500.0).round(1),
        cy=bt.st_hex_center_y(col("cell"), 500.0).round(1),
    )
    middles = centres.to_pydict()
    print(middles)
    # A recovered centre must be within one cell of the point that produced it, or the key
    # cannot be plotted back onto a map.
    projected_xy = projected.select("x", "y").to_pydict()
    for cx, cy, x, y in zip(
        middles["cx"], middles["cy"], projected_xy["x"], projected_xy["y"], strict=True
    ):
        assert abs(cx - x) < 500.0 and abs(cy - y) < 500.0


def main() -> None:
    geohash_is_a_prefix_code_over_space()
    tiles_are_the_grid_maps_are_served_on()
    s2_cells_sort_spatially_and_are_near_equal_area()
    hexagons_remove_the_grid_bias_of_squares()


if __name__ == "__main__":
    main()

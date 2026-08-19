"""Geospatial expression throughput, on real places, against DuckDB's spatial extension.

Answers two questions the previous version of this file could not:

1. **Does per-row cost stay flat as the input grows?** It should -- every ``ST_*`` function
   is a scalar expression evaluated per row in Rust -- but "should" is not a measurement,
   and the shape of the rows/s column down each scale is what separates linear scaling from
   something quietly quadratic.
2. **How does that compare to the reference geospatial engine?** DuckDB's spatial extension
   is GEOS and PROJ underneath, which is what PostGIS and Sedona use, so it is the bar this
   surface has to clear rather than a second opinion.

Three things about the method, each of which a first attempt got wrong:

* **The data is real.** Places come from Overture Maps' public S3 release -- a genuine
  point distribution, clustered in cities and empty over oceans, which is what makes a
  grid or an index behave the way it will in production. A uniform synthetic lattice
  flatters every spatial structure there is. The extract is written once to
  ``~/bench-data/overture`` and reused; nothing is generated.
* **Both engines read the same Arrow table.** DuckDB runs on the zero-copy Arrow the
  Batcher path consumes, not on its own compressed store, so this compares execution
  against execution rather than storage against storage (the distinction
  ``TPCH_FINDINGS.md`` draws between the ``duckdb`` and ``duckdb_arrow`` bars).
* **Correctness is checked before any timing is trusted.** Each case's two results are
  compared first; a case whose engines disagree is reported as a mismatch and never timed.
  The reduction is over the expression's *output*, because a bare ``select(...).count()``
  is answered from row-count metadata without evaluating anything -- which is how an
  earlier draft reported 1.3 billion rows/s and times that *fell* as the input grew.

Where DuckDB has no equivalent (geohash, S2, slippy tiles, quadkeys) the case is timed for
Batcher alone and reported as ``n/a`` rather than dropped, because those are the functions a
lakehouse actually keys a spatial join on.

    python benchmarks/geospatial.py                  # 250k / 1M / 2M rows
    python benchmarks/geospatial.py 500000           # one explicit scale
    BENCH_OVERTURE_RELEASE=2026-07-22.0 python benchmarks/geospatial.py
"""

from __future__ import annotations

import os
import sys
import time

import batcher as bt
from envinfo import require_release_build

#: The Overture Maps release to read. Public, anonymous, and versioned, so a run is
#: reproducible against a named snapshot rather than against "whatever is current".
RELEASE = os.environ.get("BENCH_OVERTURE_RELEASE", "2026-07-22.0")
PLACES_URI = (
    f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/"
    "part-000{part:02d}-*.parquet"
)
CACHE = os.path.expanduser(os.environ.get("BENCH_OVERTURE_LOCAL", "~/bench-data/overture"))

#: How many of the release's 16 place files to draw from. Reading a prefix of one file
#: would sample one region: the files are spatially ordered, so the first two million rows
#: of part 0 span a single quadrant of the globe. Taking a slice from each of several files
#: spreads the sample across continents, which is the point of using real points at all.
PARTS = 8

#: A box over central London, chosen because the sample really does contain places inside
#: it: a predicate that rejects everything measures the reject path and nothing else.
BOX = "POLYGON((-0.2 51.45, 0.02 51.45, 0.02 51.58, -0.2 51.58, -0.2 51.45))"

#: A reference point (Trafalgar Square) for the distance cases.
REF_LON, REF_LAT = -0.1281, 51.5080


def _duckdb():
    """A DuckDB connection with httpfs and spatial loaded, or None with a reason printed."""
    try:
        import duckdb
    except ImportError:
        print("duckdb is not installed; the comparison column will read n/a")
        return None
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
        con.execute("SET s3_region='us-west-2';")
        con.execute("CREATE OR REPLACE SECRET s (TYPE s3, PROVIDER config, REGION 'us-west-2');")
    except Exception as exc:
        print(f"duckdb spatial/httpfs unavailable ({exc}); the comparison column will read n/a")
        return None
    return con


def ensure_places(con, rows: int) -> str:
    """Materialize `rows` real Overture places locally once, returning the parquet path.

    Written from several of the release's files rather than the head of one, so the sample
    spans the globe. A no-op on every run after the first.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"places_{rows}.parquet")
    if os.path.exists(path):
        return path
    per_part = max(1, rows // PARTS)
    arms = " UNION ALL ".join(
        f"(SELECT id, ST_X(geometry) AS lon, ST_Y(geometry) AS lat, "
        f"        categories.primary AS category, confidence "
        f" FROM read_parquet('{PLACES_URI.format(part=part)}') "
        f" WHERE geometry IS NOT NULL LIMIT {per_part})"
        for part in range(PARTS)
    )
    started = time.perf_counter()
    con.execute(f"COPY ({arms}) TO '{path}' (FORMAT parquet)")
    print(f"  materialized {rows:,} Overture places in {time.perf_counter() - started:.1f}s")
    return path


def load(con, rows: int):
    """The places table as Arrow, shared by both engines."""
    path = ensure_places(con, rows)
    return con.execute(f"SELECT * FROM read_parquet('{path}')").arrow().read_all()


# --------------------------------------------------------------------------- #
# Cases: (label, batcher callable, duckdb SQL or None)
#
# Each side reduces the expression's output to one number, so nothing can be skipped and
# the two are directly comparable. `p` is the DuckDB point built from the same two columns
# Batcher's `st_point` reads.
# --------------------------------------------------------------------------- #
_POINT = "ST_Point(lon, lat)"


def cases(ds):
    """Every measured case, as (label, batcher thunk, DuckDB SQL or None)."""
    point = bt.st_point(bt.col("lon"), bt.col("lat"))
    box = bt.lit(BOX)
    ref = bt.lit(f"POINT({REF_LON} {REF_LAT})")
    return [
        (
            "st_x of a constructed point",
            lambda: ds.agg(s=bt.sum(bt.st_x(point))).to_pydict()["s"][0],
            f"SELECT sum(ST_X({_POINT})) FROM t",
        ),
        (
            "st_intersects vs a box",
            lambda: ds.agg(s=bt.count_if(bt.st_intersects(point, box))).to_pydict()["s"][0],
            f"SELECT count(*) FILTER (WHERE ST_Intersects({_POINT}, "
            f"ST_GeomFromText('{BOX}'))) FROM t",
        ),
        (
            "st_distance to a point",
            lambda: ds.agg(s=bt.sum(bt.st_distance(point, ref))).to_pydict()["s"][0],
            f"SELECT sum(ST_Distance({_POINT}, ST_Point({REF_LON}, {REF_LAT}))) FROM t",
        ),
        (
            "st_transform to 3857",
            lambda: ds.agg(s=bt.sum(bt.st_x(bt.st_transform(point, 4326, 3857)))).to_pydict()["s"][
                0
            ],
            f"SELECT sum(ST_X(ST_Transform({_POINT}, 'EPSG:4326', 'EPSG:3857',"
            " always_xy := true))) FROM t",
        ),
        (
            "st_as_text (WKT render)",
            lambda: ds.agg(s=bt.max(bt.st_as_text(point))).to_pydict()["s"][0],
            f"SELECT max(ST_AsText({_POINT})) FROM t",
        ),
        # No DuckDB equivalent exists for the grid encoders, and they are the functions a
        # lakehouse keys a spatial join on -- so they are timed for Batcher alone rather
        # than left out of the picture.
        (
            "geohash_encode (p8)",
            lambda: ds.agg(
                s=bt.max(bt.geohash_encode(bt.col("lon"), bt.col("lat"), 8))
            ).to_pydict()["s"][0],
            None,
        ),
        (
            "st_s2_cell (level 15)",
            lambda: ds.agg(s=bt.max(bt.st_s2_cell(bt.col("lon"), bt.col("lat"), 15))).to_pydict()[
                "s"
            ][0],
            None,
        ),
        (
            "st_quadkey (zoom 16)",
            lambda: ds.agg(s=bt.max(bt.st_quadkey(bt.col("lon"), bt.col("lat"), 16))).to_pydict()[
                "s"
            ][0],
            None,
        ),
    ]


def _agrees(ours, theirs) -> bool:
    """Whether the two engines' reductions match, to float tolerance.

    Whitespace is stripped from a text result before comparing: the two render WKT with a
    different space after the keyword (``POINT(1 2)`` against ``POINT (1 2)``) and they are
    the same geometry, so treating that as a mismatch would drop a real case from the
    table. Every other character still has to match, so a wrong coordinate is still caught.
    """
    if ours is None or theirs is None:
        return ours is theirs
    if isinstance(ours, str) or isinstance(theirs, str):
        return "".join(str(ours).split()) == "".join(str(theirs).split())
    return abs(float(ours) - float(theirs)) <= 1e-6 * max(1.0, abs(float(theirs)))


def _time(call) -> tuple[float, object]:
    """Run once to warm, then time the second run. Returns (seconds, result)."""
    call()
    started = time.perf_counter()
    result = call()
    return time.perf_counter() - started, result


def run(con, rows: int) -> None:
    """Time every case over `rows` real places, in both engines."""
    table = load(con, rows)
    ds = bt.from_arrow(table).cache()
    ds.count()  # materialize the cache so the first case does not pay for it
    con.register("t", table)

    print(f"\n--- {table.num_rows:,} Overture places ---")
    print(f"  {'case':<30}{'batcher':>10}{'duckdb':>10}{'ratio':>8}   {'batcher':>10}")
    for label, ours_call, their_sql in cases(ds):
        our_seconds, our_value = _time(ours_call)
        rate = f"{table.num_rows / our_seconds / 1e6:.1f} M/s"
        if their_sql is None:
            print(f"  {label:<30}{our_seconds * 1000:9.1f}ms{'n/a':>10}{'':>8}   {rate:>10}")
            continue
        their_seconds, their_row = _time(lambda sql=their_sql: con.execute(sql).fetchone())
        their_value = their_row[0]
        if not _agrees(our_value, their_value):
            print(
                f"  {label:<30}  MISMATCH: batcher={our_value!r} duckdb={their_value!r}"
                "  (not timed)"
            )
            continue
        ratio = their_seconds / our_seconds if our_seconds else float("inf")
        print(
            f"  {label:<30}{our_seconds * 1000:9.1f}ms{their_seconds * 1000:9.1f}ms"
            f"{ratio:8.2f}x   {rate:>10}"
        )
    con.unregister("t")


def main() -> None:
    """Run every scale, or the scales named on the command line."""
    # A dev-profile engine is a hard stop, not a warning. `maturin develop` builds the
    # debug profile by default, and the difference is not subtle here: the same quadkey
    # case measured 32 ms release and 600 ms debug, so a table produced without this check
    # reports the build profile rather than the engine.
    require_release_build()
    con = _duckdb()
    if con is None:
        print(
            "This benchmark reads real Overture Maps places over S3 and compares against\n"
            "DuckDB's spatial extension; both need duckdb with httpfs and spatial. It does\n"
            "not fall back to generated points, because a uniform lattice would flatter\n"
            "every spatial structure being measured."
        )
        raise SystemExit(1)
    scales = [int(a) for a in sys.argv[1:]] or [250_000, 1_000_000, 2_000_000]
    for rows in scales:
        run(con, rows)
    print(
        "\nRead the rows/s column down each expression: flat means the per-row cost is\n"
        "independent of input size. The ratio column is DuckDB's time over Batcher's, so\n"
        "above 1.00x is Batcher ahead, on the same Arrow input."
    )


if __name__ == "__main__":
    main()

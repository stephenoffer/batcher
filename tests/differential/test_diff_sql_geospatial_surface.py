"""The public function library must be reachable from SQL, not just from the DataFrame.

`bc-geo` implements 113 geospatial functions and the DataFrame API exposes every one of
them. **None was reachable from SQL.** ``SELECT ST_Area(g)`` raised "unknown function
'ST_Area'" while `bt.st_area(...)` answered — and `ST_Area` is not Batcher's name for that
operation, it is *the* name: PostGIS, DuckDB spatial, Snowflake, BigQuery and the OGC
standard all spell it this way, and spatial work is written in SQL far more often than in a
DataFrame. The whole surface was invisible to the front end its users reach for.

The dispatch is derived from `plan.functions.geo.__all__` rather than tabulated, so the
first test here is the one that matters: every exported member has to be callable. A table
of 113 rows would be a copy of that list, and a copy drifts — which is exactly how the
window vocabulary lost six functions.

Values are held against DuckDB's spatial extension, which is GEOS underneath and therefore
an *independent* implementation of the same standard rather than a restatement of the same
code. Two known renderings differ and are asserted as such rather than papered over.
"""

from __future__ import annotations

import inspect
import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher._sql.parser.expressions.lowering.families import is_scalar_callable
from batcher.plan.functions import geo
from batcher.plan.functions import spatial as spatial_fns

pytestmark = pytest.mark.differential

_GEOMS = [
    "POINT(1 2)",
    "LINESTRING(0 0, 3 4)",
    "POLYGON((0 0,4 0,4 4,0 4,0 0))",
    "MULTIPOINT((0 0),(1 1))",
    "POLYGON((0 0,10 0,10 10,0 10,0 0),(2 2,4 2,4 4,2 4,2 2))",
]


@pytest.fixture
def geoms() -> pa.Table:
    return pa.table({"g": pa.array(_GEOMS), "x": pa.array([1.0] * len(_GEOMS))})


@pytest.fixture
def spatial(duck):
    """DuckDB with the spatial extension, or a skip when it cannot be installed."""
    try:
        duck.execute("INSTALL spatial; LOAD spatial;")
    except Exception as exc:  # pragma: no cover - offline / no extension available
        pytest.skip(f"duckdb spatial unavailable: {exc}")
    return duck


def _sql_argument(param: str) -> str:
    """A plausible SQL argument for a parameter, by its name in the Python signature."""
    p = param.lower()
    if "geom" in p or p in {"a", "b", "line", "other", "point", "ring"}:
        return "ST_GeomFromText(g)"
    if "geohash" in p or p == "text":
        return "'s00tw'"
    if "wkb" in p:
        return "ST_AsBinary(ST_GeomFromText(g))"
    if "json" in p:
        return "ST_AsGeoJSON(ST_GeomFromText(g))"
    if any(k in p for k in ("srid", "zoom", "level", "precision", "quad", "index")):
        return "3"
    return "1.0"


@pytest.mark.parametrize("name", sorted(geo.__all__))
def test_every_exported_geo_function_is_reachable_from_sql(geoms, name):
    """Reachability, not values: the call must get as far as the engine.

    An engine-level complaint about an *argument* is a pass — this generator picks a
    plausible argument by parameter name, not a meaningful one. What must not happen is
    "unknown function", which is the defect: a name the engine implements and the SQL
    front end cannot reach.
    """
    params = list(inspect.signature(getattr(geo, name)).parameters)
    call = f"{name}({', '.join(_sql_argument(p) for p in params)})"
    try:
        bt.sql(f"SELECT {call} AS v FROM t", t=geoms).collect()
    except NotImplementedError as exc:  # pragma: no cover - the defect this test exists for
        pytest.fail(f"{name} is not reachable from SQL: {exc}")
    except RuntimeError:
        pass  # reached the engine; the generated argument was not one it accepts


#: Functions whose SQL name normalizes to a different word than Batcher's, so the
#: underscore rule cannot reach them and an alias is required.
@pytest.mark.parametrize(
    "call",
    [
        "ST_NPoints(ST_GeomFromText(g))",
        "ST_NGeometries(ST_GeomFromText(g))",
        "ST_NInteriorRings(ST_GeomFromText(g))",
        "ST_AsWKB(ST_GeomFromText(g))",
        "ST_AsWKT(ST_GeomFromText(g))",
        "ST_MakePoint(x, x)",
    ],
)
def test_the_postgis_spellings_resolve(geoms, call):
    bt.sql(f"SELECT {call} AS v FROM t", t=geoms).collect()


#: Calls whose answer must equal DuckDB spatial's, column for column. Deliberately the
#: measures, accessors and predicates rather than the constructors: a constructor returns
#: WKB, where a rendering difference would mask a real one.
_AGREES_WITH_DUCKDB = [
    "ST_Area(ST_GeomFromText(g))",
    "ST_Length(ST_GeomFromText(g))",
    "ST_Perimeter(ST_GeomFromText(g))",
    "ST_GeometryType(ST_GeomFromText(g))",
    "ST_Dimension(ST_GeomFromText(g))",
    "ST_IsEmpty(ST_GeomFromText(g))",
    "ST_NumGeometries(ST_GeomFromText(g))",
    "ST_NPoints(ST_GeomFromText(g))",
    "ST_XMin(ST_GeomFromText(g))",
    "ST_XMax(ST_GeomFromText(g))",
    "ST_YMin(ST_GeomFromText(g))",
    "ST_YMax(ST_GeomFromText(g))",
    "ST_Intersects(ST_GeomFromText(g), ST_GeomFromText('POINT(1 2)'))",
    "ST_Distance(ST_GeomFromText(g), ST_GeomFromText('POINT(0 0)'))",
    "ST_Contains(ST_GeomFromText(g), ST_GeomFromText('POINT(1 1)'))",
]


@pytest.mark.parametrize("call", _AGREES_WITH_DUCKDB)
def test_a_geo_call_answers_what_duckdb_spatial_answers(spatial, geoms, call):
    query = f"SELECT {call} AS v FROM t"
    spatial.register("t", geoms)
    expected = [row[0] for row in spatial.execute(query).fetchall()]
    actual = bt.sql(query, t=geoms).to_pydict()["v"]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        if isinstance(want, float) and isinstance(got, float):
            assert round(got, 9) == round(want, 9), call
        else:
            assert got == want, call


def test_wkt_is_read_permissively_and_written_strictly(spatial, geoms):
    """A known, deliberate divergence in *text*, asserted rather than left to surprise.

    DuckDB writes ``POINT (1 2)`` with a space after the type name and ``MULTIPOINT (0 0,
    1 1)`` with bare points; Batcher writes ``POINT(1 2)`` (PostGIS's rendering) and
    ``MULTIPOINT((0 0), (1 1))`` (the strict OGC form, where each point-text is
    parenthesized). All four are valid WKT for the same geometry.

    What matters for interop is the read direction, and it is permissive: both spellings
    parse, and to the same geometry. So a DuckDB-written geometry round-trips through
    Batcher by value; only the string it renders back to differs.
    """
    query = "SELECT ST_AsText(ST_GeomFromText(g)) AS v FROM t"
    spatial.register("t", geoms)
    duck = [row[0] for row in spatial.execute(query).fetchall()]
    ours = bt.sql(query, t=geoms).to_pydict()["v"]
    assert duck[0] == "POINT (1 2)" and ours[0] == "POINT(1 2)"
    assert duck[3] == "MULTIPOINT (0 0, 1 1)" and ours[3] == "MULTIPOINT((0 0), (1 1))"

    # The read direction: DuckDB's rendering parses here, to the same geometry as ours.
    both = pa.table({"g": pa.array([duck[3], ours[3]])})
    read_back = bt.sql(
        "SELECT ST_NumGeometries(ST_GeomFromText(g)) AS n, "
        "ST_AsText(ST_GeomFromText(g)) AS w FROM t",
        t=both,
    ).to_pydict()
    assert read_back["n"] == [2, 2]
    assert read_back["w"][0] == read_back["w"][1]


def test_st_buffer_takes_the_standard_two_argument_form(geoms):
    """PostGIS and DuckDB make `num_seg_quarter_circle` optional at 8; Batcher's Python
    signature requires it, so the two-argument SQL form needed the documented default."""
    two = bt.sql("SELECT ST_AsText(ST_Buffer(ST_GeomFromText(g), 1.0)) AS v FROM t", t=geoms)
    three = bt.sql("SELECT ST_AsText(ST_Buffer(ST_GeomFromText(g), 1.0, 8)) AS v FROM t", t=geoms)
    assert two.to_pydict() == three.to_pydict()


def test_a_wrong_arity_names_the_arity(geoms):
    """Not "unknown function", which would send the reader after the wrong problem."""
    with pytest.raises(NotImplementedError, match="takes 1 argument"):
        bt.sql("SELECT ST_Area(ST_GeomFromText(g), 1.0) AS v FROM t", t=geoms).collect()


# --- the rigid-body family --------------------------------------------------------------
#
# `bc-spatial` implements 42 single-column rotation and pose functions and the DataFrame API
# exposes every one; none was reachable from SQL either, for the same reason and with the
# same fix. There is no DuckDB oracle for these, so they are held against rotations whose
# answer is known by construction, and against the DataFrame spelling of the same call —
# which is the invariant that matters here: one engine, two front ends, one answer.

#: Imported rather than restated: the dispatcher and this test must agree on which names are
#: in scope, and a second copy of the rule is how they would stop agreeing.
_SQL_CALLABLE_SPATIAL = sorted(
    name for name in spatial_fns.__all__ if is_scalar_callable(getattr(spatial_fns, name))
)


@pytest.fixture
def rotations() -> pa.Table:
    """The identity, and a 90-degree rotation about z, as (x, y, z, w) plus a point."""
    half = math.sqrt(0.5)
    return pa.table(
        {
            "qx": pa.array([0.0, 0.0]),
            "qy": pa.array([0.0, 0.0]),
            "qz": pa.array([0.0, half]),
            "qw": pa.array([1.0, half]),
            "px": pa.array([1.0, 1.0]),
            "py": pa.array([0.0, 0.0]),
            "pz": pa.array([0.0, 0.0]),
            "tx": pa.array([0.0, 0.0]),
            "ty": pa.array([0.0, 0.0]),
            "tz": pa.array([0.0, 0.0]),
        }
    )


@pytest.mark.parametrize("name", _SQL_CALLABLE_SPATIAL)
def test_every_scalar_spatial_function_is_reachable_from_sql(rotations, name):
    """Same reachability contract as the geospatial family, over `spatial_fns.__all__`."""
    columns = {"qx", "qy", "qz", "qw", "px", "py", "pz", "tx", "ty", "tz"}
    params = list(inspect.signature(getattr(spatial_fns, name)).parameters)
    args = [p if p in columns else "1.0" for p in params]
    try:
        bt.sql(f"SELECT {name}({', '.join(args)}) AS v FROM t", t=rotations).collect()
    except NotImplementedError as exc:  # pragma: no cover - the defect this test exists for
        pytest.fail(f"{name} is not reachable from SQL: {exc}")


#: Rotations whose answer is fixed by the mathematics, not by this implementation: the
#: identity turns by nothing, a 90-degree z-rotation turns by pi/2 and takes the x-axis onto
#: the y-axis.
@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("quat_angle(qx, qy, qz, qw)", [0.0, math.pi / 2]),
        ("quat_norm(qx, qy, qz, qw)", [1.0, 1.0]),
        ("quat_to_yaw(qx, qy, qz, qw)", [0.0, math.pi / 2]),
        ("quat_rotate_x(qx, qy, qz, qw, px, py, pz)", [1.0, 0.0]),
        ("quat_rotate_y(qx, qy, qz, qw, px, py, pz)", [0.0, 1.0]),
    ],
)
def test_a_rotation_through_sql_is_the_rotation(rotations, call, expected):
    got = bt.sql(f"SELECT {call} AS v FROM t", t=rotations).to_pydict()["v"]
    assert [round(v, 12) for v in got] == [round(v, 12) for v in expected]


@pytest.mark.parametrize("call", ["quat_angle", "quat_to_yaw", "quat_norm"])
def test_sql_and_the_dataframe_api_reach_the_same_kernel(rotations, call):
    """One engine, two front ends, one answer — which is the whole point of the wiring."""
    args = ["qx", "qy", "qz", "qw"]
    from_sql = bt.sql(f"SELECT {call}({', '.join(args)}) AS v FROM t", t=rotations)
    from_api = bt.from_arrow(rotations).select(v=getattr(bt, call)(*(bt.col(a) for a in args)))
    assert from_sql.to_pydict() == from_api.to_pydict()


@pytest.mark.parametrize(
    "call",
    [
        "distance_3d(px, py, pz, tx, ty, tz)",
        "norm_3d(px, py, pz)",
        # Exactly the declared arity, which is the case an arity check cannot catch: two
        # scalars pass the count and then fail inside the function with a bare
        # `TypeError: a batcher expression is not iterable`.
        "quat_multiply(qx, qy)",
        # A *grouped return* over scalar parameters — the other half of the same rule.
        "quat_from_euler(px, py, pz)",
    ],
)
def test_a_grouped_argument_says_so_rather_than_counting(rotations, call):
    """`distance_3d(a: Point, b: Point)` takes two *points*, not six numbers.

    "takes 2 argument(s), got 6" reads as nonsense for a caller who wrote the components
    out, and "unknown function" would be worse still for a name that plainly exists.
    """
    with pytest.raises(NotImplementedError, match="group of columns"):
        bt.sql(f"SELECT {call} AS v FROM t", t=rotations).collect()


# --- the rest of the library ------------------------------------------------------------
#
# The same wiring reaches every *scalar* function `plan.functions` exports — the text-quality,
# evaluation and statistics families as well as the two spatial ones. Each was public,
# answered through `bt.<name>(...)`, and raised "unknown function" in SQL.


def _scalar_library() -> list:
    """Every registry member that is not an aggregate, with a column argument for each."""
    from batcher._sql.parser.expressions.lowering.families import _registry

    out = []
    for fn in _registry().values():
        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        out.append((fn.__name__, fn, params))
    return sorted(out, key=lambda entry: entry[0])


@pytest.fixture
def library_table() -> pa.Table:
    return pa.table(
        {
            "g": pa.array(["a", "b"]),
            "f": pa.array([1.0, 2.0]),
            "s": pa.array(["Alpha beta gamma", "x y"]),
            "l": pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64())),
        }
    )


def _library_argument(param) -> str:
    annotation = str(param.annotation).strip("'\"").lower()
    name = param.name.lower()
    if "list" in annotation or name in {"vec", "embedding", "vector"}:
        return "l"
    if name == "s" or any(
        k in name
        for k in (
            "text",
            "string",
            "prompt",
            "candidate",
            "reference",
            "prediction",
            "answer",
            "message",
            "document",
        )
    ):
        return "s"
    return "f"


def test_the_scalar_library_answers_what_the_dataframe_api_answers(library_table):
    """One engine, two front ends, one answer — over the whole library at once.

    Parametrizing 400 cases would make the failure list unreadable and the run slow; the
    interesting number is *how many* disagree, and which. A single case is enough to fail on.
    """
    ds = bt.from_arrow(library_table)
    checked, mismatched = 0, []
    for name, fn, params in _scalar_library():
        args = [_library_argument(p) for p in params]
        try:
            expected = ds.select(v=fn(*(bt.col(a) for a in args))).to_pydict()["v"]
        except Exception:
            continue  # the generated argument is not one this function takes
        try:
            actual = bt.sql(f"SELECT {name}({', '.join(args)}) AS v FROM t", t=library_table)
        except NotImplementedError as exc:
            # Two deliberate refusals, both explained rather than "unknown function": an
            # aggregate needs the aggregate dispatch, and a function taking a Python value
            # (`bleu(candidate, reference, n: int)`) cannot take the `Expr` a SQL argument
            # lowers to. Neither is a disagreement; a *third* kind of failure would be.
            if "is an aggregate" in str(exc) or "no scalar SQL spelling" in str(exc):
                continue
            if "no scan-order aggregate" in str(exc):
                # `first`/`last` name a row in *scan order*, which a mergeable aggregate
                # cannot promise. A refusal that predates this wiring and is the right one.
                continue
            if "unsupported SQL expression" in str(exc):
                # sqlglot gave this name a *typed* node, so it is intercepted by a handler
                # upstream of the derived dispatch and never reaches it. Whatever that
                # handler decides is that handler's contract, not this one's.
                continue
            mismatched.append((name, str(exc)[:80]))
            continue
        checked += 1
        got = actual.to_pydict()["v"]
        # NaN is not equal to itself, so a plain `==` reports a difference on every float
        # column carrying one — which several of these metrics do by design.
        canon = ["nan" if isinstance(v, float) and math.isnan(v) else v for v in (got, expected)[0]]
        want = ["nan" if isinstance(v, float) and math.isnan(v) else v for v in expected]
        if canon != want and name != "current_timestamp":
            mismatched.append((name, f"api={expected} sql={got}"))
    assert checked > 80, f"only {checked} library functions were exercised — the probe broke"
    assert not mismatched, f"SQL and the DataFrame API disagree: {mismatched[:8]}"


def test_a_name_sql_already_means_keeps_its_sql_meaning(library_table):
    """The derived dispatch runs last, so it can never shadow a real SQL function.

    `contains` and `upper` exist in both vocabularies; SQL's reading has to win, or adding
    a Python function could silently change what an existing query means.
    """
    out = bt.sql(
        "SELECT upper(s) AS u, contains(s, 'Alpha') AS c FROM t", t=library_table
    ).to_pydict()
    assert out["u"] == ["ALPHA BETA GAMMA", "X Y"]
    assert out["c"] == [True, False]

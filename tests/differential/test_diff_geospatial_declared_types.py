"""Every `st_*` and `quat_*`/`se3_*` function declares the type the engine returns.

`GeoFunc` and `SpatialFunc` were the two `Expr` node kinds `infer_type` had no arm for at
all. Because `Project.available_schema` is all-or-nothing by design
(`SchemaRef.from_typed_fields`), that did not cost the geometry column its type -- it cost
**every column beside it**: a `select(area=st_area(g), id=col("id"))` reported both as
`null`, and an empty result was built from that. 155 functions across the two vocabularies,
and not one of them had a rule.

**The oracle is the engine's own execution, not DuckDB**, and deliberately so: the question
here is not what `ST_Area` should return but whether the control plane's *static* answer
agrees with the runtime one. DuckDB cannot referee that -- it has no notion of the declared
type of an unexecuted Batcher plan -- so a non-empty `collect()` is the reference, exactly
as `test_diff_aggregate_declared_types` uses it for `AGG_FNS`. A zero-row run cannot be the
reference, because an empty result is *built from* the declaration this file is checking.
The values these functions compute are DuckDB's business and are covered by the geospatial
differential tests; the types are this file's.

This file re-derives the classification from `GEO_FNS` and `SPATIAL_FNS`, so a function
added to either vocabulary fails here rather than shipping untyped. It is deliberately two
assertions rather than one:

* the *partition* check needs no engine and no data -- it just holds the five geometry sets
  against the vocabulary, and is what a newly added name trips;
* the *execution* check builds each function through its public constructor and compares the
  declared type against a real run, which is what a wrongly classified name trips.

A partition alone would let `st_area` be classified as returning text; an execution sweep
alone would quietly stop covering whatever it could no longer build.
"""

from __future__ import annotations

import inspect

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.plan.expr_ir.fn_names import GEO_FNS, SPATIAL_FNS
from batcher.plan.types.infer import geospatial

pytestmark = pytest.mark.differential

#: The two vocabulary names whose public constructor is spelled differently.
_PUBLIC_NAME = {"st_force2d": "st_force_2d", "st_force3d": "st_force_3d"}

_ROWS = pa.table(
    {
        "wkt": pa.array(["POINT(1 2)", "LINESTRING(0 0,1 1,2 2)"]),
        "wkt2": pa.array(["POINT(3 4)", "LINESTRING(1 1,2 2,3 3)"]),
        "poly": pa.array(["POLYGON((0 0,4 0,4 4,0 4,0 0))", "POLYGON((0 0,2 0,2 2,0 2,0 0))"]),
        "gh": pa.array(["u4pruyd", "ezs42"]),
        **{f"n{i}": pa.array([1.0, 0.5], pa.float64()) for i in range(10)},
        "cell": pa.array([1, 2], pa.int64()),
    }
)


@pytest.fixture(scope="module")
def dataset():
    return bt.from_arrow(_ROWS)


def _numbers(count: int) -> list:
    return [col(f"n{i}") for i in range(count)]


def _attempts(fn_name: str, arity: int):
    """Argument tuples worth trying, in the order the families actually take them."""
    geom = bt.st_geom_from_text(col("wkt"))
    geom2 = bt.st_geom_from_text(col("wkt2"))
    poly = bt.st_geom_from_text(col("poly"))
    yield [poly, *_numbers(arity - 1)] if arity else []
    yield [geom, *_numbers(arity - 1)] if arity else []
    if arity >= 2:
        yield [geom, geom2, *_numbers(arity - 2)]
        yield [poly, poly, *_numbers(arity - 2)]
    yield _numbers(arity)
    if arity >= 1:
        yield [col("wkt"), *_numbers(arity - 1)]
        yield [col("gh"), *_numbers(arity - 1)]
        # The grid readers take a *cell id* first, which is an integer rather than a
        # coordinate -- `st_hex_center_x(cell, size)`, `st_s2_cell_parent(cell, level)`.
        yield [col("cell"), *_numbers(arity - 1)]
        yield [col("cell"), 1]
    if arity >= 2:
        yield [*_numbers(arity - 1), 5]


def _declared_and_actual(dataset, node):
    built = dataset.select(v=node)
    return built.schema.field("v").type, built.collect().schema.field("v").type


def _build_and_measure(dataset, fn_name: str):
    """`(declared, actual)` for `fn_name`, or `None` if no shape builds it."""
    ctor = getattr(bt, _PUBLIC_NAME.get(fn_name, fn_name), None)
    if not callable(ctor):
        return None
    required = [
        p
        for p in inspect.signature(ctor).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    for args in _attempts(fn_name, len(required)):
        if len(args) != len(required):
            continue
        try:
            return _declared_and_actual(dataset, ctor(*args))
        except Exception:
            continue
    return None


# --- the partition, which needs no engine ------------------------------------------------

_GEO_SETS = {
    "binary": geospatial._GEO_BINARY,
    "double": geospatial._GEO_DOUBLE,
    "bool": geospatial._GEO_BOOL,
    "int64": geospatial._GEO_INT,
    "string": geospatial._GEO_STRING,
}


def test_the_geometry_sets_partition_the_vocabulary():
    """Every `GEO_FNS` member is classified exactly once, and nothing else is."""
    union: set[str] = set()
    for name, members in _GEO_SETS.items():
        overlap = union & set(members)
        assert not overlap, f"{name} also claims {sorted(overlap)}"
        union |= set(members)
    assert union == set(GEO_FNS), (
        f"unclassified: {sorted(set(GEO_FNS) - union)}; "
        f"classified but not in GEO_FNS: {sorted(union - set(GEO_FNS))}"
    )


def test_every_geometry_function_resolves_to_a_type():
    """The lookup answers for every member -- `None` is the fallback, not an answer."""
    unresolved = sorted(fn for fn in GEO_FNS if geospatial.geofunc_type(fn) is None)
    assert not unresolved, unresolved


def test_the_rigid_body_vocabulary_is_uniformly_double():
    """Stated as a property because it is one: `SPATIAL_FNS` names an output *component*."""
    assert all(geospatial.spatialfunc_type(fn) == pa.float64() for fn in SPATIAL_FNS)


# --- the execution sweep -----------------------------------------------------------------


@pytest.mark.parametrize("fn", sorted(GEO_FNS))
def test_a_geometry_function_declares_what_the_engine_returns(dataset, fn):
    measured = _build_and_measure(dataset, fn)
    assert measured is not None, (
        f"no argument shape builds {fn}, so its declared type is going unchecked rather "
        "than passing -- add the shape it needs to `_attempts`"
    )
    declared, actual = measured
    assert declared == actual, f"{fn}: declared {declared}, engine returns {actual}"


@pytest.mark.parametrize("fn", sorted(SPATIAL_FNS))
def test_a_rigid_body_function_declares_what_the_engine_returns(dataset, fn):
    measured = _build_and_measure(dataset, fn)
    assert measured is not None, f"no argument shape builds {fn}"
    declared, actual = measured
    assert declared == actual, f"{fn}: declared {declared}, engine returns {actual}"


def test_a_geometry_column_no_longer_costs_its_neighbours_their_types():
    """The symptom that made this worth 155 table entries rather than a `None` fallback."""
    rows = pa.table({"wkt": pa.array(["POINT(1 2)"]), "id": pa.array([7], pa.int64())})
    geom = bt.st_geom_from_text(col("wkt"))
    schema = bt.from_arrow(rows).select(area=bt.st_area(geom), id=col("id")).schema
    assert schema.field("area").type == pa.float64()
    assert schema.field("id").type == pa.int64()

"""SQL must reach every typed accessor the expression API has, and mean the same thing.

`.str`, `.dt`, `.list`, `.struct`, `.json`, `.map`, `.image`, `.audio` and `.video` are the
bulk of Batcher's expression surface, and SQL used to reach them through a table per family
-- so a method added to a namespace was reachable from `col(...)` and silently absent from
SQL. `lowering/accessors.py` derives the SQL vocabulary from the namespaces instead. These
tests are the other half of that: they walk the *live* namespaces, so the derivation cannot
quietly stop covering one.

The oracle is the expression API itself, compared as lowered IR rather than as results. The
SQL spelling of an accessor is supposed to *be* the accessor call, so equal IR is the exact
claim -- and it holds for every entry with no data, no engine, and no per-namespace fixture
of images and waveforms that would decide which of the 435 names actually got checked.
"""

from __future__ import annotations

import inspect

import pytest

import batcher as bt
from batcher._sql.parser.expressions.anonymous import known_names
from batcher._sql.parser.expressions.lowering.accessors import (
    _CURATED_ELSEWHERE,
    accessor_namespaces,
    accessor_vocabulary,
)
from batcher._sql.parser.expressions.lowering.signatures import STRINGS, parameter_kinds
from batcher.plan.expr_ir import Expr, col
from batcher.plan.functions.collection import element

#: One argument per parameter kind. Deliberately values every namespace accepts, so a
#: mismatch is the translation's and not the fixture's.
_SQL_ARGUMENT = {Expr: "c", str: "'a'", bool: "TRUE", int: "1", float: "1.0", STRINGS: "'a'"}
_PY_ARGUMENT = {Expr: col("c"), str: "a", bool: True, int: 1, float: 1.0, STRINGS: ["a"]}


def _accessor_methods() -> list[tuple[str, str]]:
    """Every public callable on every accessor namespace."""
    probe = col("__probe")
    found = []
    for namespace in accessor_namespaces():
        cls = type(getattr(probe, namespace))
        found += [
            (namespace, name)
            for name in sorted(n for n in dir(cls) if not n.startswith("_"))
            if callable(getattr(cls, name, None)) and not inspect.isclass(getattr(cls, name))
        ]
    return found


def test_the_namespaces_are_not_empty() -> None:
    """The walk found the surface it claims to walk.

    Without this the tests below are vacuously true the moment the accessors move. `.seq`
    is named because it is the namespace a *hardcoded* list of namespaces omitted, which is
    the failure `accessor_namespaces` exists to make impossible.
    """
    assert len(_accessor_methods()) > 400
    assert "seq" in accessor_namespaces()


def test_the_namespace_list_is_the_whole_namespace_list() -> None:
    """`accessor_namespaces` finds every namespace, checked by a second, independent route.

    It reads the *annotations* on `Expr`'s properties, which is a static reading and drops
    a namespace silently if one is ever spelled differently. This asks the built object
    what it got instead, so the two have to agree.
    """
    probe = col("__probe")
    live = {
        name
        for name in dir(type(probe))
        if not name.startswith("_")
        and type(getattr(probe, name, None)).__name__.endswith("Namespace")
    }
    assert live, "the runtime probe found no namespaces at all"
    assert live == set(accessor_namespaces())


def test_every_accessor_method_has_a_sql_name() -> None:
    """No accessor operation is reachable from `col(...)` and unreachable from SQL.

    A name reaches SQL one of two ways: the derived vocabulary, or a handler that already
    curates it -- `anonymous.py`'s tables, or one of the family modules listed in
    `_CURATED_ELSEWHERE`. The derived dispatch deliberately declines both, so that a name
    served by a curated handler at *some* arities cannot mean something else at the rest.
    `test_the_curated_names_reach_the_accessor` holds the curated half to the same meaning.
    """
    vocabulary = accessor_vocabulary()
    claimed = known_names() | _CURATED_ELSEWHERE
    missing = [
        f"{namespace}.{method}"
        for namespace, method in _accessor_methods()
        if (key := f"{namespace}{method}".replace("_", "")) not in vocabulary and key not in claimed
    ]
    assert not missing, f"no SQL spelling for: {missing}"

    # And a name reached through a curated handler has to be *pinned* to the accessor it
    # reaches, not merely claimed by something. Without this the coverage above would pass
    # for a name that answers to a completely different operation.
    #
    # An alias never reaches this list: `str.endswith` normalizes to the same key as
    # `str.ends_with`, so it is in the vocabulary and the first check already passed it.
    unverified = [
        f"{namespace}.{method}"
        for namespace, method in _accessor_methods()
        if f"{namespace}{method}".replace("_", "") not in vocabulary
        and (namespace, method) not in _CURATED_CALLS
        and (namespace, method) not in _AGGREGATE_ACCESSORS
    ]
    assert not unverified, f"curated but not pinned to an accessor: {unverified}"


#: Accessors that *reduce*. They cannot be pinned the way the others are -- the comparison
#: below builds a bare projection, and an aggregate needs a `GROUP BY` around it -- so each
#: has its own test. `test_the_aggregate_accessor_reaches_the_same_aggregate` is `.str.join`'s.
_AGGREGATE_ACCESSORS = {("str", "join")}


def test_the_aggregate_accessor_reaches_the_same_aggregate() -> None:
    """`.str.join(sep)` collapses a group to one string, which SQL spells `string_agg`.

    Compared as *results* rather than as lowered IR, because the two spellings reach the
    aggregate by different routes -- the accessor builds it directly, SQL through its
    aggregate dispatch -- so equal IR is not the claim. Equal answers is.
    """
    ds = bt.from_pydict({"g": ["a", "a", "b"], "c": ["x", "y", "z"]})
    expected = ds.group_by("g").agg(r=col("c").str.join("-")).sort("g").to_pydict()
    actual = bt.sql(
        "SELECT g, string_agg(c, '-') AS r FROM t GROUP BY g ORDER BY g", t=ds
    ).to_pydict()
    assert actual == expected


#: The accessors reached through a curated SQL name instead of the derived one, each with
#: the call that reaches it. Written out rather than derived, because the curated spelling
#: is a *choice* -- DuckDB's `list_unique` counts where `.list.unique` lists, and the JSON
#: readers take a path where the accessor takes whatever it is given -- and a table of
#: choices is the thing a test should be checking rather than recomputing.
_CURATED_CALLS: dict[tuple[str, str], tuple[str, str]] = {
    # The five `.json` readers take a JSON *path*; the curated handler normalizes a bare key.
    ("json", "array_length"): ("json_array_length(c, '$.a')", "array_length('$.a')"),
    ("json", "exists"): ("json_exists(c, '$.a')", "exists('$.a')"),
    ("json", "extract_string"): ("json_extract_string(c, '$.a')", "extract_string('$.a')"),
    ("json", "keys"): ("json_keys(c, '$.a')", "keys('$.a')"),
    ("json", "value"): ("json_value(c, '$.a')", "value('$.a')"),
    # DuckDB's `list_unique` counts; the list itself is `list_distinct`.
    ("list", "unique"): ("list_distinct(l)", "unique()"),
    # DuckDB's bounds are inclusive and 1-based; the accessor's are an offset and a length.
    ("list", "slice"): ("list_slice(l, 2, 3)", "slice(1, 2)"),
    # `strlen` is DuckDB's byte length, so `.str.len` keeps the ANSI spelling.
    ("str", "len"): ("length(c)", "len()"),
    ("str", "split"): ("string_split(c, ' ')", "split(' ')"),
    # The rest are curated under exactly their own name, and are here so "reachable" means
    # "reaches this operation" rather than "some handler answers to the name".
    ("list", "concat"): ("list_concat(l, l)", "concat(bt.col('l'))"),
    ("list", "difference"): ("list_difference(l, l)", "difference(bt.col('l'))"),
    ("list", "first"): ("list_first(l)", "first()"),
    ("list", "last"): ("list_last(l)", "last()"),
    ("list", "median"): ("list_median(l)", "median()"),
    ("list", "position"): ("list_position(l, 1)", "position(1)"),
    ("list", "has_all"): ("list_has_all(l, l)", "has_all(bt.col('l'))"),
    ("list", "has_any"): ("list_has_any(l, l)", "has_any(bt.col('l'))"),
    ("list", "intersect"): ("list_intersect(l, l)", "intersect(bt.col('l'))"),
    ("list", "union"): ("list_union(l, l)", "union(bt.col('l'))"),
    ("struct", "keys"): ("struct_keys(c)", "keys()"),
    ("map", "keys"): ("map_keys(c)", "keys()"),
    ("map", "values"): ("map_values(c)", "values()"),
    ("map", "entries"): ("map_entries(c)", "entries()"),
    ("map", "contains"): ("map_contains(c, 'a')", "contains('a')"),
    ("list", "transform"): ("list_transform(l, x -> x)", "transform(element())"),
    ("list", "filter"): ("list_filter(l, x -> x)", "filter(element())"),
}


@pytest.mark.parametrize(("namespace", "method"), sorted(_CURATED_CALLS))
def test_the_curated_names_reach_the_accessor(namespace: str, method: str) -> None:
    """A curated SQL name lowers to the same accessor the expression API exposes.

    This is the half of the coverage claim the derived dispatch does not make. Without it
    "reachable" would mean only that *some* handler answers to the name, which is not the
    same as its answering with the operation the DataFrame API spells the same way.
    """
    sql_call, python_call = _CURATED_CALLS[(namespace, method)]
    ds = bt.from_pydict({"c": ["a b"], "l": [[1, 2]]})
    subject = "l" if "(l" in sql_call else "c"
    scope = {"col": col, "bt": bt, "element": element}
    expected = eval(f"col({subject!r}).{namespace}.{python_call}", scope)
    plan = bt.sql(f"SELECT {sql_call} AS r FROM t", t=ds)._plan
    assert plan.items[0].expr.to_ir() == expected.to_ir()


def _arguments(fn):
    """The SQL argument list and the Python one for a call to `fn`, or None if unclassified."""
    kinds = parameter_kinds(fn, skip_first=True)
    if kinds is None:
        return None
    return (
        ", ".join(["c"] + [_SQL_ARGUMENT[k] for k in kinds]),
        [_PY_ARGUMENT[k] for k in kinds],
    )


@pytest.mark.parametrize(
    ("namespace", "method"), sorted((ns, m) for ns, m, _ in accessor_vocabulary().values())
)
def test_the_sql_spelling_lowers_to_the_accessor_call(namespace: str, method: str) -> None:
    """`<ns>_<method>(c, ...)` builds exactly what `col("c").<ns>.<method>(...)` builds.

    The written spelling keeps its underscores. The vocabulary is *keyed* without them, so
    that ``ST_AsText`` and ``st_as_text`` agree, but the key is a lookup form and not a
    name to hand a reader: ``strlen`` (the key for `.str.len`) is DuckDB's own byte-length
    function, which an earlier handler rightly claims, while ``str_len`` reaches the
    accessor. Writing the query from the key rather than from the name is how this test
    first reported that collision as a defect in the dispatch.

    An argument the method rejects (a string where it wants a date, a size it validates)
    makes *both* sides raise, which is still agreement -- and is why the comparison is on
    the raised type as well as on the IR.
    """
    name = f"{namespace}_{method}"
    fn = accessor_vocabulary()[f"{namespace}{method}".replace("_", "")][2]
    arguments = _arguments(fn)
    assert arguments is not None
    sql_args, py_args = arguments
    ds = bt.from_pydict({"c": ["a"]})

    try:
        expected = getattr(getattr(col("c"), namespace), method)(*py_args).to_ir()
    except Exception as exc:  # the accessor refuses these arguments
        with pytest.raises(type(exc)):
            bt.sql(f"SELECT {name}({sql_args}) AS r FROM t", t=ds)
        return

    plan = bt.sql(f"SELECT {name}({sql_args}) AS r FROM t", t=ds)._plan
    assert plan.items[0].expr.to_ir() == expected

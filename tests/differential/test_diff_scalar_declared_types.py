"""Every unary math function's *declared* output type is the one the engine produces.

`test_diff_aggregate_declared_types` does this for `AGG_FNS`, and the reason it exists
applies unchanged to the scalar vocabulary: `Project.available_schema` is what
`Dataset.schema` is answered from, what an empty result is typed from, and what the device
tier holds its own result against (`api/terminal/gpu_backend/verify.py`). A wrong answer
there is not a cosmetic one.

It is *worse* here than for an aggregate, because a projection carries several columns and
the schema is all-or-nothing by design (`SchemaRef.from_typed_fields` -- a partial schema
missing a field would be planned against rather than falling back). So one uncertain column
costs the declared type of **every** column beside it. That is what this file was written
after: `abs()` over a null-typed column made a neighbouring plain Int64 passthrough
advertise as `null` in `Dataset.schema`, and four functions plus `div` were in that state.

The sweep is derived from `MATH_FNS`, so a function added to the vocabulary with no type
rule fails here instead of quietly making every plan that uses it schema-less.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.expr_ir.core import Math2Expr, MathExpr
from batcher.plan.expr_ir.fn_names import LIST_FNS, MATH_FNS, STR_FNS
from batcher.plan.expr_ir.func_nodes import ListFunc, StrFunc

pytestmark = pytest.mark.differential

#: One column per input family the math vocabulary accepts. `z` is the one that matters and
#: the one that was missing: `null` is not `is_integer` and not `is_floating`, so every rule
#: asking that question answered "uncertain" for it.
_COLUMNS: dict[str, pa.Array] = {
    "i32": pa.array([1, 2, 3, 4], pa.int32()),
    "i64": pa.array([1, 2, 3, 4], pa.int64()),
    "f32": pa.array([1.0, 2.0, 3.0, 4.0], pa.float32()),
    "f64": pa.array([1.0, 2.0, 3.0, 4.0], pa.float64()),
    "s": pa.array(["a", "b", "a", "c"]),
    "bo": pa.array([True, False, True, True], pa.bool_()),
    "z": pa.array([None] * 4, pa.null()),
}

#: Scalar functions the control plane is allowed not to type, each with the reason. **Empty,
#: and that is the point** -- an entry costs a null-typed empty result and a `Dataset.schema`
#: that reports every sibling column as `null`. Adding one is a decision to surface.
_UNDECLARED: dict[str, str] = {}


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    return pa.table(_COLUMNS)


def _declared_and_actual(dataset) -> tuple[pa.DataType | None, pa.DataType]:
    schema = dataset._plan.available_schema()
    declared = schema.arrow.field("v").type if schema else None
    return declared, dataset.collect().schema.field("v").type


def _accepted(rows: pa.Table, fn: str) -> list[tuple[str, pa.DataType | None, pa.DataType]]:
    """Every `(column, declared, engine)` triple this function accepts."""
    out = []
    for name in _COLUMNS:
        try:
            dataset = bt.from_arrow(rows).select(v=MathExpr(fn, bt.col(name)))
            declared, actual = _declared_and_actual(dataset)
        except Exception:
            # A pair the engine has no meaning for is refused at plan build or at execution.
            # Skipping rather than listing which functions take which input is what keeps a
            # newly added function visible here instead of silently unreachable.
            continue
        out.append((name, declared, actual))
    return out


@pytest.mark.parametrize("fn", sorted(MATH_FNS))
def test_the_declared_type_is_the_engine_type(rows, fn):
    accepted = _accepted(rows, fn)
    assert accepted, f"{fn} accepted none of the input types -- the fixture cannot see it"
    for name, declared, actual in accepted:
        if declared is None:
            assert fn in _UNDECLARED, (
                f"{fn} over {name!r} declares no output type (the engine returns {actual}); "
                "give it a rule in `plan.types.infer.arithmetic`, or admit it in "
                "`_UNDECLARED` with the reason -- an uncertain column costs the declared "
                "type of every other column in the same projection, not just its own"
            )
            continue
        assert declared == actual, (
            f"{fn} over {name!r}: declared {declared}, engine returns {actual}"
        )


def test_an_uncertain_column_would_cost_its_neighbours_their_types(rows):
    """The positive control for why the sweep above is worth its runtime.

    This is the symptom that made the four missing rules user-visible, and it is stated as
    a property rather than a regression: `k` is a plain Int64 passthrough that no expression
    touches, and its declared type depends on whether `v` beside it could be typed.
    """
    dataset = bt.from_arrow(rows).select(v=MathExpr("abs", bt.col("z")), k=bt.col("i64"))
    assert dataset.schema.field("v").type == pa.float64()
    assert dataset.schema.field("k").type == pa.int64()


#: The families the binary sweep below covers. `s` and `bo` are deliberately out, and the
#: omission is a real limit rather than an oversight: the engine *coerces* a string or a
#: boolean into `round(x, n)` and into `//`, returning `double` (silent nulls for a string
#: that does not parse), where DuckDB refuses all three spellings with a `BinderException`.
#: Giving those a declared type would entrench a leniency the oracle does not have, so the
#: engine-vs-DuckDB question is the one to settle first. Every purely numeric pair, and
#: every pair involving `null`, is covered -- which is the rule this file was written for.
_BINARY_COLUMNS = ("i32", "i64", "f32", "f64", "z")


@pytest.mark.parametrize("column", _BINARY_COLUMNS)
def test_binary_round_and_division_declare_what_the_engine_returns(rows, column):
    """`round(x, n)` and `/` follow their operand, and a `null` operand has to be one."""
    for build in (
        lambda c: Math2Expr("round", c, bt.lit(2)),
        lambda c: c / bt.col("i64"),
        lambda c: bt.col("i64") / c,
    ):
        try:
            dataset = bt.from_arrow(rows).select(v=build(bt.col(column)))
            declared, actual = _declared_and_actual(dataset)
        except Exception:
            continue
        assert declared == actual, f"over {column!r}: declared {declared}, engine {actual}"


def test_the_fixture_reaches_every_input_family(rows):
    """Guard against a vacuous sweep.

    Every assertion above is "declared == engine for the pairs the engine accepts", which a
    fixture that stopped producing usable columns would satisfy by accepting nothing. No
    single function spans the fixture -- `abs` refuses a string and a boolean, which is
    correct -- so the check is that the vocabulary as a whole still reaches every family.
    """
    assert set(rows.column_names) == set(_COLUMNS)
    reached = {name for fn in MATH_FNS for name, _d, _a in _accepted(rows, fn)}
    assert reached == set(_COLUMNS), f"the vocabulary no longer reaches {set(_COLUMNS) - reached}"


# --- the string vocabulary -------------------------------------------------------------

#: A 32-byte AES key, given as the 64 hex characters `aes_encrypt` requires.
_AES_KEY = "0" * 64

#: The `(column, kwargs)` shapes a `str` function may need, tried in order until one builds.
#: Probing rather than recording which function takes which is what keeps a newly added
#: function visible here -- an unrecorded one would simply never be built, and the sweep
#: would pass by omission. `test_a_string_function_declares_what_the_engine_returns` fails
#: on "never built" for exactly that reason, and widening this list is how you fix it.
#:
#: The shapes past the first three are not padding. Each was added because a function was
#: unreachable without it, and three of the four unreachable groups turned out to be
#: *missing type rules* once they could be built at all: `bin` over an integer, `to_case`
#: with a style, and `compress`/`decompress` with a codec.
_STR_SHAPES: tuple[tuple[str, dict], ...] = (
    ("s", {}),
    ("s", {"pattern": "a"}),
    ("s", {"pattern": "a", "replacement": "x"}),
    # `hamming` is defined only between equal-length strings and refuses anything else, so
    # a one-character pattern reaches every other pattern-taking function and not it.
    ("s", {"pattern": "ab"}),
    ("z", {}),
    ("i", {}),  # chr, bin, to_base, format_bytes -- an integer rendered as text
    ("s", {"length": 2}),  # chunk, minhash
    ("s", {"pattern": "gzip"}),  # compress / decompress -- a codec
    ("s", {"pattern": "snake"}),  # to_case -- a style
    ("s", {"pattern": _AES_KEY}),  # aes_encrypt / aes_decrypt -- a key
    ("s", {"pattern": "Xxn\x00"}),  # mask_by_class -- one replacement per character class
    ("s", {"start": 1, "length": 2}),
)


@pytest.fixture(scope="module")
def text() -> pa.Table:
    return pa.table(
        {
            "s": pa.array(["ab", "bc", "ab", "cd"]),
            "z": pa.array([None] * 4, pa.null()),
            "i": pa.array([65, 66, 67, 68], pa.int64()),
        }
    )


@pytest.mark.parametrize("fn", sorted(STR_FNS))
def test_a_string_function_declares_what_the_engine_returns(text, fn):
    """`strfunc_type` answers from the function name alone, so a name it has never been
    told about returns ``None`` -- and takes every column in the projection with it.

    Seven functions were in that state, and each was a sibling of an entry already in the
    table: `json_extract`/`json_value` beside `json_extract_string`, `json_contains` and
    `json_exists` beside `json_extract_bool`, `damerau_levenshtein` beside `levenshtein`,
    and the two Jaro similarities beside `jaccard_similarity`. That is the shape a
    hand-maintained lookup fails in, which is why the sweep is derived from `STR_FNS`.
    """
    checked = 0
    for column, kwargs in _STR_SHAPES:
        try:
            dataset = bt.from_arrow(text).select(v=StrFunc(fn, bt.col(column), **kwargs))
            declared, actual = _declared_and_actual(dataset)
        except Exception:
            continue
        checked += 1
        assert declared is not None, (
            f"{fn} over {column!r} declares no output type (the engine returns {actual}); "
            "add it to the right table in `plan.types.infer.scalars`"
        )
        assert declared == actual, (
            f"{fn} over {column!r}: declared {declared}, engine returns {actual}"
        )
    assert checked, (
        f"{fn} was never built -- no shape in `_STR_SHAPES` reaches it, so this function "
        "is going untested rather than passing. Add the shape it needs."
    )


# --- the list vocabulary ---------------------------------------------------------------

#: One list column per element type. The element type is the whole question for a list
#: reduction, and it is where `sum` was wrong: it was classified with `min`/`max` as
#: element-preserving, which holds for a numeric element and for no other kind.
_LIST_COLUMNS: dict[str, pa.Array] = {
    "int64": pa.array([[1, 2], [3]], pa.list_(pa.int64())),
    "float64": pa.array([[1.0, 2.0], [3.0]], pa.list_(pa.float64())),
    "string": pa.array([["1", "2"], ["3"]], pa.list_(pa.string())),
    "bool": pa.array([[True, False], [True]], pa.list_(pa.bool_())),
    "date32": pa.array([[1, 2], [3]], pa.list_(pa.date32())),
    "timestamp": pa.array([[1, 2], [3]], pa.list_(pa.timestamp("us"))),
    "null": pa.array([[None, None], [None]], pa.list_(pa.null())),
    # `flatten` is the one member of the vocabulary that needs a *nested* list, and without
    # this column it accepted nothing at all rather than being tested.
    "nested": pa.array([[[1, 2], [3]], [[4]]], pa.list_(pa.list_(pa.int64()))),
}


@pytest.fixture(scope="module")
def lists() -> pa.Table:
    return pa.table(_LIST_COLUMNS)


@pytest.mark.parametrize("fn", sorted(LIST_FNS))
def test_a_list_function_declares_what_the_engine_returns(lists, fn):
    """Every element type this reduction accepts, held against what the engine returns.

    A pair the engine refuses -- `sum` over a Date list, say -- is skipped by *not being
    built*, never by `pytest.skip`. The difference matters: a skip would turn a genuine
    regression into a test that quietly leaves the run, and `lint-skips` reads module-level
    guards, so a mid-body one is invisible to it. The `checked` count is what closes that
    hole -- if a function stops working over every element type, this fails.
    """
    checked = 0
    for element in sorted(_LIST_COLUMNS):
        try:
            dataset = bt.from_arrow(lists).select(v=ListFunc(fn, bt.col(element)))
            declared, actual = _declared_and_actual(dataset)
        except Exception:
            continue
        checked += 1
        assert declared == actual, (
            f"{fn} over a {element} list: declared {declared}, engine returns {actual}"
        )
    assert checked, f"{fn} accepted no element type -- the fixture cannot see it"


def test_an_empty_result_is_typed_the_way_a_populated_one_is(lists):
    """The declared type is what an empty result is *made of*, so a wrong one is a wrong
    answer rather than a wrong annotation.

    `list.sum()` over a `List<String>` declared `string` and returned `double`, which made
    one query produce two column types: `double` when the filter matched and `string` when
    it did not. That is the `{collect} x {empty}` cell `CLAUDE.md` names, reached through
    the type system rather than through the operator.
    """
    dataset = bt.from_arrow(lists).select(v=ListFunc("sum", bt.col("string")))
    populated = dataset.collect().schema.field("v").type
    empty = dataset.filter(bt.col("v") > 10_000).collect().schema.field("v").type
    assert populated == empty == pa.float64()


#: Node kinds with no type rule, each with the reason it has none. Empty, and meant to
#: stay that way: every one of the five that was missing turned out to declare a wrong type.
NO_TYPE_RULE: dict[str, str] = {}


def test_every_expression_node_has_a_type_rule():
    """No `IRNode` subclass is missing an arm in `plan/types/infer/dispatch.py`.

    That module is an `isinstance` cascade -- the shape `expr_ir/walk.py` deliberately
    avoids, because "a per-type cascade silently returns the empty set for any node nobody
    added an arm for". Five of the node types had no arm and *every one* declared `null`
    for a column that collects as a real type.

    Two things make this check honest, and both were wrong in its first draft:

    * **The denominator is forced complete.** `IRNode.__subclasses__()` returns 48 types on
      a bare import and 52 once the lazily-imported `image`/`audio`/`video` namespaces have
      registered theirs, so the modules are imported here rather than left to whatever a
      previous test happened to touch. A count that depends on import order is not a count.
    * **A grouped arm counts, a bare mention does not.** Twelve types are handled by tuple
      arms such as ``isinstance(expr, (Not, IsNull, IsNotNull, IsNan, IsInf))``, so looking
      only for ``isinstance(expr, Name)`` reports them missing. Falling back to "the name
      appears in the module" is the opposite error -- every one of them appears in the
      import block, so that fallback passes a node for being imported.
    """
    import importlib
    import inspect
    import re

    import batcher.plan.types.infer.dispatch as dispatch
    from batcher.plan.expr_ir.core import IRNode

    # Imported for their side effect -- each registers its own `IRNode` subclasses, and
    # without them `__subclasses__()` is short by five (four multimodal, one `.seq`).
    # Spelled through `import_module`
    # rather than a plain `import`, because a plain one reads as unused and `ruff --fix`
    # deletes it: that happened here, the denominator silently fell back to 48, and only
    # the length assertion below caught it.
    for _side_effect in ("audio", "image", "video", "namespaces.sequence"):
        importlib.import_module(f"batcher.plan.expr_ir.{_side_effect}")

    source = inspect.getsource(dispatch)
    arms = re.findall(r"isinstance\(\s*expr\s*,\s*(\(?[^)]*\)?)", source)
    covered = {n for arm in arms for n in re.findall(r"\b([A-Z]\w+)\b", arm)}

    subclasses = sorted(c.__name__ for c in IRNode.__subclasses__())
    assert len(subclasses) >= 53, f"denominator collapsed to {len(subclasses)} node types"

    missing = sorted(n for n in subclasses if n not in covered)
    unexplained = [n for n in missing if n not in NO_TYPE_RULE]
    assert not unexplained, (
        f"{len(unexplained)} expression node type(s) have no arm in infer/dispatch.py: "
        f"{unexplained}. Each makes `Dataset.schema` advertise `null` for its column."
    )
    stale = sorted(n for n in NO_TYPE_RULE if n not in missing)
    assert not stale, f"NO_TYPE_RULE lists nodes that now have a rule (remove them): {stale}"


def test_map_from_arrays_declares_the_map_type_it_returns():
    """`map_from_arrays` declares `map<k, v>`, not `null`.

    `MakeMap` was one of five `IRNode` subclasses with no arm in
    `plan/types/infer/dispatch.py`, and that module is a 43-branch `isinstance` cascade --
    the shape `expr_ir/walk.py` avoids precisely because "a per-type cascade silently
    returns the empty set for any node nobody added an arm for". Here the fall-through was
    `None`, so `Dataset.schema` advertised `null` for a column that collects as
    `map<string, int64>`.

    That is the failure this file's own docstring describes as "not a cosmetic one": the
    declared schema is what an empty result is typed from and what the device tier holds
    its results against, and one uncertain column costs every column beside it its type.
    """
    ds = bt.from_pydict({"k": [["a", "b"]], "n": [[1, 2]]})
    out = ds.select(v=bt.map_from_arrays(bt.col("k"), bt.col("n")))
    declared, actual = _declared_and_actual(out)
    assert declared == pa.map_(pa.string(), pa.int64())
    assert declared == actual

    # The neighbour check this file exists for: an uncertain column would strip the
    # declared type off the plain passthrough sitting next to it.
    both = ds.select(v=bt.map_from_arrays(bt.col("k"), bt.col("n")), keep=bt.col("k"))
    assert both._plan.available_schema() is not None
    assert both.schema.field("keep").type == pa.list_(pa.string())


def test_the_untyped_node_kinds_declare_what_they_return():
    """All five nodes that had no type rule declare their real type.

    Spelled through the public surface that reaches each, because that is where the wrong
    type was visible: `Dataset.schema` said `null` for every one of them.

    `WindowStart` is the one worth reading twice. Its type is a **timestamp whatever the
    input was**, not the input's own type -- the plausible rule. `_sql`'s bucket lowering
    proves it: for a DATE argument it builds `Cast(WindowStart(...), "date")`, and that cast
    exists only because the window yields a timestamp. Declaring `date32` made the cast look
    redundant, it was eliminated, and `time_bucket(INTERVAL 1 DAY, DATE ...)` returned a
    timestamp -- a wrong declared type becoming a wrong result.
    """
    import datetime as _dt

    from batcher.plan.functions.temporal import window

    ds = bt.from_pydict(
        {
            "t": [_dt.datetime(2024, 1, 1, 0, 0)],
            "s": ["abc"],
            "lst": [[1, 2, 3]],
            "i": [1],
            "k": [["a"]],
            "n": [[1]],
        }
    )
    cases = {
        "WindowStart": (ds.select(v=window(bt.col("t"), "5 minutes")), pa.timestamp("us")),
        "WindowBuckets": (
            ds.select(v=window(bt.col("t"), "10 minutes", "5 minutes")),
            pa.list_(pa.timestamp("us")),
        ),
        "ListGetDyn": (ds.sql("SELECT lst[i] AS v FROM self"), pa.int64()),
        "StrFuncDyn": (ds.sql("SELECT repeat(s, i) AS v FROM self"), pa.string()),
        "MakeMap": (
            ds.select(v=bt.map_from_arrays(bt.col("k"), bt.col("n"))),
            pa.map_(pa.string(), pa.int64()),
        ),
    }
    for kind, (out, expected) in cases.items():
        declared, actual = _declared_and_actual(out)
        assert declared == expected, f"{kind} declared {declared}, expected {expected}"
        assert declared == actual, f"{kind} declared {declared} but returns {actual}"

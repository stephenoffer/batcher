"""Every accessor-namespace method declares the type the engine actually returns.

`test_diff_scalar_declared_types` sweeps the IR *vocabularies* -- `MATH_FNS`, `STR_FNS`,
`LIST_FNS` -- by constructing nodes directly. This sweeps the **public surface** instead:
every callable on `Expr` and on each accessor namespace, reached the way a user reaches it.
The two find different things, because a namespace method and the node it builds are not the
same population. Four defects lived in that gap and none of them was a missing vocabulary
entry:

* `.struct.get(name)` is documented as the subscript spelling of `.struct.field(name)` and is
  what ``s["x"]`` lowers to, but the two build different nodes and only `StructField` was
  typed -- so the same field projection declared `string` written one way and nothing at all
  written the other;
* `.list.join(sep)` (`ListJoin`) had no inference arm at all;
* `.list.add`/`.subtract`/`.multiply` (`ListZip`) had none either, though the node's own
  docstring has said ``-> List<Float64>`` since it was written;
* `bt.array(...)` had none, and its element type is a promotion the module already computes
  for `coalesce`/`greatest`/`least`.

Each cost far more than its own column: `Project.available_schema` is all-or-nothing by
design (`SchemaRef.from_typed_fields` -- a partial schema missing a field would be planned
against rather than falling back), so one untyped expression makes `Dataset.schema` report
**every** column beside it as `null`.

**The oracle is the engine's own execution, not DuckDB**, and deliberately so: the question
here is not what `ST_Area` should return but whether the control plane's *static* answer
agrees with the runtime one. DuckDB cannot referee that -- it has no notion of the declared
type of an unexecuted Batcher plan -- so a non-empty `collect()` is the reference, exactly
as `test_diff_aggregate_declared_types` uses it for `AGG_FNS`. A zero-row run cannot be the
reference, because an empty result is *built from* the declaration this file is checking.
The values these functions compute are DuckDB's business and are covered by the geospatial
differential tests; the types are this file's.

The sweep is derived from `dir()` on the live objects, so a method added tomorrow is either
exercised or fails the reachability assertion. It is not allowed to pass by omission: a
method no argument shape can build is a failure, not a silent skip.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

#: One column per type the accessors need. `ll` (a list of lists) exists only so `flatten`
#: is reachable, and `ts2`/`l2` only so the two-argument methods have a partner of the right
#: type -- without them those methods go untested rather than failing.
_ROWS = pa.table(
    {
        "i": pa.array([1, 2, 3, 4], pa.int64()),
        "f": pa.array([1.5, 2.5, 3.5, 4.5], pa.float64()),
        "s": pa.array(["ab", "bc", "ab", "cd"]),
        "j": pa.array(['{"a":1,"b":"x"}'] * 4),
        "ts": pa.array([1_600_000_000_000_000] * 4, pa.timestamp("us")),
        "ts2": pa.array([1_600_000_100_000_000] * 4, pa.timestamp("us")),
        "d": pa.array([19000, 19001, 19002, 19003], pa.date32()),
        "l": pa.array([[1, 2], [3], [4, 5], [6]], pa.list_(pa.int64())),
        "l2": pa.array([[2, 3], [4], [5, 6], [7]], pa.list_(pa.int64())),
        "ls": pa.array([["a", "b"], ["c"], ["d"], ["e"]], pa.list_(pa.string())),
        "ll": pa.array([[[1, 2], [3]], [[4]], [[5]], [[6]]], pa.list_(pa.list_(pa.int64()))),
        "st": pa.array(
            [{"x": 1, "y": "p"}] * 4, pa.struct([("x", pa.int64()), ("y", pa.string())])
        ),
        "m": pa.array([[("a", 1)]] * 4, pa.map_(pa.string(), pa.int64())),
        "bo": pa.array([True, False, True, True], pa.bool_()),
    }
)

#: A 32-byte AES key as the 64 hex characters the crypto functions require.
_AES_KEY = "0" * 64

#: Literal argument tuples, tried after the column ones. Order matters and is not arbitrary:
#: a *column* argument is tried first so a numeric method binds numerically. Literals first
#: made `//` bind to the string `"a"`, which the engine coerces and DuckDB refuses -- a real
#: divergence, but one recorded in `test_diff_scalar_declared_types` rather than here, and
#: not the thing this file is trying to measure.
_LITERAL_ARGS: tuple[tuple, ...] = (
    ("a",), (1,), ("$.a",), ("a", "b"), (1, 2), ("ab",), ("x",), ("y",), (2,), (0,),
    ("upper",), ("snake",), ("gzip",), (0.5,), (True,), ("UTC",), ("%Y-%m-%d",), (_AES_KEY,),
    ("$.b",), ("day",), ("1d",), ("09:00", "17:00"), ("UTC", "UTC"), ("float64",),
    ([1.0, 3.0],), ("a", 1), (1, "a"), ("int64",),
)  # fmt: skip

#: Columns offered as an argument, before any literal.
_COLUMN_ARGS = ("i", "f", "s", "l", "l2", "ls", "ts", "ts2", "d", "bo")

#: Window sizes for the `rolling_*_by` / `ewm_*_by` family, which takes an ordering column
#: and a window rather than a bare argument.
_WINDOW_ARGS = ("2us", "2i", 2)

#: namespace -> the receiver columns to try it on. `""` is the fluent builder itself.
_NAMESPACES: dict[str, tuple[str, ...]] = {
    "dt": ("ts", "d"),
    "json": ("j",),
    "map": ("m",),
    "struct": ("st",),
    "list": ("l", "ls", "ll"),
}


def _methods(namespace: str) -> list[str]:
    holder = getattr(col("_"), namespace) if namespace else col("_")
    return sorted(n for n in dir(holder) if not n.startswith("_"))


def _attempts(namespace: str, name: str, receivers):
    """Every `(bound method, args)` worth trying, outermost-first."""
    for receiver in receivers:
        base = getattr(col(receiver), namespace) if namespace else col(receiver)
        method = getattr(base, name, None)
        if not callable(method):
            continue
        yield method, ()
        for other in _COLUMN_ARGS:
            yield method, (col(other),)
        for args in _LITERAL_ARGS:
            yield method, args
        # `(by, window)` -- the shape the `rolling_*_by` / `ewm_*_by` family takes. Tried
        # last because it is the narrowest, and only these reach it.
        for other in _COLUMN_ARGS:
            for window in _WINDOW_ARGS:
                yield method, (col(other), window)


def _sweep(namespace: str, receivers) -> tuple[list[str], list[str]]:
    """`(reached, divergences)` over every method of `namespace`."""
    dataset = bt.from_arrow(_ROWS)
    reached, bad = [], []
    for name in _methods(namespace):
        for method, args in _attempts(namespace, name, receivers):
            try:
                built = dataset.select(v=method(*args))
                declared = built.schema.field("v").type
                actual = built.collect().schema.field("v").type
            except Exception:
                continue
            reached.append(name)
            if declared != actual:
                bad.append(
                    f"{namespace or 'Expr'}.{name}{args}: declared {declared}, engine {actual}"
                )
            break
    return reached, bad


@pytest.mark.parametrize("namespace", sorted(_NAMESPACES))
def test_every_accessor_method_declares_what_the_engine_returns(namespace):
    reached, bad = _sweep(namespace, _NAMESPACES[namespace])
    assert not bad, "\n".join(bad)
    missing = sorted(set(_methods(namespace)) - set(reached))
    assert not missing, (
        f"no argument shape reaches .{namespace}.{{{', '.join(missing)}}}, so they are going "
        "untested rather than passing -- add the shape they need to `_LITERAL_ARGS`, or the "
        "column they need to `_ROWS`"
    )


# --- the fluent builder ------------------------------------------------------------------

#: The accessor *properties*. Not callables that build an expression, and each is swept by
#: its own case above (or, for the media ones, needs real files).
_EXPR_ACCESSORS = frozenset(
    {"audio", "dt", "image", "json", "list", "map", "seq", "str", "struct", "video"}
)

#: Not expressions at all: metadata about the node rather than a column derived from it.
_EXPR_NOT_EXPRESSIONS = frozenset({"name", "tag", "to_ir", "vocab"})

#: Methods that only mean anything inside a window, so `select(v=...)` cannot build them.
#: Covered by `test_the_window_only_methods_declare_what_they_return` instead, which is why
#: they are named here rather than left in the residue.
_EXPR_WINDOW_ONLY = frozenset(
    {
        "backward_fill",
        "ewm_mean",
        "ewm_std",
        "ewm_var",
        "forward_fill",
        "interpolate",
        "rle_id",
    }
)


def test_every_expr_method_declares_what_the_engine_returns():
    reached, bad = _sweep("", ("i", "f", "s", "bo", "l"))
    assert not bad, "\n".join(bad)
    residue = sorted(
        set(_methods(""))
        - set(reached)
        - _EXPR_ACCESSORS
        - _EXPR_NOT_EXPRESSIONS
        - _EXPR_WINDOW_ONLY
    )
    assert not residue, (
        f"unreachable and unclassified: {residue}. Give them an argument shape, or classify "
        "them -- an `Expr` method nothing builds is one whose declared type nobody checks"
    )


def test_the_expr_classification_names_only_real_methods():
    """A classification that outlives the method it excuses is how coverage quietly drops."""
    live = set(_methods(""))
    for label, names in (
        ("accessors", _EXPR_ACCESSORS),
        ("not-expressions", _EXPR_NOT_EXPRESSIONS),
        ("window-only", _EXPR_WINDOW_ONLY),
    ):
        stale = sorted(names - live)
        assert not stale, f"{label} still excuses {stale}, which `Expr` no longer has"


@pytest.mark.parametrize("fn", sorted(_EXPR_WINDOW_ONLY))
def test_the_window_only_methods_declare_what_they_return(fn):
    """The other half of the classification: excused from the sweep, not from the contract."""
    rows = pa.table(
        {
            "g": pa.array(["a", "a", "b", "b"]),
            "t": pa.array([1, 2, 3, 4], pa.int64()),
            "v": pa.array([1.0, None, 3.0, None], pa.float64()),
        }
    )
    dataset = bt.from_arrow(rows)
    receiver = col("g") if fn == "rle_id" else col("v")
    method = getattr(receiver, fn)
    built = method(alpha=0.5) if fn.startswith("ewm_") else method()
    windowed = dataset.select(r=built.over(partition_by="g", order_by="t"))
    assert windowed.schema.field("r").type == windowed.collect().schema.field("r").type


def test_the_sweep_is_not_vacuous():
    """Guard the guard: every assertion above is "declared == engine for what was built", so
    a fixture that stopped building anything would satisfy all of them."""
    reached, _ = _sweep("", ("i", "f", "s", "bo", "l"))
    assert len(reached) > 150, f"the fluent builder sweep reached only {len(reached)} methods"
    for namespace, receivers in _NAMESPACES.items():
        got, _ = _sweep(namespace, receivers)
        assert got, f".{namespace} reached nothing"

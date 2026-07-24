"""Integer arguments to expression builders are validated at the API edge.

An accessor method lowers its scalar arguments straight into the JSON IR, so before this
was enforced a typo produced one of two errors, neither of which named the caller's
mistake:

* ``RuntimeError: malformed plan IR: invalid type: string "3", expected i64`` — the Rust
  deserializer refusing the plan, naming the engine's wire format, at `collect()` time
  rather than at the call;
* ``TypeError: '<' not supported between instances of 'str' and 'int'`` — from a builder
  that compared the argument (``if n < 1``) without first checking it was a number.

The contract is `PlanError`, raised while building, naming the method and the parameter.
This module sweeps the whole accessor surface rather than listing methods by hand, so a
builder added later with an unvalidated `int` parameter fails here instead of shipping.
"""

from __future__ import annotations

import inspect

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit

#: A receiver column of the right type for each accessor namespace.
_RECEIVER = {"str": "s", "dt": "t", "list": "l", "json": "j", "map": "j", "struct": "j"}

#: Value substituted for the integer parameter under test.
_NOT_AN_INT = "not-an-int"


def _int_methods():
    """Every ``(namespace, method, parameter, call_kwargs)`` with a required `int` parameter.

    Discovered by introspection so the sweep cannot fall behind the surface it guards.
    """
    found = []
    for ns, recv in _RECEIVER.items():
        acc = getattr(bt.col(recv), ns)
        for name in sorted(m for m in dir(acc) if not m.startswith("_")):
            fn = getattr(acc, name)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            required = [
                p
                for p in sig.parameters.values()
                if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            annotations = {
                p.name: (p.annotation if isinstance(p.annotation, str) else "") for p in required
            }
            targets = [p.name for p in required if annotations[p.name].strip() == "int"]
            if not targets:
                continue
            for target in targets:
                kwargs, ok = {}, True
                for p in required:
                    ann = annotations[p.name]
                    if p.name == target:
                        kwargs[p.name] = _NOT_AN_INT
                    elif "int" in ann:
                        kwargs[p.name] = 1
                    elif "str" in ann:
                        kwargs[p.name] = "a"
                    else:
                        ok = False
                if ok:
                    found.append(pytest.param(ns, name, target, kwargs, id=f"{ns}.{name}-{target}"))
    return found


_CASES = _int_methods()


def test_the_sweep_actually_found_methods_to_check():
    """A collection bug that found nothing would make every test below vacuously pass."""
    assert len(_CASES) >= 20, f"only {len(_CASES)} int parameters discovered"


@pytest.mark.parametrize(("ns", "method", "param", "kwargs"), _CASES)
def test_a_non_integer_is_rejected_as_a_plan_error(ns, method, param, kwargs):
    fn = getattr(getattr(bt.col(_RECEIVER[ns]), ns), method)
    with pytest.raises(PlanError) as excinfo:
        fn(**kwargs)
    message = str(excinfo.value)
    assert param in message, f"{ns}.{method}: message does not name {param!r}: {message}"
    assert method in message, f"{ns}.{method}: message does not name the method: {message}"


@pytest.mark.parametrize(
    "build",
    [
        lambda: bt.col("s").str.left(2.5),
        lambda: bt.col("s").str.lpad("5"),
        lambda: bt.col("s").str.substr(None),
        lambda: bt.col("l").list.get(1.0),
        lambda: bt.col("f").rolling_sum("3"),
        lambda: bt.col("f").hash_bucket(2.5),
    ],
    ids=["left-float", "lpad-str", "substr-none", "list.get-float", "rolling-str", "bucket-float"],
)
def test_the_non_accessor_builders_reject_the_same_way(build):
    """`Expr`'s own integer parameters are covered too, not just the accessors."""
    with pytest.raises(PlanError):
        build()


def test_a_bool_is_rejected_because_it_would_silently_mean_one():
    """``str.repeat(True)`` is a mistake, and `bool` being an `int` subclass hides it.

    Accepting it would repeat the string once and return a plausible answer, so this is
    the one place the validator is deliberately stricter than `isinstance(x, int)`.
    """
    with pytest.raises(PlanError, match="must be an integer"):
        bt.col("s").str.repeat(True)


def test_a_numpy_integer_is_still_accepted():
    """Rejecting non-`int` by type would break NumPy scalars, which are ordinary input."""
    np = pytest.importorskip("numpy")
    ds = bt.from_pydict({"s": ["hello"], "l": [[1, 2, 3]]})
    got = ds.select(
        a=bt.col("s").str.left(np.int64(3)),
        b=bt.col("l").list.get(np.int32(-1)),
    ).to_pydict()
    assert got == {"a": ["hel"], "b": [3]}


# --- the same contract on the `Dataset` verbs ---------------------------------------
#
# These take a row count rather than a character count, and a wrong-typed one arrives the
# same way a user's does: from a CLI flag, a config file, or a JSON payload, where "10" is
# a string. Each used to compare it (`if n < 0`) and raise a bare `TypeError`.

_DATASET_CASES = [
    pytest.param("limit", ("n",), {"n": "2"}, id="limit"),
    pytest.param("limit", ("offset",), {"n": 2, "offset": "1"}, id="limit-offset"),
    pytest.param("top_k", ("k",), {"k": "2", "by": "x"}, id="top_k"),
    pytest.param("bottom_k", ("k",), {"k": 2.5, "by": "x"}, id="bottom_k"),
    pytest.param("nlargest", ("n",), {"n": "2", "columns": "x"}, id="nlargest"),
    pytest.param("nsmallest", ("n",), {"n": "2", "columns": "x"}, id="nsmallest"),
    pytest.param("slice", ("offset",), {"offset": "0"}, id="slice-offset"),
    pytest.param("slice", ("length",), {"offset": 0, "length": "1"}, id="slice-length"),
    pytest.param("gather_every", ("n",), {"n": "2"}, id="gather_every"),
    pytest.param("coalesce", ("n",), {"n": "4"}, id="coalesce"),
    pytest.param("repartition", ("num_files",), {"num_files": "4"}, id="repartition"),
    pytest.param("sample_per_group", ("n",), {"n": "1", "by": "s"}, id="sample_per_group"),
]


@pytest.mark.parametrize(("method", "named", "kwargs"), _DATASET_CASES)
def test_a_dataset_verb_rejects_a_non_integer_row_count(method, named, kwargs):
    ds = bt.from_pydict({"x": [3, 1, 2], "s": ["a", "b", "c"]})
    with pytest.raises(PlanError) as excinfo:
        getattr(ds, method)(**kwargs)
    message = str(excinfo.value)
    assert method in message, f"message does not name the method: {message}"
    for arg in named:
        assert arg in message, f"message does not name {arg!r}: {message}"


def test_the_dataset_verbs_still_do_their_job():
    """The guards must not have changed any accepted call's behavior."""
    ds = bt.from_pydict({"x": [3, 1, 2], "s": ["a", "b", "c"]})
    assert ds.limit(2).to_pydict() == {"x": [3, 1], "s": ["a", "b"]}
    assert ds.top_k(2, "x").to_pydict() == {"x": [3, 2], "s": ["a", "c"]}
    assert ds.slice(1, 2).to_pydict() == {"x": [1, 2], "s": ["b", "c"]}
    assert ds.gather_every(2).to_pydict() == {"x": [3, 2], "s": ["a", "c"]}
    assert ds.nsmallest(2, "x").to_pydict() == {"x": [1, 2], "s": ["b", "c"]}


# --- the float-typed parameters -----------------------------------------------------
#
# These already rejected an out-of-domain value (`q=2.0`, `q=nan`), because each carries a
# check like `if not 0.0 <= q <= 1.0`. What none of them could do was survive a *string*:
# `0.0 <= "abc"` raises `TypeError` before the domain check runs, so the message named
# Python's comparison rules rather than the parameter.


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: bt.col("x").quantile("a"),
        lambda ds: bt.col("x").approx_quantile("a"),
        lambda ds: ds.approx_quantile("x", "a"),
        lambda ds: ds.approx_percentile("x", "a"),
        lambda ds: ds.sample_frac("abc"),
    ],
    ids=["quantile", "approx_quantile", "ds.approx_quantile", "approx_percentile", "sample_frac"],
)
def test_a_non_number_float_parameter_is_rejected_as_a_plan_error(build):
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(PlanError, match="must be a number"):
        build(ds)


def test_a_bool_probability_is_rejected_too():
    """``sample_frac(True)`` would silently mean "keep everything"."""
    with pytest.raises(PlanError, match="must be a number"):
        bt.col("x").quantile(True)


def test_the_domain_checks_and_the_happy_path_are_unchanged():
    """The type gate runs *before* the domain check without replacing it."""
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
    assert ds.agg(q=bt.col("x").quantile(0.5)).to_pydict() == {"q": [2.5]}
    assert ds.approx_percentile("x", 50.0) == pytest.approx(2.5)
    # An `int` is a number, so it is still accepted where a float is declared.
    assert ds.agg(q=bt.col("x").quantile(1)).to_pydict() == {"q": [4.0]}
    with pytest.raises(PlanError, match=r"\[0, 1\]"):
        bt.col("x").quantile(2.0)


# --- the module-level `bt.*` functions ----------------------------------------------
#
# `long_output_rate` and `short_output_rate` were the worst of these: they *accepted* a
# string and built a comparison of a character count against it, so the mistake became a
# plan rather than an error.

_MODULE_CASES = [
    pytest.param(lambda: bt.date_add(bt.col("d"), "x"), "date_add", "days", id="date_add"),
    pytest.param(lambda: bt.date_sub(bt.col("d"), "x"), "date_sub", "days", id="date_sub"),
    pytest.param(lambda: bt.ntile("x"), "ntile", "n", id="ntile"),
    pytest.param(lambda: bt.nth_value(bt.col("x"), "x"), "nth_value", "n", id="nth_value"),
    pytest.param(lambda: bt.range("x"), "range", "start", id="range"),
    pytest.param(
        lambda: bt.width_bucket(bt.col("x"), 0, 10, "x"), "width_bucket", "count", id="width_bucket"
    ),
    pytest.param(
        lambda: bt.long_output_rate(bt.col("t"), "x"),
        "long_output_rate",
        "min_chars",
        id="long_output_rate",
    ),
    pytest.param(
        lambda: bt.short_output_rate(bt.col("t"), "x"),
        "short_output_rate",
        "max_chars",
        id="short_output_rate",
    ),
    pytest.param(
        lambda: bt.truncate_to_token_budget(bt.col("t"), "x"),
        "truncate_to_token_budget",
        "budget",
        id="truncate_to_token_budget",
    ),
]


@pytest.mark.parametrize(("build", "func", "arg"), _MODULE_CASES)
def test_a_module_level_function_rejects_a_non_integer(build, func, arg):
    with pytest.raises(PlanError) as excinfo:
        build()
    message = str(excinfo.value)
    assert func in message, f"message does not name the function: {message}"
    assert arg in message, f"message does not name {arg!r}: {message}"


def test_the_module_level_functions_still_do_their_job():
    import datetime

    ds = bt.from_pydict({"d": [datetime.date(2024, 1, 15)], "t": ["hello world"], "x": [5.0]})
    assert ds.select(r=bt.date_add(bt.col("d"), 10)).to_pydict() == {
        "r": [datetime.date(2024, 1, 25)]
    }
    assert ds.select(r=bt.date_sub(bt.col("d"), 10)).to_pydict() == {
        "r": [datetime.date(2024, 1, 5)]
    }
    assert ds.select(r=bt.width_bucket(bt.col("x"), 0, 10, 5)).to_pydict() == {"r": [3.0]}
    assert bt.range(3).to_pydict() == {"value": [0, 1, 2]}
    assert ds.agg(r=bt.long_output_rate(bt.col("t"), 3)).to_pydict() == {"r": [1.0]}

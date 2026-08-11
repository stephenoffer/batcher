"""Built-in table functions in the FROM clause: the series generators and model scoring.

Two families live here, both of which are relations a query *reads from* rather than
operators applied to one.

**The series generators.** ``FROM generate_series(1, 5)`` and ``FROM range(3)`` are how SQL
writes a spine: the integers a report joins against, or the dates a calendar table is built
from. sqlglot parses both into one typed `GenerateSeries` node, so they differ only in
whether the end is inclusive.

Neither reached a handler. `_table` looked the name up in the *registered* function
registry, found nothing, and reported ``unknown table ''`` — the empty name coming from a
node that carries no table name at all.

**Model scoring.** ``FROM ML_PREDICT(t, 'model.pkl')`` scores a fitted model over a
relation, which is the one thing a warehouse can do that a DataFrame-only engine makes you
leave SQL for. BigQuery's ``ML.PREDICT(MODEL m, TABLE t)`` is accepted as the same thing,
because a query being ported should not have to be rewritten to run. Both spellings lower to
`Dataset.ml.predict`, so the model loads once per worker and each batch is scored as one
dense matrix — there is no second inference path here, only a second way to ask for it.

These are relation *sources*, built at plan time, the same way `bt.from_pydict` builds one.
Nothing here is a hot path and none of it touches a row: the series values come from a single
vectorized `numpy.arange`, and scoring is a `map_batches` stage the engine runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.api.session import from_arrow

__all__ = [
    "is_predict_source",
    "is_series_source",
    "predict_table",
    "relation_argument",
    "series_table",
]

#: How many rows a single generated series may produce. The relation is materialized in
#: memory, so an unbounded `range(1e12)` would exhaust the host before the plan even runs.
#: A refusal names the limit; silently truncating would be a wrong answer.
_MAX_ROWS = 100_000_000


def is_series_source(node) -> bool:
    """Whether this FROM item is ``generate_series(...)`` / ``range(...)``."""
    return isinstance(node, exp.Table) and isinstance(node.this, exp.GenerateSeries)


def _bound(node, what: str) -> int:
    """A constant integer bound, or a clear rejection."""
    if isinstance(node, (exp.Literal, exp.Neg)):
        try:
            return int(node.to_py())
        except (TypeError, ValueError):
            # Not silence — a literal that is not an integer (a string, a float with a
            # fractional part) falls through to the `PlanError` below, which names the
            # constraint. Re-raising the `ValueError` here would surface sqlglot's
            # internals instead of the rule the user actually broke.
            pass
    raise PlanError(
        f"{what} of a series must be a constant integer; the relation's size has to be "
        "known when the plan is built"
    )


def series_table(node) -> Dataset:
    """Build the relation ``generate_series`` / ``range`` denotes.

    ``generate_series(a, b)`` includes `b`; ``range(a, b)`` excludes it. Both accept an
    optional step, which may be negative for a descending series. The output column is
    named after the function, which is what DuckDB names it, unless a
    ``AS t(name)`` alias renames it.

    Args:
        node: The `exp.Table` wrapping the `GenerateSeries`.

    Returns:
        A single-column `Dataset` of the generated integers.
    """
    series = node.this
    exclusive = bool(series.args.get("is_end_exclusive"))
    start = _bound(series.args.get("start"), "the start")
    end = _bound(series.args.get("end"), "the end")
    step_node = series.args.get("step")
    step = _bound(step_node, "the step") if step_node is not None else 1
    if step == 0:
        raise PlanError("a series step of 0 would never reach its end")
    stop = end if exclusive else (end + (1 if step > 0 else -1))

    count = max(0, -(-(stop - start) // step)) if step else 0
    if count > _MAX_ROWS:
        raise PlanError(
            f"a series of {count} rows exceeds the {_MAX_ROWS} row limit; it is "
            "materialized in memory, so generate it in ranges or read it from storage"
        )

    name = "range" if exclusive else "generate_series"
    alias = node.args.get("alias")
    columns = getattr(alias, "columns", None) if alias is not None else None
    if columns:
        name = columns[0].name
    values = np.arange(start, stop, step, dtype=np.int64)
    return from_arrow(pa.table({name: pa.array(values, pa.int64())}))


# --- model scoring: ML_PREDICT(t, model) / ML.PREDICT(MODEL m, TABLE t) ----------------

#: The neutral spelling's function names. `ML.PREDICT` is BigQuery grammar and only the
#: BigQuery read dialect parses it, so the capability would otherwise be reachable only by
#: switching dialects — which is the wrong thing to ask of someone whose SQL is DuckDB's.
_PREDICT_NAMES = frozenset({"ml_predict", "predict"})

#: Settings `ML_PREDICT` forwards to `Dataset.ml.predict`, and nothing else. The scoring
#: surface is wide (batch sizes, worker counts, accelerators, retry budgets); those are
#: *execution* choices that belong to the pipeline that runs the query, not to the query
#: text, and a SQL string is the wrong place to pin a GPU count. What is listed here is what
#: changes the *answer*: which columns are features, which output the model is asked for, and
#: what the result column is called.
_PREDICT_OPTIONS = frozenset({"features", "method", "output_column"})


def relation_argument(tr, node, what: str) -> Dataset:
    """The relation a table function's argument denotes.

    A table function's input can be written four ways — a bare name, a parenthesized
    ``SELECT``, a set operation, or a subquery — and every table function has to accept all
    four or a query fails on the spelling rather than on the meaning. Shared so the built-ins
    here and the registered functions in `udf` resolve one argument the same way.

    Args:
        tr: The translator, holding the table registry.
        node: The argument's AST node.
        what: How to name the function in an error message.

    Returns:
        The relation, as a `Dataset`.

    Raises:
        PlanError: When the argument names no known table, or is not a relation at all.
    """
    if isinstance(node, exp.Subquery):
        return tr.statement(node.this)
    if isinstance(node, (exp.Select, exp.Union)):
        return tr.statement(node)
    name = node.name if isinstance(node, (exp.Column, exp.Table, exp.Identifier)) else None
    if not name:
        raise PlanError(f"{what} takes a table as its relation argument, got {node.sql()!r}")
    if name not in tr._registry:
        raise PlanError(f"unknown table {name!r}; registered: {list(tr._registry)}")
    return tr._registry[name]


def is_predict_source(node) -> bool:
    """Whether this FROM item scores a model — ``ML_PREDICT(...)`` or ``ML.PREDICT(...)``."""
    if not isinstance(node, exp.Table):
        return False
    inner = node.this
    if isinstance(inner, exp.Predict):
        return True
    return isinstance(inner, exp.Anonymous) and inner.name.lower() in _PREDICT_NAMES


def predict_table(tr, node) -> Dataset:
    """Build the relation a model-scoring table function denotes.

    Accepts both spellings and lowers them to the same `Dataset.ml.predict` call:

    * ``FROM ML_PREDICT(t, 'model.pkl')`` — relation first, model second, parsed by every
      dialect;
    * ``FROM ML.PREDICT(MODEL m, TABLE t)`` — BigQuery's own grammar, so a ported query runs
      as written under ``dialect="bigquery"``.

    The model is either a string literal naming a saved model (a path or URI) or an
    identifier naming one registered with `Session.register_model`, which is how a model
    fitted in the same process is reachable from SQL at all.

    Args:
        tr: The translator, holding the table and model registries.
        node: The `exp.Table` wrapping the call.

    Returns:
        The input relation with the prediction column(s) appended.

    Raises:
        PlanError: For an unknown model or table, a bad argument count, or an option that is
            not one of ``features``, ``method`` or ``output_column``.
    """
    inner = node.this
    if isinstance(inner, exp.Predict):
        # BigQuery: `ML.PREDICT(MODEL m, TABLE t [, STRUCT(...)])`. `this` is the model and
        # `expression` the relation — the reverse of the neutral form's order, which is the
        # only difference between the two once the arguments are named.
        source = relation_argument(tr, inner.args["expression"], "ML.PREDICT")
        model = _model(tr, inner.args["this"])
        options = _struct_options(inner.args.get("params_struct"))
        return _scored(source, model, options)
    args = list(inner.expressions)
    positional = [a for a in args if not isinstance(a, exp.Kwarg)]
    if len(positional) != 2:
        raise PlanError(
            f"{inner.name}(relation, model) takes exactly two positional arguments — the "
            f"relation to score and the model — got {len(positional)}"
        )
    source = relation_argument(tr, positional[0], inner.name)
    model = _model(tr, positional[1])
    return _scored(source, model, _kwarg_options(args))


def _model(tr, node) -> Any:
    """The model a scoring call names: a registered object, or a path/URI to a saved one."""
    if isinstance(node, exp.Literal) and node.is_string:
        return node.this
    name = node.name if isinstance(node, (exp.Column, exp.Table, exp.Identifier)) else None
    if not name:
        raise PlanError(
            f"the model argument must be a registered model name or a quoted path to a saved "
            f"model, got {node.sql()!r}"
        )
    models = getattr(tr, "_models", {})
    if name not in models:
        raise PlanError(
            f"unknown model {name!r}; registered: {sorted(models)}. Register a fitted model "
            f"with Session.register_model(name, model), or pass a quoted path to a saved one"
        )
    return models[name]


def _kwarg_options(args: list) -> dict[str, Any]:
    """The ``name => value`` settings of a neutral-form call, validated."""
    return {
        _option_name(kw.this.name): _option_value(kw.expression)
        for kw in args
        if isinstance(kw, exp.Kwarg)
    }


def _struct_options(params) -> dict[str, Any]:
    """The settings of BigQuery's trailing ``STRUCT(<value> AS <name>, ...)`` argument.

    A named struct field arrives in one of two shapes depending on how sqlglot folded it —
    an `Alias` (value then name) or a `PropertyEQ` (name then value) — so both are read here.
    They mean the same thing, and rejecting the second would refuse the spelling BigQuery's
    own parser produces.
    """
    if params is None:
        return {}
    out: dict[str, Any] = {}
    for field in getattr(params, "expressions", []) or []:
        if isinstance(field, exp.PropertyEQ):
            out[_option_name(field.this.name)] = _option_value(field.expression)
        elif isinstance(field, exp.Alias):
            out[_option_name(field.alias)] = _option_value(field.this)
        else:
            raise PlanError(
                "each setting in ML.PREDICT's STRUCT must be written `<value> AS <name>`, "
                f"got {field.sql()!r}"
            )
    return out


def _option_name(name: str) -> str:
    """`name` if it is a setting scoring understands, else a rejection that lists them."""
    lowered = name.lower()
    if lowered not in _PREDICT_OPTIONS:
        raise PlanError(
            f"unknown model-scoring setting {name!r}; supported: {sorted(_PREDICT_OPTIONS)}"
        )
    return lowered


def _option_value(node) -> Any:
    """A setting's value: a string, or a list of strings for ``features``."""
    if isinstance(node, (exp.Array, exp.Tuple)):
        return [_option_value(item) for item in node.expressions]
    if isinstance(node, exp.Literal) and node.is_string:
        return node.this
    raise PlanError(
        f"a model-scoring setting must be a string, or a list of strings for `features`, "
        f"got {node.sql()!r}"
    )


def _scored(source: Dataset, model: Any, options: dict[str, Any]) -> Dataset:
    """`source` with the model's predictions appended — the one place either spelling lands.

    Two kinds of model reach here and they are scored differently, because they *are*
    different: a `batcher.ml` estimator lowers its fitted parameters to an `Expr` the engine
    evaluates in Rust, so it scores a relation directly and never sees a matrix; anything else
    — XGBoost, LightGBM, CatBoost, scikit-learn, ONNX, or a path to a saved model — is scored
    through `Dataset.ml.predict`, which loads it once per worker and hands it one dense matrix
    per batch. Routing a native estimator through the matrix path was the first thing this did
    and it fails immediately: `LinearRegression.predict` is handed a NumPy array where it
    expects a `Dataset`.
    """
    if not _is_native_estimator(model):
        return source.ml.predict(model, **options)
    if "features" in options:
        raise PlanError(
            "`features` cannot be set for a batcher.ml estimator: it was fitted with its "
            "feature columns and scores by name. Set the columns when you build the model"
        )
    method = options.get("method", "predict")
    scorer = getattr(model, method, None)
    if scorer is None:
        raise PlanError(
            f"{type(model).__name__} has no {method!r} method; it is a batcher.ml estimator, "
            f"so `method` names one of its own methods"
        )
    scored = scorer(source)
    alias = options.get("output_column")
    produced = getattr(model, "output_column", "prediction")
    return scored.rename({produced: alias}) if alias and alias != produced else scored


def _is_native_estimator(model: Any) -> bool:
    """Whether `model` is a `batcher.ml` estimator, which scores a relation rather than a matrix.

    Decided by where the class comes from, not by which methods it has: the `Estimator`
    protocol is presence-only, and a scikit-learn estimator has `fit` and `predict` too — so
    an `isinstance` check against it would send every sklearn model down the native path and
    hand it a `Dataset` where it wants an array.
    """
    return type(model).__module__.startswith("batcher.ml.")

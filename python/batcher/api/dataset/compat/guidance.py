"""The actionable-attribute-error table for ecosystem APIs Batcher does not have.

A migrant does not read the API reference first — they type what they already know
and read the traceback. So the traceback is the documentation. `Dataset.__getattr__`
routes every failed attribute lookup through `attribute_error_for`, which answers in
one of three ways, most specific first:

1. **A known-absent ecosystem API.** ``ds.set_index("k")`` is not a gap to be filled
   later; Batcher has no row index by design, and the message says so and names the
   replacement. Silence here is what makes a migration feel like guesswork.
2. **A column name.** ``ds.amount`` is the pandas attribute-access habit. Batcher
   does not add it (a column named ``filter`` would shadow a method — the exact bug
   pandas ships), so the message points at the two unambiguous spellings.
3. **A near-miss method name.** ``ds.dropna_`` gets a `difflib` suggestion.

Everything here is message-only: no branch changes what a query computes.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

__all__ = ["attribute_error_for"]


# Ecosystem attributes Batcher deliberately does not have, and what to type instead.
# Keyed by the name a pandas/Polars/Spark user types; the value is the "why + what
# instead" half of the message. Grouped by the reason they are absent.
_UNSUPPORTED: dict[str, str] = {
    # --- there is no row index -------------------------------------------------
    "index": (
        "Batcher relations have no row index (they are unordered multisets, like SQL). "
        "For a positional column use ds.with_row_index(); to order rows use ds.sort()."
    ),
    "set_index": (
        "Batcher relations have no row index. Keep the key as an ordinary column and "
        "join or group on it: ds.join(other, on='key') / ds.group_by('key')."
    ),
    "reset_index": (
        "Batcher relations have no row index, so there is nothing to reset. "
        "For a fresh 0..n-1 column use ds.with_row_index()."
    ),
    "reindex": (
        "Batcher relations have no row index. To restrict or reorder columns use "
        "ds.select(...); to order rows use ds.sort(...)."
    ),
    "loc": (
        "Batcher has no label-based indexer. Filter rows with ds.filter(bt.col('x') > 0) "
        "and pick columns with ds.select('a', 'b') or ds[['a', 'b']]."
    ),
    "iloc": (
        "Batcher has no positional indexer. Use ds.slice(offset, length), ds.head(n), "
        "or ds[0:10] for rows, and ds.select(...) for columns."
    ),
    "at": "Batcher has no scalar indexer. Use ds.filter(...).item() for a single value.",
    "iat": "Batcher has no scalar indexer. Use ds.filter(...).item() for a single value.",
    # --- transposition needs a bounded, homogeneous frame ----------------------
    "T": (
        "Transposing needs a fully materialized, single-typed frame, which a lazy "
        "(possibly unbounded) relation is not. Collect first: ds.to_pandas().T. "
        "To reshape relationally use ds.unpivot() / ds.pivot()."
    ),
    "transpose": (
        "Transposing needs a fully materialized, single-typed frame, which a lazy "
        "(possibly unbounded) relation is not. Collect first: ds.to_pandas().T. "
        "To reshape relationally use ds.unpivot() / ds.pivot()."
    ),
    # --- per-row Python is not on the hot path ---------------------------------
    "iterrows": (
        "Batcher never runs per-row Python on the hot path. To consume rows at the "
        "end of a pipeline use ds.iter_rows(named=True), which streams batches; to "
        "compute per row use an expression (bt.col('x') * 2) or ds.map_batches()."
    ),
    "itertuples": (
        "Batcher never runs per-row Python on the hot path. To consume rows at the "
        "end of a pipeline use ds.iter_rows(), which streams batches; to compute per "
        "row use an expression (bt.col('x') * 2) or ds.map_batches()."
    ),
    "applymap": (
        "Batcher has no per-cell Python callback. Express the work as an expression "
        "over the columns, e.g. ds.with_columns(x=bt.col('x') * 2)."
    ),
    "apply": (
        "pandas `apply` has per-row and per-column meanings that do not survive a "
        "columnar engine. Use an expression (ds.with_columns(y=bt.col('x') * 2)) for "
        "column work, or ds.map_batches(fn) for a whole-Arrow-batch callback."
    ),
    # --- mutation: a Dataset is immutable --------------------------------------
    "inplace": (
        "A Dataset is immutable; every operation returns a new one. Rebind instead: "
        "ds = ds.drop_nulls()."
    ),
    "insert": (
        "A Dataset is immutable. Add a column with ds.with_columns(name=expr); "
        "column order is set by ds.select(...)."
    ),
    "pop": "A Dataset is immutable. Use ds.drop('col') to get a Dataset without a column.",
    "update": (
        "A Dataset is immutable. Derive a new one with ds.with_columns(...) or "
        "join the replacement values in with ds.join(other, on=...)."
    ),
    # --- ordered/positional operations that need an explicit order -------------
    "shift": (
        "Shifting needs an explicit row order (a relation has none). Use a window: "
        "ds.window(order_by=['t'], functions={'prev': ('lag', 'x')})."
    ),
    "diff": (
        "Differencing needs an explicit row order (a relation has none). Use a window "
        "to get the previous value, then subtract: "
        "ds.window(order_by=['t'], functions={'prev': ('lag', 'x')})"
        ".with_columns(d=bt.col('x') - bt.col('prev'))."
    ),
    "cumsum": (
        "A running total needs an explicit row order (a relation has none). Use a "
        "window: ds.window(order_by=['t'], functions={'running': ('sum', 'x')})."
    ),
    "cumprod": (
        "A running product needs an explicit row order (a relation has none) and is "
        "not a supported window aggregate; compute it with ds.map_batches()."
    ),
    "cummax": (
        "A running maximum needs an explicit row order (a relation has none). Use a "
        "window: ds.window(order_by=['t'], functions={'running': ('max', 'x')})."
    ),
    "cummin": (
        "A running minimum needs an explicit row order (a relation has none). Use a "
        "window: ds.window(order_by=['t'], functions={'running': ('min', 'x')})."
    ),
    "rolling": (
        "Rolling windows are spelled with an explicit frame: "
        "ds.window(order_by=['t'], functions={'avg3': ('avg', 'x')}, frame=(-2, 0))."
    ),
    "expanding": (
        "Expanding windows are spelled with an explicit frame: "
        "ds.window(order_by=['t'], functions={'run': ('sum', 'x')}, frame=(None, 0))."
    ),
    "resample": (
        "Time bucketing is a group_by over a truncated timestamp: "
        "ds.group_by(bucket=bt.col('t').dt.truncate('1h')).agg(bt.col('x').sum())."
    ),
    # --- naming differences that deserve a pointer rather than a silent alias ---
    "hstack": (
        "Batcher has no positional column stacking (there is no row order to align "
        "on). Add columns with ds.with_columns(...), or join on a key with "
        "ds.join(other, on='key')."
    ),
    "toPandas": "Spelled ds.to_pandas() here (PEP 8 naming throughout).",
    "printSchema": "Spelled ds.schema here; ds.info() prints a readable summary.",
    "withColumn": "Spelled ds.with_columns(name=expr) here (PEP 8 naming throughout).",
    "withColumnRenamed": "Spelled ds.rename({'old': 'new'}) here (PEP 8 naming throughout).",
    "selectExpr": "Use ds.sql('SELECT ... FROM self') or ds.select(<expressions>).",
    "createOrReplaceTempView": (
        "Batcher has no global view registry. Pass the dataset straight into "
        "bt.sql('SELECT * FROM t', t=ds), or use ds.sql('SELECT * FROM self')."
    ),
    "rdd": "Batcher has no RDD layer. Use ds.iter_batches() for Arrow batches.",
    "metrics": "Spelled ds.stats() here, which returns measured per-operator RunStats.",
    "n_partitions": (
        "Partitioning is decided at execution, not carried on the plan, so a lazy "
        "Dataset has no partition count. ds.repartition(n) sets the output layout for "
        "the next write, and ds.explain(analyze=True) reports what actually ran."
    ),
    "memory_usage_deep": "Spelled ds.memory_usage() here; it is an estimate, not a measurement.",
}


def _method_names(ds: Dataset) -> list[str]:
    """The public method and property names a user could have meant, for did-you-mean."""
    return [n for n in dir(type(ds)) if not n.startswith("_")]


def attribute_error_for(ds: Dataset, name: str) -> AttributeError:
    """Build the `AttributeError` for a failed `Dataset` attribute lookup.

    Args:
        ds: The dataset the attribute was looked up on.
        name: The attribute name that was not found.

    Returns:
        An `AttributeError` whose message explains the absence and names the
        Batcher spelling to use instead.
    """
    if name in _UNSUPPORTED:
        return AttributeError(f"Dataset has no attribute {name!r}. {_UNSUPPORTED[name]}")

    # A column name: the pandas attribute-access habit. Batcher does not add it
    # because a column named e.g. "filter" would shadow a method, so point at the
    # two spellings that can never be ambiguous.
    try:
        columns = ds.columns
    except Exception:  # pragma: no cover - a malformed plan must not mask the real error
        columns = []
    if name in columns:
        return AttributeError(
            f"Dataset has no attribute {name!r}, but it is a column. Batcher does not "
            "expose columns as attributes (a column could shadow a method). Use "
            f"ds[{name!r}] for the expression, or bt.col({name!r}) to build one."
        )

    candidates = _method_names(ds)
    close = difflib.get_close_matches(name, [*candidates, *columns], n=3, cutoff=0.7)
    msg = f"Dataset has no attribute {name!r}."
    if close:
        msg += f" Did you mean {' or '.join(repr(c) for c in close)}?"
    return AttributeError(msg)

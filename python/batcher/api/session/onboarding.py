"""Top-level `bt.<name>` migration guidance: the traceback as the documentation.

`Dataset`, `Expr`, and `GroupBy` already answer a migrant who types a method they know.
This module does the same one level up, for the names a pandas, Polars, or PySpark user
reaches for on the package itself — ``bt.DataFrame``, ``bt.SparkSession``, ``bt.scan_csv``,
``bt.read_sql``. The root package's ``__getattr__`` (PEP 562) routes a failed top-level
lookup here, so ``bt.LazyFrame`` comes back naming ``bt.read`` and the lazy `Dataset`
model instead of a bare ``module 'batcher' has no attribute`` error.

Keyed by the exact top-level name a migrant types; every replacement is a real
`batcher` function or type. Rendering is the shared `batcher._internal.errors.absent_error`.
"""

from __future__ import annotations

from collections.abc import Iterable

from batcher._internal.errors import absent_error

__all__ = ["TOP_LEVEL_UNSUPPORTED", "top_level_attribute_error"]


TOP_LEVEL_UNSUPPORTED: dict[str, str] = {
    # --- frame / series types ---------------------------------------------------------
    "DataFrame": (
        "Batcher's frame type is Dataset, and it is lazy. Build one with bt.from_pydict"
        "({...}), bt.from_pandas(df), or bt.read.*; nothing runs until a terminal op."
    ),
    "LazyFrame": (
        "A Batcher Dataset is already lazy — there is no eager/lazy split. Build one with "
        "bt.from_pydict({...}) or bt.read.* (every reader is lazy)."
    ),
    "Series": (
        "There is no separate Series type. A single column is an expression, bt.col('x'); "
        "a one-column result is a Dataset you collect with ds.to_pydict()['x']."
    ),
    "Index": "Batcher relations have no row index (they are unordered multisets, like SQL).",
    "Column": "A column reference is bt.col('x'); build derived columns with expressions.",
    "Categorical": (
        "There is no categorical constructor; store the values as a string column, and "
        "bt.col('x').label_encode() when you need integer codes."
    ),
    "Row": "Rows are plain dicts. Build data with bt.from_pylist([{'a': 1}, {'a': 2}]).",
    # --- sessions / contexts: Batcher runs in-process ---------------------------------
    "SparkSession": (
        "There is no session to start — Batcher runs in-process. Read with bt.read.* and "
        "query with bt.sql('SELECT ...', t=frame); scale out with collect(distributed=True)."
    ),
    "SparkContext": "There is no context to start. Read with bt.read.* and query with bt.sql(...).",
    "SQLContext": (
        "Query with bt.sql('SELECT ...', t=frame), or open a bt.Session() for many queries."
    ),
    "HiveContext": "Query with bt.sql('SELECT ...', t=frame), or open a bt.Session().",
    "createDataFrame": (
        "Build a Dataset with bt.from_pydict({...}), bt.from_pandas(df), or bt.from_arrow(t)."
    ),
    # --- functions / window namespaces ------------------------------------------------
    "functions": (
        "Scalar and aggregate functions are top-level: bt.col, bt.when, bt.sum, bt.lit, ..."
    ),
    "F": "Spark's F.* functions are top-level here: bt.col, bt.when, bt.sum, bt.avg, ...",
    "Window": (
        "Window functions are ds.window(partition_by=[...], order_by=[...], functions={...}), "
        "or expr.over(partition_by=[...]) on an aggregate."
    ),
    "types": "Types are dtype strings you pass to .cast('int64'); see bt.col('x').cast(...).",
    # --- lazy readers: every bt.read.* is already lazy --------------------------------
    "scan_csv": "Every reader is already lazy: bt.read.csv(path) (no scan_/read_ split).",
    "scan_parquet": "Every reader is already lazy: bt.read.parquet(path) (no scan_/read_ split).",
    "scan_ndjson": "Every reader is already lazy: bt.read.json(path) (no scan_/read_ split).",
    "scan_ipc": "Every reader is already lazy: bt.read.arrow(path) (no scan_/read_ split).",
    "scan_delta": "Every reader is already lazy: bt.read.delta(path).",
    # --- SQL / database readers -------------------------------------------------------
    "read_sql": (
        "Read a query with bt.read_database(query, uri=...) or bt.read.sql(query, uri=...)."
    ),
    "read_sql_query": "Read a query with bt.read_database(query, uri=...).",
    "read_sql_table": (
        "Read a table with bt.read_database('SELECT * FROM t', uri=...) or bt.read.sql(...)."
    ),
    # --- foreign-format readers with no native path -----------------------------------
    "read_feather": "Arrow/Feather is bt.read_ipc(path) or bt.read.arrow(path).",
    "read_html": (
        "No native HTML reader; load with pandas then bt.from_pandas(pd.read_html(url)[0])."
    ),
    "read_pickle": (
        "No native pickle reader; load with pandas then bt.from_pandas(pd.read_pickle(p))."
    ),
    "read_fwf": "No fixed-width reader; load with pandas then bt.from_pandas(pd.read_fwf(p)).",
    "read_stata": "No Stata reader; load with pandas then bt.from_pandas(pd.read_stata(p)).",
    "read_hdf": "No HDF5 reader; load with pandas then bt.from_pandas(pd.read_hdf(p)).",
    "read_spss": "No SPSS reader; load with pandas then bt.from_pandas(pd.read_spss(p)).",
    "read_clipboard": (
        "No clipboard reader; load with pandas then bt.from_pandas(pd.read_clipboard())."
    ),
    # --- value / dtype constructors ---------------------------------------------------
    "NA": (
        "Use None in Python data; a null literal is bt.lit(None), tested with "
        "bt.col('x').is_null()."
    ),
    "NaT": "Use None for a missing timestamp; a null literal is bt.lit(None).",
    "null": "A null literal is bt.lit(None); test a column with bt.col('x').is_null().",
    "to_datetime": (
        "Parse a string column with bt.col('t').str.to_datetime(...), or cast with "
        "bt.col('t').cast('timestamp[us]')."
    ),
    "to_numeric": "Cast a column with bt.col('x').cast('float64') / .cast('int64').",
    # --- reshaping / helpers that are Dataset methods or differently named ------------
    "pivot_table": (
        "Pivoting is a Dataset method: ds.pivot(index=[...], on=..., values=..., aggregate='sum')."
    ),
    "get_dummies": "One-hot encoding is a Dataset method: ds.get_dummies('col').",
    "melt": "Reshaping wide-to-long is a Dataset method: ds.unpivot(index=[...], on=[...]).",
    "json_normalize": (
        "Read nested JSON with bt.read.json(path); flatten with ds.unnest('col') or the "
        ".struct / .json accessors."
    ),
    "arange": "An integer range is bt.range(start, end) (like builtins.range).",
    "int_range": "An integer range is bt.range(start, end).",
    "datetime_range": "A timestamp range is bt.date_range(start, end=..., interval='1d').",
    "qcut": (
        "Quantile-bin with breaks from bt.col('x').quantile(...), then bt.col('x').cut(breaks=...)."
    ),
    "set_option": (
        "Configure with bt.set_config(...) / bt.config_context(...) and the bt.Config dataclasses."
    ),
    "options": (
        "Configuration lives on bt.Config; read or change it with bt.active_config() / "
        "bt.set_config(...)."
    ),
}


def top_level_attribute_error(name: str, members: Iterable[str]) -> AttributeError:
    """Build the `AttributeError` for a failed top-level ``bt.<name>`` lookup.

    Args:
        name: The top-level attribute name that was not found.
        members: The real public top-level names, for the did-you-mean fallback.

    Returns:
        An `AttributeError` that names the Batcher spelling to use instead.
    """
    return absent_error("batcher", name, TOP_LEVEL_UNSUPPORTED, members)

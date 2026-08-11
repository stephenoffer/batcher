"""The Dataset half of the migration-error table: what a migrant types, and why it is absent.

Keyed by the exact spelling a pandas, Polars, or PySpark user reaches for; the value is
the "why Batcher does not have it, and what to type instead" half of the message that
`Dataset.__getattr__` prints. Every replacement named here is a real public `Dataset`
method, `Expr` method, or top-level `batcher` function, so a migrant can copy it straight
out of the traceback. Grouped by the reason the API is absent, because the reason is what
a migrant actually needs to internalize.
"""

from __future__ import annotations

from batcher.api.dataset.compat.guidance._dataset_naming import (
    DATASET_EXPORTERS,
    DATASET_NAMING,
    DATASET_RAY_DATA,
)

__all__ = ["DATASET_UNSUPPORTED"]


# --- there is no row index (a relation is an unordered multiset, as in SQL) ----------
_NO_INDEX: dict[str, str] = {
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
    "reindex_like": "Batcher relations have no row index. Align columns with ds.select(...).",
    "set_axis": (
        "Batcher relations have no row index. Rename columns with ds.rename({...}); "
        "row order comes from ds.sort(...)."
    ),
    "rename_axis": "Batcher relations have no row/axis index to name.",
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
    "xs": "Batcher has no cross-section indexer. Use ds.filter(...) and ds.select(...).",
    "lookup": "Batcher has no positional lookup. Select the value with ds.filter(...).item().",
    "droplevel": (
        "Batcher has no MultiIndex. Columns are flat; use ds.select(...) / ds.rename(...)."
    ),
    "swaplevel": "Batcher has no MultiIndex. Columns are flat; reorder with ds.select(...).",
    "reorder_levels": "Batcher has no MultiIndex. Reorder columns with ds.select(...).",
    "nlevels": "Batcher has no MultiIndex; columns are flat. Count them with len(ds.columns).",
    "first_valid_index": (
        "Batcher relations have no row index. Filter for the first match with "
        "ds.filter(...).head(1)."
    ),
    "last_valid_index": "Batcher relations have no row index. Filter, sort, and take the last row.",
    "keys": "Iterate columns via ds.columns and index a column with ds[name].",
    "items": "Iterate columns via ds.columns and index a column with ds[name].",
    "iteritems": "Iterate columns via ds.columns and index a column with ds[name].",
    "get": "Select a column with ds['col'] (an expression) or ds.select('col') (a Dataset).",
    "bool": "Use ds.is_empty() to test for rows, or ds.count() for the row count.",
}

# --- transposition needs a bounded, homogeneous, materialized frame ------------------
_NO_TRANSPOSE: dict[str, str] = {
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
    "stack": "Reshaping wide-to-long is ds.unpivot(index=[...], on=[...]).",
    "unstack": "Reshaping long-to-wide is ds.pivot(index=[...], on=..., values=...).",
    "swapaxes": "A relation has no axes to swap. Reshape with ds.unpivot() / ds.pivot().",
    "to_xarray": "No xarray bridge. Collect first: ds.to_pandas().to_xarray().",
}

# --- per-row Python never runs on the hot path ---------------------------------------
_NO_PER_ROW: dict[str, str] = {
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
    "map_rows": "Batcher runs callbacks over Arrow batches, not rows: ds.map_batches(fn).",
    "foreach": "Batcher has no per-row action. Use ds.map_batches(fn) or ds.write.for_each(...).",
    "foreachPartition": "Spelled ds.map_batches(fn) here; each call gets one Arrow batch.",
    "mapInPandas": "Spelled ds.map_batches(fn) here; the callback gets Arrow batches (fn(batch)).",
    "mapInArrow": "Spelled ds.map_batches(fn) here; the callback already gets Arrow batches.",
}

# --- a Dataset is immutable; every operation returns a new one ------------------------
_IMMUTABLE: dict[str, str] = {
    "inplace": (
        "A Dataset is immutable; every operation returns a new one. Rebind instead: "
        "ds = ds.drop_nulls()."
    ),
    "insert": (
        "A Dataset is immutable. Add a column with ds.with_columns(name=expr); "
        "column order is set by ds.select(...)."
    ),
    "insert_column": "A Dataset is immutable. Add a column with ds.with_columns(name=expr).",
    "replace_column": "A Dataset is immutable. Replace a column with ds.with_columns(name=expr).",
    "pop": "A Dataset is immutable. Use ds.drop('col') to get a Dataset without a column.",
    "drop_in_place": (
        "A Dataset is immutable. Use ds.drop('col') to get a Dataset without a column."
    ),
    "update": (
        "A Dataset is immutable. Derive a new one with ds.with_columns(...) or "
        "join the replacement values in with ds.join(other, on=...)."
    ),
    "extend": "A Dataset is immutable. Stack rows with ds.union(other) (a new Dataset).",
    "clear": "A Dataset is immutable. For an empty, same-schema Dataset use ds.limit(0).",
    "clone": "A Dataset is already immutable; ds.copy() (an identity) is here if you want it.",
    "insert_at_idx": "A Dataset is immutable. Add a column with ds.with_columns(name=expr).",
}

# --- ordered / windowed operations need an explicit row order (a relation has none) ---
_NEEDS_ORDER: dict[str, str] = {
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
    "pct_change": (
        "A percent change needs an explicit row order (a relation has none). Use a "
        "window's lag, or bt.col('x').pct_change() inside a window's order_by."
    ),
    "cumsum": (
        "A running total needs an explicit row order (a relation has none). Use a "
        "window: ds.window(order_by=['t'], functions={'running': ('sum', 'x')})."
    ),
    "cumprod": (
        "A running product needs an explicit row order (a relation has none). Use a window: "
        "ds.window(order_by=['t'], functions={'running': ('product', 'x')})."
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
        "A row-count window is an explicit frame: "
        "ds.window(order_by=['t'], functions={'avg3': ('avg', 'x')}, frame=(-2, 0)). "
        "For a *time* window use bt.col('x').rolling_mean_by('t', '5m')."
    ),
    "expanding": (
        "Expanding windows are spelled with an explicit frame: "
        "ds.window(order_by=['t'], functions={'run': ('sum', 'x')}, frame=(None, 0))."
    ),
    "ewm": (
        "Exponentially weighted stats are expressions: bt.col('x').ewm_mean(span=n) bound with "
        ".over(order_by=[...]), or .ewm_mean_by('t', half_life) to decay by elapsed time."
    ),
    "resample": (
        "Downsampling is a group_by over a bucketed timestamp: "
        "ds.group_by(bucket=bt.window(bt.col('t'), '1h')).agg(s=bt.col('x').sum()). "
        "bt.window takes a duration; .dt.truncate takes a calendar unit ('hour', 'month')."
    ),
    "asfreq": (
        "Bucketing to a fixed frequency is "
        "ds.group_by(bucket=bt.window(bt.col('t'), '1d')).agg(...). To emit the empty "
        "buckets too, left-join onto a bt.date_range grid -- see the time-series user guide."
    ),
    "upsample": (
        "Build the grid and left-join onto it: bt.date_range(lo, hi, interval='1h') gives the "
        "rows, ds.join(..., how='left') attaches what you have, and "
        "bt.col('x').interpolate() / .forward_fill() fills the gaps. See the time-series "
        "user guide."
    ),
    "group_by_dynamic": (
        "Time-window grouping is ds.group_by(bucket=bt.window(bt.col('t'), '1h')).agg(...); "
        "pass a slide to bt.window for overlapping windows, or use "
        "ds.session_window(time_col, gap) for gap-based sessions."
    ),
    "interpolate": (
        "Interpolation is an expression: bt.col('x').interpolate() bound with "
        ".over(order_by=[...]); .forward_fill() / .backward_fill() hold a value flat instead."
    ),
    "ffill": (
        "Forward fill is bt.col('x').forward_fill(); combine with a window's order_by for "
        "ordered fill."
    ),
    "pad": (
        "Forward fill is bt.col('x').forward_fill(); combine with a window's order_by for "
        "ordered fill."
    ),
    "bfill": (
        "Backward fill is bt.col('x').backward_fill(); combine with a window's order_by for "
        "ordered fill."
    ),
    "backfill": (
        "Backward fill is bt.col('x').backward_fill(); combine with a window's order_by for "
        "ordered fill."
    ),
    "asof": "An as-of join is ds.join_asof(other, on='t', by=['key']).",
    "at_time": (
        "Filter on the clock time: ds.filter(bt.col('t').dt.is_between_time('09:00', '09:00'))."
    ),
    "between_time": (
        "Filter on the clock time: ds.filter(bt.col('t').dt.is_between_time('09:00', '17:00')); "
        "it wraps past midnight, which an hour comparison does not."
    ),
    "tz_convert": "Convert a timezone with bt.col('t').dt.convert_timezone('UTC').",
    "tz_localize": "Attach a timezone with bt.col('t').dt.convert_timezone('UTC').",
    "truncate": (
        "Trim rows by a boundary column with ds.filter(...), or by position with "
        "ds.slice(offset, length)."
    ),
    "idxmax": (
        "There is no row index. For the row itself use ds.sort('x', descending=True).head(1); "
        "for the argmax within a group use bt.col('x').arg_max()."
    ),
    "idxmin": (
        "There is no row index. For the row itself use ds.sort('x').head(1); "
        "for the argmin within a group use bt.col('x').arg_min()."
    ),
}

# --- reductions that exist per-expression, reached through .agg(...) at frame level ---
_AGG_REDUCTIONS: dict[str, str] = {
    "prod": "Spelled ds.product('x') here (or bt.col('x').product() inside ds.agg(...)).",
    "skew": "Spelled ds.skewness('x') here (or bt.col('x').skewness() inside ds.agg(...)).",
    "kurt": "Spelled ds.kurtosis('x') here (or bt.col('x').kurtosis() inside ds.agg(...)).",
    "sem": "Standard error of the mean is bt.sem(bt.col('x')) inside ds.agg(...).",
    "corrwith": "Pairwise correlation is ds.corr('a', 'b'); the full matrix is ds.corr_matrix().",
    "dot": "A matrix product is not a relational op. Use ds.to_numpy() then NumPy.",
    "nunique_approx": (
        "Approximate distinct count is bt.col('x').approx_n_unique(), or ds.approx_n_unique."
    ),
}

# --- selection / predicate idioms that map onto expressions --------------------------
_PREDICATES: dict[str, str] = {
    "where": (
        "A conditional value is bt.when(cond).then(a).otherwise(b); to keep rows use "
        "ds.filter(cond)."
    ),
    "mask": "The inverse of where: bt.when(~cond).then(a).otherwise(b), or ds.filter(~cond).",
    "isin": "Membership is ds.filter(bt.col('x').is_in(['a', 'b'])).",
    "replace": "Value replacement is bt.col('x').replace({old: new}) inside ds.with_columns(...).",
    "duplicated": (
        "Flag repeats with bt.col('key').is_duplicated() inside ds.with_columns(...) or "
        "ds.filter(...)."
    ),
    "factorize": "Dense integer codes are bt.col('x').label_encode().",
    "cut": "Binning is bt.col('x').cut(breaks=[...]) inside ds.with_columns(...).",
    "qcut": (
        "Quantile binning: derive breaks from bt.col('x').quantile(...), then "
        "bt.col('x').cut(breaks=...)."
    ),
    "combine_first": (
        "Fill a column's nulls from another via a join, then coalesce: "
        "ds.join(other, on='key').with_columns(x=bt.coalesce(bt.col('x'), bt.col('x_right')))."
    ),
    "combine": (
        "Combine two frames by joining on a key, then compute with expressions over both columns."
    ),
    "align": (
        "Align two frames by joining on the shared key: ds.join(other, on='key', how='outer')."
    ),
    "compare": (
        "Compare results with ds.equals(other); for a row-level diff, join and compare columns."
    ),
    "fill_nan": (
        "Replace NaN (distinct from null) with bt.col('x').fill_nan(0) inside ds.with_columns(...)."
    ),
    "drop_nans": (
        "ds.drop_nulls() drops nulls, not NaN. Drop NaN with ds.filter(bt.col('x').is_not_nan())."
    ),
    "is_duplicated": (
        "Flag duplicate rows with bt.col('key').is_duplicated() in ds.with_columns(...)."
    ),
    "is_unique": "Flag unique rows with bt.col('key').is_unique() in ds.with_columns(...).",
}

# --- reshaping and horizontal helpers that live at top level or on ds ------------------
_RESHAPE: dict[str, str] = {
    "pivot_table": "Spelled ds.pivot(index=[...], on=..., values=..., aggregate='sum') here.",
    "wide_to_long": "Reshape wide-to-long with ds.unpivot(index=[...], on=[...]).",
    "merge_ordered": (
        "Join on a key with ds.join(other, on='key'); for time alignment use ds.join_asof(...)."
    ),
    "to_dummies": "One-hot encoding is ds.get_dummies('col').",
    "get_column": "Select a column with ds['x'] (an expression) or ds.select('x') (a Dataset).",
    "get_columns": "Select columns with ds.select('a', 'b') or ds[['a', 'b']].",
    "iter_columns": "Iterate over ds.columns and index each with ds[name].",
    "to_series": (
        "For a single column use ds.select('x').to_arrow().column(0), or the expression ds['x']."
    ),
    "row": "Read rows at the end of a pipeline with ds.iter_rows(named=True) or ds.to_pylist().",
    "rows": "Materialize rows with ds.to_pylist(); stream them with ds.iter_rows().",
    "fold": (
        "Reduce across columns with bt.fold_horizontal(fn, [bt.col('a'), bt.col('b')]) in a select."
    ),
    "max_horizontal": (
        "Row-wise max across columns is bt.max_horizontal('a', 'b') in ds.select(...)."
    ),
    "min_horizontal": (
        "Row-wise min across columns is bt.min_horizontal('a', 'b') in ds.select(...)."
    ),
    "sum_horizontal": (
        "Row-wise sum across columns is bt.sum_horizontal('a', 'b') in ds.select(...)."
    ),
    "mean_horizontal": (
        "Row-wise mean across columns is bt.mean_horizontal('a', 'b') in ds.select(...)."
    ),
    "hash_rows": "A per-row hash column is bt.hash_rows(...) in ds.with_columns(...).",
    "partition_by": (
        "For output layout use ds.write.parquet(partition_by=[...]); to process per group "
        "use ds.group_by(...).agg(...) or a window."
    ),
    "explode_multiple": "Explode a list column with ds.explode('col').",
}

# --- storage / chunking / execution knobs Batcher manages for you --------------------
_MANAGED: dict[str, str] = {
    "rechunk": "Arrow chunk layout is managed internally; there is nothing to rechunk.",
    "shrink_to_fit": "Memory layout is managed internally; there is nothing to shrink.",
    "n_chunks": "Arrow chunk layout is managed internally and not exposed on a lazy plan.",
    "set_sorted": "The optimizer detects and exploits sortedness; there is no manual sorted flag.",
    "estimated_size": "An in-memory size estimate is ds.memory_usage().",
    "to_init_repr": (
        "No round-trippable repr; repr(ds) shows the schema, ds.glimpse() shows a preview."
    ),
    "infer_objects": (
        "Columns are already Arrow-typed; nothing to infer. Change types with ds.cast({...})."
    ),
    "convert_dtypes": "Columns are already Arrow-typed. Change types with ds.cast({...}).",
    "checkpoint": "Materialize and reuse a result with ds.cache().",
    "localCheckpoint": "Materialize and reuse a result with ds.cache().",
    "unpersist": "Caching is scoped to the plan; there is no manual unpersist.",
    "storageLevel": "Batcher manages spill and caching; there is no storage level to set.",
    "hint": (
        "The optimizer (Kyber) chooses join strategy and build side; inspect it with ds.explain()."
    ),
    "sameSemantics": "Compare two results with ds.equals(other).",
    "semanticHash": "Compare two results with ds.equals(other).",
    "sink_parquet": "Every write already streams: ds.write.parquet(path).",
    "sink_csv": "Every write already streams: ds.write.csv(path).",
    "sink_ipc": "Every write already streams: ds.write.arrow(path).",
    "sink_ndjson": "Every write already streams: ds.write.json(path).",
}


def _merge(*tables: dict[str, str]) -> dict[str, str]:
    """Flatten the reason-grouped tables into the one lookup __getattr__ uses."""
    out: dict[str, str] = {}
    for table in tables:
        out.update(table)
    return out


#: The full pandas/Polars/Spark/Ray Data → Batcher redirect table for `Dataset.__getattr__`.
DATASET_UNSUPPORTED: dict[str, str] = _merge(
    _NO_INDEX,
    _NO_TRANSPOSE,
    _NO_PER_ROW,
    _IMMUTABLE,
    _NEEDS_ORDER,
    _AGG_REDUCTIONS,
    _PREDICATES,
    _RESHAPE,
    _MANAGED,
    DATASET_NAMING,
    DATASET_RAY_DATA,
    DATASET_EXPORTERS,
)

"""The GroupBy half of the migration-error table.

A pandas `GroupBy` and a Spark `GroupedData` carry a much wider surface than Batcher's,
because pandas lets you loop, transform, and apply arbitrary Python per group. Batcher's
`GroupBy` is deliberately narrow: it aggregates. Keyed by the method a migrant types; the
value says why it is absent and which relational spelling replaces it. Every replacement
is a real `GroupBy` method, `Dataset` method, or window call.
"""

from __future__ import annotations

__all__ = ["GROUPBY_UNSUPPORTED"]


GROUPBY_UNSUPPORTED: dict[str, str] = {
    # --- per-group Python: caps the job at one machine, so Batcher does not offer it ---
    "apply": (
        "A GroupBy has no per-group Python callback (it would materialize one frame per "
        "key and cap the job at one machine). Aggregate with .agg(...), keep every row "
        "with a window (ds.window(partition_by=[...], functions={...})), or drop to "
        "ds.map_batches() after a shuffle."
    ),
    "transform": (
        "A broadcast-back-to-rows transform is a window, not a group_by: "
        "ds.window(partition_by=['k'], functions={'gmean': ('avg', 'x')}) adds the group "
        "statistic to every row."
    ),
    "filter": (
        "Filtering whole groups by an aggregate: compute the aggregate as a window and "
        "filter on it, e.g. ds.window(partition_by=['k'], functions={'n': ('count', 'x')})"
        ".filter(bt.col('n') > 2)."
    ),
    "pipe": "Chain off the Dataset instead: ds.group_by('k').agg(...).pipe(fn).",
    # --- per-group ordered / positional operations ------------------------------------
    "cumcount": (
        "A within-group running index is a window's row_number: "
        "ds.window(partition_by=['k'], order_by=['t'], functions={'i': ('row_number',)})."
    ),
    "ngroup": (
        "A dense group id is bt.col('k').label_encode() in ds.with_columns(...); "
        "for a per-group counter use a window's row_number."
    ),
    "rank": (
        "Per-group rank is a window: ds.window(partition_by=['k'], order_by=['x'], "
        "functions={'r': ('rank',)})."
    ),
    "shift": (
        "Per-group shift is a window's lag: ds.window(partition_by=['k'], order_by=['t'], "
        "functions={'prev': ('lag', 'x')})."
    ),
    "diff": (
        "Per-group diff is a window's lag then a subtraction: "
        "ds.window(partition_by=['k'], order_by=['t'], functions={'prev': ('lag', 'x')})."
    ),
    "pct_change": (
        "Per-group percent change is a window: partition_by the key, order_by the time, "
        "and use lag or bt.col('x').pct_change()."
    ),
    "cumsum": (
        "A per-group running total is a window: ds.window(partition_by=['k'], "
        "order_by=['t'], functions={'run': ('sum', 'x')})."
    ),
    "cummax": (
        "A per-group running max is a window: ds.window(partition_by=['k'], "
        "order_by=['t'], functions={'run': ('max', 'x')})."
    ),
    "cummin": (
        "A per-group running min is a window: ds.window(partition_by=['k'], "
        "order_by=['t'], functions={'run': ('min', 'x')})."
    ),
    "rolling": (
        "A per-group rolling window: ds.window(partition_by=['k'], order_by=['t'], "
        "functions={'avg3': ('avg', 'x')}, frame=(-2, 0))."
    ),
    "expanding": (
        "A per-group expanding window: ds.window(partition_by=['k'], order_by=['t'], "
        "functions={'run': ('sum', 'x')}, frame=(None, 0))."
    ),
    "fillna": (
        "Fill per group by broadcasting a window aggregate, or fill globally with "
        "ds.fill_null(...)."
    ),
    "ffill": "Forward fill within a group is a window over the ordered key; see ds.window(...).",
    "bfill": "Backward fill within a group is a window over the ordered key; see ds.window(...).",
    # --- materializing a single group -------------------------------------------------
    "get_group": (
        "There is no per-group frame to fetch. Filter for the group instead: "
        "ds.filter(bt.col('k') == value)."
    ),
    "groups": (
        "There is no group-to-rows mapping. Filter for a key with ds.filter(bt.col('k') == value)."
    ),
    "indices": "There is no group-to-index mapping (a relation has no row index).",
    "ngroups": "For the number of groups use ds.select('k').distinct().count().",
    "nth": (
        "The nth row per group is a window: ds.window(partition_by=['k'], order_by=['t'], "
        "functions={'i': ('row_number',)}).filter(bt.col('i') == n)."
    ),
    "describe": (
        "Aggregate the stats you need explicitly: .agg(mean=bt.col('x').mean(), "
        "std=bt.col('x').std(), ...)."
    ),
    "value_counts": "Counts per group: add the column to the keys — ds.group_by('k', 'x').len().",
    "cov": "Aggregate covariance with .agg(c=bt.covar_samp(bt.col('a'), bt.col('b'))).",
    "corr": "Aggregate correlation with .agg(r=bt.corr(bt.col('a'), bt.col('b'))).",
    "sem": "Aggregate standard error with .agg(s=bt.sem(bt.col('x'))).",
    "skew": "Aggregate skewness with .agg(s=bt.col('x').skew()).",
    "all": "Aggregate a boolean per group with .agg(ok=bt.col('flag').all()).",
    "any": "Aggregate a boolean per group with .agg(hit=bt.col('flag').any()).",
    "count_distinct": "Distinct count per group is .n_unique() or .agg(n=bt.col('x').n_unique()).",
    "aggregate": "Spelled .agg(...) here.",
    # --- Spark GroupedData naming -----------------------------------------------------
    "pivot": (
        "Batcher's pivot is a Dataset method, not a grouped one: "
        "ds.pivot(index=['k'], on='col', values='v', aggregate='sum')."
    ),
    "cube": "There is no CUBE; union several ds.group_by(...).agg(...) at different key sets.",
    "rollup": "There is no ROLLUP; union several ds.group_by(...).agg(...) at nested key sets.",
    "applyInPandas": "Spelled ds.map_batches(fn) after grouping, or .agg(...) for reductions.",
    "cogroup": "Co-grouping two frames is a join on the key: ds.join(other, on='k').",
}

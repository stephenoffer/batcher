"""The GroupBy half of the migration-error table.

A pandas `GroupBy` and a Spark `GroupedData` carry a much wider surface than Batcher's,
because pandas lets you loop, transform, and apply arbitrary Python per group. Batcher's
`GroupBy` is deliberately narrow: it aggregates. Keyed by the method a migrant types; the
value says why it is absent and which relational spelling replaces it. Every replacement
is a real `GroupBy` method, `Dataset` method, or window call.
"""

from __future__ import annotations

__all__ = ["GROUPBY_UNSUPPORTED"]


#: Where every spelling of "run my Python function over each group" now points.
#:
#: The advice this replaces was "drop to ds.map_batches() after a shuffle", and it was
#: wrong in a way that returns wrong answers rather than raising: `map_batches` sees
#: arbitrary batches, and a group is not confined to one of them. Measured over 50k rows
#: and 20 keys, **every** key's rows spanned more than one `map_batches` call, so a
#: per-group callback written that way silently ran once per fragment of each group.
#: `repartition(by=)` does not fix it either — it lays out output files and leaves the
#: batching untouched.
_PER_GROUP_PYTHON = (
    "Spelled ds.group_by('k').map_groups(fn) here: fn receives one whole group as a "
    "pyarrow RecordBatch (pass batch_format='pandas' for a frame). Do NOT call "
    "map_batches straight after grouping — a group spans several batches, so the callback "
    "would silently see fragments. For a plain reduction use .agg(...), and to broadcast a "
    "group statistic back onto every row use ds.window(partition_by=['k'], functions={...})."
)


GROUPBY_UNSUPPORTED: dict[str, str] = {
    # --- per-group Python: caps the job at one machine, so Batcher does not offer it ---
    "apply": _PER_GROUP_PYTHON,
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
    "applyInPandas": _PER_GROUP_PYTHON,
    "cogroup": "Co-grouping two frames is a join on the key: ds.join(other, on='k').",
}

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
    "all": "Aggregate a boolean per group with .agg(ok=bt.col('flag').bool_and()).",
    "any": "Aggregate a boolean per group with .agg(hit=bt.col('flag').bool_or()).",
    "aggregate": "Spelled .agg(...) here.",
    # --- Spark GroupedData naming -----------------------------------------------------
    "pivot": (
        "Batcher's pivot is a Dataset method, not a grouped one: "
        "ds.pivot(index=['k'], on='col', values='v', aggregate='sum')."
    ),
    "applyInPandas": _PER_GROUP_PYTHON,
    "cogroup": "Co-grouping two frames is a join on the key: ds.join(other, on='k').",
    # --- pandas GroupBy names with no Batcher equivalent on the grouped object ---------
    #
    # Batcher's GroupBy aggregates and nothing else, so these all answer by pointing at the
    # expression or the Dataset that does the work. Each was a bare AttributeError with a
    # fuzzy "did you mean" that pointed somewhere wrong: `idxmax` suggested `max`, which
    # returns the value rather than the row it came from.
    "idxmax": (
        "There is no row index to return. For the row itself, order and take the first "
        "per group: ds.group_by('g').first('a', order_by='a'). For the value alone, "
        ".agg(m=bt.col('a').max())."
    ),
    "idxmin": (
        "There is no row index to return. For the row itself: "
        "ds.group_by('g').last('a', order_by='a'). For the value alone, "
        ".agg(m=bt.col('a').min())."
    ),
    "cumprod": (
        "A running product is a window, not an aggregate, because it returns one row per "
        "input row: ds.with_columns(r=bt.col('a').cum_prod().over("
        "partition_by=['g'], order_by='a'))."
    ),
    "ewm": (
        "Exponentially weighted statistics are window expressions: "
        "ds.with_columns(r=bt.col('a').ewm_mean(alpha=0.5).over("
        "partition_by=['g'], order_by='t'))."
    ),
    "corrwith": (
        "Correlating two columns per group is an aggregate over both: "
        "ds.group_by('g').agg(r=bt.corr(bt.col('a'), bt.col('b')))."
    ),
    "resample": (
        "Resampling is a second grouping key, not a grouped method. Truncate the timestamp "
        "and name it: ds.group_by('g', hour=bt.col('t').dt.truncate('hour')).agg("
        "n=bt.col('a').mean())."
    ),
    "sample": (
        "Sampling is a Dataset operation: ds.sample(fraction=0.1). To sample within each "
        "group, rank inside the group and filter: ds.with_columns(r=bt.col('a').rank()"
        ".over(partition_by=['g'])).filter(bt.col('r') <= 5)."
    ),
    "take": (
        "There are no row positions to take. Order within the group and limit: "
        "ds.group_by('g').first('a', order_by='a')."
    ),
    "ohlc": (
        "Open/high/low/close is four aggregates over an ordered column: "
        "ds.group_by('g').agg(o=bt.col('a').first(order_by='t'), h=bt.col('a').max(), "
        "lo=bt.col('a').min(), c=bt.col('a').last(order_by='t'))."
    ),
    "dtypes": "Column types belong to the frame, not the grouping: read ds.schema.",
    "ndim": "A grouped relation is always two-dimensional; there is nothing to ask.",
    "level": (
        "Batcher has no index levels, so there is no level to group by. Group by the "
        "column itself: ds.group_by('g')."
    ),
    "grouper": (
        "Batcher exposes no grouper object. The keys you grouped by are the `keys` "
        "property, ds.group_by('g').keys, and the grouping is only realized by .agg(...)."
    ),
    "plot": "Batcher does not plot. Collect the aggregate and hand it to your plotting library.",
    "boxplot": "Batcher does not plot. Collect the aggregate and hand it to your plotting library.",
    "hist": (
        "Batcher does not plot. For the counts a histogram would draw, bucket and "
        "aggregate: ds.group_by('g', bucket=bt.col('a').floor()).agg(n=bt.count())."
    ),
}

"""The migration-error table for expression idioms Batcher does not have on `Expr`.

`compat` binds the pandas/Polars *spellings that do* map onto an `Expr` (the aliases in
`names`/`operators`). This module is the other half: when a migrant reaches for a Series
or `Expr` method Batcher deliberately does not carry — a per-element Python UDF, a
row-order-dependent op with no window, a Dataset-level reshape — `Expr.__getattr__` raises
an error that names the Batcher spelling instead of a bare `AttributeError`.

Keyed by the exact name a pandas/Polars user types; the value is the "why it is absent,
what to type instead" half of the message. Every replacement is a real `Expr` method, a
typed-accessor method (``.list``/``.str``), a `Dataset` method, or a top-level `batcher`
function. Rendering is the shared `batcher._internal.errors.absent_error`, so an `Expr`
typo and a `Dataset` typo read identically.
"""

from __future__ import annotations

import re

from batcher._internal.errors import absent_error, public_members

__all__ = ["EXPR_UNSUPPORTED", "expr_attribute_error"]


EXPR_UNSUPPORTED: dict[str, str] = {
    # --- per-element Python: never runs on the hot path -------------------------------
    "map_elements": (
        "Batcher has no per-element Python callback. Express the work with operators "
        "(bt.col('x') * 2, bt.when(...).then(...)), or for a genuine Python function run "
        "it over Arrow batches with ds.map_batches(fn) or a registered bt.udf."
    ),
    "map_batches": (
        "On an expression this is not supported. Run a batch callback at the Dataset "
        "level: ds.map_batches(fn), where fn receives a whole Arrow batch."
    ),
    "map_dict": "Remap values with the .map accessor: bt.col('x').map.get({old: new}).",
    "replace_strict": "Strict value remap is the .map accessor: bt.col('x').map.get({old: new}).",
    "apply": (
        "Batcher has no per-element apply. Use operators (bt.col('x') * 2) or "
        "ds.map_batches(fn) for a whole-Arrow-batch Python callback."
    ),
    "transform": (
        "Express the transform with operators, e.g. bt.col('x') * 2 in ds.with_columns(...)."
    ),
    # --- Dataset-level operations wrongly reached through an expression ----------------
    "filter": (
        "Filter rows at the Dataset level: ds.filter(bt.col('x') > 0). Inside a list use "
        ".list.filter(...)."
    ),
    "sort": "Sort rows at the Dataset level: ds.sort('x'). Inside a list use .list.sort().",
    "sort_by": "Sort rows at the Dataset level: ds.sort('x', descending=True).",
    "rint": "Round half to even is bt.col('x').round(mode='half_to_even').",
    "explode": "Explode a list column at the Dataset level: ds.explode('x').",
    "value_counts": (
        "Value counts is a Dataset op: ds.value_counts('x'), or ds.group_by('x').len()."
    ),
    "unique": (
        "Distinct values are ds.select('x').distinct(); count them with "
        "bt.col('x').count_distinct()."
    ),
    "unique_counts": "Per-value counts are ds.value_counts('x').",
    "drop_nulls": (
        "Drop nulls at the Dataset level: ds.drop_nulls('x'), or filter bt.col('x').is_not_null()."
    ),
    "drop_nans": (
        "Drop the rows holding NaN with ds.drop_nans('x'); .fill_nan(v) replaces them in place."
    ),
    "gather": (
        "Positional gather is not an expression op. Use ds.limit(n, offset=o) / "
        "ds.gather_every(...)."
    ),
    "take": "Positional take is not an expression op. Use ds.limit(n, offset=o).",
    "head": "On a list use .list.head(n); at the Dataset level use ds.limit(n).",
    "tail": "On a list use .list.slice(-n, n); at the Dataset level use ds.tail(n).",
    "reverse": "On a list use .list.reverse(); at the Dataset level use ds.reverse().",
    "flatten": "Flatten a list column with .list.flatten(), or explode it with ds.explode('x').",
    "slice": (
        "Ambiguous on an expression: on a list use .list.slice(off, len); on a string "
        ".str.slice(off, len)."
    ),
    # --- conditional / null idioms ----------------------------------------------------
    "where": "A conditional value is bt.when(cond).then(bt.col('x')).otherwise(other).",
    "mask": "The inverse of where: bt.when(cond).then(other).otherwise(bt.col('x')).",
    "if_else": "Spelled bt.when(cond).then(a).otherwise(b) here.",
    "case_when": "Chain conditions: bt.when(c1).then(a).when(c2).then(b).otherwise(c).",
    "coalesce": "Spelled bt.coalesce(bt.col('a'), bt.col('b')) (a top-level function) here.",
    "combine_first": "Fill nulls from another column with bt.coalesce(bt.col('a'), bt.col('b')).",
    # --- argmax / positional stats ----------------------------------------------------
    "argmax": (
        "Spelled bt.col('x').arg_max(order_by=...) here (the value at another column's max "
        "is max_by)."
    ),
    "argmin": (
        "Spelled bt.col('x').arg_min(order_by=...) here (the value at another column's min "
        "is min_by)."
    ),
    "idxmax": "The argmax index is bt.col('x').arg_max(order_by=...).",
    "idxmin": "The argmin index is bt.col('x').arg_min(order_by=...).",
    # --- clipping / casting naming ----------------------------------------------------
    "clip_lower": "Spelled bt.col('x').clip(lower=lo) here.",
    "clip_upper": "Spelled bt.col('x').clip(upper=hi) here.",
    "to_physical": (
        "Reinterpret the storage type with bt.col('x').cast('int64') (or the target dtype)."
    ),
    "reinterpret": "Change type with bt.col('x').cast('int64') / .try_cast(...).",
    # --- row-order-dependent stats need a window --------------------------------------
    "search_sorted": (
        "Binary search over a column is not an expression op; use a join or ds.map_batches()."
    ),
    "rle": (
        "Run-length encoding as a struct is not built in; number the runs with "
        "bt.col('x').rle_id().over(order_by=...) and group by the result."
    ),
    "dot": "The dot product of two vector columns is bt.col('a').list.dot(bt.col('b')).",
    "reshape": "Reshape a flat column into lists with the .list accessor or a group_by array_agg.",
    "extend_constant": (
        "Pad a list with the .list accessor; a scalar is broadcast automatically in operators."
    ),
    # --- name namespace (Polars .name.* / .keep_name) ---------------------------------
    "keep_name": (
        "Rename with bt.col('x').alias('name'); by default a projection keeps the source name."
    ),
    "prefix": (
        "Add a name prefix with bt.col('x').alias('pre_' + 'x'); rename many with "
        "ds.rename(lambda c: ...)."
    ),
    "suffix": (
        "Add a name suffix with bt.col('x').alias('x_suf'); rename many with ds.rename(lambda "
        "c: ...)."
    ),
    "map_alias": "Rename with bt.col('x').alias(...); rename many columns with ds.rename(fn).",
    # --- accessor namespaces renamed or absent ----------------------------------------
    "arr": (
        "The list/array accessor is spelled .list here: bt.col('x').list.sum() (Polars renamed "
        ".arr)."
    ),
    "bin": (
        "There is no binary accessor; use the .str accessor for text or .list for array columns."
    ),
    "cat": (
        "There is no categorical accessor; store values as a string column and use "
        "bt.col('x').label_encode() for integer codes."
    ),
}


# --- pandas Series methods -------------------------------------------------------------
#
# Every name below comes from `pandas.Series`; Batcher's `Expr` already covers Polars' own
# `Expr` surface. A pandas migrant is a supported path (`python -m batcher.migrate`), and a
# bare `AttributeError` there is worse than it looks: `Expr.__getattr__` falls back to a
# fuzzy "did you mean", which for these produced actively wrong suggestions -- `corr` ->
# `zscore`, `dtype` -> `dt`. Naming the real answer is the point.

#: pandas gives each operator a method alias plus a reflected form. Arithmetic on an `Expr`
#: is Python's own operators, reflected order included, so one sentence answers the family.
_ARITHMETIC = {
    name: (
        f"Batcher spells this with the operator: bt.col('a') {symbol} bt.col('b'), or "
        f"bt.col('a') {symbol} 2 against a literal. The reflected order works too "
        f"(2 {symbol} bt.col('a')), so there is no r-prefixed spelling to learn."
    )
    for name, symbol in (
        ("multiply", "*"),
        ("rmul", "*"),
        ("subtract", "-"),
        ("rsub", "-"),
        ("radd", "+"),
        ("rdiv", "/"),
        ("rtruediv", "/"),
        ("rfloordiv", "//"),
        ("rpow", "**"),
        ("rdivmod", "//"),
    )
}

#: An expression describes a column; it holds no data and has no dtype until a plan gives
#: it one. These all ask the *value* questions, which a Dataset answers.
_NOT_DATA = {
    name: (
        f"An expression describes a computation, not a column of data, so it has no "
        f"`{name}`. Build a Dataset and ask it: ds.with_columns(r=bt.col('a') * 2) then "
        f"ds.schema, ds.count() or ds.to_pydict()."
    )
    for name in (
        "array",
        "attrs",
        "axes",
        "dtype",
        "dtypes",
        "empty",
        "flags",
        "hasnans",
        "info",
        "items",
        "keys",
        "memory_usage",
        "nbytes",
        "ndim",
        "shape",
        "size",
        "sparse",
        "values",
        "view",
        "ravel",
        "equals",
        "copy",
        "pop",
        "update",
        "compare",
        "convert_dtypes",
        "infer_objects",
        "describe",
        "plot",
        "aggregate",
        "combine",
    )
}

#: Exporting is a terminal on a Dataset. An expression has nothing to export.
_EXPORTERS = {
    name: (
        f"`{name}` is a Dataset terminal, not an expression method. Put the expression in "
        f"a plan first: ds.with_columns(r=bt.col('a') * 2), then use ds.write.* to write "
        f"or ds.to_pydict() / ds.to_arrow() to materialize."
    )
    for name in (
        "to_clipboard",
        "to_csv",
        "to_dict",
        "to_excel",
        "to_hdf",
        "to_json",
        "to_markdown",
        "to_numpy",
        "to_pickle",
        "to_sql",
        "to_string",
        "to_xarray",
    )
}

#: A relation is an unordered multiset with no row index, exactly as in SQL.
_NO_INDEX = {
    name: (
        f"Batcher relations have no row index, so there is no `{name}`. Keep the key as an "
        f"ordinary column and select, filter or join on it; for a positional column use "
        f"ds.with_row_index(), and to order rows use ds.sort(...)."
    )
    for name in (
        "index",
        "iloc",
        "loc",
        "iat",
        "xs",
        "reset_index",
        "sort_index",
        "reindex",
        "reindex_like",
        "droplevel",
        "swaplevel",
        "reorder_levels",
        "rename_axis",
        "set_axis",
        "set_flags",
        "swapaxes",
        "transpose",
        "squeeze",
        "unstack",
        "drop",
        "add_prefix",
        "add_suffix",
        "first_valid_index",
        "last_valid_index",
    )
}

_SERIES_ONLY: dict[str, str] = {
    **_ARITHMETIC,
    **_NOT_DATA,
    **_EXPORTERS,
    **_NO_INDEX,
    # --- statistics that are frame-level, because they read two columns at once --------
    "corr": (
        "Correlation reads two columns, so it is a Dataset terminal or an aggregate: "
        "ds.corr('a', 'b') for the scalar, or ds.agg(r=bt.corr(bt.col('a'), bt.col('b')))."
    ),
    "cov": (
        "Covariance reads two columns: ds.cov('a', 'b') for the scalar, or inside an "
        "aggregate over a group with ds.group_by('g').agg(...)."
    ),
    "autocorr": (
        "Autocorrelation is correlation against a lagged copy, and the lag needs an "
        "explicit order: ds.with_columns(prev=bt.col('a').shift(1).over(order_by='t')) "
        "then ds.corr('a', 'prev')."
    ),
    # --- null handling, under Batcher's names -----------------------------------------
    "dropna": (
        "Dropping nulls is a row operation: ds.drop_nulls('a'), or filter explicitly with "
        "ds.filter(bt.col('a').is_not_null())."
    ),
    "ffill": (
        "Forward fill needs an explicit row order, because a relation has none: "
        "ds.with_columns(filled=bt.col('a').forward_fill().over(order_by='t'))."
    ),
    "pad": (
        "pandas' `pad` is forward fill, and it needs a row order: "
        "ds.with_columns(filled=bt.col('a').forward_fill().over(order_by='t'))."
    ),
    "bfill": (
        "Backward fill needs an explicit row order: "
        "ds.with_columns(filled=bt.col('a').backward_fill().over(order_by='t'))."
    ),
    # --- ordering and position ---------------------------------------------------------
    "argsort": (
        "Batcher has no positional argsort, because a relation has no row positions to "
        "return. To order rows use ds.sort('a'); for a rank within the values use "
        "bt.col('a').rank(); for the extreme row itself use "
        "ds.sort('a', descending=True).limit(1)."
    ),
    "searchsorted": (
        "There are no row positions to search for. Express the comparison instead: "
        "bt.col('a') >= value, or count with ds.filter(bt.col('a') < value).count()."
    ),
    "nlargest": (
        "The n largest values are bt.col('a').top_k(n), or ds.sort('a', descending=True).limit(n)."
    ),
    "nsmallest": "The n smallest values are ds.sort('a').limit(n).",
    "is_monotonic_decreasing": (
        "Monotonicity is a property of an ordered frame, not of an expression. Sort and "
        "compare against the shifted column: ds.sort('t').with_columns("
        "down=bt.col('a') <= bt.col('a').shift(1).over(order_by='t'))."
    ),
    "factorize": (
        "Assigning each distinct value an integer code is an encoder: "
        "batcher.ml.preprocessors.OrdinalEncoder('a').fit_transform(ds)."
    ),
    # --- grouping and reshaping belong to the Dataset ----------------------------------
    "groupby": (
        "Grouping is a Dataset operation: ds.group_by('g').agg(n=bt.col('a').sum()). An "
        "expression can carry a group with bt.col('a').sum().over(partition_by=['g'])."
    ),
    # --- temporal: a calendar unit, a timezone, or a window ----------------------------
    "tz_localize": (
        "Attaching a timezone is a cast, and the parametrized name uses parentheses: "
        "bt.col('t').cast('timestamp(us, UTC)'). To move an already-aware timestamp "
        "between zones use bt.col('t').dt.convert_timezone('UTC', 'Europe/Paris')."
    ),
    "tz_convert": (
        "Converting between timezones names both ends: "
        "bt.col('t').dt.convert_timezone('UTC', 'Europe/Paris')."
    ),
    "to_period": (
        "Period arithmetic is truncation to a calendar unit: "
        "bt.col('t').dt.truncate('month'), then bt.col('t').dt.strftime(...) to label it."
    ),
    "to_timestamp": (
        "Converting to a timestamp is a cast, bt.col('t').cast('timestamp(us)') -- the "
        "parametrized dtype names use parentheses -- or bt.col('t').dt.timestamp() from "
        "an epoch value."
    ),
    "at_time": (
        "Selecting a time of day is a predicate on the extracted part: "
        "ds.filter(bt.col('t').dt.hour() == 9)."
    ),
    "between_time": (
        "A time-of-day range is bt.col('t').dt.is_between_time('09:00', '17:00'), used as "
        "a filter predicate."
    ),
    "asfreq": (
        "There is no frequency index to conform to. Bucket by a calendar unit and "
        "aggregate, naming the derived key: "
        "ds.group_by(hour=bt.col('t').dt.truncate('hour')).agg(n=bt.col('a').mean())."
    ),
    "resample": (
        "Resampling is a truncate plus a group_by, with the derived key named: "
        "ds.group_by(hour=bt.col('t').dt.truncate('hour')).agg(n=bt.col('a').mean())."
    ),
    "asof": (
        "An as-of lookup is a join: ds.join_asof(other, on='t'), which matches each left "
        "row to the most recent right row at or before it."
    ),
}

EXPR_UNSUPPORTED.update(_SERIES_ONLY)


def expr_attribute_error(expr: object, name: str) -> AttributeError:
    """Build the `AttributeError` for a failed `Expr` attribute lookup.

    Args:
        expr: The expression the attribute was looked up on.
        name: The attribute name that was not found.

    Returns:
        An `AttributeError` that explains the absence and names the Batcher spelling,
        accessor, or Dataset method to use instead.
    """
    return absent_error("Expr", name, EXPR_UNSUPPORTED, public_members(type(expr)), receiver="Expr")


# --- typed-accessor migration tables -------------------------------------------------
# pandas/Polars string methods a migrant types on ``.str`` that Batcher spells
# differently or does not carry. Every replacement is a real ``.str`` / ``.list`` method.
STR_UNSUPPORTED: dict[str, str] = {
    "cat": (
        "Concatenate columns with bt.concat_str(...) / bt.concat_ws(sep, ...); a list column "
        "joins with .list.join(sep)."
    ),
    "center": "Pad both sides with .str.lpad(width) then .str.rpad(width).",
    "count": "Count occurrences with .str.count_matches(pattern) or .str.count_char(ch).",
    "decode": "Byte decoding is not exposed; string columns are already UTF-8 text.",
    "encode": "Byte encoding is not exposed; string columns are already UTF-8 text.",
    "extractall": "Spelled .str.extract_all(pattern) here.",
    "find": "The index of a substring is .str.position(sub).",
    "findall": "All matches are .str.extract_all(pattern).",
    "fullmatch": (
        ".str.match(pattern) anchors the start only, as pandas .str.match does; "
        "anchor the end yourself with .str.regexp_matches('^(?:pattern)$')."
    ),
    "get": "The i-th character is .str.slice(i, 1).",
    "index": "The index of a substring is .str.position(sub).",
    "isdecimal": "Spelled .str.is_numeric() here.",
    "islower": "Spelled .str.is_lower() here.",
    "isnumeric": "Spelled .str.is_numeric() here.",
    "istitle": "There is no is_title; compare against .str.to_titlecase().",
    "isupper": "Spelled .str.is_upper() here.",
    "normalize": (
        "Unicode normalization is not exposed; .str.normalize_whitespace() collapses whitespace."
    ),
    "pad": "Left/right pad with .str.lpad(width) / .str.rpad(width).",
    "partition": (
        "Split on the first delimiter with .str.split_part(sep, 1) (and part 2 for the tail)."
    ),
    "rfind": "The index of a substring is .str.position(sub).",
    "rindex": "The index of a substring is .str.position(sub).",
    "rpartition": "Split on a delimiter with .str.split_part(sep, n).",
    "rsplit": "Split with .str.split(sep); pick a piece with .str.split_part(sep, n).",
    "slice_replace": "Replace a slice with .str.overlay(replacement, start, length).",
    "swapcase": "There is no swapcase; combine .str.upper() and .str.lower() as needed.",
    "wrap": "Line wrapping is a display concern; use ds.map_batches() if you need it.",
    "get_dummies": "One-hot from a delimited column: split it, then ds.get_dummies(...).",
    "casefold": "Spelled .str.lower() here.",
}

# Polars/Daft list (``.arr``/``.list``) methods a migrant types that Batcher spells
# differently or does not carry. Every replacement is a real ``.list`` or `Dataset` method.
LIST_UNSUPPORTED: dict[str, str] = {
    "eval": "Apply an expression to each element with .list.transform(...).",
    "to_struct": "There is no list-to-struct; index elements with .list.get(i) into named columns.",
    "to_array": "List columns are already array-typed; there is no separate array conversion.",
    "explode": "Explode a list into rows at the Dataset level: ds.explode('x').",
    # No `gather` entry: `.list.gather(indices)` is a real method now, and an entry here would
    # shadow it — the redirect fires on a missing attribute, so listing a method that exists
    # is dead weight that reads as "we do not carry this". `take` is Polars' older spelling of
    # the same operation, so it redirects to `gather` rather than to the scalar accessors,
    # which index one position instead of taking a column of them.
    "take": "Positional take is .list.gather(indices), taking a column of positions per row.",
    "sort_desc": "Sort descending with .list.sort(descending=True); the nulls stay last.",
    "count_matches": (
        "Count occurrences by filtering and .list.len(), or test with .list.contains(x)."
    ),
    "set_symmetric_difference": (
        "Symmetric difference is .list.difference(b) both ways, then .list.union(...)."
    ),
    "shift": "Shifting elements within a list is not built in; explode, window, and re-aggregate.",
    "sample": "Sampling within a list is not built in; explode then ds.sample(...).",
    "tail": "The last n elements are .list.slice(-n, n); the last one is .list.last().",
    "all": "Reduce a boolean list with .list.min() (all true == min 1); or explode and aggregate.",
    "any": "Reduce a boolean list with .list.max() (any true == max 1); or explode and aggregate.",
}

# pandas/Polars datetime methods a migrant types on ``.dt`` that Batcher spells differently.
DT_UNSUPPORTED: dict[str, str] = {
    "isocalendar": "ISO parts are separate: .dt.iso_year() and .dt.week().",
    "time": "Extract time parts with .dt.hour(), .dt.minute(), .dt.second().",
    "timetz": (
        "Extract time parts with .dt.hour()/.dt.minute()/.dt.second(); shift zones with "
        ".dt.convert_timezone(...)."
    ),
    "to_period": (
        "Bucket to a period with .dt.truncate('month') -- a calendar unit name, not a "
        "duration -- then group on it. For a fixed-width bucket use bt.window(col, '1h')."
    ),
    "total_seconds": (
        "Seconds between two timestamps is bt.col('a').dt.seconds_between(bt.col('b'))."
    ),
    "tz_convert": "Spelled .dt.convert_timezone('UTC') here.",
    "tz_localize": "Attach or change a timezone with .dt.convert_timezone('UTC').",
}


def accessor_attribute_error(
    accessor: object, label: str, name: str, table: dict[str, str]
) -> AttributeError:
    """Build the `AttributeError` for a failed typed-accessor lookup (``.str``/``.dt``).

    Args:
        accessor: The accessor instance the attribute was looked up on.
        label: The accessor label for the message, e.g. ``"'.str' accessor"``.
        name: The attribute name that was not found.
        table: The known-absent methods for this accessor.

    Returns:
        An `AttributeError` naming the Batcher accessor method to use instead, or a
        `Did you mean ...?` against the accessor's real methods for a near miss.
    """
    namespace = re.search(r"'\.(\w+)' accessor", label)
    receiver = f"Expr.{namespace.group(1)}" if namespace else None
    return absent_error(label, name, table, public_members(type(accessor)), receiver=receiver)

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

from batcher._internal.errors import absent_error

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
    "explode": "Explode a list column at the Dataset level: ds.explode('x').",
    "value_counts": (
        "Value counts is a Dataset op: ds.value_counts('x'), or ds.group_by('x').len()."
    ),
    "unique": (
        "Distinct values are ds.select('x').distinct(); count them with bt.col('x').n_unique()."
    ),
    "unique_counts": "Per-value counts are ds.value_counts('x').",
    "drop_nulls": (
        "Drop nulls at the Dataset level: ds.drop_nulls('x'), or filter bt.col('x').is_not_null()."
    ),
    "drop_nans": (
        "Drop NaN with ds.filter(bt.col('x').is_not_nan()); .fill_nan(v) replaces them in place."
    ),
    "gather": (
        "Positional gather is not an expression op. Use ds.slice(...) / ds.gather_every(...)."
    ),
    "take": "Positional take is not an expression op. Use ds.slice(...) / ds.head(n).",
    "head": "On a list use .list.head(n); at the Dataset level use ds.head(n).",
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
    "argmax": "Spelled bt.col('x').arg_max() here.",
    "argmin": "Spelled bt.col('x').arg_min() here.",
    "idxmax": "The argmax index is bt.col('x').arg_max().",
    "idxmin": "The argmin index is bt.col('x').arg_min().",
    "peak_max": "Local maxima are not built in; compare against lag/lead with a window.",
    "peak_min": "Local minima are not built in; compare against lag/lead with a window.",
    # --- clipping / casting naming ----------------------------------------------------
    "clip_lower": "Spelled bt.col('x').clip_min(lo) here.",
    "clip_upper": "Spelled bt.col('x').clip_max(hi) here.",
    "to_physical": (
        "Reinterpret the storage type with bt.col('x').cast('int64') (or the target dtype)."
    ),
    "reinterpret": "Change type with bt.col('x').cast('int64') / .try_cast(...).",
    # --- row-order-dependent stats need a window --------------------------------------
    "ewm_mean": (
        "Exponentially weighted mean is not built in; use bt.col('x').rolling_mean(...) in a "
        "window."
    ),
    "ewm_std": (
        "Exponentially weighted std is not built in; use bt.col('x').rolling_std(...) in a window."
    ),
    "ewm_var": (
        "Exponentially weighted var is not built in; use bt.col('x').rolling_var(...) in a window."
    ),
    "interpolate": (
        "Interpolation is not built in; use bt.col('x').forward_fill() / .backward_fill()."
    ),
    "search_sorted": (
        "Binary search over a column is not an expression op; use a join or ds.map_batches()."
    ),
    "rle": "Run-length encoding is not built in; detect run boundaries with a lag window.",
    "rle_id": "Run ids are not built in; derive them from a lag window and a running sum.",
    "entropy": "Entropy is not a built-in reducer; compute it from value counts (ds.value_counts).",
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


def expr_attribute_error(expr: object, name: str) -> AttributeError:
    """Build the `AttributeError` for a failed `Expr` attribute lookup.

    Args:
        expr: The expression the attribute was looked up on.
        name: The attribute name that was not found.

    Returns:
        An `AttributeError` that explains the absence and names the Batcher spelling,
        accessor, or Dataset method to use instead.
    """
    members = [n for n in dir(type(expr)) if not n.startswith("_")]
    return absent_error("Expr", name, EXPR_UNSUPPORTED, members)


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
    "findall": "All matches are .str.extract_all(pattern) or .str.regexp_extract_all(pattern).",
    "fullmatch": ".str.match(pattern) anchors a full-string match.",
    "get": "The i-th character is .str.slice(i, 1).",
    "index": "The index of a substring is .str.position(sub).",
    "isdecimal": "Spelled .str.isdigit() / .str.is_numeric() here.",
    "islower": "Spelled .str.is_lower() here.",
    "isnumeric": "Spelled .str.is_numeric() here.",
    "istitle": "There is no is_title; compare against .str.to_titlecase().",
    "isupper": "Spelled .str.is_upper() here.",
    "join": (
        "Join a list column's items with .list.join(sep); concatenate columns with "
        "bt.concat_str(...)."
    ),
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
    "gather": "Index elements with .list.get(i) or .list.element_at(i).",
    "take": "Index elements with .list.get(i) or .list.element_at(i).",
    "count_matches": (
        "Count occurrences by filtering and .list.len(), or test with .list.contains(x)."
    ),
    "set_symmetric_difference": (
        "Symmetric difference is .list.set_difference(a, b) both ways, then .list.union(...)."
    ),
    "shift": "Shifting elements within a list is not built in; explode, window, and re-aggregate.",
    "concat": "Concatenate list columns with .list.union(...); for strings use bt.concat_str(...).",
    "sample": "Sampling within a list is not built in; explode then ds.sample(...).",
    "tail": "The last n elements are .list.slice(-n, n); the last one is .list.last().",
    "drop_nulls": "Drop nulls inside a list by exploding first: ds.explode('x').drop_nulls('x').",
    "all": "Reduce a boolean list with .list.min() (all true == min 1); or explode and aggregate.",
    "any": "Reduce a boolean list with .list.max() (any true == max 1); or explode and aggregate.",
}

# pandas/Polars datetime methods a migrant types on ``.dt`` that Batcher spells differently.
DT_UNSUPPORTED: dict[str, str] = {
    "ceil": "Rounding up is not built in; .dt.truncate(unit) / .dt.floor(unit) round down.",
    "isocalendar": "ISO parts are separate: .dt.iso_year() and .dt.week().",
    "round": "Rounding to a unit is not built in; .dt.truncate(unit) / .dt.floor(unit) truncate.",
    "time": "Extract time parts with .dt.hour(), .dt.minute(), .dt.second().",
    "timetz": (
        "Extract time parts with .dt.hour()/.dt.minute()/.dt.second(); shift zones with "
        ".dt.convert_timezone(...)."
    ),
    "to_period": "Bucket to a period with .dt.truncate('1mo'), then group on it.",
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
    members = [n for n in dir(type(accessor)) if not n.startswith("_")]
    return absent_error(label, name, table, members)

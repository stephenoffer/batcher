"""The input type each aggregate, window function, and temporal expression accepts.

`plan.logical.aggregate` already says what type an aggregate *produces*; this says what it
can be *given*. The two belong together, and only one of them existed: an aggregate whose
input type the engine cannot handle was discovered by the engine, at execution, as a bare
``RuntimeError: aggregate sum is not supported for column type Utf8``. On an interactive
query that is a poor error. On a PB-scale job it is hours of scan thrown away to learn
something the schema knew before the first byte was read -- and `Dataset.schema` reported a
type for the column all along (``sum`` of a string declared ``string``; ``mean`` of one
declared ``double``), so the plan claimed an output the query could never produce.

Worse than the late errors are the pairs the engine *accepts* and answers meaninglessly.
``PRODUCT`` over a ``timestamp`` multiplies raw epoch microseconds into a 1e30 double;
``PRODUCT``/``COVAR``/``KAHAN_SUM`` over a string return **all nulls** with no diagnostic at
all. DuckDB rejects every one of those, and a silent null is exactly the failure mode the
engine's contract calls out as the dangerous one: it passes every gate while being wrong.

So the domains below are the *semantic* ones -- what the aggregate means -- rather than a
transcription of what the accumulators happen to tolerate. They are narrower than the
engine's tolerance in precisely the places where the engine's tolerance produces nonsense,
and never narrower anywhere else: `tests/unit/test_aggregate_input_domains.py` runs every
(aggregate, type) pair through the real engine and fails if this module rejects something
the engine answers correctly.

Two divergences from DuckDB are deliberate and stated rather than hidden. DuckDB computes
``AVG``/``QUANTILE`` over a temporal column; Batcher's accumulators do not, so those are
rejected here with the cast that makes them work. DuckDB computes ``SUM`` over a boolean;
Batcher's do not.

Neutral layer: imports only `pyarrow`.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["aggregate_domain_error", "key_domain_error", "window_domain_error"]


def _is_numeric(dt: pa.DataType) -> bool:
    """Integer, floating, or decimal -- a value arithmetic is *meaningful* on.

    `pa.types.is_numeric` is not this: it counts booleans, and the whole point of the
    numeric domain is that ``PRODUCT`` over a flag column is a category error rather than a
    product of zeros and ones.
    """
    return pa.types.is_integer(dt) or pa.types.is_floating(dt) or pa.types.is_decimal(dt)


def _is_real(dt: pa.DataType) -> bool:
    """Integer or floating -- the domain of the accumulators that work in `f64`."""
    return pa.types.is_integer(dt) or pa.types.is_floating(dt)


def _is_orderable(dt: pa.DataType) -> bool:
    """A type the engine's comparison kernels order: anything but a nested one."""
    return not (
        pa.types.is_list(dt)
        or pa.types.is_large_list(dt)
        or pa.types.is_fixed_size_list(dt)
        or pa.types.is_struct(dt)
        or pa.types.is_map(dt)
    )


#: Aggregates whose input must be numeric, and the hint each one's failure carries.
#:
#: `sum` and `mean` accumulate in decimal when given one, so they take the full numeric
#: domain. The rest work in `f64` and reject a decimal input, so they take `_is_real` --
#: which is a real gap against DuckDB (`STDDEV(decimal)` runs there) rather than a semantic
#: boundary, and is stated as such in the message.
_NUMERIC: dict[str, str] = dict.fromkeys(("sum", "mean"), "numeric")
_REAL: dict[str, str] = dict.fromkeys(
    (
        "median",
        "quantile",
        "quantile_disc",
        "approx_quantile",
        "stddev",
        "var",
        "mad",
        "product",
        "skewness",
        "kurtosis",
        "kurtosis_pop",
        "kahan_sum",
        "corr",
        "covar_pop",
        "covar_samp",
        "aun",
        "l_count",
        "n_length",
    ),
    "real",
)
_BITWISE = ("bit_and", "bit_or", "bit_xor")
_BOOLEAN = ("bool_and", "bool_or")
#: Aggregates that pick or compare a value, so they need one the engine can order.
_ORDERED = ("min", "max", "any_value")

# The cast that most often turns a rejected column into an accepted one, by input family.
_CAST_HINT = {
    "temporal": "cast it to int64 first if you mean to aggregate the underlying instant",
    "boolean": "cast it to int64 first if you mean to count true as 1",
    "string": "cast it to a numeric type first if it holds numbers as text",
}


def _family(dt: pa.DataType) -> str:
    if pa.types.is_temporal(dt) or pa.types.is_duration(dt):
        return "temporal"
    if pa.types.is_boolean(dt):
        return "boolean"
    if pa.types.is_string(dt) or pa.types.is_large_string(dt):
        return "string"
    return "other"


def _reject(func: str, column: str, dt: pa.DataType, domain: str) -> str:
    hint = _CAST_HINT.get(_family(dt))
    tail = f"; {hint}" if hint else ""
    return f"aggregate {func!r} needs {domain} input, but {column} is {dt}{tail}"


def aggregate_domain_error(func: str, column: str, dt: pa.DataType) -> str | None:
    """Why `func` cannot be computed over a `dt` column, or None when it can.

    A `None` return is not a promise the query succeeds -- it means this rule has nothing
    to say about the pair. Aggregates with no domain restriction (``count``, ``list_agg``,
    ``histogram``, ``mode``, ``arg_min``…) are simply absent from the tables above.

    The ``null`` type is accepted by all but the boolean pair (see the comment below).

    Args:
        func: The aggregate's IR tag (`plan.ir_tags.AGG_FNS`).
        column: How to name the offending input in the message.
        dt: The input column's Arrow type, already widened to what the engine sees.

    Returns:
        A message naming the column, its type, and the fix, or None when the pair is fine.
    """
    if pa.types.is_null(dt):
        # Accepted by every aggregate but the boolean pair. An all-null column has no
        # values to be the wrong type, and rejecting it would fail an empty partition of a
        # perfectly valid relation. `bool_and`/`bool_or` are the exception because the
        # accumulators genuinely cannot run on one: the group's values are materialized as
        # Int64, and the engine's own answer is the baffling "aggregate bool_and is not
        # supported for column type Int64" for a column the user knows is `null`.
        if func in _BOOLEAN:
            return (
                f"aggregate {func!r} needs boolean input, but {column} is all-null with no "
                f"type of its own; cast it to bool so the empty case is still a boolean"
            )
        return None
    if func in _NUMERIC and not _is_numeric(dt):
        return _reject(func, column, dt, "numeric")
    if func in _REAL and not _is_real(dt):
        domain = "integer or floating-point" if pa.types.is_decimal(dt) else "numeric"
        return _reject(func, column, dt, domain)
    if func in _BITWISE and not pa.types.is_integer(dt):
        return _reject(func, column, dt, "integer")
    if func in _BOOLEAN and not pa.types.is_boolean(dt):
        return _reject(func, column, dt, "boolean")
    if func in _ORDERED and not _is_orderable(dt):
        return f"aggregate {func!r} needs a comparable input, but {column} is {dt}"
    return None


#: A window function's name against the aggregate whose input domain it shares.
#:
#: The window vocabulary and the aggregate vocabulary overlap without being the same set,
#: and the two spell one function differently: `WINDOW_AGGREGATES` has ``avg`` where
#: `AGG_FNS` has ``mean``. Mapping instead of re-declaring keeps the domains defined once,
#: so a rule change reaches both surfaces.
_WINDOW_AS_AGGREGATE = {
    "avg": "mean",
    "stddev": "stddev",
    "var": "var",
    "sum": "sum",
    "product": "product",
    "min": "min",
    "max": "max",
    "bit_and": "bit_and",
    "bit_or": "bit_or",
    "bit_xor": "bit_xor",
    "bool_and": "bool_and",
    "bool_or": "bool_or",
    # The exponentially-weighted statistics and the gap filler are weighted means over the
    # ordered prefix, so they need the same real-valued input `mean` does. They have no
    # aggregate spelling of their own, which is why they borrow one here.
    "ewm_mean": "mean",
    "ewm_var": "var",
    "ewm_std": "stddev",
    "interpolate": "mean",
}


def window_domain_error(func: str, column: str, dt: pa.DataType) -> str | None:
    """Why window function `func` cannot run over a `dt` column, or None when it can.

    The window forms of the aggregates compute the same statistic over a frame, so they
    have the same input domain -- and the same two failure modes when given the wrong one.
    Ranking (`row_number`, `rank`, …) and the positional value functions (`lag`, `lead`,
    `first_value`, `forward_fill`, …) take any type by construction, so they are absent
    from the map and this returns None for them.

    Args:
        func: The window function's IR tag (`plan.ir_tags.WINDOW_FUNCS`).
        column: How to name the offending input in the message.
        dt: The input column's Arrow type, already widened to what the engine sees.

    Returns:
        A message naming the column, its type, and the fix, or None when the pair is fine.
    """
    equivalent = _WINDOW_AS_AGGREGATE.get(func)
    if equivalent is None:
        return None
    problem = aggregate_domain_error(equivalent, column, dt)
    if problem is None:
        return None
    # Reported under the name the user wrote, not the aggregate it borrows the domain from.
    return problem.replace(f"aggregate {equivalent!r}", f"window function {func!r}", 1)


# There is deliberately no `temporal_domain_error` here, and it is worth saying why so it is
# not re-derived a third time.
#
# A build-time refusal of a *string* input to the `.dt` surface was added on the premise that
# the engine answers such a column with all nulls. It does not: `eval/temporal/date.rs` parses
# a text column as a timestamp first, uniformly and on purpose, so `year('2016-07-30')` is a
# real answer rather than a silent null. That hoist is documented in the engine as accepting
# *more* than the DuckDB oracle rather than answering differently from it -- the one direction
# a compatibility convenience may go -- and the Spark dialect surface depends on it
# (`tests/unit/test_sql_spark_dialect_names.py`).
#
# So the refusal rejected queries the engine answers correctly, and only an unparseable *value*
# yields a null, exactly as a `TRY_CAST` would. The type alone cannot distinguish the two at
# plan time, which is why this is a runtime cast and not a domain rule.


def _contains_map(dt: pa.DataType) -> bool:
    """Whether `dt` is a `map`, or nests one at any depth.

    Nesting is what makes this a walk rather than a type check: `struct<m: map<..>>` and
    `list<map<..>>` fail exactly as a bare `map` does, because the row encoder descends into
    the child types to build the comparable byte string.
    """
    if pa.types.is_map(dt):
        return True
    return any(_contains_map(dt.field(i).type) for i in range(dt.num_fields))


def key_domain_error(column: str, dt: pa.DataType, operation: str) -> str | None:
    """Why `column` cannot be a grouping / dedup / join key, or `None` when it can.

    Grouping, `DISTINCT`, and hash joins all identify rows by encoding the key columns into
    a single comparable byte string (`bc-runtime`'s `keys`, over arrow-rs's row format).
    That encoder handles every type the engine otherwise supports — including the nested
    ones people assume it would not: `list`, `large_list`, `fixed_size_list`, `struct`,
    `list<struct>`, and dictionary-encoded columns all key correctly. The single exception
    is `map`, which has no canonical ordering of its entries and therefore no stable
    encoding, and it is rejected wherever it appears, including nested inside a `struct` or
    a `list`.

    Without this the rejection arrived from Rust as

        RuntimeError: Not yet implemented: Row format support not yet implemented for:
        [SortField { options: SortOptions { descending: false, nulls_first: true }, data_type

    — an internal dump, truncated mid-struct, naming neither the column nor the operation,
    and arriving *after the scan*. That is the same failure this module's
    `aggregate_domain_error` was written to remove for aggregate inputs, and the same one
    `plan.logical.join._validate_key_types` removed for mismatched join keys; this is the
    third case, and the one both of those left behind.

    A `sort` is deliberately not covered: sorting a `map` column works, because the sort
    path compares values directly rather than through the row encoder.

    Args:
        column: The key column's name, for the message.
        dt: Its Arrow type.
        operation: What is being attempted, e.g. ``"group_by"`` — named in the message so
            the reader is not left to infer which clause is at fault.

    Returns:
        An actionable message, or `None` when the type can be a key.
    """
    if not _contains_map(dt):
        return None
    nested = "" if pa.types.is_map(dt) else " (nested inside it)"
    return (
        f"{operation}: column {column!r} is {dt}, and a map{nested} cannot be a key — its "
        f"entries have no canonical order, so there is no stable way to tell two maps "
        f"apart. Key on something derived from it instead, such as "
        f"`col({column!r}).map.keys()`, `col({column!r}).map.values()`, or a specific "
        f"lookup, or cast it to a list of structs."
    )

"""What a Python function registered for SQL looks like to the translator.

A `Session` records one of these per `register_function` call; `_sql` reads them when it
meets an unknown function name in a query. The record lives on the `api` side of the
boundary because `_sql` is a front-end built *on* `api`, so the dependency has to point
that way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

__all__ = [
    "NULL_HANDLING",
    "RegisteredFunction",
    "resolve_type",
    "validate_null_handling",
    "validate_options",
]

#: DuckDB `create_function`'s two `null_handling` modes, accepted by the scalar form.
NULL_HANDLING = ("special", "default")


@dataclass(frozen=True)
class RegisteredFunction:
    """A Python function registered for use in SQL, plus how it lowers to `map_batches`.

    `table` selects the call form: a table function ``SELECT * FROM f(t)`` (whole
    relation in, relation out) when true, else a scalar function ``SELECT f(x)``
    hoisted into a column-materializing `map_batches`. `vectorized` (scalar form)
    chooses whether `fn` receives whole Arrow arrays or one row at a time; `per_row`
    is the table-form analogue (``ds.map`` vs ``ds.map_batches``). `null_handling`
    (scalar form) is DuckDB's ``create_function`` switch: ``"special"`` hands NULL to `fn`,
    ``"default"`` answers NULL for any row with a NULL argument without calling `fn`.
    """

    name: str
    fn: Callable
    table: bool
    per_row: bool
    vectorized: bool
    result_type: pa.DataType | None
    output_columns: tuple[str, ...] | None
    config: dict[str, Any] = field(default_factory=dict)
    null_handling: str = "special"


def resolve_type(result_type: str | pa.DataType | None) -> pa.DataType | None:
    """Resolve a declared result type (an Arrow type or its string alias) to a type.

    Args:
        result_type: An Arrow type, an alias such as ``"int64"``, or `None`.

    Returns:
        The resolved Arrow type, or `None` when none was declared.
    """
    if result_type is None or isinstance(result_type, pa.DataType):
        return result_type
    return pa.type_for_alias(result_type)


#: Options a user reaches for that mean "make this an aggregate". SQL aggregates are not a
#: thing `register_function` can express — an aggregate needs a mergeable
#: partial/combine/finalize form in the engine, not a Python callable over one batch — so
#: naming the alternative is the whole of the help that can be given.
_AGGREGATE_OPTIONS = ("aggregate", "agg", "is_aggregate")


def validate_options(name: str, options: dict[str, Any], *, table: bool, per_row: bool) -> None:
    """Reject a `register_function` option the chosen call form would silently drop.

    `register_function` takes ``**config`` so a table function can forward `map_batches`
    options, and that catch-all swallowed everything else in silence: a misspelled
    ``result_typ``, a ``num_gpus`` on the scalar form (where the config was never read at
    all), an ``aggregate=True`` that looked like it registered a UDAF and then failed at
    query time inside pyarrow. Each of those is a query that runs and is wrong, or an option
    the user believes is in force and is not.

    Args:
        name: The SQL function name, for the message.
        options: The extra keywords the caller passed.
        table: Whether it is being registered as a table function.
        per_row: Whether a table function is applied row-at-a-time.

    Raises:
        PlanError: If any option cannot take effect for this call form.
    """
    if not options:
        return
    from batcher._internal.errors import PlanError

    for key in options:
        if key in _AGGREGATE_OPTIONS:
            raise PlanError(
                f"register_function({name!r}): {key}= is not supported — a SQL aggregate needs "
                f"a mergeable partial/combine/finalize form, which a Python callable over one "
                f"batch cannot provide. Use ds.group_by(...).agg(...) with a built-in "
                f"aggregate, or ds.group_by(...).map_groups(fn) for arbitrary Python per group."
            )
    if not table:
        raise PlanError(
            f"register_function({name!r}): a scalar function takes no map_batches options, so "
            f"{sorted(options)} would be ignored. Pass table=True to register a table function "
            f"(SELECT * FROM {name}(t)), which forwards them."
        )
    # The unknown-key half is shared with the `@udf` decorator, which forwards the same bag
    # to the same place — one list of valid options, read off the live signature.
    from batcher.api.dataset._options import validate_map_options

    validate_map_options(f"register_function({name!r})", options, per_row=per_row)


def validate_null_handling(name: str, null_handling: object, *, table: bool) -> None:
    """Reject a `null_handling` that is not DuckDB's, or one the table form cannot honour.

    Args:
        name: The SQL function name, for the message.
        null_handling: The requested mode.
        table: Whether it is being registered as a table function.

    Raises:
        PlanError: If the mode is unknown, or ``"default"`` is asked of a table function,
            whose `fn` receives whole rows rather than arguments.
    """
    from batcher._internal.errors import PlanError

    if null_handling not in NULL_HANDLING:
        raise PlanError(
            f"register_function({name!r}): null_handling must be one of "
            f"{list(NULL_HANDLING)}, got {null_handling!r}"
        )
    if table and null_handling != "special":
        raise PlanError(
            f"register_function({name!r}): null_handling applies to a scalar function's "
            "arguments; a table function receives whole rows. Drop it, or handle NULLs in fn."
        )

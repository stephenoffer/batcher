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

__all__ = ["RegisteredFunction", "resolve_type"]


@dataclass(frozen=True)
class RegisteredFunction:
    """A Python function registered for use in SQL, plus how it lowers to `map_batches`.

    `table` selects the call form: a table function ``SELECT * FROM f(t)`` (whole
    relation in, relation out) when true, else a scalar function ``SELECT f(x)``
    hoisted into a column-materializing `map_batches`. `vectorized` (scalar form)
    chooses whether `fn` receives whole Arrow arrays or one row at a time; `per_row`
    is the table-form analogue (``ds.ml.map`` vs ``ds.ml.map_batches``).
    """

    name: str
    fn: Callable
    table: bool
    per_row: bool
    vectorized: bool
    result_type: pa.DataType | None
    output_columns: tuple[str, ...] | None
    config: dict[str, Any] = field(default_factory=dict)


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

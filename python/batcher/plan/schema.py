"""`SchemaRef` — a thin wrapper making `pyarrow.Schema` the source of truth.

Every plan node, contract, and operator declares its schema as a `SchemaRef`.
There is deliberately no parallel type system: Arrow types ARE Batcher's types,
so a column dtype, a cross-FFI batch, and a Rust kernel all agree by
construction. Schema compatibility is validated at plan-build time (fail fast,
before any work is scheduled).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pyarrow as pa

from batcher._internal.errors import did_you_mean

__all__ = ["SchemaRef", "column_name", "placeholder_schema", "suggest_columns"]


def placeholder_schema(names: list[str]) -> pa.Schema:
    """Null-typed placeholders carrying just `names` — the last-resort empty-result schema.

    Used only when a zero-batch result's types cannot be inferred (an opaque `map_batches`
    output). Prefer `plan.logical.empty_result_schema`, which falls back to this.

    Args:
        names: The column names the result is expected to carry.

    Returns:
        A schema with one null-typed field per name.
    """
    return pa.schema([pa.field(name, pa.null()) for name in names])


def suggest_columns(name: str, available: list[str]) -> str:
    """A '; did you mean ...' hint for an unknown column, or '' when nothing is close.

    Turns a bare 'no column x' into an actionable message (the Postgres / Polars
    affordance). Returns up to three suggestions in this clause-shaped form, which is
    what its six call sites concatenate onto an existing sentence.

    The matching itself is `batcher._internal.errors.did_you_mean` — deliberately not a
    second `difflib` call here. That helper ranks a case-only difference first and also
    catches the abbreviation case (``"cust"`` for ``"customer_id"``) that a bare
    similarity ratio scores below any usable cutoff, and keeping one implementation is
    what stops column suggestions and format/option suggestions from drifting apart.

    Args:
        name: The column name the user supplied.
        available: The column names that do exist.

    Returns:
        ``"; did you mean 'x' or 'y'?"``, or ``""`` when nothing is close enough.

    Examples:
        .. doctest::

            >>> from batcher.plan.schema import suggest_columns
            >>> suggest_columns("nmae", ["name", "age"])
            "; did you mean 'name'?"
            >>> suggest_columns("zzz", ["name", "age"])
            ''
    """
    matches = did_you_mean(name, available)
    if not matches:
        return ""
    return f"; did you mean {' or '.join(repr(n) for n in matches)}?"


def column_name(value: object, *, arg: str, api: str) -> str:
    """`value` as a column *name*, or a `PlanError` saying what to pass instead.

    Twenty-five public methods take a column name as a string — ``ds.sum("v")``,
    ``ds.with_watermark("ts", ...)``, ``ds.session_window("ts", ...)`` — and a caller
    arriving from Polars reaches for ``ds.sum(col("v"))`` instead. Every one of them then
    did the same thing: tested ``value not in ds.columns``, which evaluates ``Expr.__eq__``
    against each name, builds an expression, and asks it for a truth value. The error the
    user saw was ``the truth value of an Expr is ambiguous; use & | ~ to combine
    predicates`` — a message about boolean operators, from a call that used none, naming
    neither the method nor the argument.

    An expression is refused rather than accepted because these methods are the ones an
    expression genuinely cannot express: the name is a *key* the operator looks up in a
    schema (a watermark's event-time column, a session's ordering column), not a value to
    compute. Where a derived value is meaningful, compute it first with ``with_columns``
    and pass the new name.

    Args:
        value: What the caller passed.
        arg: The parameter's name, for the message (e.g. ``"time_col"``).
        api: The method's name, for the message (e.g. ``"with_watermark"``).

    Returns:
        `value` unchanged, once it is known to be a string.

    Examples:
        .. doctest::

            >>> from batcher.plan.schema import column_name
            >>> column_name("ts", arg="time_col", api="with_watermark")
            'ts'
    """
    if isinstance(value, str):
        return value
    from batcher._internal.errors import PlanError
    from batcher.plan.expr_ir.core import Expr

    if isinstance(value, Expr):
        named = getattr(value, "name", None)
        hint = (
            f" — pass {named!r}"
            if isinstance(named, str) and named
            else " — pass the column's name"
        )
        raise PlanError(
            f"{api}(): {arg} must be a column name, not an expression{hint}. This "
            "argument names a column the operator looks up in the schema, so a computed "
            "expression has nothing to look up; derive it with with_columns(...) first "
            "and pass the new name."
        )
    raise PlanError(f"{api}(): {arg} must be a column name (str), not {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class SchemaRef:
    """An immutable handle to a `pyarrow.Schema`."""

    arrow: pa.Schema

    @property
    def names(self) -> list[str]:
        return list(self.arrow.names)

    def field(self, name: str) -> pa.Field:
        idx = self.arrow.get_field_index(name)
        if idx < 0:
            hint = suggest_columns(name, self.names)
            raise KeyError(f"no column named {name!r} in schema {self.names}{hint}")
        return self.arrow.field(idx)

    def has(self, name: str) -> bool:
        return self.arrow.get_field_index(name) >= 0

    @classmethod
    def from_arrow(cls, schema: pa.Schema) -> SchemaRef:
        return cls(schema)

    @classmethod
    def from_typed_fields(
        cls, fields: Iterable[tuple[str, pa.DataType | None]]
    ) -> SchemaRef | None:
        """Build a schema from `(name, type)` pairs, or `None` if any type is unknown.

        The all-or-nothing rule is the point, and it is why this is shared rather than
        written per node: `available_schema` promises names *and* types, so one column whose
        type cannot be inferred makes the whole schema unknown. A node that instead dropped
        the uncertain column would hand callers a plausible schema that is missing a field,
        and they would plan against it rather than falling back to a zero-row execution.
        `Projection` and `Aggregate` each had their own copy of that loop.

        Args:
            fields: `(name, type)` pairs in output order; a `None` type means "not inferable".

        Returns:
            The schema, or `None` when any pair's type is `None`.
        """
        out: list[pa.Field] = []
        for name, dtype in fields:
            if dtype is None:
                return None
            out.append(pa.field(name, dtype))
        return cls.from_arrow(pa.schema(out))

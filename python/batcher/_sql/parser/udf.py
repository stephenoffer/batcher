"""Registered-Python-function support for the SQL translator.

Python cannot run inside the engine's Rust expression evaluator, so a function
registered with `Session.register_function` lowers to a `MapBatches` stage instead
of an `Expr`. Two forms:

* table function — ``SELECT * FROM f(t)`` — resolved in `clauses._table` via
  `_apply_table_function`: the whole relation flows through `ds.ml.map_batches`.
* scalar function — ``SELECT f(x)`` / ``WHERE f(x)`` — resolved by `_hoist_udfs`,
  a pre-pass that materializes each call into a synthetic column (a `map_batches`
  that appends one column) and rewrites the call to reference that column, so the
  remaining scalar translation sees a plain column.

The per-call adapters are module-level, picklable classes (like
`api.dataset.callbacks._RowMap`) so Ray can ship them to workers. Functions take
the translator instance (`tr`) as their first argument.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError


def _const_value(node) -> Any:
    """The Python value of a literal SQL argument (constant UDF args bind directly)."""
    from sqlglot import expressions as exp

    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Null):
        return None
    if node.is_string:
        return node.this
    text = node.this
    return float(text) if ("." in text or "e" in text.lower()) else int(text)


def _to_array(result: Any, result_type: pa.DataType | None) -> pa.Array:
    """Coerce a vectorized UDF's return value to a single Arrow array."""
    if isinstance(result, pa.ChunkedArray):
        result = result.combine_chunks()
    if isinstance(result, pa.Array):
        return result.cast(result_type) if result_type is not None else result
    return pa.array(result, type=result_type)


class _SqlUdfBatch:
    """Vectorized scalar UDF: ``fn(array, ...) -> array``, appended as one column."""

    __slots__ = ("arg_cols", "const_args", "fn", "out_col", "result_type")

    def __init__(self, fn, arg_cols, const_args, out_col, result_type) -> None:
        self.fn = fn
        self.arg_cols = arg_cols  # list[(position, column_name)]
        self.const_args = const_args  # list[(position, value)]
        self.out_col = out_col
        self.result_type = result_type

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        args = _ordered_args(batch, self.arg_cols, self.const_args)
        col = _to_array(self.fn(*args), self.result_type)
        return batch.append_column(self.out_col, col)


class _SqlUdfRow:
    """Per-row scalar UDF: ``fn(*scalars) -> scalar``, appended as one column."""

    __slots__ = ("arg_cols", "const_args", "fn", "out_col", "result_type")

    def __init__(self, fn, arg_cols, const_args, out_col, result_type) -> None:
        self.fn = fn
        self.arg_cols = arg_cols
        self.const_args = const_args
        self.out_col = out_col
        self.result_type = result_type

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        n = len(self.arg_cols) + len(self.const_args)
        columns = {name: batch.column(name).to_pylist() for _, name in self.arg_cols}
        out = []
        for row in range(batch.num_rows):
            args: list[Any] = [None] * n
            for pos, name in self.arg_cols:
                args[pos] = columns[name][row]
            for pos, val in self.const_args:
                args[pos] = val
            out.append(self.fn(*args))
        return batch.append_column(self.out_col, pa.array(out, type=self.result_type))


def _ordered_args(batch: pa.RecordBatch, arg_cols, const_args) -> list[Any]:
    """Reassemble the positional argument list from columns + bound constants."""
    args: list[Any] = [None] * (len(arg_cols) + len(const_args))
    for pos, name in arg_cols:
        args[pos] = batch.column(name)
    for pos, val in const_args:
        args[pos] = val
    return args


def _make_adapter(rf, arg_cols, const_args, out_col):
    cls = _SqlUdfBatch if rf.vectorized else _SqlUdfRow
    return cls(rf.fn, arg_cols, const_args, out_col, rf.result_type)


def _is_registered_scalar(tr, node) -> bool:
    """Whether `node` is a call to a registered *scalar* (non-table) function."""
    from sqlglot import expressions as exp

    if not isinstance(node, exp.Anonymous):
        return False
    rf = tr._functions.get(node.name)
    return rf is not None and not rf.table


def contains_registered_scalar(tr, node) -> bool:
    """Whether any subexpression of `node` calls a registered scalar function."""
    from sqlglot import expressions as exp

    if node is None:
        return False
    return any(_is_registered_scalar(tr, n) for n in node.find_all(exp.Anonymous))


def _hoist_udfs(tr, ds, clause_nodes):
    """Materialize registered scalar-function calls in `clause_nodes` as columns.

    Each call is rewritten to a reference to a synthetic column computed by an
    inserted `map_batches`; nested calls hoist deepest-first so an inner result is a
    plain column by the time the outer call is processed. Returns
    ``(ds, roots)`` — the dataset with the `map_batches` stages prepended and the
    (possibly-replaced) clause nodes, since a call that is itself a clause root is
    swapped out and the caller must use the replacement.
    """
    from sqlglot import expressions as exp

    roots = list(clause_nodes)
    calls = []
    for root in roots:
        if root is None:
            continue
        calls.extend(n for n in root.find_all(exp.Anonymous) if _is_registered_scalar(tr, n))
    # find_all is pre-order (parents first); reverse so nested calls hoist first.
    for call in reversed(calls):
        ds, replacement = _hoist_one(tr, ds, call)
        for i, root in enumerate(roots):
            if root is call:
                roots[i] = replacement
    return ds, roots


def _hoist_one(tr, ds, call):
    from sqlglot import expressions as exp

    rf = tr._functions[call.name]
    arg_cols: list[tuple[int, str]] = []
    const_args: list[tuple[int, Any]] = []
    for i, arg in enumerate(call.expressions):
        if isinstance(arg, (exp.Literal, exp.Boolean, exp.Null)):
            const_args.append((i, _const_value(arg)))
        elif isinstance(arg, exp.Column):
            arg_cols.append((i, arg.name))
        else:
            tmp = f"__bc_udf_arg{tr._udf_n}_{i}"
            ds = ds.with_columns(**{tmp: tr._scalar(arg)})
            arg_cols.append((i, tmp))

    out_col = f"__bc_udf_{tr._udf_n}"
    tr._udf_n += 1
    adapter = _make_adapter(rf, arg_cols, const_args, out_col)
    ds = ds.ml.map_batches(adapter, output_columns=[*ds.columns, out_col])
    replacement = exp.column(out_col)
    if call.parent is not None:
        call.replace(replacement)
    return ds, replacement


def _apply_table_function(tr, anon, rf):
    """Apply a registered table function ``f(t)`` (``SELECT * FROM f(t)``)."""
    from sqlglot import expressions as exp

    if len(anon.expressions) != 1:
        raise PlanError(f"table function {rf.name!r} takes exactly one table argument")
    arg = anon.expressions[0]
    if isinstance(arg, exp.Subquery):
        src = tr.statement(arg.this)
    elif isinstance(arg, (exp.Select, exp.Union)):
        src = tr.statement(arg)
    elif isinstance(arg, exp.Column):
        if arg.name not in tr._registry:
            raise PlanError(f"unknown table {arg.name!r}; registered: {list(tr._registry)}")
        src = tr._registry[arg.name]
    else:
        raise PlanError(f"table function {rf.name!r} argument must be a table name or subquery")

    out_cols = list(rf.output_columns) if rf.output_columns is not None else None
    if rf.per_row:
        return src.ml.map(rf.fn, output_columns=out_cols)
    return src.ml.map_batches(rf.fn, output_columns=out_cols, **rf.config)

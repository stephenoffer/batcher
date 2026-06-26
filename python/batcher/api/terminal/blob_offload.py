"""Automatic blob offload placement around pipeline breakers.

When a `large_binary` column (the signal for large per-row payloads — what media
materialization and `download_dataset` produce) flows through a `Sort`, the whole
payload is carried in the sort's buffers and spill files even though the sort only
touches its keys. This opt-in transform rewrites such a sort to

    Sort(input)  →  materialize( Sort( offload(input) ) )

so the payload rides through the breaker as a tiny content-addressed URI handle and
is read back right after — the same offload/materialize the explicit
`Dataset.offload_blobs`/`materialize_blobs` build, placed automatically. It is
schema-transparent (the handle column is private and dropped on materialize) and
result-identical (offload∘materialize is the identity), so it only trades blob bytes
crossing the breaker for the re-read.

This lives in `api` — the only layer allowed to wire `io` (the offload/materialize
functions) into a plan; `kyber`/`carbonite`/`core` cannot import `io`. It runs as a
plan-preparation step (off by default, gated on `ExecutionConfig.auto_offload_blobs`),
before optimization, so both the single-node and distributed paths see the rewrite.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import pyarrow as pa

from batcher.io.formats.multimodal.blob import (
    default_blob_root,
    materialize_and_drop_handle,
    offload_blob_bytes,
)
from batcher.plan.expr_ir import referenced_columns
from batcher.plan.logical import Join, LogicalPlan, MapBatches, Sort, Union

__all__ = ["insert_blob_offload", "maybe_insert_blob_offload"]

# A private handle-column name so the rewrite never collides with a user column
# (including reference mode's own ``uri``).
_AUTO_HANDLE = "__blob_handle__"


def maybe_insert_blob_offload(plan: LogicalPlan) -> LogicalPlan:
    """Apply `insert_blob_offload` when ``ExecutionConfig.auto_offload_blobs`` is on, else
    return `plan` unchanged — the one place the terminal path consults the flag."""
    from batcher.config import active_config

    if active_config().execution.auto_offload_blobs:
        return insert_blob_offload(plan)
    return plan


def insert_blob_offload(
    plan: LogicalPlan, *, root: str | None = None, batch_size: int = 8
) -> LogicalPlan:
    """Rewrite eligible breakers to offload their large-payload columns out of line."""
    return _rewrite(plan, root or default_blob_root(), batch_size)


def _rewrite(node: LogicalPlan, root: str, batch_size: int) -> LogicalPlan:
    node = _recurse_children(node, root, batch_size)
    if isinstance(node, Sort):
        column = _offloadable_blob_column(node)
        if column is not None:
            offloaded = _offload_node(node.input, column, root, batch_size)
            sorted_node = dataclasses.replace(node, input=offloaded)
            return _materialize_node(sorted_node, column, batch_size)
    return node


def _recurse_children(node: LogicalPlan, root: str, batch_size: int) -> LogicalPlan:
    """Apply the rewrite to a node's children, preserving identity when nothing changed."""
    if isinstance(node, Join):
        left, right = _rewrite(node.left, root, batch_size), _rewrite(node.right, root, batch_size)
        if left is node.left and right is node.right:
            return node
        return dataclasses.replace(node, left=left, right=right)
    if isinstance(node, Union):
        inputs = tuple(_rewrite(i, root, batch_size) for i in node.inputs)
        if all(a is b for a, b in zip(inputs, node.inputs, strict=True)):
            return node
        return dataclasses.replace(node, inputs=inputs)
    if hasattr(node, "input"):
        child = _rewrite(node.input, root, batch_size)
        return node if child is node.input else dataclasses.replace(node, input=child)
    return node


def _offloadable_blob_column(sort: Sort) -> str | None:
    """The name of a `large_binary` input column the sort does not key on, or None.

    Requires inferable input types (P2 `available_schema`); a sort already fed by an
    offload (an opaque `MapBatches`, schema ``None``) is skipped, so the rewrite is
    idempotent and never double-offloads.
    """
    schema = sort.input.available_schema()
    if schema is None:
        return None
    key_cols: set[str] = set()
    for key in sort.keys:
        key_cols |= referenced_columns(key.expr)
    for field in schema.arrow:
        if pa.types.is_large_binary(field.type) and field.name not in key_cols:
            return field.name
    return None


def _offload_node(input_plan: LogicalPlan, column: str, root: str, batch_size: int) -> MapBatches:
    cols = input_plan.available_columns()
    out_cols = (*cols, _AUTO_HANDLE) if _AUTO_HANDLE not in cols else tuple(cols)
    fn = partial(offload_blob_bytes, root=root, src=column, uri_col=_AUTO_HANDLE)
    return MapBatches(input_plan, fn, batch_size=batch_size, output_columns=out_cols)


def _materialize_node(sorted_plan: LogicalPlan, column: str, batch_size: int) -> MapBatches:
    # Restore the original schema: payload back in `column`, the private handle gone.
    out_cols = tuple(c for c in sorted_plan.available_columns() if c != _AUTO_HANDLE)
    fn = partial(materialize_and_drop_handle, uri_col=_AUTO_HANDLE, into=column)
    return MapBatches(sorted_plan, fn, batch_size=batch_size, output_columns=out_cols)

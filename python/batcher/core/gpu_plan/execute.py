"""Replay a matched plan on a dataframe backend — the executor behind the GPU entry points.

Every function here is written against `DfBackend`, so the same code runs on cuDF (the
accelerated backend, on a GPU worker) and on pandas (the head-runnable check against the
native CPU engine). The `execute_cudf_*` wrappers are the only places that name cuDF, and
they exist so a caller cannot accidentally take the verification backend to production.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.core.gpu_plan.backend import DfBackend, Unsupported
from batcher.core.gpu_plan.eligibility import JOIN_HOW
from batcher.core.gpu_plan.ops import apply_op

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "execute_cudf_join",
    "execute_cudf_plan",
    "execute_cudf_union",
    "run_chain",
    "run_join",
    "run_union",
]

_LEFT = "L__"
_RIGHT = "R__"


def _cudf() -> DfBackend:
    import cudf

    return DfBackend(cudf)


def run_chain(table: pa.Table, ops: list[dict], be: DfBackend):
    """Replay an operator chain on a dataframe built from `table`."""
    df = be.from_arrow(table)
    for op in ops:
        df = apply_op(df, op, be)
    return df


def execute_cudf_plan(table: pa.Table, ops: list[dict]) -> pa.Table:
    """Replay a translated operator chain on the GPU via cuDF, returning Arrow.

    Args:
        table: The input rows.
        ops: The bottom-up operator IR chain from `gpu_plan_ops`.

    Returns:
        The chain's result as an Arrow table.

    Raises:
        Unsupported: For an operator or expression outside the translated subset, which the
            caller turns into a CPU-engine fallback.
    """
    be = _cudf()
    return be.to_arrow(run_chain(table, ops, be))


def run_join(left_t, right_t, left_ops, right_ops, join_ir: dict, ops: list[dict], be: DfBackend):
    """Replay both input chains, join them, then replay the chain above the join.

    Each side's columns are prefixed before the merge so same-named columns never collide;
    the join's `output` spec then selects by side and aliases, reproducing the exact columns
    and order the CPU engine would produce.
    """
    left = run_chain(left_t, left_ops, be)
    right = run_chain(right_t, right_ops, be)
    how = JOIN_HOW[join_ir["join_type"]]
    if how in ("semi", "anti"):
        out = _semi_join(left, right, join_ir, keep=how == "semi")
    else:
        out = _equi_join(left, right, join_ir, how, be)
    for op in ops:
        out = apply_op(out, op, be)
    return out


def _equi_join(left, right, join_ir: dict, how: str, be: DfBackend):
    lg = left.add_prefix(_LEFT)
    rg = right.add_prefix(_RIGHT)
    merged = lg.merge(
        rg,
        left_on=[_LEFT + k for k in join_ir["left_keys"]],
        right_on=[_RIGHT + k for k in join_ir["right_keys"]],
        how=how,
    )
    cols = {}
    for o in join_ir["output"]:
        src = (_LEFT if o["side"] == "left" else _RIGHT) + o["name"]
        cols[o["alias"]] = merged[src].reset_index(drop=True)
    return be.lib.DataFrame(cols)


def _semi_join(left, right, join_ir: dict, *, keep: bool):
    """A semi/anti join as a key-membership filter over the left side.

    A `merge` cannot express either one: it would duplicate a left row per matching right row
    (semi keeps one) and has no mode that keeps the non-matching rows alone (anti). Membership
    also gets the null key right for free, since null is not a member of anything — which is
    the answer both semi and anti want.
    """
    lkeys = join_ir["left_keys"]
    rkeys = join_ir["right_keys"]
    if len(lkeys) != 1:
        # A composite key needs a tuple-valued membership test, which neither backend offers
        # without materializing a joint key column of an inferred type.
        raise Unsupported("semi/anti join on a composite key")
    member = left[lkeys[0]].isin(right[rkeys[0]])
    mask = member.fillna(False)
    out = left[mask if keep else ~mask].reset_index(drop=True)
    return out.rename(columns={o["name"]: o["alias"] for o in join_ir["output"]})[
        [o["alias"] for o in join_ir["output"]]
    ]


def execute_cudf_join(
    left_t: pa.Table,
    right_t: pa.Table,
    left_ops: list[dict],
    right_ops: list[dict],
    join_ir: dict,
    ops: list[dict],
) -> pa.Table:
    """Run a join of two translated chains plus the chain above it on the GPU via cuDF.

    Args:
        left_t: The left input's rows.
        right_t: The right input's rows.
        left_ops: The left input chain's operator IR.
        right_ops: The right input chain's operator IR.
        join_ir: The join node's IR.
        ops: The operator chain above the join.

    Returns:
        The join's result as an Arrow table.
    """
    be = _cudf()
    return be.to_arrow(run_join(left_t, right_t, left_ops, right_ops, join_ir, ops, be))


def run_union(tables: list, input_ops: list[list[dict]], distinct: bool, ops, be: DfBackend):
    """Replay each input's chain, concatenate them, optionally deduplicate, then run `ops`."""
    frames = [run_chain(t, o, be) for t, o in zip(tables, input_ops, strict=True)]
    out = be.concat(frames)
    if distinct:
        out = out.drop_duplicates().reset_index(drop=True)
    for op in ops:
        out = apply_op(out, op, be)
    return out


def execute_cudf_union(
    tables: list, input_ops: list[list[dict]], distinct: bool, ops: list[dict]
) -> pa.Table:
    """Run a union of translated chains plus the chain above it on the GPU via cuDF.

    Args:
        tables: Each input's rows.
        input_ops: Each input chain's operator IR, positionally matching `tables`.
        distinct: Whether the union deduplicates.
        ops: The operator chain above the union.

    Returns:
        The union's result as an Arrow table.
    """
    be = _cudf()
    return be.to_arrow(run_union(tables, input_ops, distinct, ops, be))

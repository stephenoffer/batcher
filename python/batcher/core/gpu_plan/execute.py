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
from batcher.core.gpu_plan.ops import apply_op, distinct_rows, fold_zero

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "execute_cudf_join",
    "execute_cudf_plan",
    "execute_cudf_union",
    "run_chain",
    "run_join",
    "run_ops",
    "run_union",
]

_LEFT = "L__"
_RIGHT = "R__"
#: Suffix of the synthetic key component that keeps a null key from matching (`_null_key_marker`).
_NULL_KEY = "__bt_nullkey"


def _cudf() -> DfBackend:
    import cudf

    return DfBackend(cudf)


def run_chain(table: pa.Table, ops: list[dict], be: DfBackend):
    """Replay an operator chain on a dataframe built from `table`."""
    return run_ops(be.from_arrow(table), ops, be)


def run_ops(df, ops: list[dict], be: DfBackend):
    """Replay an operator chain on a dataframe that is already on the backend.

    The entry point for a frame the caller obtained without going through Arrow — a shard the
    device read for itself. Kept as one loop shared with `run_chain` so the two ways of
    getting a frame cannot drift into two ways of executing one.

    Args:
        df: The frame to transform.
        ops: The bottom-up operator IR chain.
        be: The dataframe backend to compute on.

    Returns:
        The chain's result, as a frame on `be`.
    """
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
        out = _semi_join(left, right, join_ir, be, keep=how == "semi")
    else:
        out = _equi_join(left, right, join_ir, how, be)
    for op in ops:
        out = apply_op(out, op, be)
    return out


def _equi_join(left, right, join_ir: dict, how: str, be: DfBackend):
    lg = left.add_prefix(_LEFT)
    rg = right.add_prefix(_RIGHT)
    lkeys = [_LEFT + k for k in join_ir["left_keys"]]
    rkeys = [_RIGHT + k for k in join_ir["right_keys"]]
    # A null key matches nothing, not even another null. Both dataframe libraries' `merge`
    # matches it to itself, which invents rows an inner join must not produce and, worse,
    # pairs up two rows an outer join was supposed to report as unmatched. Adding one
    # synthetic key component fixes every join type at once, because the merge then does the
    # rest of the work itself: an inner join drops the rows, an outer keeps them unmatched.
    lg[_LEFT + _NULL_KEY] = _null_key_marker(lg, lkeys, side=0)
    rg[_RIGHT + _NULL_KEY] = _null_key_marker(rg, rkeys, side=1)
    merged = lg.merge(
        rg,
        left_on=[*lkeys, _LEFT + _NULL_KEY],
        right_on=[*rkeys, _RIGHT + _NULL_KEY],
        how=how,
    )
    cols = {}
    for o in join_ir["output"]:
        src = (_LEFT if o["side"] == "left" else _RIGHT) + o["name"]
        cols[o["alias"]] = merged[src].reset_index(drop=True)
    return be.lib.DataFrame(cols)


def _null_key_marker(frame, keys: list[str], *, side: int):
    """A synthetic key component under which a null key matches nothing.

    `-1` wherever the row's key is complete, so the two sides agree there and the real keys
    decide the match. A side-specific `0` or `1` wherever any key component is null — a value
    the other side never carries on any row, so such a row cannot match a complete key, and two
    null keys cannot match each other either.

    Any component being null makes the whole key null, which is SQL's rule: a comparison with
    an unknown is unknown, and one unknown column is enough to make the row's key unknown.
    """
    missing = frame[keys[0]].isna()
    for key in keys[1:]:
        missing = missing | frame[key].isna()
    return missing.astype("int8") * (side + 1) - 1


def _semi_join(left, right, join_ir: dict, be: DfBackend, *, keep: bool):
    """A semi/anti join as a key-membership filter over the left side.

    A `merge` cannot express either one: it would duplicate a left row per matching right row
    (semi keeps one) and has no mode that keeps the non-matching rows alone (anti).

    A null left key is *not* a member, however many nulls the right side holds, because null
    equals nothing including itself. `isin` disagrees — it treats a null as an ordinary value
    and finds it — so nullness is subtracted from the membership rather than relied upon. The
    consequence of getting this wrong runs in opposite directions for the two joins, which is
    why it is worth stating: a semi join gains rows it should have dropped, and an anti join
    drops the rows that are most often the point of running one.
    """
    lkeys = join_ir["left_keys"]
    rkeys = join_ir["right_keys"]
    if len(lkeys) != 1:
        # A composite key needs a tuple-valued membership test, which neither backend offers
        # without materializing a joint key column of an inferred type.
        raise Unsupported("semi/anti join on a composite key")
    # Folded on both sides: `isin` compares by hash, so a left `0.0` would not find a right
    # `-0.0` — the same two-zeros split the group key and DISTINCT have, arriving through a
    # third door.
    probe = fold_zero(left[lkeys[0]], be)
    member = probe.isin(fold_zero(right[rkeys[0]], be))
    mask = member.fillna(False) & ~probe.isna()
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
        # The same fold DISTINCT needs: a UNION deduplicates rows, so it decides identity.
        out = distinct_rows(out, be)
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

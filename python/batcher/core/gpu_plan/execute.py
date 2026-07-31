"""Replay a matched plan on a dataframe backend — the executor behind the GPU entry points.

Every function here is written against `DfBackend`, so the same code runs on cuDF (the
accelerated backend, on a GPU worker) and on pandas (the head-runnable check against the
native CPU engine). The `execute_cudf_*` wrappers are the only places that name cuDF, and
they exist so a caller cannot accidentally take the verification backend to production.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.core.gpu_plan.backend import DfBackend
from batcher.core.gpu_plan.eligibility import JOIN_HOW
from batcher.core.gpu_plan.ops import apply_op, distinct_rows, fold_zero

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "execute_cudf_join",
    "execute_cudf_plan",
    "execute_cudf_union",
    "join_frames",
    "run_chain",
    "run_join",
    "run_join_frames",
    "run_ops",
    "run_union",
    "run_union_frames",
    "union_frames",
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
    return run_join_frames(
        be.from_arrow(left_t), be.from_arrow(right_t), left_ops, right_ops, join_ir, ops, be
    )


def run_join_frames(left, right, left_ops, right_ops, join_ir: dict, ops: list[dict], be):
    """`run_join` for inputs already on the backend — a side the device read for itself.

    The two inputs are different relations, so one may arrive from the device reader and the
    other from the host one without the mismatch that matters. Within a single shard the two
    readers' schemas have to agree because the pieces are concatenated; across the two sides of
    a join there is nothing to concatenate, and each side is only required to be itself.

    Args:
        left: The left input, already a frame on `be`.
        right: The right input, already a frame on `be`.
        left_ops: The left input chain's operator IR.
        right_ops: The right input chain's operator IR.
        join_ir: The join node's IR.
        ops: The operator chain above the join.
        be: The dataframe backend to compute on.

    Returns:
        The join's result, as a frame on `be`.
    """
    left = run_ops(left, left_ops, be)
    right = run_ops(right, right_ops, be)
    return run_ops(join_frames(left, right, join_ir, be), ops, be)


def join_frames(left, right, join_ir: dict, be: DfBackend):
    """Join two frames that are already on the backend — the join kernel itself.

    Split out from `run_join_frames` so the tree executor, whose inputs are whole sub-plans
    rather than chains over scans, reaches the same kernel instead of restating it. A second
    statement of the null-key rule or the semi/anti membership test is the one way the linear
    and tree forms could ever disagree about a join.

    Args:
        left: The left input, already a frame on `be`.
        right: The right input, already a frame on `be`.
        join_ir: The join node's IR.
        be: The dataframe backend to compute on.

    Returns:
        The joined frame, carrying the columns and order `join_ir["output"]` asks for.
    """
    how = JOIN_HOW[join_ir["join_type"]]
    if how in ("semi", "anti"):
        return _semi_join(left, right, join_ir, be, keep=how == "semi")
    return _equi_join(left, right, join_ir, how, be)


def union_frames(frames: list, distinct: bool, be: DfBackend):
    """Concatenate frames already on the backend, deduplicating when the union asks for it.

    Args:
        frames: The inputs, already frames on `be`.
        distinct: Whether the union deduplicates.
        be: The dataframe backend to compute on.

    Returns:
        The concatenated (and optionally deduplicated) frame.
    """
    out = be.concat(frames)
    if distinct:
        # The same fold DISTINCT needs: a UNION deduplicates rows, so it decides identity.
        out = distinct_rows(out, be)
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
    mask = (
        _member_of(left, right, lkeys[0], rkeys[0], be)
        if len(lkeys) == 1
        else _member_of_tuple(left, right, lkeys, rkeys, be)
    )
    out = left[mask if keep else ~mask].reset_index(drop=True)
    return out.rename(columns={o["name"]: o["alias"] for o in join_ir["output"]})[
        [o["alias"] for o in join_ir["output"]]
    ]


def _member_of(left, right, lkey: str, rkey: str, be: DfBackend):
    """Membership on a single key, as a mask over the left rows.

    Folded on both sides: `isin` compares by hash, so a left `0.0` would not find a right
    `-0.0` — the same two-zeros split the group key and DISTINCT have, arriving through a third
    door. Nullness is then subtracted, because `isin` treats a null as an ordinary value and
    finds it.
    """
    probe = fold_zero(left[lkey], be)
    return probe.isin(fold_zero(right[rkey], be)).fillna(False) & ~probe.isna()


def _member_of_tuple(left, right, lkeys: list[str], rkeys: list[str], be: DfBackend):
    """Membership on a composite key, as a mask over the left rows.

    `isin` tests one column, so a multi-column key needs a different mechanism: merge the left
    key tuples against the *deduplicated* right ones and ask which found a partner. Deduplicating
    is what makes this a membership test rather than a join — without it a left row would come
    back once per matching right row, which is the fan-out a semi join exists to avoid.

    The left row's position is carried through the merge and sorted back afterwards. A merge
    does not promise to preserve the left frame's order, and on the host backend it happens to,
    which is the shape of bug that passes every test here and reorders on the device.

    A star-schema anti-join on `(date, store)` is the reason this is worth having at all: the
    whole plan used to go to the CPU engine over the key having two columns.
    """
    pos = "__bt_pos"
    probe = be.lib.DataFrame({f"__bt_k{i}": fold_zero(left[k], be) for i, k in enumerate(lkeys)})
    key_names = list(probe.columns)
    probe[_NULL_KEY] = _null_key_marker(probe, key_names, side=0)
    probe[pos] = range(len(left))
    keys = be.lib.DataFrame({f"__bt_k{i}": fold_zero(right[k], be) for i, k in enumerate(rkeys)})
    keys[_NULL_KEY] = _null_key_marker(keys, key_names, side=1)
    present = "__bt_present"
    keys = keys.drop_duplicates()
    keys[present] = 1
    merged = probe.merge(keys, on=[*key_names, _NULL_KEY], how="left").sort_values(pos)
    # `notna` rather than a filled boolean: an unmatched row's marker is missing, which *is*
    # the answer, and filling it first would ask the library to pick a dtype for a column that
    # only ever holds one value and a hole.
    return merged[present].notna().reset_index(drop=True)


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
    return run_union_frames([be.from_arrow(t) for t in tables], input_ops, distinct, ops, be)


def run_union_frames(frames: list, input_ops: list[list[dict]], distinct: bool, ops, be):
    """`run_union` for inputs already on the backend — inputs the device read for itself.

    A union *does* concatenate its inputs, so the schemas here must agree — but that is a
    property of the relations being unioned, which the plan already required, not of which
    reader produced them. Each input is separately either device-readable or not, and the
    device reader declines rather than approximating, so a mixed set still concatenates.

    Args:
        frames: The inputs, already frames on `be`.
        input_ops: Each input chain's operator IR, positionally matching `frames`.
        distinct: Whether the union deduplicates.
        ops: The operator chain above the union.
        be: The dataframe backend to compute on.

    Returns:
        The union's result, as a frame on `be`.
    """
    reduced = [run_ops(f, o, be) for f, o in zip(frames, input_ops, strict=True)]
    return run_ops(union_frames(reduced, distinct, be), ops, be)


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

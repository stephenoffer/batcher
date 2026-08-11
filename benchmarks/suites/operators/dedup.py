"""Operator-mix: deduplication over TPC-H ``lineitem`` — whole-row and keyed.

Two operators wear one name. **Whole-row** `DISTINCT` collapses rows agreeing on every
selected column and carries no payload; **keyed** dedup (`unique(subset=…)`, SQL's `DISTINCT ON`)
keeps one whole row per key and carries every other column with it. They have different
costs, different scaling, and different weak spots, so both are measured — and each at both
ends of the cardinality range, which is the axis the engines' path choices actually turn on.

The keyed cases are the ones that matter most in practice (one row per order, per user, per
document) and the ones this suite existed without: the dedup family had no benchmark at all,
so the operator that most workloads spend real time in was the one nothing measured.

Cardinality is chosen from `lineitem`'s own columns rather than a synthetic key:
`l_returnflag` has 3 values, `l_shipmode` 7, and `l_orderkey` roughly a quarter of the rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from registry import suite

from .base import cannot_run, sql_fanout, with_native

if TYPE_CHECKING:
    from context import Context

dedup = suite("ops-dedup", dataset="operators")

# The keyed-ordered dedup is spelled as an ordered window (`row_number() OVER (PARTITION BY
# … ORDER BY …)`), so Daft meets exactly the failure `ops-window` already documents: it is
# SIGKILLed on the 6M-row `lineitem` rather than raising, and the kill takes the *runner*
# with it. Measured here alone in a fresh process (exit 137) while Batcher answers in 153 ms
# and DuckDB in 100 ms; inside the suite it truncated the `operators` run at case 7 of 18,
# so eleven cases and five working engines reported nothing at all.
_DAFT_ORDERED_WINDOW_OOM = (
    "OOM: daft is SIGKILLed on an ordered 6M-row window (the row_number() this case is "
    "spelled with); the kill takes the runner, not the query"
)


@dedup.case("op-distinct-low-card")
def distinct_low_card(ctx: Context):
    """`DISTINCT l_returnflag, l_linestatus` — whole-row dedup down to a handful of rows.

    The shape where a per-morsel pre-reduction pays for itself many times over: each morsel
    collapses to at most six rows, so whatever merges them afterwards has nothing to do.
    """
    sql = "SELECT DISTINCT l_returnflag, l_linestatus FROM lineitem"

    def pyarrow(t: pa.Table) -> pa.Table:
        cols = ["l_returnflag", "l_linestatus"]
        return t.select(cols).group_by(cols).aggregate([])

    def ray(rd) -> pa.Table:
        cols = ["l_returnflag", "l_linestatus"]
        df = rd.select_columns(cols).groupby(cols).count().to_pandas()
        return pa.Table.from_pandas(df[cols], preserve_index=False)

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@dedup.case("op-distinct-high-card")
def distinct_high_card(ctx: Context):
    """`DISTINCT l_orderkey` — whole-row dedup where the pre-reduction buys nothing.

    A near-unique key, so a per-morsel partial reduces almost no rows yet still hashes every
    one of them. This is the case a single-pass hash-partitioned dedup exists for, and the
    one where paying for both passes shows up as a straight doubling.
    """
    sql = "SELECT DISTINCT l_orderkey FROM lineitem"

    def pyarrow(t: pa.Table) -> pa.Table:
        return t.select(["l_orderkey"]).group_by(["l_orderkey"]).aggregate([])

    def ray(rd) -> pa.Table:
        df = rd.select_columns(["l_orderkey"]).groupby("l_orderkey").count().to_pandas()
        return pa.Table.from_pandas(df[["l_orderkey"]], preserve_index=False)

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@dedup.case("op-distinct-limit")
def distinct_limit(ctx: Context):
    """`DISTINCT l_orderkey` under a `LIMIT` — the dedup that stops once `k` groups exist.

    The one *asymptotic* gap the competitor review found, from the other side: without the
    fusion a `DISTINCT` under a `LIMIT` consumes its whole input before the limit throws
    nearly all of it away, so the work is proportional to the **input** to answer a question
    about `k` rows. `RelOp::Distinct` carries the limit, and the dedup stops as soon as it
    holds `k` of them.

    The key is near-unique on purpose: `k` distinct values then turn up in the first morsel,
    which is where the early exit is worth the most. The low-cardinality counterpart is
    `op-distinct-low-card` above, where the exit cannot fire (there are fewer than `k`
    distinct values in the whole column) and the dense direct-map path is what wins instead.

    **Why the count wrapper.** `SELECT DISTINCT g … LIMIT 5` does not say *which* five rows,
    and the engines genuinely disagree: Batcher keeps the first five in input order, because
    invariant #7 requires one node and many to agree, while DuckDB keeps whichever five its
    threads reach first. Comparing the rows would gate a correctness failure on a choice SQL
    leaves open, so the case counts them instead — deterministic at five, and every engine
    still has to do the same limited dedup to answer it.
    """
    # The derived table is aliased because Polars' SQL requires it; DuckDB and Batcher
    # accept it either way, so the alias costs nothing and keeps a third engine in the row.
    sql = "SELECT count(*) AS c FROM (SELECT DISTINCT l_orderkey FROM lineitem LIMIT 5) AS d"

    def pyarrow(t: pa.Table) -> pa.Table:
        distinct = t.select(["l_orderkey"]).group_by(["l_orderkey"]).aggregate([])
        return pa.table({"c": pa.array([distinct.slice(0, 5).num_rows], type=pa.int64())})

    def ray(rd) -> pa.Table:
        df = rd.select_columns(["l_orderkey"]).groupby("l_orderkey").count().limit(5).to_pandas()
        return pa.table({"c": pa.array([len(df)], type=pa.int64())})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


#: The payload a keyed dedup carries, plus the key and the ordering column. Held to a handful
#: of columns rather than `lineitem`'s sixteen so the case measures the dedup and not the scan.
_KEYED_COLS = ["l_orderkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_shipmode"]


@dedup.case("op-dedup-keyed-ordered")
def dedup_keyed_ordered(ctx: Context):
    """One row per `l_orderkey`, the one with the smallest `l_linenumber`, payload carried.

    The canonical real dedup: collapse an event log to its earliest row per key. Expressed in
    SQL as `QUALIFY row_number() = 1`, which is what an engine without a dedup operator has to
    do — a full sort of every partition and a rank column over the whole relation, to select
    one row from each. An engine that recognizes the shape as a per-key minimum does one hash
    pass instead, and the gap between those two is what this case measures.
    """
    cols = ", ".join(_KEYED_COLS)
    # The derived table is aliased (`) AS ranked`) because standard SQL requires it and
    # Polars enforces it — without the alias Polars answers `SQLSyntaxError: derived tables
    # must have aliases` and drops out of the comparison entirely, which reads as "Polars
    # has no result" when in fact it was never asked a question it could parse.
    sql = (
        f"SELECT {cols} FROM ("
        f"  SELECT {cols}, row_number() OVER ("
        "     PARTITION BY l_orderkey ORDER BY l_linenumber"
        "  ) AS rn FROM lineitem"
        ") AS ranked WHERE rn = 1"
    )

    def pyarrow(t: pa.Table) -> pa.Table:
        # Acero has no per-partition ranking, so sort the whole table by (key, order) and take
        # each key's first row — the same answer, and the only shape available here.
        import pyarrow.compute as pc

        sub = t.select(_KEYED_COLS)
        order = pc.sort_indices(
            sub, sort_keys=[("l_orderkey", "ascending"), ("l_linenumber", "ascending")]
        )
        sorted_sub = sub.take(order)
        keys = sorted_sub.column("l_orderkey").to_numpy(zero_copy_only=False)
        first = [0, *(i for i in range(1, len(keys)) if keys[i] != keys[i - 1])]
        return sorted_sub.take(pa.array(first, pa.int64()))

    fns = cannot_run(sql_fanout(ctx, sql), "daft", _DAFT_ORDERED_WINDOW_OOM)
    return with_native(ctx, fns, pyarrow=pyarrow)


@dedup.case("op-dedup-keyed-unordered")
def dedup_keyed_unordered(ctx: Context):
    """One row per `(l_orderkey, l_linenumber)` with no ordering — the unordered dedup path.

    `keep="any"` is the default and the cheapest path: with nothing to compare, the engine
    keeps whichever row it saw first per key. That also makes the surviving row unspecified,
    so a case keyed on a non-unique column would have no oracle — the engines would disagree
    on the payload and the harness would (rightly) refuse to time it.

    Keying on `lineitem`'s **primary key** removes the ambiguity without removing the work:
    `(l_orderkey, l_linenumber)` occurs once per row, so every engine must return every row
    and the answer is exact, while the operator still hashes and gathers the entire relation.
    That is the unordered path at its worst case, which is the number worth having.

    The uniqueness is an assumption about the data, and it is load-bearing — `DISTINCT ON (pk)`
    and `SELECT DISTINCT` agree only because of it. It is not silent: were the generator ever
    to emit a duplicate pair, the two spellings would return different row counts and the
    harness's correctness gate would fail the case rather than time a mismatched comparison.
    """
    cols = ", ".join(_KEYED_COLS)
    key = "l_orderkey, l_linenumber"
    fns = {}
    for name, run in ctx.sql_runners().items():
        # `DISTINCT ON` is the keyed operator spelled directly; Batcher and DuckDB take it.
        # For the rest, `SELECT DISTINCT` over the same columns returns the identical rows
        # *because the key is unique*, so the comparison stays exact while each engine runs
        # the dedup it actually has.
        query = (
            f"SELECT DISTINCT ON ({key}) {cols} FROM lineitem"
            if name in ("batcher", "duckdb")
            else f"SELECT DISTINCT {cols} FROM lineitem"
        )
        fns[name] = lambda run=run, query=query: run(query)

    def pyarrow(t: pa.Table) -> pa.Table:
        return t.select(_KEYED_COLS).group_by(_KEYED_COLS).aggregate([])

    return with_native(ctx, fns, pyarrow=pyarrow)

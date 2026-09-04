"""Distributed equivalence and timing benchmark.

Exercises the mergeable execution path (partial / combine / finalize over a hash
shuffle) by running each query two ways: single-node and across several
partitions via ``collect(distributed=True, num_partitions=...)``. The mergeable
algebra guarantees the two results are identical, so this benchmark asserts that
equivalence first and only then reports timings. It is the single-node == many-
partition invariant from the engine contract, measured.

Run:
    source .venv/bin/activate
    python3 benchmarks/run.py --benchmark distributed          # TPC-H scale 1, 8 partitions
    python3 benchmarks/internals/distributed.py                # the same, as a script
    python3 benchmarks/internals/distributed.py 10 16          # scale 10, 16 partitions

Reads the public TPC-H tables (``sources/`` — no data is generated). Requires the
optional ``ray`` extra; without it the benchmark exits cleanly with a skip message.
"""

from __future__ import annotations

import os
import sys
import time

# Executed as a script, sys.path[0] is this package dir; the harness core it shares with
# every other benchmark (context / engines / harness) lives one level up. Same bootstrap
# as ``benchmarks/iso/run.py``. No-op when imported through ``run.py --benchmark``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import batcher as bt
import engines as engines_mod
from batcher import col, count, row_number
from context import Context
from envinfo import machine_fingerprint, require_quiet_box, require_release_build
from harness import bench, order_violation, results_match

try:
    import ray  # noqa: F401

    HAVE_RAY = True
except ImportError:
    HAVE_RAY = False

DEFAULT_SCALE = 1.0


def build_plans(ctx: Context):
    """Return (name, plan, ordered_by) triples; each plan is a non-collected Dataset.

    These mirror the mergeable operators the distributed path composes from
    ``partial / combine / finalize``, over the real TPC-H ``lineitem`` ⋈ ``orders``
    tables. ``.claude/rules/rust-engine.md`` names five that MUST be mergeable —
    aggregation, join, distinct, window, top-N — and until now this benchmark built only
    the first three.

    The two that were missing are the two worth having most. ``CLAUDE.md``'s list of
    things that "passed every gate while being wrong" opens with ``sort(descending=True)``
    returning unsorted data under spill, which is precisely a sort whose distributed form
    diverged; and a window function's frame is the hardest thing in the set to keep
    correct across a shuffle. A benchmark that exists to measure the mergeable-algebra
    invariant should not have been silent on both.

    ``ordered_by`` is the third element because the comparison this feeds
    (:func:`harness.results_match`) sorts both sides. Adding a sort plan without it would
    add a case that *cannot fail*: the checker would sort away the very property the plan
    was added to test. See :func:`_order_note`.

    An empty ``ordered_by`` is therefore a claim in its own right — that this plan's output
    order is not promised — and it leaves the plan policed by the multiset check alone. Where
    that is genuinely weaker than the query's guarantee, the fix is to put a **unique key** in
    the projection rather than to assert an order the engine does not owe: a full-row multiset
    over a unique key is a row-for-row comparison. ``window-rank`` is the case, and says so.
    """
    lineitem = ctx.handle("lineitem", "batcher")
    # orders keyed to lineitem so the equi-join is on a shared column name.
    orders = ctx.handle("orders", "batcher").rename({"o_orderkey": "l_orderkey"})
    return [
        (
            "groupby-agg",
            lineitem.group_by("l_returnflag").agg(s=col("l_extendedprice").sum(), n=count()),
            (),
        ),
        (
            "groupby-2key",
            lineitem.group_by("l_returnflag", "l_linestatus").agg(
                s=col("l_quantity").sum(), n=count()
            ),
            (),
        ),
        (
            "join+groupby",
            lineitem.join(orders, on="l_orderkey", how="inner")
            .group_by("o_orderpriority")
            .agg(s=col("l_extendedprice").sum(), n=count()),
            (),
        ),
        ("distinct", lineitem.select("l_returnflag", "l_linestatus").distinct(), ()),
        # Full sort on one fixed-width key. Descending on purpose: the recorded silent
        # failure was `sort(descending=True)` specifically, and a partitioned sort has to
        # get the range-partition boundaries the right way round to answer it.
        (
            "sort-desc",
            lineitem.select("l_orderkey", "l_extendedprice").sort(
                "l_extendedprice", "l_orderkey", descending=True
            ),
            (("l_extendedprice", True), ("l_orderkey", True)),
        ),
        # Top-N. A distributed top-N must take a local top-N per partition and merge, so it
        # can be right about *which* rows survive and wrong about their order, or the
        # reverse — the multiset check catches the first and only the order check the second.
        (
            "top-n",
            lineitem.select("l_orderkey", "l_linenumber", "l_extendedprice")
            .sort("l_extendedprice", "l_orderkey", "l_linenumber", descending=[True, False, False])
            .limit(100),
            (("l_extendedprice", True), ("l_orderkey", False), ("l_linenumber", False)),
        ),
        # Window. The frame is what a shuffle can break: rows of one partition key must all
        # land together and be ordered within the partition before the frame is evaluated.
        #
        # ``ordered_by`` stays empty, and deliberately: a window promises nothing about its
        # *output* order, so asserting one would be inventing a contract the engine does not
        # owe. That leaves the plan policed by the multiset comparison alone — which is
        # complete **here**, and the reason is worth stating because it is not general.
        # ``rank`` is a pure function of the partition key and the order key, and both are in
        # the projection: rows agreeing on them get the same rank by definition, so no rank
        # can be attached to the wrong row without changing a projected value. The generic
        # weakness of a multiset check — two errors cancelling, one row carrying the value
        # another should have had — has nothing to grip on.
        (
            "window-rank",
            lineitem.select("l_returnflag", "l_orderkey", "l_extendedprice").with_columns(
                r=col("l_extendedprice")
                .rank()
                .over(partition_by="l_returnflag", order_by="l_extendedprice")
            ),
            (),
        ),
        # The window shape where the multiset check is **not** complete on its own, and the
        # one the ordered-bucket-offset algebra actually exercises.
        #
        # ``row_number`` is not a function of the partition and order keys: it has to *break*
        # ties, and which tied row gets which number is undefined in SQL — so single-node and
        # distributed legitimately differ there (`.claude/rules/python-control-plane.md`
        # states it as one of the two exceptions to "distributed == single-node"). Two things
        # follow, and the plan needs both:
        #
        # * the ORDER BY is made fully determining (``l_extendedprice``, then the primary key),
        #   so there is no tie left to break and the two paths must agree exactly. Without
        #   this the case would fail for a reason that is not a defect;
        # * ``l_linenumber`` joins ``l_orderkey`` in the projection, making every row unique,
        #   so the full-row multiset :func:`harness.results_match` builds *is* a row-for-row
        #   comparison. That is what catches a number attached to the wrong row — the failure
        #   a multiset alone lets cancel. No ordering is claimed and no `rid` is minted; the
        #   table's own primary key does it.
        #
        # A running ``sum`` rides along because it is the other half of the algebra: the rank
        # functions are offset by a row *count* and the folds by an accumulated *value*, and
        # only one of those two arithmetics was being exercised.
        (
            "window-rownum-runsum",
            lineitem.select(
                "l_returnflag", "l_orderkey", "l_linenumber", "l_extendedprice"
            ).with_columns(
                n=row_number().over(
                    partition_by="l_returnflag",
                    order_by=["l_extendedprice", "l_orderkey", "l_linenumber"],
                ),
                s=col("l_extendedprice")
                .sum()
                .over(
                    partition_by="l_returnflag",
                    order_by=["l_extendedprice", "l_orderkey", "l_linenumber"],
                ),
            ),
            (),
        ),
    ]


def _order_note(table, ordered_by) -> str | None:
    """``None`` when ``table`` is ordered as ``ordered_by`` says it must be, else why not.

    :func:`harness.results_match` compares row *multisets* — it sorts both sides — so it is
    structurally unable to see an ``ORDER BY`` that did not happen. That is fine for the
    aggregate and join plans above, whose results are unordered sets. It is the whole point
    for ``sort-desc`` and ``top-n``: without this, a distributed sort that returned the
    right rows in the wrong order would be reported *identical* to the single-node result,
    and the benchmark that exists to prove the mergeable-algebra invariant would be
    certifying the one operator with a recorded history of breaking silently.
    """
    if not ordered_by:
        return None
    return order_violation(table, list(ordered_by))


def run(scale: float = DEFAULT_SCALE, num_partitions: int = 8, runs: int = 3) -> int:
    """Run the single-node vs many-partition equivalence + timing benchmark."""
    if not HAVE_RAY:
        print("ray is not installed; skipping distributed benchmark.")
        print("Install the optional extra with:  uv pip install -e '.[ray]'")
        return 0

    print(f"Batcher distributed benchmark  (engine {bt.engine_version()})")
    print(f"TPC-H scale = {scale}, num_partitions = {num_partitions}, best-of-{runs}\n")

    t0 = time.perf_counter()
    ctx = Context.build("tpch", scale, engines_mod.resolve(["batcher"]))
    print(f"loaded data in {time.perf_counter() - t0:.2f}s\n")

    rows = []
    any_mismatch = False
    for name, plan, ordered_by in build_plans(ctx):
        print(f"running {name} ...", flush=True)
        single = plan.collect()
        dist = plan.collect(distributed=True, num_partitions=num_partitions)
        ok, msg = results_match(single, dist)
        # Both sides are order-checked, not just the distributed one. A single-node sort
        # that came back unordered is the same defect and would otherwise be invisible
        # here, because `results_match` would happily call two identically-unsorted
        # results equal.
        for side, table in (("single-node", single), ("distributed", dist)):
            violation = _order_note(table, ordered_by)
            if violation is not None:
                ok, msg = False, f"{side} {violation}"
                break
        if not ok:
            any_mismatch = True

        def run_dist(p=plan):
            return p.collect(distributed=True, num_partitions=num_partitions)

        sn_ms = bench(plan.collect, runs=runs)
        di_ms = bench(run_dist, runs=runs)
        speedup = f"{sn_ms / di_ms:.2f}x" if di_ms else "-"
        rows.append((name, sn_ms, di_ms, speedup, "OK" if ok else f"MISMATCH: {msg}"))

    print()
    headers = ["query", "single_ms", "dist_ms", "single/dist", "status"]
    widths = [len(h) for h in headers]
    table = [[n, f"{s:.1f}", f"{d:.1f}", sp, st] for (n, s, d, sp, st) in rows]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        return "  ".join(
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells)
        )

    print(fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in table:
        print(fmt(row))

    print()
    if any_mismatch:
        print("Distributed result diverged from single-node. This is a correctness bug.")
        return 1
    print("Distributed results match single-node on every query.")
    return 0


def main() -> int:
    # Refuse to time a dev-profile engine: it is 8-60x slower, so a number taken from one
    # compares an unoptimized Batcher against release competitors. `BENCH_ALLOW_DEBUG_BUILD=1`
    # overrides deliberately.
    require_release_build()
    # Print the machine before any number: a timing is only reproducible beside the
    # box that produced it, and this file's own history has ratios quoted across four
    # different machines as if they were comparable.
    print(machine_fingerprint())
    # ...and refuse a contended one: a neighbour's load is not a fact about any
    # engine. `BENCH_ALLOW_BUSY_BOX=1` overrides.
    require_quiet_box()
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCALE
    num_partitions = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    return run(scale, num_partitions)


if __name__ == "__main__":
    raise SystemExit(main())

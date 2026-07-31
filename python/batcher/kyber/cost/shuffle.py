"""The `net` axis — what a plan costs to move across a cluster.

`Cost` has carried a `net` field and `CostWeights` a `net` weight (2.0 — a shuffled byte
costs twice a local one) since the cost model existed, and **nothing ever wrote to it**.
Every plan Kyber has ever ranked was ranked as though the network were free.

On one machine that is exactly right, and this module returns zero for it. On a cluster it
is the term that decides the plan. Two join orders with identical CPU can differ by a
factor of the fact table in bytes on the wire; an aggregate whose input is already
co-partitioned on its group keys needs no shuffle at all while its sibling needs a full
one; and a build side small enough to replicate turns an all-to-all into a one-to-many.
None of those distinctions were visible to the enumerator, so it could not prefer the
cheaper side of any of them except by accident through the row counts.

## The two terms, and why the second one exists

**Volume.** Bytes on the wire, `rows x width`, discounted by the share that never leaves
the node it started on. A hash repartition across `W` workers sends each row to one of `W`
buckets and one of those buckets is local, so `1/W` of the data stays home. At `W = 2` half
the shuffle is free and at `W = 10,000` essentially none of it is — which is the honest
shape, and it is why a shuffle that is nearly free on a four-node cluster is not.

**Fan-out.** A shuffle between `P` producers and `R` reducers creates `P x R` fragments,
and that product is what actually stops a shuffle-heavy plan from reaching ten thousand
nodes: at `P = R = 10,000` it is a hundred million fragments to open, track, and drain,
each carrying its own Arrow IPC framing, before a single useful byte moves. A model with
only the volume term says a plan with three shuffle stages costs three times one with a
single stage; the real ratio at fleet scale is far worse, and the quadratic term is what
makes the optimizer prefer a plan that shuffles once over one that shuffles three times
even when the bytes are equal.

This is also precisely why a **broadcast** join is not just "a cheaper shuffle": it is
`W` fragments rather than `W^2`, so its advantage grows with the square of the cluster.

## What this deliberately does not know

Kyber sees `HardwareProfile.worker_count`, not the partition count `dist` will finally
choose (`SchedulingEnvelope.n_tasks`), and it must not — the scheduling grant is derived
*from* the annotated plan, so reading it here would invert the dependency. Worker count is
the right proxy: the two track each other, and the term only has to *rank* plans.
"""

from __future__ import annotations

from batcher.kyber.properties import hash_partitioned_on
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Join,
    Limit,
    LogicalPlan,
    Sort,
    Union,
    Window,
)

__all__ = ["fanout_bytes", "net_cost", "shuffle_bytes"]

# Wire bytes charged per shuffle fragment, as a floor.
#
# Every (producer, reducer) pair exchanges at least one Arrow IPC message: an 8-byte
# continuation marker, a 4-byte metadata length, and a flatbuffer `Message` describing the
# schema and the batch, padded to an 8-byte boundary. For a modest schema that flatbuffer
# alone runs to a couple of hundred bytes, so 256 is a floor rather than an estimate.
#
# It stands in for more than framing. The fragment also costs a connection to establish, a
# credit window to negotiate (`bc-transport`), and an entry in the driver's tracking table.
# Those are not bytes, and pricing them in bytes is an encoding, not a measurement — stated
# plainly here rather than buried, because the term's *shape* (quadratic in the fleet) is
# what matters to the ranking and its constant only sets where the knee falls.
_FRAGMENT_BYTES = 256.0


def shuffle_bytes(rows: float, row_bytes: float, workers: int) -> float:
    """Bytes on the wire when `rows` rows are hash-repartitioned across `workers`.

    Discounted by the `1/W` share that hashes to the bucket already on its own node, so a
    two-worker shuffle moves half its data and a ten-thousand-worker shuffle moves all of it.

    Args:
        rows: Rows entering the shuffle.
        row_bytes: Average bytes per row.
        workers: Workers the data is repartitioned across.

    Returns:
        Bytes crossing the network, `0.0` on a single worker.
    """
    if workers <= 1 or rows <= 0.0 or row_bytes <= 0.0:
        return 0.0
    return rows * row_bytes * (1.0 - 1.0 / workers)


def fanout_bytes(producers: int, reducers: int) -> float:
    """Fixed wire cost of the `producers x reducers` fragments a shuffle opens.

    The term that makes an all-to-all shuffle superlinear in the fleet, and the reason a
    broadcast (`W` fragments) beats a repartition (`W^2`) by a margin that grows with the
    cluster rather than staying constant.

    Args:
        producers: Tasks writing into the shuffle.
        reducers: Tasks reading out of it.

    Returns:
        Bytes charged for fragment framing and setup, `0.0` for a local exchange.
    """
    if producers <= 1 and reducers <= 1:
        return 0.0
    return _FRAGMENT_BYTES * max(1, producers) * max(1, reducers)


def _already_partitioned_on(node: LogicalPlan, keys: tuple[str, ...]) -> bool:
    """Whether `node` already delivers a partitioning that makes a shuffle on `keys` free.

    The containment runs the same direction `properties.satisfies` documents: rows
    partitioned by `hash(a)` keep every `(a, b)` group whole, so a *subset* of the required
    keys suffices and a superset does not. Answering "no" costs at most an over-charged
    shuffle in the ranking; answering "yes" wrongly would under-charge a plan that does have
    to move its data, so the empty (unclaimed) partitioning is treated as no guarantee.
    """
    delivered = hash_partitioned_on(node)
    return bool(delivered) and bool(keys) and set(delivered) <= set(keys)


def _col_names(items) -> tuple[str, ...]:
    """The bare-column names among `items`, or `()` if any entry is a computed expression.

    A partitioning can only be claimed on columns a consumer can name. One computed key and
    the whole key set is unusable, which is the conservative direction.
    """
    names = [i.alias for i in items if isinstance(i.expr, Col)]
    return tuple(names) if len(names) == len(items) else ()


def net_cost(
    node: LogicalPlan,
    rows_of,
    row_bytes_of,
    workers: int,
    locality: float = 1.0,
    stats_of=None,
) -> float:
    """Bytes `node` moves across the network, excluding its inputs.

    Zero for every map-only operator (`Scan`, `Filter`, `Project`, `MapBatches`, `Unnest`)
    and zero everywhere on a single worker, so a single-node plan is ranked exactly as it
    was before this axis existed.

    Args:
        node: The plan node being priced.
        rows_of: Callable returning a node's estimated output row count.
        row_bytes_of: Callable returning a node's estimated average row width in bytes.
        workers: Workers the plan will run across; `1` for single-node.
        locality: What a byte of this exchange costs relative to a cross-rack byte, from
            `cost.locality.locality_factor`. `1.0` — the default, and what an unreadable fleet
            yields — charges every byte at the network rate, which is exactly what this axis
            charged before the tier model existed. Applied uniformly to volume *and* fan-out,
            because a fragment that never leaves its host opens no connection and negotiates no
            credit window either.
        stats_of: Callable returning a node's `RelStats`, or `None` to skip the straggler
            term. Only a partitioned window reads it — the one shuffling shape whose skew the
            engine can neither pre-reduce away nor salt (see `cost.imbalance`). `None`, and an
            unmeasured partition column, both leave the cost exactly as it was.

    Returns:
        Estimated cross-rack-equivalent bytes on the `net` axis.
    """
    if workers <= 1:
        return 0.0
    flat = _flat_net_cost(node, rows_of, row_bytes_of, workers)
    return flat * max(0.0, locality) * _straggler(node, stats_of, workers)


def _straggler(node: LogicalPlan, stats_of, workers: int) -> float:
    """How much longer this exchange takes than a balanced one, `1.0` when it is balanced.

    Applied only to a partitioned window. An aggregate and a distinct pre-reduce their hot key
    to one partial row per worker before shuffling, and a join's hot values are salted across
    reducers by `dist`; charging either for skew would penalize the mechanism that removes it.
    A window can do neither — its frame spans a whole partition, so the hot partition lands
    whole on one worker and the stage waits for it.
    """
    if stats_of is None or not isinstance(node, Window) or not node.partition_keys:
        return 1.0
    keys = tuple(k.name for k in node.partition_keys if isinstance(k, Col))
    if len(keys) != len(node.partition_keys):
        return 1.0
    try:
        from batcher.kyber.cost.imbalance import partition_imbalance

        stats = stats_of(node.input)
        return partition_imbalance([dict(stats.column(k).mcv or {}) for k in keys], workers)
    except Exception as exc:  # pragma: no cover - cost must never break a plan
        from batcher._internal.logging import note_suppressed

        note_suppressed("kyber", "price a window's partition skew", exc)
        return 1.0


def _flat_net_cost(node: LogicalPlan, rows_of, row_bytes_of, workers: int) -> float:
    """`net_cost` before the locality re-pricing — every byte charged at the network rate."""
    if isinstance(node, Aggregate):
        return _aggregate_net(node, rows_of, row_bytes_of, workers)
    if isinstance(node, Distinct):
        return _distinct_net(node, rows_of, row_bytes_of, workers)
    if isinstance(node, Join):
        return _join_net(node, rows_of, row_bytes_of, workers)
    if isinstance(node, Sort):
        return _sort_net(node, rows_of, row_bytes_of, workers)
    if isinstance(node, Window):
        return _window_net(node, rows_of, row_bytes_of, workers)
    if isinstance(node, Union) and node.distinct:
        rows = sum(rows_of(i) for i in node.inputs)
        return shuffle_bytes(rows, row_bytes_of(node), workers) + fanout_bytes(workers, workers)
    if isinstance(node, Limit):
        # A global limit gathers `k` rows from every worker to one place and re-applies the
        # limit there. Tiny in bytes, but it is a real `W -> 1` exchange and costing it at
        # exactly zero is what would let a plan stack limits for free.
        return shuffle_bytes(rows_of(node) * workers, row_bytes_of(node), workers) + fanout_bytes(
            workers, 1
        )
    return 0.0


def _aggregate_net(node: Aggregate, rows_of, row_bytes_of, workers: int) -> float:
    """Bytes a hash aggregate shuffles, after its partial pre-aggregation.

    The mergeable form is what makes this small: each worker aggregates locally first, so
    what crosses the wire is its *partial* state, not its input. A worker can emit no more
    than its share of the input rows and no more than the global group count, so the total
    partial volume is `min(in_rows, workers x groups)` — which is why a two-group aggregate
    over a trillion rows shuffles two rows per worker and a group-per-row aggregate shuffles
    everything. Costing it at the input volume, as a model without `partial -> combine` would,
    over-charges the first case by the whole relation.
    """
    keys = _col_names(node.group_keys)
    if keys and _already_partitioned_on(node.input, keys):
        return 0.0
    groups = max(1.0, rows_of(node))
    partial_rows = min(rows_of(node.input), groups * workers)
    return shuffle_bytes(partial_rows, row_bytes_of(node), workers) + fanout_bytes(
        workers, workers if keys else 1
    )


def _distinct_net(node: Distinct, rows_of, row_bytes_of, workers: int) -> float:
    """A `Distinct`'s shuffle: the same partial-then-combine shape an aggregate has.

    Distinct is an aggregate with every output column as a group key and no measures, so it
    pre-reduces to its local distinct set before shuffling and is bounded the same way.
    """
    keys = tuple(node.available_columns() or ())
    if keys and _already_partitioned_on(node.input, keys):
        return 0.0
    partial_rows = min(rows_of(node.input), max(1.0, rows_of(node)) * workers)
    return shuffle_bytes(partial_rows, row_bytes_of(node), workers) + fanout_bytes(workers, workers)


def _join_net(node: Join, rows_of, row_bytes_of, workers: int) -> float:
    """Bytes a join moves, at whichever of its two strategies is cheaper.

    A **broadcast** replicates the build side to every worker: `W - 1` copies of the build
    bytes and `W` fragments, with the probe side never moving. A **repartition** co-locates
    both sides by the join key: both sides on the wire and `W^2` fragments. The engine picks
    between them (`rules/selection.py`, against `broadcast_max_bytes`), so the cost model
    charges the minimum — the same reasoning `join_op_cost` already applies to build-side
    orientation, and for the same reason: ranking an order by a strategy the physical plan
    will not use ranks it by a cost nobody pays.

    A join whose inputs are *already* co-partitioned on its keys moves nothing.
    """
    keys = tuple(node.left_keys or ())
    co_partitioned = keys and _already_partitioned_on(node.left, keys)
    if co_partitioned and _already_partitioned_on(node.right, keys):
        return 0.0
    build_rows, build_width = rows_of(node.right), row_bytes_of(node.right)
    # A non-inner join may not replicate its *left* side, but the build side is the right one
    # in every orientation the planner emits, so a broadcast of the right side stays legal.
    broadcast = build_rows * build_width * (workers - 1) + fanout_bytes(1, workers)
    repartition = (
        shuffle_bytes(build_rows, build_width, workers)
        + shuffle_bytes(rows_of(node.left), row_bytes_of(node.left), workers)
        + fanout_bytes(workers, workers)
    )
    return min(broadcast, repartition)


def _sort_net(node: Sort, rows_of, row_bytes_of, workers: int) -> float:
    """Bytes a global sort moves.

    A **top-N** is the cheap case and the reason to keep the limit fused: each worker ranks
    locally and forwards only its own `k` rows, so `W x k` rows reach one place — bounded by
    the limit, not by the input. A full sort has no such bound and must range-partition the
    whole relation, so it pays the full volume plus an all-to-all fan-out.
    """
    width = row_bytes_of(node)
    rows = rows_of(node.input)
    if node.limit:
        gathered = min(rows, float(node.limit) * workers)
        return shuffle_bytes(gathered, width, workers) + fanout_bytes(workers, 1)
    return shuffle_bytes(rows, width, workers) + fanout_bytes(workers, workers)


def _window_net(node: Window, rows_of, row_bytes_of, workers: int) -> float:
    """Bytes a window function moves.

    A partitioned window shuffles by its `PARTITION BY` keys, and moves nothing when its
    input already delivers that partitioning. An *unpartitioned* window is the expensive one
    and is worth costing honestly: its frame spans the whole relation, so every row must
    reach a single worker — a `W -> 1` gather of the entire input, which is the shape that
    makes a global `row_number()` the operator that will not scale.
    """
    width = row_bytes_of(node)
    rows = rows_of(node.input)
    keys = tuple(k.name for k in node.partition_keys if isinstance(k, Col))
    if len(keys) != len(node.partition_keys):
        keys = ()
    if not keys:
        return shuffle_bytes(rows, width, workers) + fanout_bytes(workers, 1)
    if _already_partitioned_on(node.input, keys):
        return 0.0
    return shuffle_bytes(rows, width, workers) + fanout_bytes(workers, workers)

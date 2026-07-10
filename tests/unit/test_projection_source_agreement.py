"""`required_columns_per_source` must be sound for the plan it is handed.

The optimizer computes `PhysicalPlan.source_projections` from the *final* plan — the very
one the engine executes. So the rule is: a scan must supply every column the surviving
plan's expressions reference. A `Project` evaluates each of its items and a `Join` emits
each of its output columns, whether or not anything above consumes them. Pruning what is
unconsumed is `rewrite_projection`'s job, and it has already run by then.

Narrowing the analysis by a "needed columns" set instead produced two bugs, both of which
read the source one column short and died with ``unknown column`` at execution:

* `_rewrite` keeps one `Project` item when nothing downstream consumes any (a projection
  still has to emit the rows a ``COUNT(*)`` above it counts) — the analysis assumed none.
* A later rule can delete the only consumer of a projection item, leaving it dead in the
  plan but still evaluated. `Filter` elimination over an empty relation does exactly this.

Both only bit sources that honor a read projection (Parquet and friends), which is why no
in-memory test caught them.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.kyber.rules.projections import required_columns_per_source
from batcher.plan.expr_ir import Col, count
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Filter,
    Join,
    JoinOutputCol,
    Project,
    Projection,
    Scan,
)
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

_SCHEMA = SchemaRef.from_arrow(
    pa.schema([("id", pa.int64()), ("email", pa.string()), ("region", pa.string())])
)
_RIGHT = SchemaRef.from_arrow(pa.schema([("id", pa.int64()), ("seq", pa.int64())]))


def _count_over(plan):
    return Aggregate(plan, (), (AggregateSpec("n", count()),))


def test_a_projection_reads_the_columns_its_expressions_reference():
    scan = Scan(0, _SCHEMA)
    project = Project(scan, (Projection("e", Col("email").str.upper()),))
    assert required_columns_per_source(_count_over(project)) == {0: ["email"]}


def test_a_projection_item_nothing_consumes_is_still_evaluated_and_still_read():
    """The regression: a dead item is computed, so the scan must supply what it reads.

    `_rewrite` would normally have pruned `e` here. When a later rule leaves such an item
    behind, the analysis must not assume it was pruned.
    """
    scan = Scan(0, _SCHEMA)
    project = Project(
        scan,
        (Projection("id", Col("id")), Projection("e", Col("email").str.upper())),
    )
    assert required_columns_per_source(_count_over(project)) == {0: ["id", "email"]}


def test_a_filter_below_the_projection_contributes_its_columns():
    """The governed-scan and CDC shape: Project(Filter(Scan)) under a count."""
    scan = Scan(0, _SCHEMA)
    plan = Project(
        Filter(scan, Col("region") == "EU"),
        (Projection("e", Col("email").str.upper()),),
    )
    assert required_columns_per_source(_count_over(plan)) == {0: ["email", "region"]}


def test_a_join_reads_every_column_of_its_declared_output():
    """A join emits `output` regardless of what consumes it, so each side must supply it."""
    left = Project(Scan(0, _SCHEMA), (Projection("id", Col("id")),))
    right = Project(Scan(1, _RIGHT), (Projection("id", Col("id")), Projection("s", Col("seq"))))
    join = Join(
        left,
        right,
        ("id",),
        ("id",),
        "left",
        (
            JoinOutputCol("left", "id", "id"),
            JoinOutputCol("right", "s", "s"),
        ),
    )
    # `s` feeds nothing above, but the join still produces it from the right's `seq`.
    assert required_columns_per_source(Project(join, (Projection("id", Col("id")),))) == {
        0: ["id"],
        1: ["id", "seq"],
    }


def test_a_scan_that_needs_nothing_still_reads_one_column_to_preserve_cardinality():
    assert required_columns_per_source(_count_over(Scan(0, _SCHEMA))) == {0: ["id"]}


def test_columns_are_read_in_schema_order_not_reference_order():
    """The projection is applied to the source as-is, so its order is the output order."""
    scan = Scan(0, _SCHEMA)
    plan = Project(Filter(scan, Col("email") == "a"), (Projection("r", Col("region")),))
    assert required_columns_per_source(plan) == {0: ["email", "region"]}


def test_the_optimizer_still_prunes_unread_columns_end_to_end():
    """Soundness must not cost pruning: after `rewrite_projection` the reads are minimal."""
    import batcher as bt
    from batcher import kyber

    ds = (
        bt.from_pydict({"id": [1], "email": ["a"], "region": ["EU"]})
        .filter(bt.col("region") == "EU")
        .select(id=bt.col("id"), e=bt.col("email").str.upper())
    )
    opt, _, _ = kyber.optimize_full(ds._plan, sources=ds._sources, hub=None, source_stats=None)
    # `e` is consumed, `id` is consumed, and nothing reads a column beyond those + `region`.
    assert set(opt.source_projections[0]) == {"id", "email", "region"}

    counted = _count_over(ds._plan)
    opt2, _, _ = kyber.optimize_full(counted, sources=ds._sources, hub=None, source_stats=None)
    # A COUNT(*) consumes no projected column, so `e` — and with it `email` — is pruned
    # away entirely. What remains is the filter's `region` plus the one item the pruned
    # projection has to keep in order to emit rows to count.
    assert set(opt2.source_projections[0]) == {"id", "region"}


def test_a_source_that_reads_nothing_still_hands_the_engine_a_schema():
    """An empty read must be one zero-row batch, not zero batches.

    A batch is the only carrier of a schema across the FFI boundary, and every pipeline
    breaker needs its input's schema even over zero rows. Sources disagree on the empty
    case — an in-memory table yields a batch, a zero-row Parquet file yields none — so
    `read_source` normalizes it. Reading nothing is routine: an incremental batch with no
    changes, a table whose rows were all deleted, a partition pruned away.
    """
    import pyarrow as pa

    from batcher.io.source import read_source

    class _EmptySource:
        bounded = True

        def schema(self):
            return pa.schema([("id", pa.int64()), ("v", pa.string())])

        def read(self, projection=None):
            return []

    batches = read_source(_EmptySource())
    assert len(batches) == 1
    assert batches[0].num_rows == 0
    assert batches[0].schema.names == ["id", "v"]

    projected = read_source(_EmptySource(), ["v"])
    assert projected[0].schema.names == ["v"]

"""`runtime_join_filter`'s pruning predicate must reach the *source*, not stop above it.

The rule adds `key BETWEEN other_min AND other_max` to a join's prunable side, and its
whole payoff is stated in its own docstring: "when the prunable side is a scan, the added
`Filter` is captured by `required_predicates_per_source` at lowering and pushed to the
source, so zonemaps prune whole row-groups / Hive partitions".

"When the prunable side is a scan" was doing silent work in that sentence. The rule runs in
`Phase.ENFORCE` -- after every pushdown pass -- so it inserts at the join's input and nothing
runs afterwards to sink what it added. Kyber's *own* column pruner puts a `Project` above the
scan whenever a query reads fewer columns than the table has, which is most SQL. So on those
plans the filter landed on top of the projection, `_collect_scan_predicates` dropped its
pending predicate at that `Project`, and the dynamic partition pruning the docstring promises
silently did not happen. The DataFrame surface, which projects for itself and leaves the
pruner nothing to add, got it; `bt.sql()` did not.

The fix is in the *walk*, not the plan. An earlier version relocated the `Filter` beneath the
projection instead, which also moves the filter's gather below it -- so a non-selective
predicate materializes full-width rows the projection was about to discard. Measured at **10%
slower on TPC-H q18**, whose runtime filter is `c_custkey BETWEEN 1 AND 149999` over a
150,000-row table and removes exactly one row. Note the estimator is no defence there: it
reports that predicate at 0.19 selectivity when it keeps 99.9993% of the rows.

**No correctness test can see this.** The engine keeps the `Filter` either way, so the rows
are identical whether or not the predicate reaches the source -- only the bytes read differ.
That is why this asserts on `required_predicates_per_source` (what the source is actually
handed) rather than on a result, and why it is a unit test rather than a differential one.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

#: Wide enough that the query reads fewer columns than the table has, which is what makes
#: Kyber's column pruner insert the `Project` this test exists to sink past. With a narrow
#: table the projection is the identity, is eliminated, and the plan cannot exhibit the bug.
_PAD = 6


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> tuple[str, str]:
    d = tmp_path_factory.mktemp("sip")
    fact, dim = str(d / "fact.parquet"), str(d / "dim.parquet")
    rows = 40_000
    pq.write_table(
        pa.table(
            {
                "k": pa.array(range(rows), pa.int64()),
                "v": pa.array([1.0] * rows),
                **{f"pad{i}": pa.array([0.0] * rows) for i in range(_PAD)},
            }
        ),
        fact,
    )
    # Covers a narrow slice of the fact key range, so the derived BETWEEN is selective and
    # `_narrows` admits it. A dimension spanning the whole range makes the rule decline.
    pq.write_table(pa.table({"dk": pa.array(range(200), pa.int64())}), dim)
    return fact, dim


def _pushed(fact: str, dim: str) -> list[str]:
    """What each operator reports handing to its source, from the machine-readable profile.

    Read through `explain(format="json")` rather than the rendered tree: the tree is a
    human-facing rendering and parsing it is how a sibling test in this suite silently
    decayed into a tautology when the renderer grew a header and glyphs.

    It must also go through a real `Session`. A bare `Optimizer()` has no bound sources and
    therefore no column statistics, so `_narrows` cannot prove the dimension's range is
    tighter and `runtime_join_filter` declines outright -- against which both arms of this
    test read empty and it proves nothing. That is not hypothetical; it is what the first
    version of this file did.
    """
    sess = bt.Session()
    sess.register("fact", bt.read.parquet(fact))
    sess.register("dim", bt.read.parquet(dim))
    ds = sess.sql("SELECT sum(v) AS s FROM fact, dim WHERE k = dk")
    doc = json.loads(ds.explain(format="json"))
    return [op["pushed"] for op in doc["ops"] if op.get("pushed")]


def test_the_derived_range_is_handed_to_the_source(tables) -> None:
    """The fact source is pre-filtered by the dimension's key range."""
    fact, dim = tables
    pushed = _pushed(fact, dim)

    assert pushed, "no operator handed its source a predicate; the join filter did not land"
    assert any("k" in p and "199" in p for p in pushed), (
        f"the fact scan was not handed the dimension's key range: {pushed}"
    )


def test_a_pruning_projection_does_not_block_it(tables, monkeypatch) -> None:
    """The regression guard, and the arm that fails before the fix.

    Restores the pre-fix behaviour -- insert the filter at the join's input and leave it
    there -- and asserts the source is then handed nothing. Without this arm its sibling
    passes for any plan that happens to have no `Project` in the way, and would keep
    passing if the fix were reverted.
    """
    import batcher.kyber.rules.projections as proj

    fact, dim = tables
    # The pre-fix behaviour: the walk gives up at any `Project`, so the pending predicate
    # never reaches the scan and the source is told nothing.
    monkeypatch.setattr(proj, "_across_passthrough", lambda node, pending: None)

    assert not _pushed(fact, dim), (
        "the pre-fix arm still reached the source, so this file no longer reproduces the "
        "bug it guards and its sibling proves nothing"
    )

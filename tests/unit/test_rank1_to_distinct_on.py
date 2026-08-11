"""Plan-shape unit tests for `rank1_window_to_distinct_on`.

The rewrite turns `row_number() OVER (PARTITION BY k ORDER BY o) = 1` — which
`qualify_to_partition_topn` has already folded into `rank_limit=1` — into a `DISTINCT ON`.
That matters because `rank_limit` is applied *after* the ranking is computed, so the window
path fully sorts every partition to answer what is really a per-key argmin.

These tests assert the plan *shape* (this file) and the tie-breaking that shape can move;
`tests/differential/test_diff_rank1_distinct_on.py` holds it against DuckDB.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.fusion import rank1_window_to_distinct_on
from batcher.plan.logical import Distinct, Project, Window


def _ranked(fn: str = "row_number", *, partition_by=("k",), order_by=("v",)):
    """A window plan over a small keyed relation, before any filter."""
    return bt.from_pydict({"k": [1, 1, 2], "v": [3, 1, 2], "p": ["a", "b", "c"]}).window(
        partition_by=list(partition_by), order_by=list(order_by), functions={"r": fn}
    )


def _fused(**kwargs):
    """The `rank_limit=1` window the rewrite consumes, as `QUALIFY r = 1` produces it."""
    from batcher.kyber.rules.fusion import qualify_to_partition_topn

    plan = _ranked(**kwargs).filter(col("r") == 1)._plan
    fused = qualify_to_partition_topn(plan, None)
    assert isinstance(fused, Window) and fused.rank_limit == 1
    return fused


def test_rule_registered():
    assert "rank1_window_to_distinct_on" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_rank1_row_number_becomes_a_distinct_on():
    out = rank1_window_to_distinct_on(_fused(), None)
    assert isinstance(out, Project)
    assert isinstance(out.input, Distinct)
    assert out.input.keys == ("k",)
    assert [k.expr.name for k in out.input.order] == ["v"]


def test_the_rank_column_survives_as_the_literal_one():
    """The window appended `r`; `Distinct` does not, so the projection has to restore it.

    Every surviving row has rank 1 by construction, so the literal is exact rather than a
    stand-in — and without it the rewrite would silently narrow the output schema for
    whatever reads `r` above.
    """
    out = rank1_window_to_distinct_on(_fused(), None)
    names = [item.alias for item in out.items]
    assert names == ["k", "v", "p", "r"]
    assert bt.from_pydict({"k": [1, 1, 2], "v": [3, 1, 2], "p": ["a", "b", "c"]}).window(
        partition_by=["k"], order_by=["v"], functions={"r": "row_number"}
    ).filter(col("r") == 1).to_pydict()["r"] == [1, 1]


def test_rank_and_dense_rank_are_declined():
    """`rank = 1` keeps ties, so it can admit several rows per partition — not a DISTINCT ON."""
    for fn in ("rank", "dense_rank"):
        assert rank1_window_to_distinct_on(_fused(fn=fn), None) is None


def test_a_limit_above_one_is_declined():
    """Only the top-*1* case is a `DISTINCT ON`; `rank_limit=2` keeps two rows per key."""
    from batcher.kyber.rules.fusion import qualify_to_partition_topn

    fused = qualify_to_partition_topn(_ranked().filter(col("r") <= 2)._plan, None)
    assert fused.rank_limit == 2
    assert rank1_window_to_distinct_on(fused, None) is None


def test_an_unpartitioned_window_is_declined():
    """`Distinct` with no keys is a whole-row dedup, which is a different operator."""
    plan = (
        bt.from_pydict({"v": [3, 1, 2]})
        .window(partition_by=[], order_by=["v"], functions={"r": "row_number"})
        .filter(col("r") == 1)
        ._plan
    )
    from batcher.kyber.rules.fusion import qualify_to_partition_topn

    assert rank1_window_to_distinct_on(qualify_to_partition_topn(plan, None), None) is None


def test_a_computed_partition_key_is_declined():
    """`Distinct.keys` are column *names*, so an expression key cannot be expressed."""
    plan = (
        bt.from_pydict({"k": [1, 1, 2], "v": [3, 1, 2]})
        .with_columns(kk=col("k") + 1)
        .window(partition_by=[col("kk") * 2], order_by=["v"], functions={"r": "row_number"})
        .filter(col("r") == 1)
        ._plan
    )
    from batcher.kyber.rules.fusion import qualify_to_partition_topn

    fused = qualify_to_partition_topn(plan, None)
    assert rank1_window_to_distinct_on(fused, None) is None


def test_the_rewrite_preserves_the_result():
    """Semantics preservation, which is the point: the plan changes and the rows do not.

    The order key is unique within each partition here, so there is exactly one correct
    answer and the tie-breaking freedom the two operators have cannot hide a difference.
    """
    ds = bt.from_pydict(
        {
            "k": [1, 1, 1, 2, 2, 3],
            "v": [3, 1, 2, 9, 4, 7],
            "p": ["a", "b", "c", "d", "e", "f"],
        }
    )
    got = (
        ds.window(partition_by=["k"], order_by=["v"], functions={"r": "row_number"})
        .filter(col("r") == 1)
        .select("k", "v", "p")
        .to_pydict()
    )
    rows = sorted(zip(got["k"], got["v"], got["p"], strict=True))
    assert rows == [(1, 1, "b"), (2, 4, "e"), (3, 7, "f")]


def test_a_descending_order_key_still_picks_the_extreme():
    """`ORDER BY v DESC` selects the per-key maximum, so the direction must be carried."""
    ds = bt.from_pydict({"k": [1, 1, 2, 2], "v": [3, 1, 9, 4], "p": ["a", "b", "c", "d"]})
    got = (
        ds.window(partition_by=["k"], order_by=[("v", True)], functions={"r": "row_number"})
        .filter(col("r") == 1)
        .select("k", "v", "p")
        .to_pydict()
    )
    assert sorted(zip(got["k"], got["v"], got["p"], strict=True)) == [(1, 3, "a"), (2, 9, "c")]

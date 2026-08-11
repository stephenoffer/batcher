"""A key constant the *statistics* prove is mirrored across an inner join — results unchanged.

`infer_join_predicates` mirrors a `key OP literal` constraint it can read off the other side's
`Filter`. That misses the shape where the constant is only visible in the data: a dimension
whose key column genuinely holds one value, with no predicate in the plan saying so. Nothing
gets mirrored, and the fact-table scan reads every row group.

`infer_join_predicate_from_constant_key` closes that, sourcing the constant from
`stats.constants.constant_value` — EXACT provenance, `min == max`, no nulls — instead of from
the plan text. Databricks reach the same rewrite at runtime instead (VLDB 2024 §5.2: a
completed stage holding one row folds the join condition to a constant and pushes it into the
other scan to prune files).

The rewrite adds a predicate the join already enforces, so it must be invisible in the answer.
These tests are the proof, and the fixtures are chosen so a wrong version would *not* be
invisible: nulls on the probe key, a probe value absent from the dimension, duplicate keys on
both sides (so a mistaken semi-join-like reduction would change multiplicity), and every join
type — because the rule must fire for `inner` and decline for the rest.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same

# `dim.k` is a genuine constant (7) with no filter saying so — that is the whole point.
# `fact.k` carries a null, a non-matching value, and a duplicate of the constant.
DIM = pa.table({"k": [7, 7, 7], "label": ["a", "b", "c"]})
FACT = pa.table({"k": [7, 7, 9, None], "v": [10, 20, 30, 40]})
# A dimension that is *not* constant, so the rule must not fire and nothing may change.
DIM2 = pa.table({"k": [7, 8], "label": ["a", "b"]})


@pytest.fixture
def star(duck):
    duck.register("dim", DIM)
    duck.register("fact", FACT)
    duck.register("dim2", DIM2)


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        # The shape the rule exists for: `fact.k = 7` is derivable only from dim's data.
        "SELECT f.k, f.v, d.label FROM fact f JOIN dim d ON f.k = d.k",
        # Mirrored the other way — the constant side on the left.
        "SELECT f.k, f.v, d.label FROM dim d JOIN fact f ON d.k = f.k",
        # Duplicates on both sides: the join multiplies rows, and the added equality must
        # not change that count.
        "SELECT count(*) AS n FROM fact f JOIN dim d ON f.k = d.k",
        # An aggregate over the joined result, where a dropped row would show up as a sum.
        "SELECT sum(f.v) AS s, count(*) AS n FROM fact f JOIN dim d ON f.k = d.k",
        # Non-constant dimension: the rule declines, and the answer is the control.
        "SELECT f.k, f.v, d.label FROM fact f JOIN dim2 d ON f.k = d.k",
        # Outer joins keep their unmatched rows, so the constraint must NOT transfer. A rule
        # that fired here would silently drop the null and the non-matching fact rows.
        "SELECT f.k, f.v, d.label FROM fact f LEFT JOIN dim d ON f.k = d.k",
        "SELECT f.k, f.v, d.label FROM dim d LEFT JOIN fact f ON d.k = f.k",
        "SELECT f.k, f.v, d.label FROM fact f FULL JOIN dim d ON f.k = d.k",
        # A three-way join, so the inference has to compose rather than fire once.
        "SELECT f.k, f.v FROM fact f JOIN dim d ON f.k = d.k JOIN dim2 e ON f.k = e.k",
    ],
)
def test_constant_key_inference_preserves_results(duck, star, query):
    assert_same(bt.sql(query, dim=DIM, fact=FACT, dim2=DIM2).collect(), duck.sql(query))


@pytest.mark.differential
def test_the_rule_actually_fires_and_only_where_it_should(tmp_path):
    """Pin the rewrite itself — otherwise the equality tests above pass vacuously.

    Written against Parquet rather than in-memory tables on purpose: the constant has to be
    *proved*, and the proof this rule consumes is a footer's ``min == max``, which an
    in-memory source does not carry. Counting firings rather than pattern-matching the plan
    keeps the assertion about the rule instead of about explain formatting.

    The `dim2` half is what makes it a test of the rule: a two-valued dimension proves nothing
    and must produce no firing at all, so a version that invented predicates from thin air
    fails here even though every result-equality test above would still pass.
    """
    from batcher.kyber.registry import DEFAULT_REGISTRY

    pq.write_table(DIM, tmp_path / "dim.parquet")
    pq.write_table(DIM2, tmp_path / "dim2.parquet")
    pq.write_table(FACT, tmp_path / "fact.parquet")

    rule = next(
        r for r in DEFAULT_REGISTRY.rules() if r.name == "infer_join_predicate_from_constant_key"
    )
    original, fired = rule.node_fn, []

    def counting(node, ctx):
        out = original(node, ctx)
        if out is not None and out is not node:
            fired.append(node)
        return out

    object.__setattr__(rule, "node_fn", counting)
    try:
        session = bt.Session()
        for name in ("dim", "dim2", "fact"):
            session.register(name, bt.read.parquet(str(tmp_path / f"{name}.parquet")))

        session.sql("SELECT f.k, f.v FROM fact f JOIN dim d ON f.k = d.k").explain()
        assert len(fired) == 1, f"constant dimension must infer the key equality: {fired}"

        fired.clear()
        session.sql("SELECT f.k, f.v FROM fact f JOIN dim2 d ON f.k = d.k").explain()
        assert fired == [], f"a two-valued dimension proves nothing: {fired}"
    finally:
        object.__setattr__(rule, "node_fn", original)

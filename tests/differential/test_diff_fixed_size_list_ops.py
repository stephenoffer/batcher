"""The `.list` namespace over a fixed-size list (an embedding column) — vs DuckDB.

A vector column is a `FixedSizeList` — that is how Arrow and Parquet store one, what
`pa.list_(pa.float32(), n)` builds, and what DuckDB's `ARRAY` type maps to. Half the
`.list` namespace accepted it and half rejected it, on the *same column*:

* accepted: `sum`, `l2_norm`, `normalize`, `softmax`, `dot`, `cosine_similarity`, `sort`,
  `reverse`, `unique`, `arg_sort`, `cum_sum`, `diff`, `median`, `n_unique`;
* rejected with `expected a List argument, got FixedSizeList`: `get`, `slice`, `contains`,
  `position`, `first`, `last`, `concat`, `intersect`, `transform`, `filter`, `join`.

So an embedding could be normalized and summed but not **subscripted** — `e[0]` on a
vector column failed. The split was not a design decision; the vector kernels had grown a
coercion helper (`list_ops::coerce::as_var_list`) and the indexing half never used it.
`require_list` now routes through the same helper, so `.list` means one thing for both
encodings.

The fixture is deliberately a `float32` fixed-size list rather than a float64 one, because
that is the width an embedding actually arrives in and it exercises the widening cast
rather than a no-op.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher import col

VECTOR = pa.list_(pa.float32(), 3)

# (name, Batcher builder, the DuckDB query over the same values as a plain LIST)
CASES = [
    ("get", lambda: col("e").list.get(0), "SELECT k, e[1] r FROM t ORDER BY k"),
    ("first", lambda: col("e").list.first(), "SELECT k, list_first(e) r FROM t ORDER BY k"),
    ("last", lambda: col("e").list.last(), "SELECT k, list_last(e) r FROM t ORDER BY k"),
    ("slice", lambda: col("e").list.slice(0, 2), "SELECT k, e[1:2] r FROM t ORDER BY k"),
    (
        "position",
        lambda: col("e").list.position(1.0),
        "SELECT k, list_position(e, 1.0) r FROM t ORDER BY k",
    ),
    (
        "contains",
        lambda: col("e").list.contains(1.0),
        "SELECT k, list_contains(e, 1.0) r FROM t ORDER BY k",
    ),
]


@pytest.fixture
def vectors(duck):
    """The same values twice: fixed-size for Batcher, variable-size for the oracle.

    DuckDB's list functions take a `LIST`, so the comparison table is the plain encoding —
    the claim being tested is that Batcher answers *identically* for the fixed-size one.
    """
    fixed = pa.table({"k": [0, 1], "e": pa.array([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]], type=VECTOR)})
    variable = pa.table({"k": [0, 1], "e": [[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]]})
    duck.register("t", variable)
    return fixed


@pytest.mark.differential
@pytest.mark.parametrize(("name", "build", "query"), CASES)
def test_a_fixed_size_list_answers_what_a_list_answers(duck, vectors, name, build, query):
    got = bt.from_arrow(vectors).select(k=col("k"), r=build()).sort("k").collect()
    assert_same_ordered(got, duck.sql(query))


@pytest.mark.differential
def test_an_embedding_can_be_subscripted(vectors):
    """The headline regression: `e[0]` on a vector column raised `expected a List
    argument, got FixedSizeList` while `e` could be normalized in the same query."""
    rows = bt.from_arrow(vectors).select(r=col("e").list.get(0)).to_pydict()["r"]
    assert rows == [3.0, 6.0]


@pytest.mark.differential
def test_the_halves_of_the_namespace_agree_on_one_column(vectors):
    """The inconsistency stated directly: a reduction and an index on the same column."""
    out = bt.from_arrow(vectors).select(
        norm=col("e").list.l2_norm(),  # was accepted
        head=col("e").list.get(0),  # was rejected
    )
    rows = out.to_pydict()
    assert rows["head"] == [3.0, 6.0]
    assert rows["norm"][0] == pytest.approx(14.0**0.5)


@pytest.mark.differential
def test_a_variable_list_is_unaffected(duck):
    """The negative test: coercing must not change the already-working encoding."""
    t = pa.table({"k": [0, 1], "e": [[3.0, 1.0, 2.0], [6.0, 5.0]]})
    duck.register("v", t)
    query = "SELECT k, e[1] r FROM v ORDER BY k"
    got = bt.from_arrow(t).select(k=col("k"), r=col("e").list.get(0)).sort("k").collect()
    assert_same_ordered(got, duck.sql(query))


@pytest.mark.differential
def test_a_non_list_argument_still_names_the_function(vectors):
    """Widening must not swallow a genuine type error."""
    with pytest.raises(Exception, match=r"list|List"):
        bt.from_pydict({"n": [1, 2]}).select(r=col("n").list.get(0)).collect()

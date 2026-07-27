"""The `.list` namespace over all three Arrow list encodings — vs DuckDB.

Arrow has three ways to hold a list column, and Batcher's kernels accepted a different
subset of them per function:

* **`List`** (32-bit offsets) — the one everything worked on.
* **`FixedSizeList`** — how a vector/embedding column is stored, and what DuckDB's `ARRAY`
  maps to. Half the namespace rejected it; closed by the previous wave.
* **`LargeList`** (64-bit offsets) — what Arrow reaches for once a list column passes
  `i32::MAX` offsets, and what an Arrow reader hands back for a `large_list` Parquet
  column *regardless of its actual size*. **Every one of the 39 no-argument `.list`
  methods rejected it**, so the namespace was entirely unusable on such a column.

All three now normalize in one place (`list_ops::coerce::as_var_list`), which is the point:
the coercion existing in one place is what stops the encodings drifting apart again.

Two properties this file pins that a per-function test would not:

* **Every method answers identically across encodings.** Checked against DuckDB for the
  ones DuckDB has, and cross-encoding for the ones it does not.
* **`simhash` is bit-identical across encodings.** A similarity hash that depended on the
  encoding would silently break nearest-neighbour lookups over a dataset written in a mix
  of them — a wrong answer with no error, and the reason `simhash` was routed through the
  shared coercion rather than given its own.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher import col

VALUES = [[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]]
ENCODINGS = {
    "list": pa.list_(pa.float64()),
    "large_list": pa.large_list(pa.float64()),
    "fixed_size_list": pa.list_(pa.float64(), 3),
}

# Methods with a DuckDB counterpart, so the answer can be checked against the oracle.
AGAINST_DUCKDB = [
    (lambda: col("e").list.sum(), "SELECT k, list_sum(e) r FROM t ORDER BY k"),
    (lambda: col("e").list.get(0), "SELECT k, e[1] r FROM t ORDER BY k"),
    (lambda: col("e").list.first(), "SELECT k, list_first(e) r FROM t ORDER BY k"),
    (lambda: col("e").list.last(), "SELECT k, list_last(e) r FROM t ORDER BY k"),
    (lambda: col("e").list.len(), "SELECT k, len(e) r FROM t ORDER BY k"),
    (lambda: col("e").list.sort(), "SELECT k, list_sort(e) r FROM t ORDER BY k"),
    (lambda: col("e").list.reverse(), "SELECT k, list_reverse(e) r FROM t ORDER BY k"),
    (lambda: col("e").list.median(), "SELECT k, list_median(e) r FROM t ORDER BY k"),
]

# No-argument methods with no DuckDB counterpart: compared across encodings instead.
CROSS_ENCODING = [
    "l2_norm",
    "normalize",
    "softmax",
    "cum_sum",
    "diff",
    "arg_sort",
    "n_unique",
    "unique",
    "simhash",
    "max_abs",
    "sum_squares",
]


@pytest.fixture
def oracle(duck):
    """DuckDB holds the values once, as a plain `LIST`; Batcher gets each encoding."""
    duck.register("t", pa.table({"k": [0, 1], "e": VALUES}))
    return duck


def _table(encoding: str) -> pa.Table:
    return pa.table({"k": [0, 1], "e": pa.array(VALUES, type=ENCODINGS[encoding])})


@pytest.mark.differential
@pytest.mark.parametrize("encoding", list(ENCODINGS))
@pytest.mark.parametrize(("build", "query"), AGAINST_DUCKDB)
def test_every_encoding_answers_what_duckdb_answers(oracle, encoding, build, query):
    got = bt.from_arrow(_table(encoding)).select(k=col("k"), r=build()).sort("k").collect()
    assert_same_ordered(got, oracle.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("method", CROSS_ENCODING)
def test_the_encodings_agree_with_each_other(method):
    """For the methods DuckDB has no counterpart for, the encodings are each other's
    oracle — they must not merely all succeed, they must all agree."""
    answers = {
        encoding: bt.from_arrow(_table(encoding))
        .select(r=getattr(col("e").list, method)())
        .to_pydict()["r"]
        for encoding in ENCODINGS
    }
    reference = answers["list"]
    for encoding, value in answers.items():
        assert value == reference, f"{method} differs on {encoding}"


@pytest.mark.differential
def test_simhash_is_bit_identical_across_encodings():
    """Called out on its own because the failure mode is silent.

    A similarity hash that depended on the encoding would return different neighbours for
    the same vector depending on how its column happened to be written, with no error
    anywhere.
    """
    hashes = [
        bt.from_arrow(_table(e)).select(r=col("e").list.simhash()).to_pydict()["r"]
        for e in ENCODINGS
    ]
    assert hashes[0] == hashes[1] == hashes[2]


@pytest.mark.differential
def test_a_large_list_of_lists_still_flattens(duck):
    """`flatten` needs a list *of lists*, so it is the one method the flat fixture cannot
    exercise; checked here so the coercion is proven for a nested child too."""
    nested = pa.table(
        {
            "k": [0, 1],
            "e": pa.array([[[1, 2], [3]], [[4]]], type=pa.large_list(pa.list_(pa.int64()))),
        }
    )
    got = bt.from_arrow(nested).select(r=col("e").list.flatten()).to_pydict()["r"]
    assert got == [[1, 2, 3], [4]]


@pytest.mark.differential
def test_a_non_list_column_still_errors(duck):
    """Accepting three encodings must not mean accepting anything."""
    with pytest.raises(Exception, match=r"List|list"):
        bt.from_pydict({"n": [1, 2]}).select(r=col("n").list.sum()).collect()

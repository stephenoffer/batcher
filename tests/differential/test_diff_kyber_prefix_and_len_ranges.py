"""`prefix_predicates_to_range` and `len_zero_to_empty_string` preserve results — vs DuckDB.

`like_prefix_to_range` has turned `LIKE 'abc%'` into `col >= 'abc' AND col < 'abd'` for a
while. These two rules give the *other* spellings of the same predicates the same
treatment:

* `starts_with(col, 'abc')` and `substr(col, 1, 3) = 'abc'` — the DataFrame spellings,
  which are what a Batcher user actually writes, and which scanned every row group while
  the SQL form skipped them;
* `len(col) = 0` → `col = ''`, which unwraps the column so zone maps, blooms and source
  pushdown can see it at all.

The asymmetry was invisible precisely because every form returned the right rows. So this
file checks the rows, and `tests/unit/test_kyber_prefix_len_rules.py` checks that the
range actually appears.

The fixture carries the boundary cases a prefix range gets wrong if the bound is off by
one: a value equal to the prefix, one just past it, one just below, the empty string, and
a null.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

PREDICATES = [
    ("s LIKE 'ab%'", lambda: col("s").str.starts_with("ab")),
    ("substr(s, 1, 2) = 'ab'", lambda: col("s").str.substr(1, 2) == "ab"),
    ("substr(s, 1, 3) = 'abc'", lambda: col("s").str.substr(1, 3) == "abc"),
    # A literal whose length disagrees with the substring length is not a prefix test;
    # it is unsatisfiable, and must stay that way rather than becoming a range.
    ("substr(s, 1, 2) = 'abc'", lambda: col("s").str.substr(1, 2) == "abc"),
    ("length(s) = 0", lambda: col("s").str.len() == 0),
    ("length(s) <> 0", lambda: col("s").str.len() != 0),
    ("NOT (s LIKE 'ab%')", lambda: ~col("s").str.starts_with("ab")),
    (
        "s LIKE 'ab%' AND length(s) <> 0",
        lambda: col("s").str.starts_with("ab") & (col("s").str.len() != 0),
    ),
]


@pytest.fixture
def strings(duck):
    t = pa.table(
        {
            "s": [
                "ab",  # equal to the prefix
                "abc",  # inside the range
                "abd",  # inside the range
                "ac",  # just past the upper bound
                "aa",  # just below the lower bound
                "",  # empty
                None,  # null
                "ABC",  # case matters
            ]
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(("sql", "build"), PREDICATES)
def test_filter_result_is_unchanged(duck, strings, sql, build):
    assert_same(
        bt.from_arrow(strings).filter(build()).collect(), duck.sql(f"SELECT s FROM t WHERE {sql}")
    )


@pytest.mark.differential
@pytest.mark.parametrize(("sql", "build"), PREDICATES)
def test_the_predicate_as_a_projected_value_keeps_its_nulls(duck, strings, sql, build):
    """A filter cannot distinguish NULL from FALSE — both drop the row. Projecting can,
    and a range built from a null-propagating predicate must stay null-propagating."""
    out = bt.from_arrow(strings).select(s=col("s"), r=build()).collect()
    assert_same(out, duck.sql(f"SELECT s, ({sql}) r FROM t"))


@pytest.mark.differential
def test_the_prefix_range_agrees_with_the_like_form_it_mirrors(duck, strings):
    """The three spellings must select exactly the same rows, which is the whole claim."""
    ds = bt.from_arrow(strings)
    by_starts_with = ds.filter(col("s").str.starts_with("ab")).to_pydict()["s"]
    by_substr = ds.filter(col("s").str.substr(1, 2) == "ab").to_pydict()["s"]
    by_like = ds.filter(col("s").str.like("ab%")).to_pydict()["s"]
    assert by_starts_with == by_substr == by_like == ["ab", "abc", "abd"]

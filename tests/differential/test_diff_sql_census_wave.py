"""SQL names the function census reported missing, and what closing each one costs.

The DuckDB, Spark and Daft censuses (`docs/architecture/internals/competitor_parity_census.md`)
re-run here reported 104 names the engine could not reach. Most needed a kernel; this file
covers the ones that needed only a *composition* of kernels the engine already had, which
is the cheapest class to close and the easiest to close wrongly — a plausible composition
that disagrees with the reference on one input is worse than the clear refusal it replaced.

So every case runs the argument as a **column**, not a literal, so constant folding cannot
answer it without the kernel; and the inputs are chosen for the edge each composition can
get wrong rather than for the happy path:

* `signbit` is not `x < 0` (it is true for `-0.0`) and not `1/x < 0` either (that reads
  `-inf` as false), so the negative zero and both infinities are here;
* `try_divide` is Spark's null-on-zero-divisor, which the engine's IEEE `/` answers with
  an infinity, so the zero divisor is the whole test;
* `printf` has no per-conversion formatting, so a width or precision must **refuse**
  rather than silently drop the padding;
* `quote` escapes with a backslash the way Spark does, not by doubling the way a SQL
  literal does, and it has to escape before it wraps.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


@pytest.fixture
def nums(duck):
    tbl = pa.table(
        {
            "x": pa.array(
                [1.5, -1.5, 0.0, -0.0, float("nan"), float("inf"), float("-inf"), None],
                pa.float64(),
            ),
            "a": pa.array([3.0, 3.0, 1.0, -1.0, 0.0, 2.0, 5.0, None], pa.float64()),
            "b": pa.array([2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, None], pa.float64()),
        }
    )
    duck.register("nums", tbl)
    return tbl


def test_signbit_matches_duckdb_on_negative_zero_and_the_infinities(duck, nums):
    """The two inputs that separate `signbit` from `x < 0` and from `1/x < 0`."""
    out = bt.sql("SELECT signbit(x) AS s FROM nums", nums=bt.from_arrow(nums))
    assert_same(out.collect(), duck.sql("SELECT signbit(x) AS s FROM nums"))


def test_signbit_reads_negative_zero_as_signed_and_nan_as_not():
    """Spelled out, because `assert_same` against DuckDB cannot show which row is which."""
    tbl = pa.table({"x": pa.array([-0.0, 0.0, float("nan"), float("-inf")], pa.float64())})
    out = bt.sql("SELECT signbit(x) AS s FROM t", t=bt.from_arrow(tbl)).to_pydict()
    assert out["s"] == [True, False, False, True]


def test_list_select_gathers_by_one_based_position(duck):
    """DuckDB's `list_select` indexes from 1, allows repeats, and may reorder."""
    tbl = pa.table({"l": [[10, 20, 30], [1, 2, 3]], "i": [[1, 3], [3, 1, 1]]})
    duck.register("t", tbl)
    out = bt.sql(
        "SELECT list_select(l, i) AS a, array_select(l, i) AS b FROM t", t=bt.from_arrow(tbl)
    )
    assert_same(
        out.collect(), duck.sql("SELECT list_select(l, i) AS a, array_select(l, i) AS b FROM t")
    )


def test_format_interpolates_duckdbs_brace_template(duck):
    tbl = pa.table({"k": ["a", "b"], "v": pa.array([1, 2], pa.int64())})
    duck.register("t", tbl)
    out = bt.sql("SELECT format('{} = {}', k, v) AS f FROM t", t=bt.from_arrow(tbl))
    assert_same(out.collect(), duck.sql("SELECT format('{} = {}', k, v) AS f FROM t"))


def test_printf_and_format_string_reach_the_same_builder():
    """The three spellings differ only in how the template marks its holes."""
    src = bt.from_pydict({"k": ["a"], "v": [3]})
    duckdb_style = bt.sql("SELECT format('{}-{}', k, v) AS f FROM t", t=src).to_pydict()
    printf_style = bt.sql("SELECT printf('%s-%d', k, v) AS f FROM t", t=src).to_pydict()
    spark_style = bt.sql(
        "SELECT format_string('%s-%d', k, v) AS f FROM t", t=src, dialect="spark"
    ).to_pydict()
    assert duckdb_style == printf_style == spark_style == {"f": ["a-3"]}


def test_printf_truncates_a_percent_d_the_way_c_does():
    """`%d` is an integer conversion, so a float argument truncates toward zero."""
    src = bt.from_pydict({"v": [3.7, -3.7]})
    assert bt.sql("SELECT printf('%d', v) AS f FROM t", t=src).to_pydict() == {"f": ["3", "-3"]}


def test_printf_escapes_a_doubled_percent():
    src = bt.from_pydict({"v": [50]})
    assert bt.sql("SELECT printf('%d%% done', v) AS f FROM t", t=src).to_pydict() == {
        "f": ["50% done"]
    }


def test_printf_refuses_a_width_or_precision_rather_than_dropping_it():
    """A padded conversion must raise: an unpadded answer is a plausible wrong string."""
    src = bt.from_pydict({"v": [1.5]})
    with pytest.raises(NotImplementedError, match="width or precision"):
        bt.sql("SELECT printf('%05.2f', v) AS f FROM t", t=src).to_pydict()


def test_printf_refuses_a_conversion_it_cannot_reproduce():
    src = bt.from_pydict({"v": [255]})
    with pytest.raises(NotImplementedError, match="not supported"):
        bt.sql("SELECT printf('%x', v) AS f FROM t", t=src).to_pydict()


def test_quote_escapes_with_a_backslash_the_way_spark_does():
    """Spark precedes an embedded quote with a backslash; a SQL literal doubles it."""
    src = bt.from_pydict({"s": ["Don't", "a'b'c", "plain", None]})
    out = bt.sql("SELECT quote(s) AS q FROM t", t=src, dialect="spark").to_pydict()
    assert out["q"] == ["'Don\\'t'", "'a\\'b\\'c'", "'plain'", None]


def test_try_divide_is_null_on_a_zero_divisor_where_plain_division_is_infinite(nums):
    """The one behaviour `try_divide` has that `/` does not."""
    src = bt.from_arrow(nums)
    plain = bt.sql("SELECT a / b AS r FROM nums", nums=src).to_pydict()["r"]
    guarded = bt.sql("SELECT try_divide(a, b) AS r FROM nums", nums=src, dialect="spark")
    guarded = guarded.to_pydict()["r"]
    assert plain[1] == float("inf") and guarded[1] is None
    # Where the divisor is non-zero the two agree exactly.
    assert plain[0] == guarded[0] == 1.5
    assert guarded[-1] is None  # a null divisor stays null, not an error


def test_try_adds_siblings_still_refuse_rather_than_wrap():
    """`try_add`/`try_subtract`/`try_multiply` differ from `+`/`-`/`*` only on overflow.

    The engine wraps there rather than raising, so there is nothing for a composition to
    test for — and returning the wrapped number where Spark returns NULL would be exactly
    the plausible-wrong-answer the census exists to catch. They keep refusing.
    """
    src = bt.from_pydict({"a": [1], "b": [2]})
    for name in ("try_add", "try_subtract", "try_multiply"):
        with pytest.raises(NotImplementedError):
            bt.sql(f"SELECT {name}(a, b) AS r FROM t", t=src, dialect="spark").to_pydict()

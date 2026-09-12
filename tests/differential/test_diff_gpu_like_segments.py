"""Multi-wildcard `LIKE` on the device translator, against DuckDB and against the engine.

`LIKE '%special%requests%'` is TPC-H q13's filter and q16's, and until now the device tier
declined it: two literal segments between wildcards is not a shape any single `.str` call
expresses. Both queries paid a full round trip to a device to find that out — and q17-shaped
plans paid two, one for the fan-out and one for the single-worker retry.

The reduction has two ways to be silently wrong, and both are checked here rather than argued:

* **Order.** `contains('a') & contains('b')` accepts `"ba"`; `LIKE '%a%b%'` does not.
* **A newline.** The regex spelling `.*a.*b.*` reads the same in the engine's Rust, in pandas
  and in cuDF, and all three exclude `\\n` from `.` — while SQL's `%` spans one. That is a
  wrong answer on a device that a pandas replay of a regex-based translation would also
  produce, so no amount of host testing would have found it.

DuckDB is the oracle for the semantics; the translator's own output is checked against the
engine's for the same rows, which is the device tier's contract (same rows, same names, same
types).
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher.api.terminal.gpu_backend.verify import compare_results
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain

pytestmark = pytest.mark.differential

#: Values that separate ordered matching from unordered, and that carry the newline `%` spans.
VALUES = [
    "special requests",
    "special X requests",
    "requests special",
    "ab",
    "ba",
    "aXbYc",
    "abcabc",
    "ab\ncd",
    "",
    None,
]

PATTERNS = ["%a%b%", "%special%requests%", "a%b%c", "a%b", "%a%b", "%ab%cd%", "%a%a%"]


@pytest.fixture
def t(duck):
    tbl = pa.table({"s": pa.array(VALUES, type=pa.string())})
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize("pattern", PATTERNS)
def test_multi_segment_like_matches_duckdb(duck, t, pattern):
    out = bt.from_arrow(t).select(m=col("s").str.like(pattern)).collect()
    assert_same(out, duck.sql(f"SELECT s LIKE '{pattern}' AS m FROM t"))


@pytest.mark.parametrize("pattern", PATTERNS)
def test_the_device_translator_agrees_with_the_engine(t, pattern):
    """The translated form, run through the same backend a device runs, against the engine."""
    ds = bt.from_arrow(t).select(m=col("s").str.like(pattern))
    matched = gpu_plan_ops(ds._plan)
    assert matched is not None, f"{pattern!r} must reach the translator for this to test it"
    _scan, ops = matched
    be = DfBackend(pd)
    # `run_chain` takes the Arrow table, exactly as the shard task hands it one.
    translated = be.to_arrow(run_chain(t, ops, be))
    # `compare_results` is what `gpu_shadow_verify` uses at runtime, so this test holds the
    # translation to the same standard a real device result is held to — same rows *and* the
    # same column types, which is the device tier's characteristic defect.
    why = compare_results(translated, ds.collect())
    assert why is None, why


def test_a_wildcard_spans_a_newline_on_the_translator(t):
    """The divergence a regex reduction produces and a host replay of it would not reveal."""
    ds = bt.from_arrow(t).filter(col("s").str.like("%ab%cd%"))
    assert ds.to_pydict()["s"] == ["ab\ncd"]


def test_order_is_respected_end_to_end(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.like("%a%b%")).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s LIKE '%a%b%'"))
    assert "ba" not in out.column("s").to_pylist()

"""Differential tests vs DuckDB for `.str` trim whitespace class and `concat_ws` nulls.

Two gaps the earlier string suites missed:

* argument-less ``trim``/``ltrim``/``rtrim`` must strip only the Unicode ``Zs``
  space-separator category (like DuckDB), NOT the C0 control whitespace
  (tab/newline/CR/…) that Rust's ``str::trim`` also removes; and
* ``concat_ws`` over rows where every value argument is NULL must yield the empty
  string, not NULL (DuckDB skips NULL args and returns ``''`` when none remain).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col


@pytest.mark.differential
def test_trim_strips_only_space_separators(duck):
    # Rows mixing control whitespace (tab/newline/CR/VT/FF) with Zs separators
    # (ASCII space, NBSP U+00A0, ideographic space U+3000). DuckDB keeps the
    # controls and strips only the space separators.
    tbl = pa.table(
        {
            "s": pa.array(
                [
                    "\t\n",  # pure control whitespace -> unchanged
                    " \u00a0x\u3000 ",  # NBSP / ideographic space around 'x'
                    "\tx\t",  # tabs kept
                    "  hi  ",  # plain ASCII spaces
                    "\r\nline\r\n",  # CR/LF kept
                    "",
                    None,
                ]
            )
        }
    )
    duck.register("t", tbl)
    out = (
        bt.from_arrow(tbl)
        .select(
            tr=col("s").str.trim(),
            lt=col("s").str.lstrip(),
            rt=col("s").str.rstrip(),
        )
        .collect()
    )
    expected = duck.sql("SELECT trim(s) tr, ltrim(s) lt, rtrim(s) rt FROM t")
    assert_same(out, expected)


@pytest.mark.differential
def test_concat_ws_all_null_args_is_empty_string(duck):
    tbl = pa.table(
        {
            "a": pa.array(["x", None, None, "p"], pa.string()),
            "b": pa.array(["1", None, "q", None], pa.string()),
        }
    )
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).select(v=bt.concat_ws("-", col("a"), col("b"))).collect()
    expected = duck.sql("SELECT concat_ws('-', a, b) v FROM t")
    assert_same(out, expected)

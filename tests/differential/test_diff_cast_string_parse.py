"""String-parsing casts must match DuckDB: whitespace trimming and the bool set.

Two DuckDB divergences in Batcher's `VARCHAR → X` casts, both pinned here:

* **Whitespace.** DuckDB trims leading/trailing whitespace before parsing a string
  into a numeric or temporal value (``'  12  '::BIGINT`` = 12, ``' 3.14 '::DOUBLE`` =
  3.14, ``' 2024-01-05 '::DATE``). Arrow's kernel does not, so the strict cast errored
  and ``TRY_CAST`` silently NULLed the padded value — data loss on the advertised
  safe-ingest path.
* **Booleans.** Arrow's ``Utf8 → Boolean`` kernel is far looser than DuckDB: it trims
  whitespace, accepts ``on``/``off``, and even matches a *prefix* (``'tru'`` → true),
  so ``TRY_CAST('tru' AS BOOLEAN)`` returned ``true`` where DuckDB returns NULL — a
  silent wrong non-null value. DuckDB accepts exactly (case-insensitive, no trimming)
  ``{true, t, 1, yes, y}`` / ``{false, f, 0, no, n}``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same

pytestmark = pytest.mark.differential


def test_try_cast_string_to_int_trims_whitespace(duck):
    """``TRY_CAST(' 12 ' AS BIGINT)`` = 12, not NULL — every C-`isspace` char, both sides."""
    vals = ["  12  ", "\t-7\n", " +5", "\x0b\x0c 9 \r", "   ", "", "1 2", "abc", None]
    t = pa.table({"s": pa.array(vals, pa.string())})
    duck.register("wsi", t)
    out = bt.from_arrow(t).select(o=col("s").try_cast("int64")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(s AS BIGINT) o FROM wsi"))


def test_try_cast_string_to_double_trims_whitespace(duck):
    """``TRY_CAST(' 3.14 ' AS DOUBLE)`` = 3.14; padded floats parse like DuckDB."""
    vals = [" 2.75 ", "  -0.5", "\t1e3\n", "  ", "", " nan ", "1 2", None]
    t = pa.table({"s": pa.array(vals, pa.string())})
    duck.register("wsf", t)
    out = bt.from_arrow(t).select(o=col("s").try_cast("float64")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(s AS DOUBLE) o FROM wsf"))


def test_try_cast_string_to_date_trims_whitespace(duck):
    """``TRY_CAST(' 2024-01-05 ' AS DATE)`` parses the padded ISO date like DuckDB."""
    vals = [" 2024-01-05 ", "2024-01-05", "\t2020-12-31\n", "  ", "nope", None]
    t = pa.table({"s": pa.array(vals, pa.string())})
    duck.register("wsd", t)
    out = bt.from_arrow(t).select(o=col("s").try_cast("date32")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(s AS DATE) o FROM wsd"))


def test_strict_cast_padded_int_does_not_error(duck):
    """Strict ``CAST('  42  ' AS BIGINT)`` = 42 (padded value is valid, no raise)."""
    t = pa.table({"s": pa.array(["  42  ", "\t-3\n", " 7"], pa.string())})
    duck.register("wsp", t)
    out = bt.from_arrow(t).select(o=col("s").cast("int64")).collect()
    assert_same(out, duck.sql("SELECT CAST(s AS BIGINT) o FROM wsp"))


def test_try_cast_string_to_bool_matches_duckdb_set(duck):
    """String→bool accepts exactly DuckDB's set: no prefix, no `on`/`off`, no trimming."""
    vals = [
        "true", "TRUE", "t", "1", "yes", "Y",
        "false", "F", "0", "no", "n",
        " true ", "on", "off", "tru", "2", "", None,
    ]  # fmt: skip
    t = pa.table({"s": pa.array(vals, pa.string())})
    duck.register("wsb", t)
    out = bt.from_arrow(t).select(o=col("s").try_cast("bool")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(s AS BOOLEAN) o FROM wsb"))


def test_strict_cast_string_to_bool_errors_on_prefix():
    """Strict ``CAST('tru' AS BOOLEAN)`` errors — arrow would silently return ``true``."""
    t = pa.table({"s": pa.array(["tru"], pa.string())})
    with pytest.raises(Exception):  # noqa: B017 — engine raises on the invalid token
        bt.from_arrow(t).select(o=col("s").cast("bool")).collect()
